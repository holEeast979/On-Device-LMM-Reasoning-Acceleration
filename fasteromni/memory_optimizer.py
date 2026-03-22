"""显存优化模块（技术点 4）— 模型无关设计

分层递进策略：
- Layer 0 (P0)：PyTorch 分配器调优 — expandable_segments + max_split_size
- Layer 1 (P1)：LLM Prefill 前 defragment — gc.collect() + empty_cache()

两层均基于 PyTorch 标准 API，不依赖任何特定模型的方法名或内部结构，
可即插即用于任意视频多模态 LLM（Qwen2.5-Omni、LLaVA-Video、InternVL 等）。

核心思路：视觉编码器（ViT）处理不同帧数的视频时，产生不同尺寸的中间 tensor，
释放后在 PyTorch CUDA 缓存中留下碎片化的 reserved 块。LLM prefill 阶段需要
分配大块连续 KV cache 时失败，导致 OOM。

解法：
- Layer 0 从分配器层面减少碎片产生
- Layer 1 在 LLM prefill 前（即 ViT 碎片产生后）释放缓存，腾出连续空间

Usage:
    from fasteromni.memory_optimizer import configure_allocator, install_defrag_hook

    # Layer 0: 在模型加载前设置分配器
    configure_allocator()

    # Layer 1: 模型加载后安装 defrag hook（自动探测 LLM 子模块）
    hook = install_defrag_hook(model)
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ─── Layer 0: 分配器调优 ──────────────────────────────────

def configure_allocator(
    expandable_segments: bool = True,
    max_split_size_mb: int = 128,
) -> None:
    """配置 PyTorch CUDA 内存分配器（模型无关，即插即用）。

    必须在第一次 CUDA 分配之前调用（即 load_model() 之前）。

    Args:
        expandable_segments: 使用虚拟地址映射，segment 可扩展合并，
            直接消除外部碎片。PyTorch 2.1+ 支持。
        max_split_size_mb: 防止大 block 被分割成小碎片。
    """
    # 警告：如果 CUDA 已初始化，环境变量设置可能无效
    if torch.cuda.is_available():
        try:
            if torch.cuda.memory_allocated() > 0 or torch.cuda.memory_reserved() > 0:
                print(
                    "[memory-opt] WARNING: CUDA already has allocations. "
                    "Allocator config may not take effect. "
                    "Call configure_allocator() before any CUDA operation.",
                    flush=True,
                )
        except Exception:
            pass

    # 解析已有配置，逐 key 检查避免重复
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    existing_keys = set()
    if existing:
        for part in existing.split(","):
            key = part.split(":")[0].strip()
            if key:
                existing_keys.add(key)

    new_parts = []
    if expandable_segments and "expandable_segments" not in existing_keys:
        new_parts.append("expandable_segments:True")
    if max_split_size_mb > 0 and "max_split_size_mb" not in existing_keys:
        new_parts.append(f"max_split_size_mb:{max_split_size_mb}")

    if not new_parts:
        print(f"[memory-opt] PYTORCH_CUDA_ALLOC_CONF already configured: {existing}", flush=True)
        return

    conf = f"{existing},{','.join(new_parts)}" if existing else ",".join(new_parts)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
    print(f"[memory-opt] PYTORCH_CUDA_ALLOC_CONF={conf}", flush=True)


# ─── 显存快照工具 ──────────────────────────────────────────

@dataclass
class MemorySnapshot:
    """单次显存快照。"""
    label: str = ""
    timestamp: float = 0.0

    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    free_mb: float = 0.0  # reserved - allocated（可重用碎片）
    fragmentation_ratio: float = 0.0  # free / reserved

    device_total_mb: float = 0.0
    device_used_mb: float = 0.0


def take_snapshot(label: str = "", device: Optional[torch.device] = None) -> MemorySnapshot:
    """采集当前 CUDA 显存快照。"""
    snap = MemorySnapshot(label=label, timestamp=time.perf_counter())

    if not torch.cuda.is_available():
        return snap

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free = reserved - allocated

    snap.allocated_mb = allocated / (1024 * 1024)
    snap.reserved_mb = reserved / (1024 * 1024)
    snap.free_mb = free / (1024 * 1024)
    snap.fragmentation_ratio = (free / reserved) if reserved > 0 else 0.0

    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_mem
        snap.device_total_mb = total / (1024 * 1024)
        snap.device_used_mb = snap.reserved_mb
    except Exception:
        pass

    return snap


def print_snapshot(snap: MemorySnapshot) -> None:
    """打印显存快照到 stdout。"""
    frag_pct = snap.fragmentation_ratio * 100
    print(
        f"[memory-opt] {snap.label}: "
        f"allocated={snap.allocated_mb:.0f}MB, "
        f"reserved={snap.reserved_mb:.0f}MB, "
        f"fragmentation={frag_pct:.1f}%",
        flush=True,
    )


# ─── Layer 1: Defragment ──────────────────────────────────

def defragment(force: bool = False, threshold: float = 0.30) -> float:
    """释放 PyTorch CUDA 缓存中的碎片化内存。

    Args:
        force: 始终执行清理（忽略阈值）。
        threshold: 碎片率超过此值时才执行清理（默认 30%）。

    Returns:
        清理前的碎片率。
    """
    if not torch.cuda.is_available():
        return 0.0

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    frag_ratio = ((reserved - allocated) / reserved) if reserved > 0 else 0.0

    if force or frag_ratio > threshold:
        gc.collect()
        torch.cuda.empty_cache()

    return frag_ratio


# ─── LLM 子模块自动探测 ───────────────────────────────────

# 常见视频多模态 LLM 的 backbone 路径（按优先级排列）
_LLM_ATTR_PATHS: List[Tuple[str, ...]] = [
    # Qwen2.5-Omni: model.thinker.model
    ("thinker", "model"),
    # LLaVA-Video / LLaVA-OneVision: model.language_model
    ("language_model",),
    # InternVL2: model.language_model
    ("language_model",),
    # mPLUG-Owl / VideoChat: model.llama_model 或 model.model
    ("llama_model",),
    ("model",),
    # Phi-3-Vision: model.model
    ("model",),
]


def _find_llm_module(model: nn.Module) -> Optional[nn.Module]:
    """自动探测模型中的 LLM backbone 子模块。

    遍历常见的属性路径，返回第一个找到的 nn.Module。
    支持 Qwen2.5-Omni、LLaVA、InternVL、mPLUG-Owl 等主流架构。

    Args:
        model: 顶层模型对象。

    Returns:
        LLM backbone 子模块，找不到则返回 None。
    """
    for attr_path in _LLM_ATTR_PATHS:
        current = model
        found = True
        for attr in attr_path:
            if hasattr(current, attr):
                current = getattr(current, attr)
            else:
                found = False
                break
        if found and isinstance(current, nn.Module) and current is not model:
            return current
    return None


# ─── Layer 1: LLM Prefill Defrag Hook（模型无关）──────────

class PrefillDefragHook:
    """在 LLM prefill 前注入 defragment（模型无关，即插即用）。

    使用 PyTorch 标准的 register_forward_pre_hook 挂在 LLM backbone 上，
    仅在 prefill（首次 forward，seq_len > 1）时触发碎片清理。

    工作原理：
    - 所有视频 LLM 的推理流程都是 ViT → Projector → LLM
    - ViT 编码后产生碎片，LLM prefill 需要大块连续 KV cache
    - 在 LLM forward 入口处清理碎片，正好卡在两者之间
    - 仅 prefill 触发（seq_len > 1），decode 阶段（seq_len=1）跳过

    泛化性：
    - 不依赖任何模型特定的方法名
    - 通过 _find_llm_module() 自动探测 LLM 子模块
    - 也支持手动指定 llm_module 参数
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        llm_module: Optional[nn.Module] = None,
        force_defrag: bool = True,
        threshold: float = 0.30,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.force_defrag = force_defrag
        self.threshold = threshold
        self.verbose = verbose

        # 自动探测或使用指定的 LLM 模块
        if llm_module is not None:
            self._llm_module = llm_module
        else:
            self._llm_module = _find_llm_module(model)

        self._handle: Optional[torch.utils.hooks.RemovableHook] = None
        self._enabled = False

        # 统计
        self.defrag_count: int = 0
        self.total_defrag_ms: float = 0.0
        self.skipped_count: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """安装 prefill defrag hook。"""
        if self._enabled:
            return

        if self._llm_module is None:
            print(
                "[memory-opt] WARNING: Could not find LLM module. "
                "Defrag hook not installed. "
                "Pass llm_module= explicitly.",
                flush=True,
            )
            return

        def _pre_hook(module: nn.Module, args: Any, kwargs: Any) -> None:
            # 判断是否为 prefill（seq_len > 1）
            seq_len = self._detect_seq_len(args, kwargs)
            if seq_len <= 1:
                # decode 阶段，跳过
                return

            # Prefill 阶段，执行 defrag
            t0 = time.perf_counter()
            frag_ratio = defragment(
                force=self.force_defrag,
                threshold=self.threshold,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if self.force_defrag or frag_ratio > self.threshold:
                self.defrag_count += 1
                self.total_defrag_ms += elapsed_ms
                if self.verbose:
                    print(
                        f"[memory-opt] defrag before prefill: "
                        f"seq_len={seq_len}, "
                        f"frag={frag_ratio*100:.1f}%, "
                        f"cleanup={elapsed_ms:.1f}ms",
                        flush=True,
                    )
            else:
                self.skipped_count += 1

        self._handle = self._llm_module.register_forward_pre_hook(
            _pre_hook, with_kwargs=True
        )
        self._enabled = True

        if self.verbose:
            module_name = self._llm_module.__class__.__name__
            print(f"[memory-opt] PrefillDefragHook installed on {module_name}", flush=True)

    @staticmethod
    def _detect_seq_len(args: Any, kwargs: Any) -> int:
        """从 forward 参数中探测序列长度。

        兼容多种模型的 forward 签名：
        - input_ids: (batch, seq_len) — 大多数 LLM
        - inputs_embeds: (batch, seq_len, hidden) — 多模态融合后
        """
        # 优先从 kwargs 检测
        for key in ("input_ids", "inputs_embeds"):
            tensor = kwargs.get(key)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                if tensor.ndim == 2:
                    return int(tensor.shape[1])  # (batch, seq)
                if tensor.ndim == 3:
                    return int(tensor.shape[1])  # (batch, seq, hidden)

        # fallback: 从 positional args 检测
        if args:
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.ndim >= 2:
                    return int(arg.shape[1])

        return 0

    def disable(self) -> None:
        """移除 hook。"""
        if not self._enabled:
            return
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._enabled = False

    def stats(self) -> Dict[str, Any]:
        """返回 defrag 统计信息。"""
        return {
            "defrag_count": self.defrag_count,
            "total_defrag_ms": self.total_defrag_ms,
            "avg_defrag_ms": (
                self.total_defrag_ms / self.defrag_count
                if self.defrag_count > 0
                else 0.0
            ),
            "skipped_count": self.skipped_count,
        }


# ─── 便捷 API ──────────────────────────────────────────────

def install_defrag_hook(
    model: nn.Module,
    *,
    llm_module: Optional[nn.Module] = None,
    force_defrag: bool = True,
    threshold: float = 0.30,
    verbose: bool = True,
) -> PrefillDefragHook:
    """安装 LLM prefill defrag hook（模型无关，即插即用）。

    自动探测 LLM backbone 子模块，在 prefill 前注入碎片清理。
    支持 Qwen2.5-Omni、LLaVA-Video、InternVL 等主流视频多模态模型。

    Args:
        model: 顶层模型对象。
        llm_module: 手动指定 LLM 子模块（可选，默认自动探测）。
        force_defrag: 始终在 prefill 前执行 defrag（推荐 True）。
        threshold: 仅在 force_defrag=False 时生效。
        verbose: 打印日志。

    Returns:
        已安装的 PrefillDefragHook 实例。
    """
    hook = PrefillDefragHook(
        model,
        llm_module=llm_module,
        force_defrag=force_defrag,
        threshold=threshold,
        verbose=verbose,
    )
    hook.enable()
    return hook
