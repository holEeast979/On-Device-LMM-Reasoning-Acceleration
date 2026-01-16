"""
Spec: vllm-comparison
vLLM vs HuggingFace TTFT 对比实验

目的：证明 vLLM 主要优化 KV cache / 批处理，不优化 encoder/prefill 阶段
     在单用户单请求场景下，TTFT 与 HF 相当

限制：vLLM 对 Qwen2.5-Omni 只支持 Thinker (无音频)，所以对比只用视频输入

使用方法:
    # 仅测 HF
    python benchmark/run.py vllm-comparison \
        --model-dir /path/to/Qwen2.5-Omni-7B \
        --manifest /path/to/manifest.csv \
        --out-dir ./results \
        --backend hf

    # 仅测 vLLM
    python benchmark/run.py vllm-comparison \
        --model-dir /path/to/Qwen2.5-Omni-7B \
        --manifest /path/to/manifest.csv \
        --out-dir ./results \
        --backend vllm

    # 两者都测 (需要分两次运行，避免显存冲突)
    python benchmark/run.py vllm-comparison --backend hf ...
    python benchmark/run.py vllm-comparison --backend vllm ...
    python benchmark/specs/vllm_comparison.py --merge-results ./results/vllm-comparison
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.runner import BenchmarkRunner
from benchmark.unified_runner import UnifiedRunner


SPEC_NAME = "vllm-comparison"

# 分层采样配置: (min_sec, max_sec, category, count)
# 总计10个视频，动态帧采样2FPS，5090 32GB安全上限30s(60帧)
DURATION_BUCKETS = [
    (5, 12, "tier1", 2),      # 5-12s = 10-24帧
    (12, 18, "tier2", 3),     # 12-18s = 24-36帧
    (18, 24, "tier3", 3),     # 18-24s = 36-48帧
    (24, 32, "tier4", 2),     # 24-32s = 48-64帧
]


def register_subcommand(subparsers, common_parser) -> None:
    p = subparsers.add_parser(
        SPEC_NAME,
        parents=[common_parser],
        help="Compare vLLM vs HuggingFace TTFT for video-only inference"
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm", "both"],
        default="hf",
        help="Which backend to use (hf, vllm, or both - runs sequentially)"
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization (default 0.85)"
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM max model length (default 32768)"
    )
    p.add_argument(
        "--max-video-duration",
        type=float,
        default=300.0,
        help="Max video duration in seconds for selection (default 5min)"
    )
    p.add_argument(
        "--no-stratified",
        action="store_true",
        default=False,
        help="Disable stratified sampling, use random sampling instead"
    )
    p.set_defaults(_spec_run=run)


def get_video_duration_ffprobe(video_path: str) -> float:
    """使用 ffprobe 获取视频时长"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info["format"]["duration"])
    except Exception:
        pass
    return 0.0


def select_videos(
    args: argparse.Namespace,
    runner: BenchmarkRunner,
    max_duration: float = 32.0,
) -> List[Dict[str, Any]]:
    """
    分层采样选择视频样本
    
    按时长分桶，每个桶内随机采样指定数量，确保覆盖不同时长段
    """
    if not os.path.exists(str(args.manifest)):
        raise SystemExit(f"manifest not found: {args.manifest}")

    all_rows = runner.load_manifest_csv(str(args.manifest), n_samples=0, seed=int(args.seed))
    
    # 收集所有可用视频及其时长
    usable = []
    for r in all_rows:
        raw_v = r.get("video_path", None)
        v = UnifiedRunner.resolve_media_path(str(args.manifest), None if raw_v is None else str(raw_v))
        
        if not v or not os.path.exists(v):
            continue
        
        duration = r.get("duration", 0)
        if not duration or duration <= 0:
            duration = get_video_duration_ffprobe(v)
        
        if duration <= 0 or duration > max_duration:
            continue
        
        rr = dict(r)
        rr["video_path"] = v
        rr["duration"] = float(duration)
        usable.append(rr)
    
    if not usable:
        raise SystemExit("no usable videos found")
    
    # 分层采样
    n_samples = int(getattr(args, "n_samples", 10))
    use_stratified = not getattr(args, "no_stratified", False)
    
    if not use_stratified or n_samples <= 0:
        # 回退到随机采样
        if n_samples <= 0 or n_samples >= len(usable):
            return usable
        df = pd.DataFrame(usable)
        sub = df.sample(n=n_samples, random_state=int(args.seed))
        return sub.to_dict(orient="records")
    
    # 按时长分桶（按视频文件去重，不同问题算同一视频）
    rng = np.random.default_rng(int(args.seed))
    selected = []
    used_video_files = set()  # 按视频文件去重
    
    # 先按视频文件去重
    unique_videos = {}
    for v in usable:
        vpath = v["video_path"]
        if vpath not in unique_videos:
            unique_videos[vpath] = v
    usable_unique = list(unique_videos.values())
    
    print(f"\nStratified sampling from {len(usable_unique)} unique videos (total {len(usable)} samples):")
    
    for min_sec, max_sec, category, count in DURATION_BUCKETS:
        if min_sec > max_duration:
            continue
        
        # 找到该桶内的视频（已去重）
        bucket_videos = [
            v for v in usable_unique
            if min_sec <= v["duration"] < max_sec
            and v["video_path"] not in used_video_files
        ]
        
        if not bucket_videos:
            print(f"  [{category}] {min_sec}-{max_sec}s: no videos available")
            continue
        
        # 随机采样
        n_pick = min(count, len(bucket_videos))
        indices = rng.choice(len(bucket_videos), size=n_pick, replace=False)
        
        for idx in indices:
            v = bucket_videos[idx]
            v["category"] = category
            selected.append(v)
            used_video_files.add(v["video_path"])
        
        print(f"  [{category}] {min_sec}-{max_sec}s: selected {n_pick}/{count} (available: {len(bucket_videos)})")
    
    # 如果分层采样数量不足，从剩余视频补充（仍需满足max_duration限制）
    if len(selected) < n_samples:
        remaining = [
            v for v in usable_unique 
            if v["video_path"] not in used_video_files 
            and v["duration"] <= max_duration  # 确保补充视频也在时长限制内
        ]
        if remaining:
            n_extra = min(n_samples - len(selected), len(remaining))
            indices = rng.choice(len(remaining), size=n_extra, replace=False)
            for idx in indices:
                v = remaining[idx]
                v["category"] = "extra"
                selected.append(v)
            print(f"  [extra] added {n_extra} more videos (duration <= {max_duration}s) to reach {len(selected)} total")
    
    # 按时长排序
    selected.sort(key=lambda x: x["duration"])
    
    print(f"Total selected: {len(selected)} videos")
    for v in selected:
        print(f"  - {v['duration']:.1f}s ({v.get('category', 'unknown')}): {os.path.basename(v['video_path'])}")
    
    return selected


def run_hf_backend(
    args: argparse.Namespace,
    runner: BenchmarkRunner,
    samples: List[Dict[str, Any]],
    out_dir: str,
) -> pd.DataFrame:
    """运行 HuggingFace 后端测试 - 带四阶段计时"""
    import torch
    from qwen_omni_utils import process_mm_info
    from utils import profiling_utils as P
    
    print("\n" + "=" * 60)
    print("Running HuggingFace Backend (with TTFT breakdown)")
    print("=" * 60)
    
    # 加载模型
    loaded = runner.load()
    model = loaded.model
    processor = loaded.processor
    
    # 获取模块 (Qwen2.5-Omni 结构)
    if hasattr(model, "thinker"):
        visual = getattr(model.thinker, "visual", None)
        llm = getattr(model.thinker, "model", model)
    else:
        visual = None
        llm = model
    audio = None  # vllm-comparison 不使用音频
    
    rows = []
    for i, s in enumerate(samples):
        video_path = str(s.get("video_path", ""))
        question = str(args.question) if args.question else str(s.get("question", "Describe what you see."))
        duration = float(s.get("duration", 0))
        sample_id = str(s.get("sample_id", os.path.basename(video_path)))
        
        print(f"\n[{i+1}/{len(samples)}] {sample_id} ({duration:.1f}s)")
        
        for r in range(int(max(1, args.repeats))):
            do_record = int(r) >= int(max(0, args.warmup))
            
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # === Preprocess ===
                t_preprocess_start = time.perf_counter()
                
                video_nframes = int(args.video_nframes) if args.video_nframes else None
                video_element = {"type": "video", "video": str(video_path)}
                if video_nframes:
                    video_element["nframes"] = video_nframes
                
                messages = [{"role": "user", "content": [video_element, {"type": "text", "text": question}]}]
                _, images, videos = process_mm_info(messages, use_audio_in_video=False)
                
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=prompt, videos=videos, return_tensors="pt", padding=True).to(model.device)
                
                torch.cuda.synchronize()
                preprocess_ms = (time.perf_counter() - t_preprocess_start) * 1000
                
                # === Setup hooks ===
                encoder_timer = P.EncoderTimer()
                encoder_timer.register(model)
                
                # === Generate (encode + prefill + decode) ===
                torch.cuda.synchronize()
                t_gen_start = time.perf_counter()
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=1,  # 只生成1个token测TTFT
                        do_sample=False,
                        return_audio=False,
                    )
                
                torch.cuda.synchronize()
                ttft_ms = (time.perf_counter() - t_gen_start) * 1000
                
                # === Collect timing ===
                visual_encoder_ms = float(encoder_timer.times["visual"][-1]) if encoder_timer.times["visual"] else 0.0
                audio_encoder_ms = 0.0  # 无音频
                encoder_ms = visual_encoder_ms + audio_encoder_ms
                prefill_ms = float(max(0.0, ttft_ms - encoder_ms))
                
                # Cleanup
                encoder_timer.remove()
                
                prompt_tokens = inputs.input_ids.shape[-1]
                
                if do_record:
                    rows.append({
                        "backend": "hf",
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "duration": duration,
                        "repeat": r,
                        "preprocess_ms": preprocess_ms,
                        "encoder_ms": encoder_ms,
                        "prefill_ms": prefill_ms,
                        "ttft_ms": preprocess_ms + ttft_ms,  # 总 TTFT = preprocess + generate(1 token)
                        "prompt_tokens": prompt_tokens,
                    })
                    print(f"    repeat={r}: preprocess={preprocess_ms:.1f}ms, encoder={encoder_ms:.1f}ms, prefill={prefill_ms:.1f}ms, TTFT={preprocess_ms + ttft_ms:.1f}ms")
                    
            except Exception as e:
                print(f"    Error: {e}")
                if do_record:
                    rows.append({
                        "backend": "hf",
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "duration": duration,
                        "repeat": r,
                        "error": str(e),
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "hf_results.csv"), index=False)
    return df


def run_vllm_backend(
    args: argparse.Namespace,
    runner: BenchmarkRunner,
    samples: List[Dict[str, Any]],
    out_dir: str,
) -> pd.DataFrame:
    """运行 vLLM 后端测试 - 带阶段计时"""
    from benchmark.vllm_backend import VLLMBackend
    
    print("\n" + "=" * 60)
    print("Running vLLM Backend (with timing breakdown)")
    print("=" * 60)
    
    # vLLM requires full dtype name (bfloat16, not bf16)
    vllm_dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    raw_dtype = str(args.dtype) if hasattr(args, "dtype") else "bfloat16"
    vllm_dtype = vllm_dtype_map.get(raw_dtype, raw_dtype)
    
    backend = VLLMBackend(
        args.model_dir,
        dtype=vllm_dtype,
        gpu_memory_utilization=float(getattr(args, "gpu_memory_utilization", 0.85)),
        max_model_len=int(getattr(args, "max_model_len", 32768)),
    )
    backend.load()
    
    rows = []
    for i, s in enumerate(samples):
        video_path = str(s.get("video_path", ""))
        question = str(args.question) if args.question else str(s.get("question", "Describe what you see."))
        duration = float(s.get("duration", 0))
        sample_id = str(s.get("sample_id", os.path.basename(video_path)))
        
        print(f"\n[{i+1}/{len(samples)}] {sample_id} ({duration:.1f}s)")
        
        for r in range(int(max(1, args.repeats))):
            do_record = int(r) >= int(max(0, args.warmup))
            
            try:
                # 使用 measure_ttft_with_breakdown 获取阶段时间
                result = backend.measure_ttft_with_breakdown(
                    video_path,
                    question,
                    video_nframes=int(args.video_nframes) if args.video_nframes else None,
                )
                
                if do_record:
                    rows.append({
                        "backend": "vllm",
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "duration": duration,
                        "repeat": r,
                        "preprocess_ms": result.get("preprocess_ms", 0),
                        "model_ms": result.get("model_ms", 0),  # encoder + prefill (vLLM 黑盒无法分解)
                        "ttft_ms": result.get("ttft_ms", 0),
                        "prompt_tokens": result.get("prompt_tokens", 0),
                    })
                    print(f"    repeat={r}: preprocess={result.get('preprocess_ms', 0):.1f}ms, model={result.get('model_ms', 0):.1f}ms, TTFT={result.get('ttft_ms', 0):.1f}ms")
                    
            except Exception as e:
                print(f"    Error: {e}")
                if do_record:
                    rows.append({
                        "backend": "vllm",
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "duration": duration,
                        "repeat": r,
                        "error": str(e),
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "vllm_results.csv"), index=False)
    return df


def merge_and_plot(out_dir: str) -> None:
    """合并结果并生成对比图"""
    hf_path = os.path.join(out_dir, "hf_results.csv")
    vllm_path = os.path.join(out_dir, "vllm_results.csv")
    
    dfs = []
    if os.path.exists(hf_path):
        dfs.append(pd.read_csv(hf_path))
    if os.path.exists(vllm_path):
        dfs.append(pd.read_csv(vllm_path))
    
    if not dfs:
        print("No results to merge")
        return
    
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(out_dir, "combined_results.csv"), index=False)
    
    # 过滤错误
    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    
    if len(df_ok) == 0:
        print("No valid results to plot")
        return
    
    # 按 backend 和 sample_id 聚合 (动态选择可用列)
    agg_dict = {"ttft_ms": "mean", "prompt_tokens": "first"}
    
    # 可选列
    for col in ["preprocess_ms", "encoder_ms", "prefill_ms", "model_ms", "total_ms", "tokens_per_sec"]:
        if col in df_ok.columns:
            agg_dict[col] = "mean"
    
    summary = df_ok.groupby(["backend", "sample_id", "duration"]).agg(agg_dict).reset_index()
    
    # 保存汇总
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    
    # 检查是否有两种 backend
    backends = summary["backend"].unique()
    if len(backends) < 2:
        print(f"Only one backend found: {backends}. Skipping comparison plot.")
        _plot_single_backend(summary, out_dir)
        return
    
    # 双 backend 对比图
    _plot_comparison(summary, out_dir)


def _plot_single_backend(summary: pd.DataFrame, out_dir: str) -> None:
    """单个 backend 的结果图 - TTFT 分解堆叠图"""
    backend = summary["backend"].iloc[0]
    
    # 按 prompt_tokens 排序
    summary = summary.sort_values("prompt_tokens")
    n = len(summary)
    x = np.arange(n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # === 左图: TTFT 堆叠 ===
    preprocess = summary["preprocess_ms"].fillna(0).values / 1000 if "preprocess_ms" in summary.columns else np.zeros(n)
    encoder = summary["encoder_ms"].fillna(0).values / 1000 if "encoder_ms" in summary.columns else np.zeros(n)
    prefill = summary["prefill_ms"].fillna(0).values / 1000 if "prefill_ms" in summary.columns else np.zeros(n)
    model = summary["model_ms"].fillna(0).values / 1000 if "model_ms" in summary.columns else np.zeros(n)
    
    bottom = np.zeros(n)
    if preprocess.any():
        ax1.bar(x, preprocess, width=0.6, bottom=bottom, label="Preprocess", color="#2ecc71")
        bottom += preprocess
    if encoder.any():
        ax1.bar(x, encoder, width=0.6, bottom=bottom, label="Encoder", color="#3498db")
        bottom += encoder
    if prefill.any():
        ax1.bar(x, prefill, width=0.6, bottom=bottom, label="Prefill", color="#9b59b6")
        bottom += prefill
    if model.any():
        ax1.bar(x, model, width=0.6, bottom=bottom, label="Model", color="#e74c3c")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(t)}" for t in summary["prompt_tokens"]], rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Prompt Tokens")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title(f"{backend.upper()} TTFT Breakdown (Stacked)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    # 上方X轴: duration
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(x)
    ax1_top.set_xticklabels([f"{d:.0f}s" for d in summary["duration"]], fontsize=8)
    ax1_top.set_xlabel("Duration")
    
    # === 右图: TTFT 百分比分解 ===
    total = preprocess + encoder + prefill + model
    total = np.where(total == 0, 1, total)  # 避免除零
    
    bottom = np.zeros(n)
    if preprocess.any():
        ax2.bar(x, preprocess / total * 100, width=0.6, bottom=bottom, label="Preprocess", color="#2ecc71")
        bottom += preprocess / total * 100
    if encoder.any():
        ax2.bar(x, encoder / total * 100, width=0.6, bottom=bottom, label="Encoder", color="#3498db")
        bottom += encoder / total * 100
    if prefill.any():
        ax2.bar(x, prefill / total * 100, width=0.6, bottom=bottom, label="Prefill", color="#9b59b6")
        bottom += prefill / total * 100
    if model.any():
        ax2.bar(x, model / total * 100, width=0.6, bottom=bottom, label="Model", color="#e74c3c")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(t)}" for t in summary["prompt_tokens"]], rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("Prompt Tokens")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title(f"{backend.upper()} TTFT Breakdown (%)")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    # 上方X轴: duration
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(x)
    ax2_top.set_xticklabels([f"{d:.0f}s" for d in summary["duration"]], fontsize=8)
    ax2_top.set_xlabel("Duration")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{backend}_results.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {backend}_results.png")


def _plot_comparison(summary: pd.DataFrame, out_dir: str) -> None:
    """HF vs vLLM TTFT 对比图 - 并排堆叠柱状图（绝对值 + 百分比）"""
    hf_data = summary[summary["backend"] == "hf"].sort_values("prompt_tokens")
    vllm_data = summary[summary["backend"] == "vllm"].sort_values("prompt_tokens")
    
    # 找共同的 sample_id
    common_samples = set(hf_data["sample_id"]) & set(vllm_data["sample_id"])
    if not common_samples:
        print("No common samples between HF and vLLM")
        return
    
    hf_data = hf_data[hf_data["sample_id"].isin(common_samples)].sort_values("prompt_tokens")
    vllm_data = vllm_data[vllm_data["sample_id"].isin(common_samples)]
    vllm_data = vllm_data.set_index("sample_id").loc[hf_data["sample_id"]].reset_index()
    
    n = len(hf_data)
    x = np.arange(n)
    width = 0.35
    
    # === 获取 HF 阶段时间 (秒) ===
    hf_preprocess = hf_data["preprocess_ms"].fillna(0).values / 1000 if "preprocess_ms" in hf_data.columns else np.zeros(n)
    hf_encoder = hf_data["encoder_ms"].fillna(0).values / 1000 if "encoder_ms" in hf_data.columns else np.zeros(n)
    hf_prefill = hf_data["prefill_ms"].fillna(0).values / 1000 if "prefill_ms" in hf_data.columns else np.zeros(n)
    hf_ttft = hf_data["ttft_ms"].values / 1000
    
    # === 获取 vLLM 阶段时间 (秒) ===
    # vLLM 只有 preprocess 和 model (encoder+prefill 无法分解)
    vllm_preprocess = vllm_data["preprocess_ms"].fillna(0).values / 1000 if "preprocess_ms" in vllm_data.columns else np.zeros(n)
    vllm_model = vllm_data["model_ms"].fillna(0).values / 1000 if "model_ms" in vllm_data.columns else np.zeros(n)
    vllm_ttft = vllm_data["ttft_ms"].values / 1000
    
    # 颜色定义
    colors = {
        "preprocess": "#2ecc71",  # 绿色
        "encoder": "#3498db",     # 蓝色
        "prefill": "#9b59b6",     # 紫色
        "model": "#e74c3c",       # 红色 (vLLM encoder+prefill)
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== 图1: 绝对值堆叠柱状图 ==========
    ax = axes[0]
    
    # HF 堆叠 (左边柱)
    bottom_hf = np.zeros(n)
    ax.bar(x - width/2, hf_preprocess, width, bottom=bottom_hf, label="Preprocess", color=colors["preprocess"])
    bottom_hf += hf_preprocess
    ax.bar(x - width/2, hf_encoder, width, bottom=bottom_hf, label="Encoder", color=colors["encoder"])
    bottom_hf += hf_encoder
    ax.bar(x - width/2, hf_prefill, width, bottom=bottom_hf, label="Prefill", color=colors["prefill"])
    
    # vLLM 堆叠 (右边柱)
    bottom_vllm = np.zeros(n)
    ax.bar(x + width/2, vllm_preprocess, width, bottom=bottom_vllm, color=colors["preprocess"])
    bottom_vllm += vllm_preprocess
    ax.bar(x + width/2, vllm_model, width, bottom=bottom_vllm, label="Encoder+Prefill", color=colors["model"])
    
    # X轴：每组显示 token 数量，下方两行分别标注 HF 和 vLLM
    ax.set_xticks(x)
    # 主标签：token 数量 + 换行 + HF vLLM
    xlabels = [f"{int(t)}\nHF  vLLM" for t in hf_data["prompt_tokens"]]
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel("Prompt Tokens", fontsize=11)
    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.set_title("TTFT Breakdown: HF vs vLLM (Absolute)", fontsize=12, fontweight="bold")
    
    # 上方X轴: duration
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"{d:.0f}s" for d in hf_data["duration"]], fontsize=9)
    ax_top.set_xlabel("Video Duration", fontsize=11)
    
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    # ========== 图2: 百分比堆叠柱状图 ==========
    ax = axes[1]
    
    # HF 百分比
    hf_total = hf_preprocess + hf_encoder + hf_prefill
    hf_total = np.where(hf_total == 0, 1, hf_total)  # 避免除零
    hf_pre_pct = hf_preprocess / hf_total * 100
    hf_enc_pct = hf_encoder / hf_total * 100
    hf_prefill_pct = hf_prefill / hf_total * 100
    
    # vLLM 百分比
    vllm_total = vllm_preprocess + vllm_model
    vllm_total = np.where(vllm_total == 0, 1, vllm_total)
    vllm_pre_pct = vllm_preprocess / vllm_total * 100
    vllm_model_pct = vllm_model / vllm_total * 100
    
    # HF 堆叠
    bottom_hf = np.zeros(n)
    ax.bar(x - width/2, hf_pre_pct, width, bottom=bottom_hf, label="Preprocess", color=colors["preprocess"])
    bottom_hf += hf_pre_pct
    ax.bar(x - width/2, hf_enc_pct, width, bottom=bottom_hf, label="Encoder", color=colors["encoder"])
    bottom_hf += hf_enc_pct
    ax.bar(x - width/2, hf_prefill_pct, width, bottom=bottom_hf, label="Prefill", color=colors["prefill"])
    
    # vLLM 堆叠
    bottom_vllm = np.zeros(n)
    ax.bar(x + width/2, vllm_pre_pct, width, bottom=bottom_vllm, color=colors["preprocess"])
    bottom_vllm += vllm_pre_pct
    ax.bar(x + width/2, vllm_model_pct, width, bottom=bottom_vllm, label="Encoder+Prefill", color=colors["model"])
    
    # X轴：每组显示 token 数量，下方两行分别标注 HF 和 vLLM
    ax.set_xticks(x)
    xlabels = [f"{int(t)}\nHF  vLLM" for t in hf_data["prompt_tokens"]]
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel("Prompt Tokens", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("TTFT Breakdown: HF vs vLLM (Percentage)", fontsize=12, fontweight="bold")
    
    # 上方X轴: duration
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"{d:.0f}s" for d in hf_data["duration"]], fontsize=9)
    ax_top.set_xlabel("Video Duration", fontsize=11)
    
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vllm_vs_hf_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: vllm_vs_hf_comparison.png")
    
    # 打印统计
    speedup = hf_data['ttft_ms'].mean() / vllm_data['ttft_ms'].mean()
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"HF mean TTFT:   {hf_data['ttft_ms'].mean():.1f} ms")
    print(f"vLLM mean TTFT: {vllm_data['ttft_ms'].mean():.1f} ms")
    print(f"Speedup:        {speedup:.2f}x")


def run(args: argparse.Namespace, runner: BenchmarkRunner) -> str:
    out_dir = os.path.join(str(args.out_dir), SPEC_NAME)
    os.makedirs(out_dir, exist_ok=True)
    
    # 选择视频
    max_dur = float(getattr(args, "max_video_duration", 32.0))
    samples = select_videos(args, runner, max_duration=max_dur)
    
    print(f"Selected {len(samples)} videos for comparison")
    
    # 保存样本列表
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2, default=str)
    
    backend = str(getattr(args, "backend", "hf")).lower()
    
    if backend in ("hf", "both"):
        run_hf_backend(args, runner, samples, out_dir)
    
    if backend in ("vllm", "both"):
        # 如果是 both，需要彻底释放 HF 模型后再加载 vLLM
        if backend == "both":
            import gc
            import torch
            
            # 释放 runner 中的 HF 模型
            if hasattr(runner, "_model") and runner._model is not None:
                del runner._model
                runner._model = None
            if hasattr(runner, "_proc") and runner._proc is not None:
                del runner._proc
                runner._proc = None
            if hasattr(runner, "_fe") and runner._fe is not None:
                del runner._fe
                runner._fe = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("\n[INFO] HF 模型已释放，准备加载 vLLM...")
        
        run_vllm_backend(args, runner, samples, out_dir)
    
    # 合并结果并生成图
    merge_and_plot(out_dir)
    
    return out_dir


if __name__ == "__main__":
    # 独立运行模式：合并已有结果
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--merge-results":
        merge_and_plot(sys.argv[2])
    else:
        print("Usage: python vllm_comparison.py --merge-results <out_dir>")
