"""
FasterOmni 串行 Pipeline

将 GOP 解析 → AV-LRM 打分 → I 帧解码 → 模型推理 串联起来。
支持两种模式：
- baseline: 原生 Qwen2.5-Omni 全量帧推理
- sparse:   GOP 稀疏化后只用 I 帧推理

Usage:
    from fasteromni.pipeline import SparseInferencePipeline
    pipe = SparseInferencePipeline(model_dir="/path/to/model")
    result = pipe.run_sparse(video_path, question, keep_ratio=0.5)
"""
from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.modules.gop_parser import parse_gops, GOPAnalysis
from fasteromni.modules.audio_energy import extract_audio_energy_per_gop, extract_audio_from_video
from fasteromni.modules.sparse import score_gops, select_gops, get_selection_summary, ScoredGOP
from fasteromni.modules.frame_decoder import decode_i_frames


@dataclass
class PipelineResult:
    """单次推理的完整结果"""
    mode: str                           # "baseline" or "sparse"
    video_path: str
    question: str

    # 模型输出
    output_text: str = ""

    # 耗时分解 (ms)
    gop_parse_ms: float = 0.0
    audio_extract_ms: float = 0.0
    scoring_ms: float = 0.0
    i_frame_decode_ms: float = 0.0
    preprocess_ms: float = 0.0          # 包括 process_mm_info 或 I 帧解码
    tokenize_ms: float = 0.0            # processor tokenize
    audio_feature_ms: float = 0.0       # whisper feature extraction
    generate_ms: float = 0.0             # model.generate() 延迟（max_new_tokens=1 时即为 TTFT）
    total_ms: float = 0.0               # 端到端总时间

    # Token 统计
    visual_tokens: int = 0
    audio_tokens: int = 0
    total_tokens: int = 0
    num_frames_input: int = 0           # 实际送入模型的帧数

    # 稀疏化统计（仅 sparse 模式）
    total_gops: int = 0
    selected_gops: int = 0
    total_frames: int = 0
    keep_ratio_actual: float = 0.0

    # 错误
    error: Optional[str] = None


class SparseInferencePipeline:
    """FasterOmni 串行推理 Pipeline"""

    def __init__(
        self,
        model_dir: str = "/root/autodl-tmp/Qwen2.5-Omni-7B",
        dtype: str = "bf16",
    ):
        self.model_dir = model_dir
        self.dtype = dtype
        self._model = None
        self._proc = None
        self._fe = None

    def load_model(self):
        """加载模型（只加载一次）"""
        if self._model is not None:
            return

        import common as C
        from transformers import WhisperFeatureExtractor

        print("Loading Qwen2.5-Omni model...", flush=True)
        t0 = time.perf_counter()
        model, proc = C.load_qwen25_omni(self.model_dir, self.dtype)
        fe = WhisperFeatureExtractor.from_pretrained(self.model_dir)
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"Model loaded in {load_ms:.0f}ms", flush=True)

        self._model = model
        self._proc = proc
        self._fe = fe

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sync_devices(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run_baseline(
        self,
        video_path: str,
        question: str,
        max_new_tokens: int = 256,
        max_frames: int = 0,
    ) -> PipelineResult:
        """
        Baseline 推理：原生 Qwen2.5-Omni 处理全量视频。

        使用 process_mm_info 做视频解码 + 音频提取，
        然后走标准的 processor → model.generate 路径。
        """
        self.load_model()
        result = PipelineResult(mode="baseline", video_path=video_path, question=question)
        t_start = time.perf_counter()

        try:
            from qwen_omni_utils import process_mm_info
            import common as C

            model = self._model
            proc = self._proc
            fe = self._fe

            # 1. 视频 + 音频提取
            t0 = time.perf_counter()
            video_ele = {"type": "video", "video": str(video_path)}
            if max_frames > 0:
                video_ele["nframes"] = max_frames
            conversation = [{"role": "user", "content": [video_ele, {"type": "text", "text": question}]}]
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
            result.preprocess_ms = (time.perf_counter() - t0) * 1000

            # 2. Tokenize（音频直接传入 processor，由 processor 生成音频占位 token + 特征）
            t0 = time.perf_counter()
            text = proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            audios_list = list(audios) if audios else None
            inputs = proc(
                text=text, videos=videos, audio=audios_list,
                use_audio_in_video=True, return_tensors="pt", padding=True,
            )
            inputs = inputs.to(model.device)
            result.tokenize_ms = (time.perf_counter() - t0) * 1000

            # 记录帧数
            if videos and len(videos) > 0:
                v = videos[0]
                if hasattr(v, 'shape'):
                    result.num_frames_input = v.shape[0]
                elif isinstance(v, list):
                    result.num_frames_input = len(v)

            # 4. Token 统计
            if "input_ids" in inputs:
                from utils.profiling_utils import get_mm_token_ids_from_tokenizer
                tok = getattr(proc, "tokenizer", proc)
                mm = get_mm_token_ids_from_tokenizer(tok)
                ids = inputs["input_ids"][0]
                vision_ids = set(int(x) for x in mm.get("vision_special_token_ids", []) or [])
                audio_ids = set(int(x) for x in mm.get("audio_special_token_ids", []) or [])
                result.visual_tokens = sum(int((ids == tid).sum().item()) for tid in vision_ids) if vision_ids else 0
                result.audio_tokens = sum(int((ids == tid).sum().item()) for tid in audio_ids) if audio_ids else 0
                result.total_tokens = int(ids.shape[-1])

            # 4. Generate
            # 注意：generate_ms 是 model.generate() 的完整耗时
            # 当 max_new_tokens=1 时，generate_ms 即为真正的 TTFT
            self._sync_devices()
            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_audio=False,
                )
            self._sync_devices()
            result.generate_ms = (time.perf_counter() - t0) * 1000

            # 解码输出
            result.output_text = proc.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

        except torch.cuda.OutOfMemoryError:
            result.error = "OOM"
            self._clear_gpu()
        except Exception as e:
            result.error = str(e)

        result.total_ms = (time.perf_counter() - t_start) * 1000
        self._clear_gpu()
        return result

    def run_sparse(
        self,
        video_path: str,
        question: str,
        max_new_tokens: int = 256,
        alpha: float = 0.5,
        keep_ratio: float = 0.5,
        variance_threshold: float = 0.02,
        min_gop_frames: int = 10,
        skip_audio: bool = False,
        max_frames: int = 32,
        max_audio_sec: float = 0,
    ) -> PipelineResult:
        """
        稀疏化推理：GOP 解析 → AV-LRM 打分 → I 帧解码 → 推理。

        只将选中 GOP 的 I 帧送入模型，大幅减少 visual token。
        """
        self.load_model()
        result = PipelineResult(mode="sparse", video_path=video_path, question=question)
        t_start = time.perf_counter()

        try:
            import common as C
            from qwen_omni_utils import fetch_video, smart_resize, process_audio_info
            from torchvision.transforms.functional import resize
            from torchvision.transforms import InterpolationMode

            model = self._model
            proc = self._proc
            fe = self._fe

            # === Step 1: GOP 解析 ===
            t0 = time.perf_counter()
            gop_analysis = parse_gops(video_path)
            result.gop_parse_ms = (time.perf_counter() - t0) * 1000
            result.total_frames = gop_analysis.total_frames
            result.total_gops = gop_analysis.num_gops

            # === Step 2: 音频提取 + 能量计算 ===
            t0 = time.perf_counter()
            audio_waveform, sr = extract_audio_from_video(video_path, sr=16000)
            audio_energies = extract_audio_energy_per_gop(video_path, gop_analysis.gops, audio=audio_waveform)
            result.audio_extract_ms = (time.perf_counter() - t0) * 1000

            # === Step 3: AV-LRM 打分 + 选择 ===
            t0 = time.perf_counter()
            scored_gops = score_gops(gop_analysis.gops, audio_energies,
                                     alpha=alpha, min_gop_frames=min_gop_frames)
            scored_gops = select_gops(scored_gops, keep_ratio=keep_ratio,
                                      variance_threshold=variance_threshold)
            result.scoring_ms = (time.perf_counter() - t0) * 1000

            summary = get_selection_summary(scored_gops)
            result.selected_gops = summary["selected_gops"]
            result.keep_ratio_actual = summary["keep_ratio_actual"]

            # === Step 4: I 帧解码 ===
            t0 = time.perf_counter()
            i_frames, decode_ms = decode_i_frames(video_path, scored_gops)
            # 如果 I 帧数超过 max_frames，等间隔降采样
            if max_frames > 0 and len(i_frames) > max_frames:
                indices = np.linspace(0, len(i_frames) - 1, max_frames).astype(int)
                i_frames = [i_frames[i] for i in indices]
            result.i_frame_decode_ms = decode_ms
            result.num_frames_input = len(i_frames)
            result.preprocess_ms = result.gop_parse_ms + result.audio_extract_ms + \
                                   result.scoring_ms + result.i_frame_decode_ms

            if not i_frames:
                result.error = "No I-frames decoded"
                result.total_ms = (time.perf_counter() - t_start) * 1000
                return result

            # === Step 5: 构造模型输入 ===
            # 将 I 帧转为 video tensor (T, C, H, W)，走 video 处理路径
            # 保证与 baseline 使用相同的 tokenization 逻辑
            t0 = time.perf_counter()

            # 转为 tensor 并 resize（复用 fetch_video 的 resize 逻辑）
            from qwen_omni_utils.v2_5.vision_process import (
                IMAGE_FACTOR, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS,
                VIDEO_TOTAL_PIXELS, FRAME_FACTOR, smart_resize,
            )
            from torchvision.transforms.functional import resize as tv_resize
            from torchvision.transforms import InterpolationMode

            nframes = len(i_frames)
            first = i_frames[0]
            width, height = first.size  # PIL: (W, H)

            # 计算 max_pixels（与 fetch_video 一致）
            max_pixels = max(
                min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR),
                int(VIDEO_MIN_PIXELS * 1.05)
            )
            resized_height, resized_width = smart_resize(
                height, width,
                factor=IMAGE_FACTOR,
                min_pixels=VIDEO_MIN_PIXELS,
                max_pixels=max_pixels,
            )

            # PIL → tensor (T, C, H, W) float32
            frame_tensors = []
            for img in i_frames:
                arr = np.array(img)  # (H, W, C) uint8
                t = torch.from_numpy(arr).permute(2, 0, 1).float()  # (C, H, W)
                frame_tensors.append(t)
            video_tensor = torch.stack(frame_tensors)  # (T, C, H, W)

            # Resize（与 fetch_video 一致）
            video_tensor = tv_resize(
                video_tensor,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()

            # 使用 video conversation template
            video_ele = {"type": "video", "video": str(video_path)}
            conversation = [{"role": "user", "content": [video_ele, {"type": "text", "text": question}]}]

            # 音频截断：避免长视频音频 token 过多导致 OOM
            # max_audio_sec=0 表示自动匹配选中帧的时间跨度
            if audio_waveform is not None and not skip_audio:
                if max_audio_sec > 0:
                    max_samples = int(max_audio_sec * sr)
                else:
                    # 自动计算：取选中 GOP 的最大结束时间
                    selected = [sg for sg in scored_gops if sg.selected]
                    if selected:
                        max_end = max(sg.gop.end_time_sec or 0 for sg in selected)
                        max_samples = int((max_end + 1.0) * sr)  # +1s 余量
                    else:
                        max_samples = len(audio_waveform)
                if len(audio_waveform) > max_samples:
                    audio_waveform = audio_waveform[:max_samples]

            audio_arg = None if skip_audio else ([audio_waveform] if audio_waveform is not None else None)
            text = proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = proc(
                text=text, videos=[video_tensor], audio=audio_arg,
                use_audio_in_video=True, return_tensors="pt", padding=True,
            )
            inputs = inputs.to(model.device)
            result.tokenize_ms = (time.perf_counter() - t0) * 1000

            # === Step 7: Token 统计 ===
            if "input_ids" in inputs:
                from utils.profiling_utils import get_mm_token_ids_from_tokenizer
                tok = getattr(proc, "tokenizer", proc)
                mm = get_mm_token_ids_from_tokenizer(tok)
                ids = inputs["input_ids"][0]
                vision_ids = set(int(x) for x in mm.get("vision_special_token_ids", []) or [])
                audio_ids = set(int(x) for x in mm.get("audio_special_token_ids", []) or [])
                result.visual_tokens = sum(int((ids == tid).sum().item()) for tid in vision_ids) if vision_ids else 0
                result.audio_tokens = sum(int((ids == tid).sum().item()) for tid in audio_ids) if audio_ids else 0
                result.total_tokens = int(ids.shape[-1])

            # === Step 8: Generate ===
            # generate_ms 是完整 generate 耗时；max_new_tokens=1 时即为 TTFT
            self._sync_devices()
            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_audio=False,
                )
            self._sync_devices()
            result.generate_ms = (time.perf_counter() - t0) * 1000

            # 解码输出
            result.output_text = proc.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

        except torch.cuda.OutOfMemoryError:
            result.error = "OOM"
            self._clear_gpu()
        except Exception as e:
            result.error = str(e)
            import traceback
            traceback.print_exc()

        result.total_ms = (time.perf_counter() - t_start) * 1000
        self._clear_gpu()
        return result


def print_result(r: PipelineResult) -> None:
    """打印推理结果摘要"""
    print(f"\n{'='*70}")
    print(f"Mode: {r.mode} | Video: {os.path.basename(r.video_path)}")
    if r.error:
        print(f"ERROR: {r.error}")
        return

    print(f"Question: {r.question[:80]}...")
    print(f"Answer: {r.output_text[:200]}")
    print(f"\n--- Timing (ms) ---")
    print(f"  Preprocess:     {r.preprocess_ms:>8.1f}")
    if r.mode == "sparse":
        print(f"    GOP parse:    {r.gop_parse_ms:>8.1f}")
        print(f"    Audio+Score:  {r.audio_extract_ms + r.scoring_ms:>8.1f}")
        print(f"    I-frame dec:  {r.i_frame_decode_ms:>8.1f}")
    print(f"  Tokenize:       {r.tokenize_ms:>8.1f}")
    print(f"  Audio feature:  {r.audio_feature_ms:>8.1f}")
    print(f"  Generate:       {r.generate_ms:>8.1f}")
    print(f"  Total:          {r.total_ms:>8.1f}")

    print(f"\n--- Tokens ---")
    print(f"  Visual: {r.visual_tokens} | Audio: {r.audio_tokens} | Total: {r.total_tokens}")
    print(f"  Frames input: {r.num_frames_input}")

    if r.mode == "sparse":
        print(f"\n--- Sparsification ---")
        print(f"  GOPs: {r.selected_gops}/{r.total_gops} | "
              f"Keep ratio: {r.keep_ratio_actual:.1%}")
