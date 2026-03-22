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
import json as _json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.modules.gop_parser import parse_gops, GOPAnalysis
from fasteromni.modules.audio_energy import extract_audio_energy_per_gop, extract_audio_from_video
from fasteromni.modules.sparse import score_gops, select_gops, get_selection_summary, ScoredGOP
from fasteromni.modules.frame_decoder import decode_i_frames


def _ffprobe_info(path: str, timeout: int = 15) -> dict:
    """获取视频元数据（fps、总帧数、分辨率）"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration", "-of", "json", path,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}")
    info = _json.loads(r.stdout)
    stream = info["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rfr = stream.get("r_frame_rate", "30/1")
    num, den = rfr.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    nb = stream.get("nb_frames", "N/A")
    if nb not in ("N/A", "0"):
        total_frames = int(nb)
    else:
        dur = float(stream.get("duration", 0)
                     or info.get("format", {}).get("duration", 0) or 0)
        total_frames = max(1, int(dur * fps))
    return {"width": width, "height": height, "fps": fps,
            "total_frames": total_frames}


def _extract_frames_ffmpeg(path: str, indices: list, width: int, height: int,
                           timeout: int = 60) -> "list[Image.Image]":
    """通过 ffmpeg select 滤镜提取指定帧，返回 PIL Image 列表"""
    select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-vf", f"select='{select_expr}'",
        "-vsync", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {r.returncode}")
    raw = r.stdout
    frame_size = width * height * 3
    n_got = len(raw) // frame_size
    frames = []
    for i in range(n_got):
        fb = raw[i * frame_size:(i + 1) * frame_size]
        arr = np.frombuffer(fb, dtype=np.uint8).reshape(height, width, 3)
        frames.append(Image.fromarray(arr))
    return frames


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
    kr_requested: float = 0.0
    kr_adaptive: float = 0.0

    # 错误
    error: Optional[str] = None


@dataclass
class SelectedFrames:
    """帧选择的输出 = 统一推理引擎的输入"""
    conversation: list
    videos: list
    audio: Optional[list]
    use_audio_in_video: bool
    num_frames_input: int

    preprocess_ms: float = 0.0
    tokenize_overhead_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    def _frames_to_video_tensor(self, frames_pil: List[Image.Image]) -> torch.Tensor:
        """将 PIL Image 列表转为 video tensor (T, C, H, W) float32，已 resize。"""
        # 转为 tensor 并 resize（复用 fetch_video 的 resize 逻辑）
        from qwen_omni_utils.v2_5.vision_process import (
            IMAGE_FACTOR, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS,
            VIDEO_TOTAL_PIXELS, FRAME_FACTOR, smart_resize,
        )
        from torchvision.transforms.functional import resize as tv_resize
        from torchvision.transforms import InterpolationMode

        nframes = len(frames_pil)
        first = frames_pil[0]
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
        for img in frames_pil:
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
        return video_tensor

    def _count_tokens(self, inputs) -> Tuple[int, int, int]:
        """统计 visual / audio / total tokens"""
        if "input_ids" not in inputs:
            return 0, 0, 0
        from utils.profiling_utils import get_mm_token_ids_from_tokenizer
        tok = getattr(self._proc, "tokenizer", self._proc)
        mm = get_mm_token_ids_from_tokenizer(tok)
        ids = inputs["input_ids"][0]
        vision_ids = set(int(x) for x in mm.get("vision_special_token_ids", []) or [])
        audio_ids = set(int(x) for x in mm.get("audio_special_token_ids", []) or [])
        visual = sum(int((ids == tid).sum().item()) for tid in vision_ids) if vision_ids else 0
        audio = sum(int((ids == tid).sum().item()) for tid in audio_ids) if audio_ids else 0
        total = int(ids.shape[-1])
        return visual, audio, total

    def _run_inference(
        self,
        selected: SelectedFrames,
        result: PipelineResult,
        max_new_tokens: int = 256,
    ) -> None:
        """统一推理引擎：tokenize → token统计 → generate → decode"""
        model = self._model
        proc = self._proc
        result.num_frames_input = selected.num_frames_input

        # 2. Tokenize（音频直接传入 processor，由 processor 生成音频占位 token + 特征）
        t0 = time.perf_counter()
        text = proc.apply_chat_template(selected.conversation, tokenize=False, add_generation_prompt=True)
        if result.mode == "baseline":
            print(f"  [baseline] step2: proc() starting", flush=True)
        inputs = proc(
            text=text, videos=selected.videos, audio=selected.audio,
            use_audio_in_video=selected.use_audio_in_video, return_tensors="pt", padding=True,
        )
        if result.mode == "baseline":
            print(f"  [baseline] step2: proc() done ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)
        inputs = inputs.to(model.device)
        result.tokenize_ms = selected.tokenize_overhead_ms + (time.perf_counter() - t0) * 1000

        # Token 统计
        result.visual_tokens, result.audio_tokens, result.total_tokens = self._count_tokens(inputs)

        # Generate
        # generate_ms 是 model.generate() 的完整耗时
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

    def _select_baseline(
        self,
        video_path: str,
        question: str,
        max_frames: int = 0,
    ) -> SelectedFrames:
        from qwen_omni_utils import process_mm_info

        # 1. 视频 + 音频提取
        t0 = time.perf_counter()
        video_ele = {"type": "video", "video": str(video_path)}
        if max_frames > 0:
            video_ele["nframes"] = max_frames
        conversation = [{"role": "user", "content": [video_ele, {"type": "text", "text": question}]}]
        print(f"  [baseline] step1: process_mm_info starting for {os.path.basename(video_path)}", flush=True)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        print(f"  [baseline] step1: done ({(time.perf_counter()-t0)*1000:.0f}ms)", flush=True)
        preprocess_ms = (time.perf_counter() - t0) * 1000

        # 记录帧数
        num_frames_input = 0
        if videos and len(videos) > 0:
            v = videos[0]
            if hasattr(v, "shape"):
                num_frames_input = int(v.shape[0])
            elif isinstance(v, list):
                num_frames_input = len(v)

        audios_list = list(audios) if audios else None
        return SelectedFrames(
            conversation=conversation,
            videos=videos,
            audio=audios_list,
            use_audio_in_video=True,
            num_frames_input=num_frames_input,
            preprocess_ms=preprocess_ms,
        )

    def _select_sparse(
        self,
        video_path: str,
        question: str,
        alpha: float = 0.5,
        keep_ratio: float = 0.5,
        variance_threshold: float = 0.02,
        min_gop_frames: int = 10,
        skip_audio: bool = False,
        max_frames: int = 32,
        max_audio_sec: float = 0,
    ) -> SelectedFrames:
        # === Step 1: GOP 解析 ===
        t0 = time.perf_counter()
        gop_analysis = parse_gops(video_path)
        gop_parse_ms = (time.perf_counter() - t0) * 1000
        total_frames = gop_analysis.total_frames
        total_gops = gop_analysis.num_gops

        # === Step 2: 音频提取 + 能量计算 ===
        t0 = time.perf_counter()
        audio_waveform, sr = extract_audio_from_video(video_path, sr=16000)
        audio_energies = extract_audio_energy_per_gop(video_path, gop_analysis.gops, audio=audio_waveform)
        audio_extract_ms = (time.perf_counter() - t0) * 1000

        # === Step 3: AV-LRM 打分 + 选择 ===
        t0 = time.perf_counter()
        scored_gops = score_gops(gop_analysis.gops, audio_energies,
                                 alpha=alpha, min_gop_frames=min_gop_frames)
        valid_gops_list = [sg for sg in scored_gops if sg.combined_score >= 0]
        n_valid = len(valid_gops_list)
        if max_frames > 0 and n_valid > 0:
            kr_adaptive = min(keep_ratio, max_frames / n_valid)
        else:
            kr_adaptive = keep_ratio
        scored_gops = select_gops(scored_gops, keep_ratio=kr_adaptive,
                                  variance_threshold=variance_threshold)
        scoring_ms = (time.perf_counter() - t0) * 1000

        summary = get_selection_summary(scored_gops)
        selected_gops = summary["selected_gops"]
        keep_ratio_actual = summary["keep_ratio_actual"]

        # === Step 4: I 帧解码 ===
        t0 = time.perf_counter()
        i_frames, decode_ms = decode_i_frames(video_path, scored_gops)
        # 如果 I 帧数超过 max_frames，等间隔降采样
        if max_frames > 0 and len(i_frames) > max_frames:
            indices = np.linspace(0, len(i_frames) - 1, max_frames).astype(int)
            i_frames = [i_frames[i] for i in indices]
        i_frame_decode_ms = decode_ms
        num_frames_input = len(i_frames)
        preprocess_ms = gop_parse_ms + audio_extract_ms + scoring_ms + i_frame_decode_ms

        metadata = {
            "gop_parse_ms": gop_parse_ms,
            "audio_extract_ms": audio_extract_ms,
            "scoring_ms": scoring_ms,
            "i_frame_decode_ms": i_frame_decode_ms,
            "total_gops": total_gops,
            "selected_gops": selected_gops,
            "total_frames": total_frames,
            "keep_ratio_actual": keep_ratio_actual,
            "kr_requested": keep_ratio,
            "kr_adaptive": kr_adaptive,
            "adaptive_triggered": kr_adaptive < keep_ratio,
        }

        # 当只有 1 个有效 GOP 时，音频与 1 帧视频在 processor 中可能无法对齐
        # 自动跳过音频，避免 Qwen2.5-Omni processor 的 StopIteration
        if len(valid_gops_list) <= 1:
            skip_audio = True

        if not i_frames:
            return SelectedFrames(
                conversation=[],
                videos=[],
                audio=None,
                use_audio_in_video=(not skip_audio),
                num_frames_input=num_frames_input,
                preprocess_ms=preprocess_ms,
                metadata={**metadata, "selection_error": "No I-frames decoded"},
            )

        # === Step 5: 构造模型输入 ===
        # 将 I 帧转为 video tensor (T, C, H, W)，走 video 处理路径
        # 保证与 baseline 使用相同的 tokenization 逻辑
        t0 = time.perf_counter()
        video_tensor = self._frames_to_video_tensor(i_frames)

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
        tokenize_overhead_ms = (time.perf_counter() - t0) * 1000

        return SelectedFrames(
            conversation=conversation,
            videos=[video_tensor],
            audio=audio_arg,
            use_audio_in_video=(not skip_audio),
            num_frames_input=num_frames_input,
            preprocess_ms=preprocess_ms,
            tokenize_overhead_ms=tokenize_overhead_ms,
            metadata=metadata,
        )

    def _select_naive(
        self,
        video_path: str,
        question: str,
        strategy: str = "uniform",
        keep_ratio: float = 0.5,
        max_frames: int = 32,
        seed: int = 42,
        skip_audio: bool = False,
    ) -> SelectedFrames:
        # === Step 1: GOP 解析（确定帧数 K，与 sparse 匹配）===
        t0 = time.perf_counter()
        gop_analysis = parse_gops(video_path)
        gop_parse_ms = (time.perf_counter() - t0) * 1000
        total_gops = gop_analysis.num_gops
        total_frames = gop_analysis.total_frames

        valid_gops = [g for g in gop_analysis.gops if g.num_frames >= 10]
        n_valid = len(valid_gops)
        if max_frames > 0 and n_valid > 0:
            kr_adaptive = min(keep_ratio, max_frames / n_valid)
        else:
            kr_adaptive = keep_ratio
        K = max(1, math.ceil(n_valid * kr_adaptive))
        if max_frames > 0:
            K = min(K, max_frames)
        selected_gops = K
        keep_ratio_actual = (K / n_valid if n_valid else 0)

        # === Step 2: 帧选择（策略相关）===
        audio_max_end = None
        if strategy == "iframe_uniform":
            n_valid = len(valid_gops)
            if n_valid > 0:
                gop_indices = np.linspace(
                    0, n_valid - 1, min(K, n_valid)).astype(int)
            else:
                gop_indices = np.array([], dtype=int)
            selected_set = set(gop_indices.tolist())
            scored_all = [
                ScoredGOP(gop=g, visual_score=0, audio_score=0,
                          combined_score=0, selected=(i in selected_set))
                for i, g in enumerate(valid_gops)
            ]
            t0 = time.perf_counter()
            frames_pil, decode_ms = decode_i_frames(
                video_path, scored_all)
            if max_frames > 0 and len(frames_pil) > max_frames:
                idx = np.linspace(
                    0, len(frames_pil) - 1, max_frames).astype(int)
                frames_pil = [frames_pil[j] for j in idx]
            i_frame_decode_ms = decode_ms
            sel_gops = [valid_gops[i] for i in gop_indices]
            if sel_gops:
                audio_max_end = max(
                    g.end_time_sec or 0 for g in sel_gops)

        elif strategy in ("uniform", "random"):
            t0 = time.perf_counter()
            info = _ffprobe_info(video_path)
            total = info["total_frames"]
            if strategy == "uniform":
                frame_indices = np.linspace(
                    0, total - 1, K).astype(int).tolist()
            else:
                rng = np.random.RandomState(seed)
                frame_indices = sorted(
                    rng.choice(
                        total, size=min(K, total), replace=False
                    ).tolist()
                )
            frames_pil = _extract_frames_ffmpeg(
                video_path, frame_indices,
                info["width"], info["height"])
            i_frame_decode_ms = (time.perf_counter() - t0) * 1000
        else:
            raise ValueError(f"Unknown naive strategy: {strategy}")

        num_frames_input = len(frames_pil)
        metadata = {
            "gop_parse_ms": gop_parse_ms,
            "audio_extract_ms": 0.0,
            "i_frame_decode_ms": i_frame_decode_ms,
            "total_gops": total_gops,
            "selected_gops": selected_gops,
            "total_frames": total_frames,
            "keep_ratio_actual": keep_ratio_actual,
            "kr_requested": keep_ratio,
            "kr_adaptive": kr_adaptive,
            "adaptive_triggered": kr_adaptive < keep_ratio,
        }
        if not frames_pil:
            preprocess_ms = gop_parse_ms + i_frame_decode_ms
            return SelectedFrames(
                conversation=[],
                videos=[],
                audio=None,
                use_audio_in_video=(not skip_audio),
                num_frames_input=num_frames_input,
                preprocess_ms=preprocess_ms,
                metadata={**metadata, "selection_error": "No frames extracted"},
            )

        # 当只有 1 个有效 GOP 时，音频与 1 帧视频在 processor 中可能无法对齐
        # 自动跳过音频，避免 Qwen2.5-Omni processor 的 StopIteration
        if len(valid_gops) <= 1:
            skip_audio = True

        # === Step 3: 音频提取 ===
        t0 = time.perf_counter()
        audio_waveform, sr = extract_audio_from_video(
            video_path, sr=16000)
        audio_extract_ms = (time.perf_counter() - t0) * 1000

        if (audio_waveform is not None and not skip_audio
                and audio_max_end is not None):
            max_samples = int((audio_max_end + 1.0) * sr)
            if len(audio_waveform) > max_samples:
                audio_waveform = audio_waveform[:max_samples]

        preprocess_ms = (gop_parse_ms
                         + audio_extract_ms
                         + i_frame_decode_ms)
        metadata["audio_extract_ms"] = audio_extract_ms

        # === Step 4: 构造模型输入（与 sparse 路径一致）===
        t0 = time.perf_counter()
        video_tensor = self._frames_to_video_tensor(frames_pil)

        video_ele = {"type": "video", "video": str(video_path)}
        conversation = [{"role": "user", "content": [
            video_ele, {"type": "text", "text": question}]}]

        audio_arg = (None if skip_audio
                     else ([audio_waveform]
                           if audio_waveform is not None else None))
        tokenize_overhead_ms = (time.perf_counter() - t0) * 1000

        return SelectedFrames(
            conversation=conversation,
            videos=[video_tensor],
            audio=audio_arg,
            use_audio_in_video=(not skip_audio),
            num_frames_input=num_frames_input,
            preprocess_ms=preprocess_ms,
            tokenize_overhead_ms=tokenize_overhead_ms,
            metadata=metadata,
        )

    def _select_text_only(self, video_path: str, question: str) -> SelectedFrames:
        """text_only: 1 黑帧 + 无音频 → 测量语言先验下界"""
        # 1. 创建 1 个 640×480 纯黑帧
        black_frame = Image.new("RGB", (640, 480), (0, 0, 0))
        video_tensor = self._frames_to_video_tensor([black_frame])

        # 2. 构造 conversation（需要 video 元素让模板格式正确）
        video_ele = {"type": "video", "video": str(video_path)}
        conversation = [{"role": "user", "content": [video_ele, {"type": "text", "text": question}]}]

        return SelectedFrames(
            conversation=conversation,
            videos=[video_tensor],
            audio=None,
            use_audio_in_video=False,
            num_frames_input=1,
            preprocess_ms=0.0,
        )

    def _select_audio_only(self, video_path: str, question: str) -> SelectedFrames:
        """audio_only: 1 黑帧 + 真实音频 → 测量音频贡献"""
        # 1. 创建 1 个 640×480 纯黑帧
        black_frame = Image.new("RGB", (640, 480), (0, 0, 0))
        video_tensor = self._frames_to_video_tensor([black_frame])

        # 2. 提取真实音频（完整，不截断，与 baseline 一致）
        t0 = time.perf_counter()
        audio_waveform, sr = extract_audio_from_video(video_path, sr=16000)
        audio_extract_ms = (time.perf_counter() - t0) * 1000
        audio_arg = [audio_waveform] if audio_waveform is not None else None

        # 3. 构造 conversation
        video_ele = {"type": "video", "video": str(video_path)}
        conversation = [{"role": "user", "content": [video_ele, {"type": "text", "text": question}]}]

        return SelectedFrames(
            conversation=conversation,
            videos=[video_tensor],
            audio=audio_arg,
            use_audio_in_video=True,
            num_frames_input=1,
            preprocess_ms=audio_extract_ms,
            metadata={"audio_extract_ms": audio_extract_ms},
        )

    def _select_video_only(self, video_path: str, question: str, max_frames: int = 0) -> SelectedFrames:
        """video_only: 真实视频 + 无音频 → 测量视觉贡献"""
        from qwen_omni_utils import process_mm_info

        # 复用 baseline 的视频提取逻辑，但不提取音频
        t0 = time.perf_counter()
        video_ele = {"type": "video", "video": str(video_path)}
        if max_frames > 0:
            video_ele["nframes"] = max_frames
        conversation = [{"role": "user", "content": [video_ele, {"type": "text", "text": question}]}]
        # use_audio_in_video=False → process_mm_info 不提取音频
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        preprocess_ms = (time.perf_counter() - t0) * 1000

        # 记录帧数
        num_frames_input = 0
        if videos and len(videos) > 0:
            v = videos[0]
            if hasattr(v, "shape"):
                num_frames_input = int(v.shape[0])
            elif isinstance(v, list):
                num_frames_input = len(v)

        return SelectedFrames(
            conversation=conversation,
            videos=videos,
            audio=None,
            use_audio_in_video=False,
            num_frames_input=num_frames_input,
            preprocess_ms=preprocess_ms,
        )

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
            selected = self._select_baseline(video_path, question, max_frames=max_frames)
            result.preprocess_ms = selected.preprocess_ms
            result.num_frames_input = selected.num_frames_input
            self._run_inference(selected, result, max_new_tokens=max_new_tokens)

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
            selected = self._select_sparse(
                video_path=video_path,
                question=question,
                alpha=alpha,
                keep_ratio=keep_ratio,
                variance_threshold=variance_threshold,
                min_gop_frames=min_gop_frames,
                skip_audio=skip_audio,
                max_frames=max_frames,
                max_audio_sec=max_audio_sec,
            )

            result.gop_parse_ms = float(selected.metadata.get("gop_parse_ms", 0.0))
            result.audio_extract_ms = float(selected.metadata.get("audio_extract_ms", 0.0))
            result.scoring_ms = float(selected.metadata.get("scoring_ms", 0.0))
            result.i_frame_decode_ms = float(selected.metadata.get("i_frame_decode_ms", 0.0))
            result.total_frames = int(selected.metadata.get("total_frames", 0))
            result.total_gops = int(selected.metadata.get("total_gops", 0))
            result.selected_gops = int(selected.metadata.get("selected_gops", 0))
            result.keep_ratio_actual = float(selected.metadata.get("keep_ratio_actual", 0.0))
            result.kr_requested = float(selected.metadata.get("kr_requested", 0.0))
            result.kr_adaptive = float(selected.metadata.get("kr_adaptive", 0.0))
            result.preprocess_ms = selected.preprocess_ms
            result.num_frames_input = selected.num_frames_input

            if "selection_error" in selected.metadata:
                result.error = str(selected.metadata["selection_error"])
                result.total_ms = (time.perf_counter() - t_start) * 1000
                return result

            self._run_inference(selected, result, max_new_tokens=max_new_tokens)

        except torch.cuda.OutOfMemoryError:
            result.error = "OOM"
            self._clear_gpu()
        except Exception as e:
            result.error = repr(e) if not str(e) else str(e)
            import traceback
            traceback.print_exc()

        result.total_ms = (time.perf_counter() - t_start) * 1000
        self._clear_gpu()
        return result

    def run_naive(
        self,
        video_path: str,
        question: str,
        strategy: str = "uniform",
        max_new_tokens: int = 256,
        keep_ratio: float = 0.5,
        max_frames: int = 32,
        seed: int = 42,
        skip_audio: bool = False,
    ) -> PipelineResult:
        """
        Naive baseline 推理：不使用 AV-LRM 打分的帧选择策略。

        策略：
        - uniform: 从全视频等间隔采 K 帧
        - random: 从全视频随机采 K 帧
        - iframe_uniform: 等间隔选 K 个 GOP 的 I 帧（不打分）

        K = min(ceil(N_valid_gops × keep_ratio), max_frames)，
        保证与 sparse 方法使用相同帧数。
        """
        self.load_model()
        result = PipelineResult(
            mode=f"naive_{strategy}", video_path=video_path, question=question)
        t_start = time.perf_counter()

        try:
            selected = self._select_naive(
                video_path=video_path,
                question=question,
                strategy=strategy,
                keep_ratio=keep_ratio,
                max_frames=max_frames,
                seed=seed,
                skip_audio=skip_audio,
            )

            result.gop_parse_ms = float(selected.metadata.get("gop_parse_ms", 0.0))
            result.audio_extract_ms = float(selected.metadata.get("audio_extract_ms", 0.0))
            result.i_frame_decode_ms = float(selected.metadata.get("i_frame_decode_ms", 0.0))
            result.total_gops = int(selected.metadata.get("total_gops", 0))
            result.selected_gops = int(selected.metadata.get("selected_gops", 0))
            result.total_frames = int(selected.metadata.get("total_frames", 0))
            result.keep_ratio_actual = float(selected.metadata.get("keep_ratio_actual", 0.0))
            result.kr_requested = float(selected.metadata.get("kr_requested", 0.0))
            result.kr_adaptive = float(selected.metadata.get("kr_adaptive", 0.0))
            result.preprocess_ms = selected.preprocess_ms
            result.num_frames_input = selected.num_frames_input

            if "selection_error" in selected.metadata:
                result.error = str(selected.metadata["selection_error"])
                result.total_ms = (
                    time.perf_counter() - t_start) * 1000
                return result

            self._run_inference(selected, result, max_new_tokens=max_new_tokens)

        except torch.cuda.OutOfMemoryError:
            result.error = "OOM"
            self._clear_gpu()
        except Exception as e:
            result.error = repr(e) if not str(e) else str(e)
            import traceback
            traceback.print_exc()

        result.total_ms = (time.perf_counter() - t_start) * 1000
        self._clear_gpu()
        return result

    def run_text_only(
        self,
        video_path: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> PipelineResult:
        self.load_model()
        result = PipelineResult(mode="text_only", video_path=video_path, question=question)
        t_start = time.perf_counter()

        try:
            selected = self._select_text_only(video_path, question)
            result.preprocess_ms = selected.preprocess_ms
            result.num_frames_input = selected.num_frames_input
            self._run_inference(selected, result, max_new_tokens=max_new_tokens)

        except torch.cuda.OutOfMemoryError:
            result.error = "OOM"
            self._clear_gpu()
        except Exception as e:
            result.error = str(e)

        result.total_ms = (time.perf_counter() - t_start) * 1000
        self._clear_gpu()
        return result

    def run_audio_only(
        self,
        video_path: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> PipelineResult:
        self.load_model()
        result = PipelineResult(mode="audio_only", video_path=video_path, question=question)
        t_start = time.perf_counter()

        try:
            selected = self._select_audio_only(video_path, question)
            result.audio_extract_ms = float(selected.metadata.get("audio_extract_ms", 0.0))
            result.preprocess_ms = selected.preprocess_ms
            result.num_frames_input = selected.num_frames_input
            self._run_inference(selected, result, max_new_tokens=max_new_tokens)

        except torch.cuda.OutOfMemoryError:
            result.error = "OOM"
            self._clear_gpu()
        except Exception as e:
            result.error = str(e)

        result.total_ms = (time.perf_counter() - t_start) * 1000
        self._clear_gpu()
        return result

    def run_video_only(
        self,
        video_path: str,
        question: str,
        max_new_tokens: int = 256,
        max_frames: int = 0,
    ) -> PipelineResult:
        self.load_model()
        result = PipelineResult(mode="video_only", video_path=video_path, question=question)
        t_start = time.perf_counter()

        try:
            selected = self._select_video_only(video_path, question, max_frames=max_frames)
            result.preprocess_ms = selected.preprocess_ms
            result.num_frames_input = selected.num_frames_input
            self._run_inference(selected, result, max_new_tokens=max_new_tokens)

        except torch.cuda.OutOfMemoryError:
            result.error = "OOM"
            self._clear_gpu()
        except Exception as e:
            result.error = str(e)

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
