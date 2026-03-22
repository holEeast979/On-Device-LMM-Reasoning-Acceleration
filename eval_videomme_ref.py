"""
Video-MME 评估脚本

选择题格式，评估零歧义（pred == gt）。
支持 baseline / sparse / sparse-no-audio 三种模式。

Usage:
    # 快速验证（5 个视频）
    python fasteromni/eval_videomme.py --max-videos 5

    # 完整评估（全部已下载视频）
    python fasteromni/eval_videomme.py

    # 消融实验
    python fasteromni/eval_videomme.py --sweep keep_ratio
"""
from __future__ import annotations

# === Monkey-patch: 防止音频提取死锁 ===
#
# 死锁根因（两个独立阻塞点）：
#   1. process_audio_info() → _check_if_video_has_audio() → av.open() 无超时
#   2. process_audio_info() → librosa.load(video_path) → audioread → ffmpeg 子进程挂死
#
# 修复策略：
#   - 替换 librosa.load 为 subprocess+timeout 版本（兜底）
#   - 替换 process_audio_info 为完全绕过 av.open/librosa 的安全版本（根治）
#
import librosa
import subprocess
import numpy as np
import os

_SAFE_AUDIO_SR = 16000
_SAFE_TIMEOUT = 30  # seconds

def _safe_extract_audio(path: str, sr: int = _SAFE_AUDIO_SR,
                        offset: float = 0.0, duration: float = None) -> np.ndarray:
    """用 subprocess ffmpeg 提取音频，带超时保护。失败返回 0.1s 静音。"""
    try:
        cmd = ["ffmpeg", "-y"]
        if offset > 0:
            cmd += ["-ss", str(offset)]
        cmd += ["-i", path, "-vn", "-acodec", "pcm_f32le",
                "-ar", str(sr), "-ac", "1", "-f", "f32le", "pipe:1"]
        if duration is not None:
            cmd += ["-t", str(duration)]
        r = subprocess.run(cmd, capture_output=True, timeout=_SAFE_TIMEOUT)
        if r.returncode == 0 and len(r.stdout) > 0:
            return np.frombuffer(r.stdout, dtype=np.float32)
    except subprocess.TimeoutExpired:
        print(f"[WARN] ffmpeg audio extraction timeout ({_SAFE_TIMEOUT}s) for {path}", flush=True)
    except Exception as e:
        print(f"[WARN] ffmpeg audio extraction failed: {e}", flush=True)
    return np.zeros(int(sr * 0.1), dtype=np.float32)


# 1. 替换 librosa.load（兜底，防止任何其他地方调用原始 librosa.load）
_original_librosa_load = librosa.load

def _safe_librosa_load(path, sr=22050, mono=True, offset=0.0, duration=None, **kwargs):
    """带超时的 librosa.load 替代"""
    if isinstance(path, np.ndarray):
        return _original_librosa_load(path, sr=sr, mono=mono, offset=offset, duration=duration, **kwargs)
    if isinstance(path, str) and not path.startswith(("http://", "https://", "data:")):
        return _safe_extract_audio(path, sr=sr, offset=offset, duration=duration), sr
    return _original_librosa_load(path, sr=sr, mono=mono, offset=offset, duration=duration, **kwargs)

librosa.load = _safe_librosa_load


# 2. 替换 process_audio_info（根治：完全绕过 av.open 和 librosa.load）
def _safe_process_audio_info(conversations, use_audio_in_video):
    """
    安全版 process_audio_info，绕过所有可能挂死的 C 扩展调用。

    原版死锁点：
      - _check_if_video_has_audio() → av.open() 无超时
      - librosa.load(video_path) → audioread.ffdec.FFmpegAudioFile → ffmpeg 子进程挂死
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    path = ele.get("audio", ele.get("audio_url"))
                    if path is None:
                        raise ValueError(f"Unknown audio {ele}")
                    audio_start = ele.get("audio_start", 0.0)
                    audio_end = ele.get("audio_end", None)
                    if isinstance(path, np.ndarray):
                        audios.append(
                            path[int(_SAFE_AUDIO_SR * audio_start):
                                 None if audio_end is None else int(_SAFE_AUDIO_SR * audio_end)]
                        )
                    else:
                        dur = (audio_end - audio_start) if audio_end is not None else None
                        audios.append(_safe_extract_audio(path, sr=_SAFE_AUDIO_SR,
                                                          offset=audio_start, duration=dur))
                elif use_audio_in_video and ele["type"] == "video":
                    path = ele.get("video", ele.get("video_url"))
                    if path is None:
                        raise ValueError(f"Unknown video {ele}")
                    audio_start = ele.get("video_start", 0.0)
                    audio_end = ele.get("video_end", None)
                    dur = (audio_end - audio_start) if audio_end is not None else None
                    audios.append(_safe_extract_audio(path, sr=_SAFE_AUDIO_SR,
                                                      offset=audio_start, duration=dur))
    if len(audios) == 0:
        audios = None
    return audios

# 注入到所有引用点
import qwen_omni_utils.v2_5.audio_process as _ap_mod
import qwen_omni_utils.v2_5 as _v25_mod
import qwen_omni_utils as _qu_mod
_ap_mod.process_audio_info = _safe_process_audio_info
_v25_mod.process_audio_info = _safe_process_audio_info
if hasattr(_qu_mod, 'process_audio_info'):
    _qu_mod.process_audio_info = _safe_process_audio_info


# 3. 替换 fetch_video（根治：用 ffmpeg subprocess 替代 torchvision/decord/av）
#
# 死锁点：torchvision.io.read_video() / decord 是 C 扩展，SIGALRM 无法打断。
# 某些视频导致 C 扩展内部 futex_wait 永久阻塞。
#
import json as _json
import math as _math
import torch as _torch
from PIL import Image as _Image
from qwen_omni_utils.v2_5.vision_process import (
    smart_nframes, smart_resize, FRAME_FACTOR,
)
# 兼容不同版本 qwen_omni_utils
try:
    from qwen_omni_utils.v2_5.vision_process import IMAGE_FACTOR
except ImportError:
    IMAGE_FACTOR = 28
try:
    from qwen_omni_utils.v2_5.vision_process import (
        VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS,
    )
except ImportError:
    from qwen_omni_utils.v2_5.vision_process import (
        VIDEO_MIN_TOKEN_NUM, VIDEO_MAX_TOKEN_NUM,
    )
    VIDEO_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * IMAGE_FACTOR * IMAGE_FACTOR
    VIDEO_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * IMAGE_FACTOR * IMAGE_FACTOR
    VIDEO_TOTAL_PIXELS = 24883200

_VIDEO_TIMEOUT = 60  # seconds


def _ffprobe_video_info(path: str, timeout: int = 15) -> dict:
    """用 ffprobe 获取视频元数据（fps、总帧数、分辨率）。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}")
    info = _json.loads(r.stdout)
    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    # Parse fps from r_frame_rate (e.g., "30000/1001")
    rfr = stream.get("r_frame_rate", "30/1")
    num, den = rfr.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    # Total frames: try nb_frames, else compute from duration
    nb_frames = stream.get("nb_frames", "N/A")
    if nb_frames != "N/A" and nb_frames != "0":
        total_frames = int(nb_frames)
    else:
        dur = float(stream.get("duration", 0) or info.get("format", {}).get("duration", 0) or 0)
        total_frames = max(1, int(dur * fps))
    return {"width": width, "height": height, "fps": fps, "total_frames": total_frames}


def _safe_fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR,
                      return_video_sample_fps: bool = False):
    """
    安全版 fetch_video：用 ffmpeg subprocess 替代 torchvision/decord。

    单次 ffmpeg 调用 + select 滤镜批量提取所有帧，比逐帧调用快 10x。
    与原版输出格式完全一致：返回 (video_tensor, sample_fps)
    其中 video_tensor 是 float32 (T, C, H, W)，已 resize。
    """
    if not isinstance(ele.get("video"), str):
        # 非文件路径（PIL 图片列表等），走原始逻辑
        from qwen_omni_utils.v2_5.vision_process import fetch_video as _orig_fv
        return _orig_fv(ele, image_factor=image_factor,
                        return_video_sample_fps=return_video_sample_fps)

    video_path = ele["video"]
    if video_path.startswith("file://"):
        video_path = video_path[7:]

    # 1. 获取视频信息
    info = _ffprobe_video_info(video_path)
    total_frames = info["total_frames"]
    video_fps = info["fps"]
    width, height = info["width"], info["height"]

    # 2. 计算需要多少帧
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    # 3. 单次 ffmpeg 调用提取所有帧（select 滤镜 + rawvideo pipe）
    indices = _torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    # 构建 select 滤镜表达式
    select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select='{select_expr}'",
        "-vsync", "vfr",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    frame_size = width * height * 3

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=_VIDEO_TIMEOUT)
        if r.returncode != 0 or len(r.stdout) < frame_size:
            raise RuntimeError(f"ffmpeg returned {r.returncode}, got {len(r.stdout)} bytes")
        raw = r.stdout
        nframes_got = len(raw) // frame_size
        frames = []
        for i in range(nframes_got):
            fb = raw[i * frame_size:(i + 1) * frame_size]
            frame = np.frombuffer(fb, dtype=np.uint8).reshape(height, width, 3)
            frames.append(_torch.from_numpy(frame.copy()).permute(2, 0, 1))
        # 如果帧数不足，用最后一帧填充
        while len(frames) < nframes:
            frames.append(frames[-1] if frames else _torch.zeros(3, height, width, dtype=_torch.uint8))
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"ffmpeg video extraction timeout ({_VIDEO_TIMEOUT}s) for {video_path}")

    video = _torch.stack(frames)  # (T, C, H, W)

    # 4. Resize（复用原版逻辑）
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height, width, factor=image_factor,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )

    from torchvision.transforms.functional import resize as _tv_resize
    from torchvision.transforms import InterpolationMode as _IM
    video = _tv_resize(
        video, [resized_height, resized_width],
        interpolation=_IM.BICUBIC, antialias=True,
    ).float()

    if return_video_sample_fps:
        return video, sample_fps
    return video


# 注入 fetch_video 到所有引用点
import qwen_omni_utils.v2_5.vision_process as _vp_mod
_vp_mod.fetch_video = _safe_fetch_video
_v25_mod.fetch_video = _safe_fetch_video
if hasattr(_qu_mod, 'fetch_video'):
    _qu_mod.fetch_video = _safe_fetch_video
# === End monkey-patch ===

import argparse
import csv
import json
import os
import re
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.pipeline import SparseInferencePipeline, PipelineResult
from fasteromni.encoder_cache import (
    EncoderCacheHook,
    patch_pipeline_run_inference,
    restore_pipeline_run_inference,
)

# ── 路径 ──────────────────────────────────────────────────

VMME_ANNOTATIONS = "/root/autodl-tmp/data/Video-MME/annotations/video_mme_test.json"
VMME_VIDEO_DIR = "/root/autodl-tmp/data/Video-MME/videos/data"


# ── 数据加载 ──────────────────────────────────────────────

@dataclass
class VideoMMESample:
    """Video-MME 单条 QA"""
    video_id: str           # e.g. "001"
    video_file_id: str      # e.g. "fFjv93ACGo8" (YouTube ID)
    video_path: str
    question_id: str
    question: str
    options: List[str]      # ["A. ...", "B. ...", "C. ...", "D. ..."]
    answer: str             # "A" / "B" / "C" / "D"
    duration: str           # "short" / "medium" / "long"
    domain: str
    sub_category: str
    task_type: str


def load_videomme_samples(max_videos: int = 0) -> List[VideoMMESample]:
    """加载 Video-MME 样本（仅已下载的视频）"""
    with open(VMME_ANNOTATIONS) as f:
        data = json.load(f)

    # 已下载的视频
    downloaded = {}
    for fname in os.listdir(VMME_VIDEO_DIR):
        if fname.endswith(".mp4"):
            vid = os.path.splitext(fname)[0]
            downloaded[vid] = os.path.join(VMME_VIDEO_DIR, fname)

    # 匹配
    samples = []
    matched_videos = set()
    for item in data:
        file_id = item["videoID"]
        if file_id not in downloaded:
            continue

        matched_videos.add(file_id)
        if max_videos > 0 and len(matched_videos) > max_videos:
            break

        samples.append(VideoMMESample(
            video_id=item["video_id"],
            video_file_id=file_id,
            video_path=downloaded[file_id],
            question_id=item["question_id"],
            question=item["question"],
            options=item["options"],
            answer=item["answer"],
            duration=item["duration"],
            domain=item["domain"],
            sub_category=item.get("sub_category", ""),
            task_type=item.get("task_type", ""),
        ))

    return samples


# ── Prompt 格式化 ─────────────────────────────────────────

def format_mcq_prompt(question: str, options: List[str]) -> str:
    """
    格式化选择题 prompt。

    输出格式：
    Question: <question>
    A. <option_a>
    B. <option_b>
    C. <option_c>
    D. <option_d>
    Answer with the option letter only (A, B, C, or D).
    """
    opts = "\n".join(options)
    return (
        f"{question}\n{opts}\n"
        f"Answer with the option letter only (A, B, C, or D)."
    )


def extract_answer_letter(output: str) -> Optional[str]:
    """
    从模型输出中提取选项字母 A/B/C/D。

    策略（按优先级）：
    1. 输出恰好是单个字母
    2. 输出以字母开头（如 "A." 或 "A. Apples"）
    3. 在输出中找第一个出现的 A/B/C/D
    """
    text = output.strip()

    # 1. 单个字母
    if text.upper() in ("A", "B", "C", "D"):
        return text.upper()

    # 2. 以字母开头
    m = re.match(r"^([A-Da-d])[.\s,)]", text)
    if m:
        return m.group(1).upper()

    # 3. 找第一个 A/B/C/D（作为独立字母出现）
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        letter = m.group(1).upper()
        if letter in ("A", "B", "C", "D"):
            return letter

    # 4. 兜底：在整个文本中找
    for ch in text.upper():
        if ch in ("A", "B", "C", "D"):
            return ch

    return None


# ── 单条评估 ──────────────────────────────────────────────

@dataclass
class EvalRecord:
    """单条评估记录"""
    question_id: str
    video_file_id: str
    duration: str
    domain: str
    task_type: str
    mode: str               # "baseline" / "sparse" / "sparse_no_audio"
    keep_ratio: float = 0.0
    alpha: float = 0.0
    gt_answer: str = ""
    pred_answer: Optional[str] = None
    pred_raw: str = ""
    correct: bool = False
    generate_ms: float = 0.0
    total_ms: float = 0.0
    visual_tokens: int = 0
    audio_tokens: int = 0
    total_tokens: int = 0
    num_frames: int = 0
    error: str = ""


class _Timeout:
    """
    单条推理超时保护（内核级）。

    threading.Timer 无法工作：GIL 被 C 扩展独占时 Python 线程全部阻塞。
    signal.SIG_DFL 方案：SIGALRM 由内核投递，SIG_DFL 直接终止进程。
    完全绕过 GIL / Python 解释器，100% 可靠。
    配合增量 CSV（已 flush），已完成数据不丢失。
    """
    def __init__(self, seconds: int = 120):
        self.seconds = seconds
        self._old_handler = None
    def __enter__(self):
        self._old_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(self.seconds)
        return self
    def __exit__(self, *args):
        signal.alarm(0)
        if self._old_handler is not None:
            signal.signal(signal.SIGALRM, self._old_handler)


def run_single(
    pipe: SparseInferencePipeline,
    sample: VideoMMESample,
    mode: str,
    keep_ratio: float = 0.5,
    alpha: float = 0.5,
    max_new_tokens: int = 16,
    max_frames: int = 64,
    timeout_sec: int = 120,
    gop_filter_mode: str = "fixed",
    min_gop_frames: int = 10,
    min_frames: int = 8,
) -> EvalRecord:
    """运行单条 QA 评估"""
    prompt = format_mcq_prompt(sample.question, sample.options)
    record = EvalRecord(
        question_id=sample.question_id,
        video_file_id=sample.video_file_id,
        duration=sample.duration,
        domain=sample.domain,
        task_type=sample.task_type,
        mode=mode,
        keep_ratio=keep_ratio,
        alpha=alpha,
        gt_answer=sample.answer,
    )

    try:
        with _Timeout(timeout_sec):
            if mode == "baseline":
                r = pipe.run_baseline(sample.video_path, prompt, max_new_tokens, max_frames=max_frames)
            elif mode == "text_only":
                r = pipe.run_text_only(sample.video_path, prompt, max_new_tokens)
            elif mode == "audio_only":
                r = pipe.run_audio_only(sample.video_path, prompt, max_new_tokens)
            elif mode == "video_only":
                r = pipe.run_video_only(sample.video_path, prompt, max_new_tokens, max_frames=max_frames)
            elif mode == "sparse":
                r = pipe.run_sparse(
                    sample.video_path, prompt, max_new_tokens,
                    alpha=alpha, keep_ratio=keep_ratio,
                    max_frames=max_frames,
                    min_frames=min_frames,
                )
            elif mode == "sparse_no_audio":
                r = pipe.run_sparse(
                    sample.video_path, prompt, max_new_tokens,
                    alpha=alpha, keep_ratio=keep_ratio,
                    skip_audio=True,
                    max_frames=max_frames,
                    min_frames=min_frames,
                )
            elif mode in ("naive_uniform", "naive_random", "naive_iframe"):
                _strategy_map = {"naive_iframe": "iframe_uniform"}
                strategy = _strategy_map.get(mode, mode.replace("naive_", ""))
                r = pipe.run_naive(
                    sample.video_path, prompt,
                    strategy=strategy,
                    max_new_tokens=max_new_tokens,
                    keep_ratio=keep_ratio,
                    max_frames=max_frames,
                    gop_filter_mode=gop_filter_mode,
                    min_gop_frames=min_gop_frames,
                    min_frames=min_frames,
                )
            elif mode == "adaptive":
                r = pipe.run_adaptive(
                    video_path=sample.video_path, question=prompt,
                    keep_ratio=keep_ratio, alpha=alpha,
                    max_frames=max_frames, min_frames=min_frames,
                    max_new_tokens=max_new_tokens,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if r.error:
                record.error = r.error
            else:
                record.pred_raw = r.output_text
                record.pred_answer = extract_answer_letter(r.output_text)
                record.correct = (record.pred_answer == record.gt_answer)
                record.generate_ms = r.generate_ms
                record.total_ms = r.total_ms
                record.visual_tokens = r.visual_tokens
                record.audio_tokens = r.audio_tokens
                record.total_tokens = r.total_tokens
                record.num_frames = r.num_frames_input

    except torch.cuda.OutOfMemoryError:
        record.error = "OOM"
        pipe._clear_gpu()
    except TimeoutError:
        record.error = "TIMEOUT"
        pipe._clear_gpu()
    except Exception as e:
        record.error = str(e)

    return record


# ── 批量评估 ──────────────────────────────────────────────

import torch

_CSV_FIELDNAMES = [
    "question_id", "video_file_id", "duration", "domain", "task_type",
    "mode", "keep_ratio", "alpha", "gt_answer", "pred_answer", "correct",
    "generate_ms", "total_ms", "visual_tokens", "audio_tokens", "total_tokens",
    "num_frames", "error", "pred_raw",
]


def run_evaluation(
    pipe: SparseInferencePipeline,
    samples: List[VideoMMESample],
    mode: str,
    keep_ratio: float = 0.5,
    alpha: float = 0.5,
    max_new_tokens: int = 16,
    max_frames: int = 64,
    incremental_csv: str = "",
    gop_filter_mode: str = "fixed",
    min_gop_frames: int = 10,
    min_frames: int = 8,
    encoder_cache: bool = False,
) -> List[EvalRecord]:
    """运行一组评估（支持自动恢复：跳过增量 CSV 中已完成的样本）"""
    # 自动恢复：读取已有增量 CSV，跳过已完成的 question_id
    completed_qids = set()
    if incremental_csv and os.path.exists(incremental_csv) and os.path.getsize(incremental_csv) > 0:
        with open(incremental_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 只认有效记录（有预测结果且有生成时间），忽略垃圾数据
                if row.get("pred_answer", "").strip() and float(row.get("generate_ms", 0)) > 0:
                    completed_qids.add(row["question_id"])
        if completed_qids:
            print(f"  [resume] Found {len(completed_qids)} completed samples in {os.path.basename(incremental_csv)}, skipping", flush=True)

    # 增量 CSV 写入：每条结果实时保存，防止中断丢数据
    csv_writer = None
    csv_file = None
    if incremental_csv:
        csv_file = open(incremental_csv, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDNAMES)
        if os.path.getsize(incremental_csv) == 0:
            csv_writer.writeheader()

    records = []
    correct = 0
    total = 0
    errors = 0
    skipped = 0

    # 按视频分组以支持预取
    from collections import OrderedDict
    video_samples = OrderedDict()
    for sample in samples:
        if sample.question_id not in completed_qids:
            if sample.video_file_id not in video_samples:
                video_samples[sample.video_file_id] = []
            video_samples[sample.video_file_id].append(sample)

    video_ids = list(video_samples.keys())

    # EncoderCache setup
    cache = None
    if encoder_cache:
        pipe.load_model()
        cache = EncoderCacheHook(pipe._model, pipe._proc)
        cache.enable()
        print(f"  [encoder-cache] Enabled", flush=True)

    try:
        for vid_idx, video_id in enumerate(video_ids):
            video_samples_list = video_samples[video_id]

            # 预取下一个视频
            if vid_idx + 1 < len(video_ids):
                next_video_id = video_ids[vid_idx + 1]
                next_sample = video_samples[next_video_id][0]
                if mode in ("naive_uniform", "naive_random", "naive_iframe"):
                    strategy = {"naive_iframe": "iframe_uniform"}.get(mode, mode.replace("naive_", ""))
                    pipe.prefetch_video(
                        next_sample.video_path, "naive",
                        strategy=strategy, question="", keep_ratio=keep_ratio,
                        max_frames=max_frames, gop_filter_mode=gop_filter_mode,
                        min_gop_frames=min_gop_frames, min_frames=min_frames
                    )
                elif mode == "sparse":
                    pipe.prefetch_video(
                        next_sample.video_path, "sparse",
                        question="", alpha=alpha, keep_ratio=keep_ratio,
                        max_frames=max_frames, min_frames=min_frames
                    )

            # EncoderCache: set cache key for this video
            original_run_inference = None
            if cache is not None:
                cache_key = cache.make_cache_key(
                    video_samples_list[0].video_path,
                    max_frames=max_frames,
                    keep_ratio=keep_ratio,
                    selection_strategy=f"{mode}_{gop_filter_mode}_mingop{min_gop_frames}",
                )
                original_run_inference = patch_pipeline_run_inference(pipe, cache, cache_key)

            # 处理当前视频的所有问题
            for sample in video_samples_list:
                rec = run_single(
                    pipe, sample, mode, keep_ratio, alpha, max_new_tokens, max_frames,
                    gop_filter_mode=gop_filter_mode, min_gop_frames=min_gop_frames,
                    min_frames=min_frames,
                )
                records.append(rec)

                # 实时写入 CSV
                if csv_writer:
                    csv_writer.writerow({k: getattr(rec, k) for k in _CSV_FIELDNAMES})
                    csv_file.flush()

                if rec.error:
                    errors += 1
                    status = f"ERR:{rec.error[:20]}"
                else:
                    total += 1
                    correct += int(rec.correct)
                    mark = "\u2713" if rec.correct else "\u2717"
                    status = f"{mark} pred={rec.pred_answer} gt={rec.gt_answer}"

                # 每条都输出进度（方便 tmux 实时查看）
                done = len(records)
                remaining = sum(len(v) for v in video_samples.values())
                acc = correct / total * 100 if total > 0 else 0
                print(f"  [{done}/{remaining}] acc={acc:.1f}% ({correct}/{total}) err={errors} "
                      f"| {sample.duration:7s} | {sample.video_file_id[:15]:15s} | {status}", flush=True)

            # EncoderCache: restore after each video
            if original_run_inference is not None:
                restore_pipeline_run_inference(pipe, original_run_inference)
                cache.clear_cache()

    finally:
        if csv_file:
            csv_file.close()
        if cache is not None:
            cache.disable()
            v_hits = cache.video_cache_hits
            v_miss = cache.video_cache_misses
            a_hits = cache.audio_cache_hits
            a_miss = cache.audio_cache_misses
            v_total = v_hits + v_miss
            a_total = a_hits + a_miss
            if v_total > 0:
                print(f"\n[EncoderCache] video: {v_hits}/{v_total} ({v_hits/v_total*100:.0f}%) "
                      f"audio: {a_hits}/{a_total} ({a_hits/a_total*100:.0f}%)", flush=True)

    # 输出预取统计
    stats = pipe.prefetch_stats()
    if stats['hits'] + stats['misses'] > 0:
        print(f"\n[Prefetch] hit_rate={stats['hit_rate']:.1%} hits={stats['hits']} misses={stats['misses']} "
              f"timeouts={stats['timeouts']} evictions={stats['evictions']}", flush=True)

    return records


# ── 结果汇总 ──────────────────────────────────────────────

def summarize_records(records: List[EvalRecord], label: str = "") -> Dict:
    """汇总评估记录"""
    valid = [r for r in records if not r.error]
    by_dur = defaultdict(list)
    for r in valid:
        by_dur[r.duration].append(r)

    summary = {
        "label": label,
        "total_samples": len(records),
        "valid_samples": len(valid),
        "errors": len(records) - len(valid),
        "overall_accuracy": sum(r.correct for r in valid) / len(valid) * 100 if valid else 0,
        "avg_generate_ms": sum(r.generate_ms for r in valid) / len(valid) if valid else 0,
        "avg_visual_tokens": sum(r.visual_tokens for r in valid) / len(valid) if valid else 0,
        "by_duration": {},
    }

    for dur in ["short", "medium", "long"]:
        recs = by_dur.get(dur, [])
        if recs:
            summary["by_duration"][dur] = {
                "count": len(recs),
                "accuracy": sum(r.correct for r in recs) / len(recs) * 100,
                "avg_generate_ms": sum(r.generate_ms for r in recs) / len(recs),
                "avg_visual_tokens": sum(r.visual_tokens for r in recs) / len(recs),
            }

    return summary


def print_summary(summaries: List[Dict]):
    """打印汇总表格"""
    print(f"\n{'='*90}")
    print(f"VIDEO-MME EVALUATION SUMMARY")
    print(f"{'='*90}")

    # 总表
    print(f"\n{'Mode':>20} | {'Accuracy':>10} | {'N':>5} | {'Err':>4} | "
          f"{'Gen(ms)':>10} | {'VisTok':>8}")
    print("-" * 75)
    for s in summaries:
        print(f"{s['label']:>20} | {s['overall_accuracy']:>9.1f}% | "
              f"{s['valid_samples']:>5} | {s['errors']:>4} | "
              f"{s['avg_generate_ms']:>9.0f} | {s['avg_visual_tokens']:>8.0f}")

    # 按时长分组
    print(f"\n{'Mode':>20} | {'short':>10} | {'medium':>10} | {'long':>10}")
    print("-" * 60)
    for s in summaries:
        parts = []
        for dur in ["short", "medium", "long"]:
            d = s["by_duration"].get(dur)
            if d:
                parts.append(f"{d['accuracy']:>5.1f}%({d['count']})")
            else:
                parts.append(f"{'N/A':>10}")
        print(f"{s['label']:>20} | {'  |  '.join(parts)}")


# ── 保存 ──────────────────────────────────────────────────

def save_results(records: List[EvalRecord], summaries: List[Dict], out_dir: str, tag: str = "eval"):
    """保存结果"""
    os.makedirs(out_dir, exist_ok=True)

    # CSV 详细记录
    csv_path = os.path.join(out_dir, f"videomme_{tag}_details.csv")
    fieldnames = [
        "question_id", "video_file_id", "duration", "domain", "task_type",
        "mode", "keep_ratio", "alpha", "gt_answer", "pred_answer", "correct",
        "generate_ms", "total_ms", "visual_tokens", "audio_tokens", "total_tokens",
        "num_frames", "error", "pred_raw",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: getattr(r, k) for k in fieldnames})

    # JSON 汇总
    json_path = os.path.join(out_dir, f"videomme_{tag}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_dir}")
    print(f"  Details: {csv_path}")
    print(f"  Summary: {json_path}")
    return csv_path


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Video-MME Evaluation")
    parser.add_argument("--max-videos", type=int, default=0,
                        help="Max videos to evaluate (0 = all downloaded)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=32,
                        help="Max frames for baseline (0=unlimited, 32 for 32GB GPU)")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", choices=["short", "medium", "long", "all"], default="all",
                        help="Filter by video duration category")
    parser.add_argument("--modes", nargs="+", default=["baseline", "sparse"],
                        choices=["baseline", "text_only", "audio_only", "video_only",
                                 "sparse", "sparse_no_audio",
                                 "naive_uniform", "naive_random", "naive_iframe",
                                 "adaptive"],
                        help="Modes to evaluate")
    parser.add_argument("--sweep", choices=["keep_ratio", "alpha", "none"], default="none",
                        help="Run ablation sweep")
    parser.add_argument("--min-frames", type=int, default=8,
                        help="Min frames floor for sparse/naive modes (prevents short-video degradation)")
    parser.add_argument("--gop-filter-mode", choices=["fixed", "adaptive"], default="fixed",
                        help="Naive iframe GOP filter mode: fixed restores scheme A, adaptive keeps scheme C")
    parser.add_argument("--min-gop-frames", type=int, default=10,
                        help="Fixed GOP threshold for naive iframe when --gop-filter-mode=fixed")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/results/fasteromni/videomme")
    parser.add_argument("--prefetch-capacity", type=int, default=2,
                        help="Prefetch buffer capacity (0=disabled, 2=default)")
    parser.add_argument("--encoder-cache", action="store_true", default=False,
                        help="Enable encoder cache (reuse ViT encoding for same video)")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("FasterOmni - Video-MME Evaluation")
    print(
        f"modes={args.modes}, keep_ratio={args.keep_ratio}, alpha={args.alpha}, "
        f"gop_filter_mode={args.gop_filter_mode}, min_gop_frames={args.min_gop_frames}, "
        f"min_frames={args.min_frames}"
    )
    print("=" * 70)

    # 加载样本
    samples = load_videomme_samples(max_videos=args.max_videos)
    if args.duration != "all":
        samples = [s for s in samples if s.duration == args.duration]
    n_vids = len(set(s.video_file_id for s in samples))
    dur_counts = defaultdict(int)
    for s in samples:
        dur_counts[s.duration] += 1
    print(f"Loaded {len(samples)} QA from {n_vids} videos")
    print(f"  short={dur_counts['short']}, medium={dur_counts['medium']}, long={dur_counts['long']}")

    # 初始化
    pipe = SparseInferencePipeline(dtype="bf16", prefetch_capacity=args.prefetch_capacity)

    all_records = []
    all_summaries = []

    # 增量 CSV 路径
    os.makedirs(args.out_dir, exist_ok=True)

    if args.sweep == "none":
        # 单配置评估 — 每个 mode 独立保存到子目录
        for mode in args.modes:
            mode_dir = os.path.join(args.out_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            inc_csv = os.path.join(mode_dir, f"{mode}_details.csv")
            print(f"\n{'='*60}")
            print(f"Running {mode.upper()} ({len(samples)} samples)")
            print(f"  Results → {mode_dir}")
            print(f"{'='*60}")
            records = run_evaluation(
                pipe, samples, mode,
                keep_ratio=args.keep_ratio, alpha=args.alpha,
                max_new_tokens=args.max_new_tokens,
                max_frames=args.max_frames,
                incremental_csv=inc_csv,
                gop_filter_mode=args.gop_filter_mode,
                min_gop_frames=args.min_gop_frames,
                min_frames=args.min_frames,
                encoder_cache=args.encoder_cache,
            )
            all_records.extend(records)
            label = f"{mode}(kr={args.keep_ratio})" if mode != "baseline" else "baseline"
            summary = summarize_records(records, label=label)
            all_summaries.append(summary)
            # 每个 mode 独立保存
            save_results(records, [summary], mode_dir, tag=mode)

        print_summary(all_summaries)
        # 同时保存汇总到根目录
        save_results(all_records, all_summaries, args.out_dir, tag="combined")

    elif args.sweep == "keep_ratio":
        # keep_ratio 消融
        keep_ratios = [0.2, 0.3, 0.5, 0.7, 0.9]

        # baseline 只跑一次
        inc_csv_base = os.path.join(args.out_dir, "baseline_inc.csv")
        print(f"\n{'='*60}")
        print(f"Running BASELINE ({len(samples)} samples)")
        print(f"  Incremental CSV → {inc_csv_base}")
        print(f"{'='*60}")
        base_records = run_evaluation(
            pipe, samples, "baseline",
            max_new_tokens=args.max_new_tokens,
            max_frames=args.max_frames,
            incremental_csv=inc_csv_base,
        )
        all_records.extend(base_records)
        base_summary = summarize_records(base_records, label="baseline")
        all_summaries.append(base_summary)

        for kr in keep_ratios:
            inc_csv_kr = os.path.join(args.out_dir, f"sparse_kr{kr}_inc.csv")
            print(f"\n{'='*60}")
            print(f"Running SPARSE kr={kr} ({len(samples)} samples)")
            print(f"  Incremental CSV → {inc_csv_kr}")
            print(f"{'='*60}")
            records = run_evaluation(
                pipe, samples, "sparse",
                keep_ratio=kr, alpha=args.alpha,
                max_new_tokens=args.max_new_tokens,
                max_frames=args.max_frames,
                incremental_csv=inc_csv_kr,
                min_frames=args.min_frames,
            )
            all_records.extend(records)
            summary = summarize_records(records, label=f"sparse(kr={kr})")
            all_summaries.append(summary)

        print_summary(all_summaries)
        save_results(all_records, all_summaries, args.out_dir, tag="ablation_kr")

    elif args.sweep == "alpha":
        alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

        print(f"\n{'='*60}")
        print(f"Running BASELINE ({len(samples)} samples)")
        print(f"{'='*60}")
        base_records = run_evaluation(
            pipe, samples, "baseline",
            max_new_tokens=args.max_new_tokens,
            max_frames=args.max_frames,
        )
        all_records.extend(base_records)
        all_summaries.append(summarize_records(base_records, label="baseline"))

        for a in alphas:
            print(f"\n{'='*60}")
            print(f"Running SPARSE alpha={a} ({len(samples)} samples)")
            print(f"{'='*60}")
            records = run_evaluation(
                pipe, samples, "sparse",
                keep_ratio=args.keep_ratio, alpha=a,
                max_new_tokens=args.max_new_tokens,
                max_frames=args.max_frames,
                min_frames=args.min_frames,
            )
            all_records.extend(records)
            all_summaries.append(summarize_records(records, label=f"sparse(a={a})"))

        print_summary(all_summaries)
        save_results(all_records, all_summaries, args.out_dir, tag="ablation_alpha")


if __name__ == "__main__":
    main()
