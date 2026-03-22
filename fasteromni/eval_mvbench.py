"""
MVBench 评估脚本

选择题格式，评估零歧义（pred letter == gt letter）。
复用 FasterOmni pipeline 的全部推理模式：
- baseline / sparse / sparse_no_audio
- text_only / audio_only / video_only
- naive_uniform / naive_random / naive_iframe

Usage:
    # 快速验证（每任务 5 题）
    python fasteromni/eval_mvbench.py --max-per-task 5 --modes baseline

    # 全量评估（核心模式）
    python fasteromni/eval_mvbench.py --modes baseline sparse naive_iframe

    # 指定任务
    python fasteromni/eval_mvbench.py --tasks action_antonym moving_count --modes baseline
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

# 兼容新版 qwen_omni_utils（使用 TOKEN_NUM 而不是 PIXELS）
try:
    from qwen_omni_utils.v2_5.vision_process import IMAGE_FACTOR
except ImportError:
    IMAGE_FACTOR = 28  # ViT patch size

try:
    from qwen_omni_utils.v2_5.vision_process import (
        VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS
    )
except ImportError:
    from qwen_omni_utils.v2_5.vision_process import (
        VIDEO_MIN_TOKEN_NUM, VIDEO_MAX_TOKEN_NUM
    )
    VIDEO_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * IMAGE_FACTOR * IMAGE_FACTOR
    VIDEO_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * IMAGE_FACTOR * IMAGE_FACTOR
    VIDEO_TOTAL_PIXELS = 24883200  # 默认值


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
import random
import re
import signal
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.pipeline import SparseInferencePipeline


# ── 默认路径 ──────────────────────────────────────────────

MVBENCH_JSON_DIR = "/root/autodl-tmp/data/MVBench/json/"
MVBENCH_VIDEO_DIR = "/root/autodl-tmp/data/MVBench/video/"
DEFAULT_SKIP_TASKS = ["episodic_reasoning", "fine_grained_pose"]

VALID_LETTERS = "ABCDE"
VALID_LETTER_SET = set(VALID_LETTERS)


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class MVBenchSample:
    task_type: str
    video_field: str
    video_path: str
    question: str
    candidates: List[str]
    answer_text: str
    answer_letter: str
    sample_id: str


@dataclass
class MVBenchRecord:
    sample_id: str
    task_type: str
    video_field: str
    mode: str
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
    adaptive_min_gop: int = 0
    min_frames: int = 0
    error: str = ""


_CSV_FIELDNAMES = [
    "sample_id", "task_type", "video_field", "mode", "keep_ratio", "alpha",
    "gt_answer", "pred_answer", "correct", "generate_ms", "total_ms",
    "visual_tokens", "audio_tokens", "total_tokens", "num_frames", 
    "adaptive_min_gop", "min_frames", "error", "pred_raw",
]


# ── 视频路径解析 ──────────────────────────────────────────

_RESOLVE_PATH_CACHE: Dict[Tuple[str, str], Optional[str]] = {}
_RESOLVE_SUBDIR_CACHE: Dict[str, List[str]] = {}
_RESOLVE_BASENAME_INDEX: Dict[str, Dict[str, str]] = {}


def _build_basename_index(video_base: str) -> Dict[str, str]:
    """递归构建 basename -> full path 索引（仅首次构建）。"""
    index: Dict[str, str] = {}
    for root_d, _dirs, files in os.walk(video_base):
        for fname in files:
            index.setdefault(fname, os.path.join(root_d, fname))
    return index


def resolve_video_path(video_field: str, video_base: str) -> Optional[str]:
    """
    解析 JSON 中的 video 字段到实际文件路径。

    策略：
    1) 若 video_field 是可用绝对路径，直接返回
    2) 直接拼接 video_base/video_field
    3) 遍历 video_base 下每个顶级子目录，拼接后检查存在
    4) 兜底：按 basename 在 video_base 递归搜索
    """
    if not video_field:
        return None

    key = (video_base, video_field)
    if key in _RESOLVE_PATH_CACHE:
        return _RESOLVE_PATH_CACHE[key]

    if os.path.isabs(video_field) and os.path.exists(video_field):
        _RESOLVE_PATH_CACHE[key] = video_field
        return video_field

    direct = os.path.join(video_base, video_field)
    if os.path.exists(direct):
        _RESOLVE_PATH_CACHE[key] = direct
        return direct

    subdirs = _RESOLVE_SUBDIR_CACHE.get(video_base)
    if subdirs is None:
        try:
            subdirs = [
                d for d in os.listdir(video_base)
                if os.path.isdir(os.path.join(video_base, d))
            ]
        except FileNotFoundError:
            subdirs = []
        _RESOLVE_SUBDIR_CACHE[video_base] = subdirs

    for subdir in subdirs:
        candidate = os.path.join(video_base, subdir, video_field)
        if os.path.exists(candidate):
            _RESOLVE_PATH_CACHE[key] = candidate
            return candidate

    basename = os.path.basename(video_field)
    if not basename:
        _RESOLVE_PATH_CACHE[key] = None
        return None

    basename_index = _RESOLVE_BASENAME_INDEX.get(video_base)
    if basename_index is None:
        basename_index = _build_basename_index(video_base)
        _RESOLVE_BASENAME_INDEX[video_base] = basename_index

    found = basename_index.get(basename)
    _RESOLVE_PATH_CACHE[key] = found
    return found


# ── 数据加载 ──────────────────────────────────────────────

def _find_answer_index(answer_text: str, candidates: List[str]) -> Optional[int]:
    """先精确匹配，再做 strip 匹配，返回候选项索引。"""
    for i, c in enumerate(candidates):
        if c == answer_text:
            return i
    norm = answer_text.strip()
    for i, c in enumerate(candidates):
        if c.strip() == norm:
            return i
    return None


def load_mvbench_samples(
    json_dir: str = MVBENCH_JSON_DIR,
    video_dir: str = MVBENCH_VIDEO_DIR,
    skip_tasks: List[str] = DEFAULT_SKIP_TASKS,
    max_per_task: int = 0,
) -> List[MVBenchSample]:
    """加载 MVBench 样本（自动跳过不可用任务 / 异常样本）。"""
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"json_dir not found: {json_dir}")
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"video_dir not found: {video_dir}")

    skip_set = set(skip_tasks)
    json_files = sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))

    samples: List[MVBenchSample] = []

    for fname in json_files:
        task_type = os.path.splitext(fname)[0]
        if task_type in skip_set:
            print(f"[skip-task] {task_type}", flush=True)
            continue

        fpath = os.path.join(json_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {fname}: {e}", flush=True)
            continue

        if not isinstance(items, list):
            print(f"[WARN] {fname} is not a list, skip", flush=True)
            continue

        loaded_this_task = 0
        for idx, item in enumerate(items):
            if max_per_task > 0 and loaded_this_task >= max_per_task:
                break

            if not isinstance(item, dict):
                print(f"[WARN] {task_type}[{idx}] is not dict, skip", flush=True)
                continue

            video_field = str(item.get("video", "")).strip()
            question = str(item.get("question", "")).strip()
            answer_text = str(item.get("answer", "")).strip()
            raw_candidates = item.get("candidates", [])

            if not video_field:
                print(f"[WARN] {task_type}[{idx}] empty video field, skip", flush=True)
                continue
            if not question:
                print(f"[WARN] {task_type}[{idx}] empty question, skip", flush=True)
                continue
            if not isinstance(raw_candidates, list) or len(raw_candidates) == 0:
                print(f"[WARN] {task_type}[{idx}] invalid candidates, skip", flush=True)
                continue

            candidates = [str(c).strip() for c in raw_candidates]
            if len(candidates) > len(VALID_LETTERS):
                print(f"[WARN] {task_type}[{idx}] has {len(candidates)} candidates (>5), skip", flush=True)
                continue

            ans_idx = _find_answer_index(answer_text, candidates)
            if ans_idx is None:
                print(
                    f"[WARN] {task_type}[{idx}] answer not in candidates: {answer_text!r}, skip",
                    flush=True,
                )
                continue

            video_path = resolve_video_path(video_field, video_dir)
            if video_path is None:
                print(
                    f"[WARN] {task_type}[{idx}] video path not found for {video_field!r}, skip",
                    flush=True,
                )
                continue

            sample = MVBenchSample(
                task_type=task_type,
                video_field=video_field,
                video_path=video_path,
                question=question,
                candidates=candidates,
                answer_text=answer_text,
                answer_letter=VALID_LETTERS[ans_idx],
                sample_id=f"{task_type}_{idx}",
            )
            samples.append(sample)
            loaded_this_task += 1

        print(f"  [{task_type}] loaded {loaded_this_task}/{len(items)}", flush=True)

    return samples


# ── Prompt & 答案提取 ─────────────────────────────────────

def format_mvbench_prompt(question: str, candidates: List[str]) -> str:
    """
    输出格式：
    {question}
    A. ...
    B. ...
    ...
    Answer with the option letter only.
    """
    lines = [question.strip()]
    for i, cand in enumerate(candidates):
        lines.append(f"{VALID_LETTERS[i]}. {cand}")
    lines.append("Answer with the option letter only.")
    return "\n".join(lines)


def extract_answer_letter(output: str) -> Optional[str]:
    """
    从模型输出中提取选项字母 A/B/C/D/E。

    策略（按优先级）:
    1) 输出恰好是单个字母
    2) 输出以字母开头（如 "A." 或 "C) ..."）
    3) 在输出中找第一个独立字母
    4) 兜底：扫描文本中的第一个合法字母
    """
    text = output.strip()
    if not text:
        return None

    upper = text.upper()

    if upper in VALID_LETTER_SET:
        return upper

    m = re.match(r"^([A-Ea-e])[.\s,)]", text)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-Ea-e])\b", text)
    if m:
        letter = m.group(1).upper()
        if letter in VALID_LETTER_SET:
            return letter

    for ch in upper:
        if ch in VALID_LETTER_SET:
            return ch

    return None


# ── 超时保护 ──────────────────────────────────────────────

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


# ── 单条评估 ──────────────────────────────────────────────

def run_single(
    pipe: SparseInferencePipeline,
    sample: MVBenchSample,
    mode: str,
    keep_ratio: float = 0.5,
    alpha: float = 0.5,
    max_new_tokens: int = 16,
    max_frames: int = 0,
    timeout_sec: int = 120,
    min_frames: int = 8,
) -> MVBenchRecord:
    """运行单条 MVBench 评估。"""
    prompt = format_mvbench_prompt(sample.question, sample.candidates)
    record = MVBenchRecord(
        sample_id=sample.sample_id,
        task_type=sample.task_type,
        video_field=sample.video_field,
        mode=mode,
        keep_ratio=keep_ratio,
        alpha=alpha,
        gt_answer=sample.answer_letter,
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
                # Legacy sparse mode: keep this branch for archived AV-LRM comparisons.
                r = pipe.run_sparse(
                    sample.video_path,
                    prompt,
                    max_new_tokens,
                    alpha=alpha,
                    keep_ratio=keep_ratio,
                    max_frames=max_frames,
                    min_frames=min_frames,
                )
            elif mode == "sparse_no_audio":
                # Legacy sparse ablation: same old path, just without audio features.
                r = pipe.run_sparse(
                    sample.video_path,
                    prompt,
                    max_new_tokens,
                    alpha=alpha,
                    keep_ratio=keep_ratio,
                    skip_audio=True,
                    max_frames=max_frames,
                    min_frames=min_frames,
                )
            elif mode in ("naive_uniform", "naive_random", "naive_iframe"):
                strategy_map = {"naive_iframe": "iframe_uniform"}
                strategy = strategy_map.get(mode, mode.replace("naive_", ""))
                r = pipe.run_naive(
                    sample.video_path,
                    prompt,
                    strategy=strategy,
                    max_new_tokens=max_new_tokens,
                    keep_ratio=keep_ratio,
                    max_frames=max_frames,
                    min_frames=min_frames,
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
                record.adaptive_min_gop = r.adaptive_min_gop
                record.min_frames = r.min_frames

    except torch.cuda.OutOfMemoryError:
        record.error = "OOM"
        pipe._clear_gpu()
    except TimeoutError:
        record.error = "TIMEOUT"
        pipe._clear_gpu()
    except Exception as e:
        record.error = str(e)

    return record


# ── 批量评估（增量 CSV + 恢复）──────────────────────────

def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes")


def _record_from_csv_row(row: dict, mode: str) -> MVBenchRecord:
    pred = row.get("pred_answer", "")
    return MVBenchRecord(
        sample_id=row.get("sample_id", ""),
        task_type=row.get("task_type", ""),
        video_field=row.get("video_field", ""),
        mode=row.get("mode", mode),
        keep_ratio=_to_float(row.get("keep_ratio", 0.0)),
        alpha=_to_float(row.get("alpha", 0.0)),
        gt_answer=row.get("gt_answer", ""),
        pred_answer=pred if str(pred).strip() else None,
        pred_raw=row.get("pred_raw", ""),
        correct=_to_bool(row.get("correct", False)),
        generate_ms=_to_float(row.get("generate_ms", 0.0)),
        total_ms=_to_float(row.get("total_ms", 0.0)),
        visual_tokens=_to_int(row.get("visual_tokens", 0)),
        audio_tokens=_to_int(row.get("audio_tokens", 0)),
        total_tokens=_to_int(row.get("total_tokens", 0)),
        num_frames=_to_int(row.get("num_frames", 0)),
        error=row.get("error", ""),
    )


def run_evaluation(
    pipe: SparseInferencePipeline,
    samples: List[MVBenchSample],
    mode: str,
    keep_ratio: float = 0.5,
    alpha: float = 0.5,
    max_new_tokens: int = 16,
    max_frames: int = 0,
    incremental_csv: str = "",
    min_frames: int = 8,
) -> List[MVBenchRecord]:
    """运行一组评估（支持断点恢复，resume key = sample_id）。"""
    completed_ids = set()
    existing_records: List[MVBenchRecord] = []

    if incremental_csv and os.path.exists(incremental_csv) and os.path.getsize(incremental_csv) > 0:
        with open(incremental_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("sample_id", "").strip()
                if not sid:
                    continue
                rec = _record_from_csv_row(row, mode)
                # 与 eval_videomme 保持一致：只把“有预测+有生成耗时”视作完成
                if rec.pred_answer and rec.generate_ms > 0:
                    completed_ids.add(sid)
                    existing_records.append(rec)

        if completed_ids:
            print(
                f"  [resume] Found {len(completed_ids)} completed samples in "
                f"{os.path.basename(incremental_csv)}, skipping",
                flush=True,
            )

    csv_writer = None
    csv_file = None
    if incremental_csv:
        os.makedirs(os.path.dirname(incremental_csv), exist_ok=True)
        csv_file = open(incremental_csv, "a", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDNAMES)
        if os.path.getsize(incremental_csv) == 0:
            csv_writer.writeheader()

    records: List[MVBenchRecord] = list(existing_records)
    correct = sum(1 for r in records if (not r.error and r.correct))
    total = sum(1 for r in records if not r.error)
    errors = sum(1 for r in records if r.error)

    done = len(existing_records)
    target = len(samples)

    for sample in samples:
        if sample.sample_id in completed_ids:
            continue

        rec = run_single(
            pipe,
            sample,
            mode,
            keep_ratio=keep_ratio,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            max_frames=max_frames,
            min_frames=min_frames,
        )
        records.append(rec)
        done += 1

        if csv_writer:
            csv_writer.writerow({k: getattr(rec, k) for k in _CSV_FIELDNAMES})
            csv_file.flush()

        if rec.error:
            errors += 1
            status = f"ERR:{rec.error[:30]}"
        else:
            total += 1
            correct += int(rec.correct)
            mark = "✓" if rec.correct else "✗"
            status = f"{mark} pred={rec.pred_answer} gt={rec.gt_answer}"

        acc = correct / total * 100 if total > 0 else 0.0
        print(
            f"  [{done}/{target}] acc={acc:.1f}% ({correct}/{total}) err={errors} "
            f"| {sample.task_type:22s} | {status}",
            flush=True,
        )

    if csv_file:
        csv_file.close()

    return records


# ── 汇总与打印 ────────────────────────────────────────────

def summarize_mvbench(records: List[MVBenchRecord], label: str) -> Dict:
    """按 task_type 汇总 MVBench 结果。"""
    valid = [r for r in records if not r.error]
    by_task: Dict[str, List[MVBenchRecord]] = defaultdict(list)
    for r in valid:
        by_task[r.task_type].append(r)

    summary = {
        "label": label,
        "total_samples": len(records),
        "valid_samples": len(valid),
        "errors": len(records) - len(valid),
        "overall_accuracy": (sum(r.correct for r in valid) / len(valid) * 100) if valid else 0.0,
        "avg_generate_ms": (sum(r.generate_ms for r in valid) / len(valid)) if valid else 0.0,
        "avg_total_ms": (sum(r.total_ms for r in valid) / len(valid)) if valid else 0.0,
        "avg_visual_tokens": (sum(r.visual_tokens for r in valid) / len(valid)) if valid else 0.0,
        "avg_audio_tokens": (sum(r.audio_tokens for r in valid) / len(valid)) if valid else 0.0,
        "avg_total_tokens": (sum(r.total_tokens for r in valid) / len(valid)) if valid else 0.0,
        "avg_num_frames": (sum(r.num_frames for r in valid) / len(valid)) if valid else 0.0,
        "per_task": {},
    }

    for task in sorted(by_task.keys()):
        recs = by_task[task]
        summary["per_task"][task] = {
            "count": len(recs),
            "accuracy": sum(r.correct for r in recs) / len(recs) * 100,
            "avg_generate_ms": sum(r.generate_ms for r in recs) / len(recs),
            "avg_visual_tokens": sum(r.visual_tokens for r in recs) / len(recs),
        }

    return summary


def print_mvbench_summary(summaries: List[Dict]):
    """打印总表和 per-task 对比表。"""
    print(f"\n{'='*50}")
    print("MVBENCH EVALUATION SUMMARY")
    print(f"{'='*50}")

    print(f"{'Mode':>14} | {'Accuracy':>10} | {'N':>5} | {'Err':>4} | {'Gen(ms)':>8} | {'VisTok':>8}")
    print("-" * 70)
    for s in summaries:
        print(
            f"{s['label']:>14} | {s['overall_accuracy']:>9.1f}% | "
            f"{s['valid_samples']:>5} | {s['errors']:>4} | "
            f"{s['avg_generate_ms']:>8.0f} | {s['avg_visual_tokens']:>8.0f}"
        )

    all_tasks = sorted({t for s in summaries for t in s.get("per_task", {}).keys()})
    if not all_tasks:
        return

    print("\nPer-Task Breakdown:")
    header = f"{'Task':>22} | " + " | ".join(f"{s['label']:>12}" for s in summaries)
    print(header)
    print("-" * len(header))
    for task in all_tasks:
        parts = []
        for s in summaries:
            d = s.get("per_task", {}).get(task)
            if d:
                parts.append(f"{d['accuracy']:>10.1f}%")
            else:
                parts.append(f"{'N/A':>10}")
        print(f"{task:>22} | " + " | ".join(parts))


# ── 保存 ──────────────────────────────────────────────────

def save_mode_summary(summary: Dict, out_dir: str, mode: str) -> str:
    path = os.path.join(out_dir, f"{mode}_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


def save_combined_summary(summaries: List[Dict], out_dir: str) -> str:
    path = os.path.join(out_dir, "mvbench_combined_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    return path


# ── Main ──────────────────────────────────────────────────

def _build_mode_label(mode: str, keep_ratio: float) -> str:
    if mode == "baseline":
        return "baseline"
    if mode == "sparse":
        return f"sparse({keep_ratio:g})"
    return mode


def main():
    parser = argparse.ArgumentParser(description="MVBench Evaluation")
    parser.add_argument("--max-per-task", type=int, default=0,
                        help="Max samples per task (0 = all 200)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames (0=unlimited, MVBench videos are short)")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Specific tasks to evaluate (default: all available)")
    parser.add_argument("--modes", nargs="+", default=["baseline", "sparse"],
                        choices=["baseline", "text_only", "audio_only", "video_only",
                                 "sparse", "sparse_no_audio",
                                 "naive_uniform", "naive_random", "naive_iframe"],
                        help="Modes to evaluate")
    parser.add_argument("--min-frames", type=int, default=8,
                        help="Min frames floor for sparse/naive modes (prevents short-video degradation)")
    parser.add_argument("--out-dir",
                        default="/root/autodl-tmp/results/fasteromni/mvbench")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("FasterOmni - MVBench Evaluation")
    print(f"modes={args.modes}, keep_ratio={args.keep_ratio}, alpha={args.alpha}, min_frames={args.min_frames}")
    print("=" * 70)

    samples = load_mvbench_samples(
        json_dir=MVBENCH_JSON_DIR,
        video_dir=MVBENCH_VIDEO_DIR,
        skip_tasks=DEFAULT_SKIP_TASKS,
        max_per_task=args.max_per_task,
    )

    if args.tasks:
        task_set = set(args.tasks)
        before = len(samples)
        samples = [s for s in samples if s.task_type in task_set]
        print(f"Applied task filter: {sorted(task_set)} -> {len(samples)}/{before} samples", flush=True)

    task_counts = defaultdict(int)
    for s in samples:
        task_counts[s.task_type] += 1

    print(f"Loaded {len(samples)} samples from {len(task_counts)} tasks", flush=True)
    if task_counts:
        detail = ", ".join(f"{k}:{task_counts[k]}" for k in sorted(task_counts.keys()))
        print(f"  {detail}", flush=True)

    pipe = SparseInferencePipeline(model_dir="/root/autodl-tmp/Qwen2.5-Omni-7B", dtype="bf16")

    os.makedirs(args.out_dir, exist_ok=True)
    all_summaries: List[Dict] = []

    for mode in args.modes:
        mode_dir = os.path.join(args.out_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        details_csv = os.path.join(mode_dir, f"{mode}_details.csv")

        print(f"\n{'='*60}")
        print(f"Running {mode.upper()} ({len(samples)} samples)")
        print(f"  Results -> {mode_dir}")
        print(f"{'='*60}")

        records = run_evaluation(
            pipe,
            samples,
            mode,
            keep_ratio=args.keep_ratio,
            alpha=args.alpha,
            max_new_tokens=args.max_new_tokens,
            max_frames=args.max_frames,
            incremental_csv=details_csv,
            min_frames=args.min_frames,
        )

        label = _build_mode_label(mode, args.keep_ratio)
        summary = summarize_mvbench(records, label=label)
        all_summaries.append(summary)

        summary_path = save_mode_summary(summary, mode_dir, mode)
        print(f"  Details: {details_csv}", flush=True)
        print(f"  Summary: {summary_path}", flush=True)

    print_mvbench_summary(all_summaries)
    combined_path = save_combined_summary(all_summaries, args.out_dir)
    print(f"\nCombined summary: {combined_path}", flush=True)


if __name__ == "__main__":
    main()
