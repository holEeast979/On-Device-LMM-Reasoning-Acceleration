"""
音频能量提取模块

从视频文件中提取音轨，按 GOP 的时间窗口切片，
计算每段的 RMS 能量作为"音频活跃度"的代理指标。

RMS 能量高 → 该时间段音频活跃（对话、音效、音乐高潮等）
RMS 能量低 → 该时间段音频平静（静音、背景噪声等）

使用方式：
    from fasteromni.modules.audio_energy import extract_audio_energy_per_gop
    energies = extract_audio_energy_per_gop("video.mp4", gops)
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np

from .gop_parser import GOPInfo


def extract_audio_from_video(video_path: str, sr: int = 16000) -> Tuple[Optional[np.ndarray], int]:
    """
    从视频中提取音频波形。

    使用 ffmpeg 提取音轨为 WAV，再用 librosa 加载。
    如果视频没有音轨，返回 (None, sr)。

    Args:
        video_path: 视频文件路径
        sr: 目标采样率

    Returns:
        (audio_waveform, sample_rate): 音频波形和采样率
    """
    import librosa

    # 先检查视频是否有音轨
    try:
        import av
        container = av.open(video_path)
        if len(container.streams.audio) == 0:
            container.close()
            return None, sr
        container.close()
    except Exception:
        return None, sr

    # 用临时文件提取音频
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",  # 不要视频
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-ac", "1",  # 单声道
            tmp_wav,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0 or not os.path.exists(tmp_wav):
            return None, sr

        audio, _ = librosa.load(tmp_wav, sr=sr)
        return audio, sr
    except Exception:
        return None, sr
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def compute_rms_for_segment(audio: np.ndarray, sr: int,
                            start_sec: float, end_sec: float) -> float:
    """
    计算音频片段的 RMS 能量。

    Args:
        audio: 完整音频波形
        sr: 采样率
        start_sec: 起始时间（秒）
        end_sec: 结束时间（秒）

    Returns:
        RMS 能量值（float）
    """
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    # 边界保护
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    if end_sample <= start_sample:
        return 0.0

    segment = audio[start_sample:end_sample]
    rms = float(np.sqrt(np.mean(segment ** 2)))
    return rms


def extract_audio_energy_per_gop(
    video_path: str,
    gops: List[GOPInfo],
    sr: int = 16000,
    audio: Optional[np.ndarray] = None,
) -> List[float]:
    """
    为每个 GOP 计算音频 RMS 能量。

    Args:
        video_path: 视频文件路径
        gops: GOP 列表（来自 gop_parser.parse_gops）
        sr: 采样率
        audio: 预提取的音频波形（可选，避免重复提取）

    Returns:
        与 gops 等长的 RMS 能量列表。如果视频无音轨，全部返回 0.0。
    """
    # 提取音频（如果未提供）
    if audio is None:
        audio, sr = extract_audio_from_video(video_path, sr=sr)

    if audio is None:
        return [0.0] * len(gops)

    energies = []
    for g in gops:
        start = g.start_time_sec if g.start_time_sec is not None else 0.0
        end = g.end_time_sec if g.end_time_sec is not None else start
        # 对最后一个 GOP，如果 end == start（未知结束时间），用音频总时长
        if end <= start:
            end = len(audio) / sr

        rms = compute_rms_for_segment(audio, sr, start, end)
        energies.append(rms)

    return energies
