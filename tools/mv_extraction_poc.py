#!/usr/bin/env python3
"""
Layer 3 Motion Vector extraction proof-of-concept.

Usage:
    python tools/mv_extraction_poc.py /path/to/video.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import av
import matplotlib.pyplot as plt
import numpy as np
from av.video.frame import PictureType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fasteromni.modules.gop_parser import GOPInfo, parse_gops


OUTPUT_DIR = "/root/autodl-tmp/results/figures"


@dataclass
class MVExtractionStats:
    method: str
    elapsed_ms: float
    decoded_frames: int
    pb_frames: int
    mv_frames: int


def _safe_timestamp_sec(frame: av.VideoFrame, time_base: float) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None:
        return float(frame.pts * time_base)
    return 0.0


def _mv_magnitude_from_array(mvs: np.ndarray) -> np.ndarray:
    if mvs.dtype.names is None:
        return np.empty((0,), dtype=np.float32)

    fields = set(mvs.dtype.names)
    if {"motion_x", "motion_y"}.issubset(fields):
        motion_x = mvs["motion_x"].astype(np.float32)
        motion_y = mvs["motion_y"].astype(np.float32)
        if "motion_scale" in fields:
            scale = mvs["motion_scale"].astype(np.float32)
            scale[scale == 0] = 1.0
            motion_x = motion_x / scale
            motion_y = motion_y / scale
        return np.sqrt(motion_x * motion_x + motion_y * motion_y)

    if {"src_x", "src_y", "dst_x", "dst_y"}.issubset(fields):
        dx = (mvs["dst_x"] - mvs["src_x"]).astype(np.float32)
        dy = (mvs["dst_y"] - mvs["src_y"]).astype(np.float32)
        return np.sqrt(dx * dx + dy * dy)

    return np.empty((0,), dtype=np.float32)


def extract_motion_vectors_pyav(video_path: str) -> Tuple[List[Dict[str, Any]], MVExtractionStats]:
    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    decoded_frames = 0
    pb_frames = 0
    mv_frames = 0

    container = av.open(video_path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base)

    codec_ctx = stream.codec_context
    try:
        options = dict(codec_ctx.options or {})
        options["flags2"] = "+export_mvs"
        codec_ctx.options = options
    except Exception:
        pass

    for frame_idx, frame in enumerate(container.decode(stream)):
        decoded_frames += 1
        pict_type = frame.pict_type
        if pict_type not in (PictureType.P, PictureType.B):
            continue
        pb_frames += 1

        ts_sec = _safe_timestamp_sec(frame, time_base)
        magnitudes = np.empty((0,), dtype=np.float32)

        for side_data in frame.side_data:
            side_data_name = str(side_data.type).upper()
            if "MOTION_VECTORS" not in side_data_name:
                continue
            mvs = side_data.to_ndarray()
            if mvs is None or len(mvs) == 0:
                continue
            magnitudes = _mv_magnitude_from_array(mvs)
            break

        if magnitudes.size == 0:
            continue

        mv_frames += 1
        rows.append(
            {
                "frame_idx": frame_idx,
                "frame_type": "P" if pict_type == PictureType.P else "B",
                "timestamp_sec": ts_sec,
                "mv_magnitude_mean": float(np.mean(magnitudes)),
                "mv_magnitude_max": float(np.max(magnitudes)),
                "mv_count": int(magnitudes.size),
            }
        )

    container.close()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = MVExtractionStats(
        method="pyav",
        elapsed_ms=elapsed_ms,
        decoded_frames=decoded_frames,
        pb_frames=pb_frames,
        mv_frames=mv_frames,
    )
    return rows, stats


def extract_motion_vectors_ffprobe(video_path: str) -> Tuple[List[Dict[str, Any]], MVExtractionStats]:
    """
    ffprobe fallback. Some ffprobe builds only expose MV side-data existence
    (without raw vector fields), so this path uses packet-size as a proxy.
    """
    t0 = time.perf_counter()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-flags2",
        "+export_mvs",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pict_type,pkt_size,side_data_list",
        "-of",
        "json",
        video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout) if proc.stdout else {}
    frames = payload.get("frames", [])

    rows: List[Dict[str, Any]] = []
    pb_frames = 0
    mv_frames = 0
    for idx, frame in enumerate(frames):
        frame_type = str(frame.get("pict_type", ""))
        if frame_type not in {"P", "B"}:
            continue
        pb_frames += 1

        side_data_list = frame.get("side_data_list", []) or []
        has_mv = any("Motion vectors" in str(entry.get("side_data_type", "")) for entry in side_data_list)
        if not has_mv:
            continue

        mv_frames += 1
        ts_sec = float(frame.get("best_effort_timestamp_time", 0.0))
        pkt_size = float(frame.get("pkt_size", 0.0))
        proxy_score = math.sqrt(max(pkt_size, 0.0))

        rows.append(
            {
                "frame_idx": idx,
                "frame_type": frame_type,
                "timestamp_sec": ts_sec,
                "mv_magnitude_mean": proxy_score,
                "mv_magnitude_max": proxy_score,
                "mv_count": 1,
            }
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = MVExtractionStats(
        method="ffprobe_proxy",
        elapsed_ms=elapsed_ms,
        decoded_frames=len(frames),
        pb_frames=pb_frames,
        mv_frames=mv_frames,
    )
    return rows, stats


def extract_motion_vectors(video_path: str) -> Tuple[List[Dict[str, Any]], MVExtractionStats]:
    try:
        rows, stats = extract_motion_vectors_pyav(video_path)
        if stats.mv_frames > 0 or stats.pb_frames == 0:
            return rows, stats
    except Exception:
        pass

    rows, stats = extract_motion_vectors_ffprobe(video_path)
    return rows, stats


def compute_gop_motion_scores(
    gops: Sequence[GOPInfo], frame_rows: Sequence[Dict[str, Any]]
) -> Tuple[List[float], List[int]]:
    per_gop_values: List[List[float]] = [[] for _ in gops]
    per_gop_mv_count = [0 for _ in gops]

    if not gops or not frame_rows:
        return [0.0 for _ in gops], per_gop_mv_count

    for row in frame_rows:
        t = float(row["timestamp_sec"])
        value = float(row["mv_magnitude_mean"])
        count = int(row["mv_count"])

        assigned_idx = None
        for idx, g in enumerate(gops):
            start = float(g.start_time_sec or 0.0)
            end = float(g.end_time_sec if g.end_time_sec is not None else float("inf"))
            is_last = idx == len(gops) - 1
            in_range = (start <= t < end) or (is_last and start <= t <= end)
            if in_range:
                assigned_idx = idx
                break

        if assigned_idx is None:
            continue

        per_gop_values[assigned_idx].append(value)
        per_gop_mv_count[assigned_idx] += count

    gop_scores = [float(np.mean(vals)) if vals else 0.0 for vals in per_gop_values]
    return gop_scores, per_gop_mv_count


def _motion_levels(scores: Sequence[float]) -> Tuple[List[str], float, float]:
    nonzero = np.array([s for s in scores if s > 0], dtype=np.float32)
    if nonzero.size == 0:
        return ["low" for _ in scores], 0.0, 0.0

    low_high = float(np.quantile(nonzero, 0.5))
    high_cut = float(np.quantile(nonzero, 0.85))
    levels = []
    for s in scores:
        if s >= high_cut and s > 0:
            levels.append("high")
        elif s >= low_high and s > 0:
            levels.append("moderate")
        else:
            levels.append("low")
    return levels, low_high, high_cut


def _plot_motion_profile(
    video_path: str,
    gops: Sequence[GOPInfo],
    frame_rows: Sequence[Dict[str, Any]],
    gop_scores: Sequence[float],
    levels: Sequence[str],
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"mv_profile_{base_name}.png")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.25,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)

    if frame_rows:
        ts = np.array([float(r["timestamp_sec"]) for r in frame_rows], dtype=np.float32)
        mags = np.array([float(r["mv_magnitude_mean"]) for r in frame_rows], dtype=np.float32)
        ax.plot(ts, mags, color="#1f77b4", linewidth=1.2, label="Frame MV mean")
        ax.scatter(ts, mags, s=8, color="#1f77b4", alpha=0.55)
    else:
        ax.text(
            0.5,
            0.5,
            "No P/B-frame MV data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )

    color_map = {"low": "#c6dbef", "moderate": "#fdd0a2", "high": "#fcae91"}
    for g, lvl in zip(gops, levels):
        start = float(g.start_time_sec or 0.0)
        end = float(g.end_time_sec if g.end_time_sec is not None else start)
        if end <= start:
            continue
        ax.axvspan(start, end, color=color_map[lvl], alpha=0.12, linewidth=0)

    for g in gops:
        if g.start_time_sec is not None:
            ax.axvline(float(g.start_time_sec), color="gray", linewidth=0.5, alpha=0.2)

    ax.set_title(f"Motion Vector Profile: {os.path.basename(video_path)}")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("MV Magnitude")
    if gop_scores:
        ymax = max(max(gop_scores), 1.0) * 1.15
        ax.set_ylim(0, ymax)
    if frame_rows:
        ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return out_path


def _print_report(
    video_path: str,
    duration_sec: float,
    total_frames: int,
    gops: Sequence[GOPInfo],
    stats: MVExtractionStats,
    gop_scores: Sequence[float],
    levels: Sequence[str],
    top_k: int = 5,
) -> None:
    per_frame_ms = stats.elapsed_ms / max(stats.decoded_frames, 1)

    print(f"Video: {os.path.basename(video_path)}")
    print(f"Duration: {duration_sec:.1f}s, Total frames: {total_frames}, GOPs: {len(gops)}")
    print()
    print(
        f"MV Extraction [{stats.method}]: {stats.elapsed_ms:.0f}ms "
        f"({per_frame_ms:.2f}ms/frame), PB={stats.pb_frames}, MV-frames={stats.mv_frames}"
    )
    if stats.pb_frames == 0:
        print("Note: No P/B frames detected (likely all-I encoding).")
    elif stats.mv_frames == 0:
        print("Note: P/B frames exist but no MV payload decoded.")
    print()

    print("GOP Motion Scores:")
    for idx, (g, score, lvl) in enumerate(zip(gops, gop_scores, levels)):
        start = float(g.start_time_sec or 0.0)
        end = float(g.end_time_sec if g.end_time_sec is not None else start)
        marker = " ★" if lvl == "high" else ""
        print(f"  GOP {idx:>3} [{start:>6.2f}s - {end:>6.2f}s]: score={score:>8.3f} ({lvl}){marker}")

    if gop_scores:
        ranked_desc = sorted(range(len(gop_scores)), key=lambda i: gop_scores[i], reverse=True)
        ranked_asc = sorted(range(len(gop_scores)), key=lambda i: gop_scores[i])
        print()
        print(f"Top-{top_k} highest motion GOPs: {ranked_desc[:top_k]}")
        print(f"Bottom-{top_k} lowest motion GOPs: {ranked_asc[:top_k]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 3 Motion Vector extraction PoC")
    parser.add_argument("video_path", type=str, help="Path to input video")
    args = parser.parse_args()

    video_path = args.video_path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    t_total = time.perf_counter()
    gop_analysis = parse_gops(video_path)
    mv_rows, mv_stats = extract_motion_vectors(video_path)
    gop_scores, _ = compute_gop_motion_scores(gop_analysis.gops, mv_rows)
    levels, _, _ = _motion_levels(gop_scores)

    _print_report(
        video_path=video_path,
        duration_sec=gop_analysis.video_duration_sec,
        total_frames=gop_analysis.total_frames,
        gops=gop_analysis.gops,
        stats=mv_stats,
        gop_scores=gop_scores,
        levels=levels,
    )

    out_path = _plot_motion_profile(
        video_path=video_path,
        gops=gop_analysis.gops,
        frame_rows=mv_rows,
        gop_scores=gop_scores,
        levels=levels,
    )
    total_ms = (time.perf_counter() - t_total) * 1000.0
    print()
    print(f"Motion profile saved to: {out_path}")
    print(f"Total pipeline time: {total_ms:.0f}ms")


if __name__ == "__main__":
    main()
