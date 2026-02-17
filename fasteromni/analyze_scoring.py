"""
AV-LRM 打分分析脚本

对视频进行 GOP 解析 + 音频能量提取 + AV-LRM 打分 + 选择，
输出打分分布和选择结果，帮助验证稀疏化策略的合理性。

Usage:
    python fasteromni/analyze_scoring.py [--num-videos N] [--alpha 0.5] [--keep-ratio 0.5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.modules.gop_parser import parse_gops, GOPAnalysis
from fasteromni.modules.audio_energy import extract_audio_energy_per_gop
from fasteromni.modules.sparse import score_gops, select_gops, get_selection_summary, print_scoring_table


VIDEO_DIRS = {
    "vmme": "/root/autodl-tmp/data/Video-MME/videos/data",
    "anet": "/root/autodl-tmp/data/ActivityNet-QA/videos",
}


def collect_video_paths(dataset: str, num_videos: int) -> list[str]:
    paths = []
    if dataset in ("all", "vmme"):
        vmme_videos = sorted(glob.glob(os.path.join(VIDEO_DIRS["vmme"], "*.mp4")))
        paths.extend(vmme_videos[:num_videos])
    if dataset in ("all", "anet"):
        anet_videos = sorted(glob.glob(os.path.join(VIDEO_DIRS["anet"], "*.mp4")))
        if not anet_videos:
            anet_videos = sorted(glob.glob(os.path.join(VIDEO_DIRS["anet"], "**", "*.mp4"), recursive=True))
        paths.extend(anet_videos[:num_videos])
    return paths


def analyze_one_video(video_path: str, alpha: float, keep_ratio: float,
                      variance_threshold: float, min_gop_frames: int,
                      verbose: bool = False) -> dict:
    """对单个视频进行完整的 GOP 解析 + 打分 + 选择"""
    fname = os.path.basename(video_path)

    # Step 1: GOP 解析
    t0 = time.perf_counter()
    gop_analysis = parse_gops(video_path)
    gop_parse_ms = (time.perf_counter() - t0) * 1000

    # Step 2a: 音频能量提取
    t0 = time.perf_counter()
    audio_energies = extract_audio_energy_per_gop(video_path, gop_analysis.gops)
    audio_extract_ms = (time.perf_counter() - t0) * 1000

    # Step 2b: 打分
    t0 = time.perf_counter()
    scored_gops = score_gops(gop_analysis.gops, audio_energies,
                             alpha=alpha, min_gop_frames=min_gop_frames)
    scoring_ms = (time.perf_counter() - t0) * 1000

    # Step 2c: 选择
    t0 = time.perf_counter()
    scored_gops = select_gops(scored_gops, keep_ratio=keep_ratio,
                              variance_threshold=variance_threshold)
    selection_ms = (time.perf_counter() - t0) * 1000

    # 汇总
    summary = get_selection_summary(scored_gops)
    summary.update({
        "video": fname,
        "duration_sec": gop_analysis.video_duration_sec,
        "resolution": f"{gop_analysis.resolution[0]}x{gop_analysis.resolution[1]}",
        "gop_parse_ms": round(gop_parse_ms, 1),
        "audio_extract_ms": round(audio_extract_ms, 1),
        "scoring_ms": round(scoring_ms, 1),
        "selection_ms": round(selection_ms, 1),
        "total_pipeline_ms": round(gop_parse_ms + audio_extract_ms + scoring_ms + selection_ms, 1),
        "alpha": alpha,
        "keep_ratio_target": keep_ratio,
    })

    if verbose:
        print(f"\n{'='*70}")
        print(f"Video: {fname} | Duration: {gop_analysis.video_duration_sec:.1f}s | "
              f"GOPs: {gop_analysis.num_gops}")
        print(f"Timing: GOP parse={gop_parse_ms:.0f}ms, Audio={audio_extract_ms:.0f}ms, "
              f"Score={scoring_ms:.0f}ms, Select={selection_ms:.0f}ms")
        print(f"Score variance: {summary['score_variance']:.4f} | "
              f"Strategy: {'Top-K' if summary['score_variance'] > variance_threshold else 'Uniform'}")
        print(f"Selected: {summary['selected_gops']}/{summary['valid_gops']} GOPs | "
              f"Frames: {summary['selected_frames']}/{summary['total_frames']} "
              f"({summary['frame_keep_ratio']:.1%})")
        print_scoring_table(scored_gops, max_rows=20)

    return summary


def print_aggregate_analysis(all_summaries: list[dict]) -> None:
    """打印跨视频的汇总分析"""
    print("\n" + "=" * 90)
    print("AGGREGATE SCORING ANALYSIS")
    print("=" * 90)

    # 分数方差分布
    variances = [s["score_variance"] for s in all_summaries]
    print("\n--- Score Variance Distribution ---")
    print(f"  Min:    {min(variances):.4f}")
    print(f"  Max:    {max(variances):.4f}")
    print(f"  Mean:   {np.mean(variances):.4f}")
    print(f"  Median: {np.median(variances):.4f}")

    # 帧保留率
    frame_ratios = [s["frame_keep_ratio"] for s in all_summaries]
    print("\n--- Frame Keep Ratio ---")
    print(f"  Min:    {min(frame_ratios):.1%}")
    print(f"  Max:    {max(frame_ratios):.1%}")
    print(f"  Mean:   {np.mean(frame_ratios):.1%}")

    # Pipeline 耗时
    pipeline_times = [s["total_pipeline_ms"] for s in all_summaries]
    print("\n--- Pipeline Overhead (ms) ---")
    print(f"  Min:    {min(pipeline_times):.0f}")
    print(f"  Max:    {max(pipeline_times):.0f}")
    print(f"  Mean:   {np.mean(pipeline_times):.0f}")

    # 每个视频的摘要
    print(f"\n--- Per-Video Summary ---")
    print(f"{'Video':>35} | {'GOPs':>5} | {'Sel':>4} | {'Variance':>8} | "
          f"{'Strategy':>8} | {'Frame%':>6} | {'Time(ms)':>8}")
    print("-" * 90)
    for s in all_summaries:
        strategy = "Top-K" if s["score_variance"] > 0.05 else "Uniform"
        print(f"{s['video'][:35]:>35} | {s['valid_gops']:>5} | {s['selected_gops']:>4} | "
              f"{s['score_variance']:>8.4f} | {strategy:>8} | "
              f"{s['frame_keep_ratio']:>6.1%} | {s['total_pipeline_ms']:>8.0f}")


def main():
    parser = argparse.ArgumentParser(description="AV-LRM Scoring Analysis")
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--dataset", choices=["all", "vmme", "anet"], default="all")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    parser.add_argument("--min-gop-frames", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/results/fasteromni/scoring-analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("FasterOmni - AV-LRM Scoring Analysis")
    print(f"alpha={args.alpha}, keep_ratio={args.keep_ratio}, "
          f"variance_threshold={args.variance_threshold}, min_gop_frames={args.min_gop_frames}")
    print("=" * 70)

    video_paths = collect_video_paths(args.dataset, args.num_videos)
    print(f"\nCollected {len(video_paths)} videos\n")

    all_summaries = []
    for i, vpath in enumerate(video_paths):
        print(f"[{i+1}/{len(video_paths)}] Processing: {os.path.basename(vpath)} ...", flush=True)
        try:
            summary = analyze_one_video(
                vpath, alpha=args.alpha, keep_ratio=args.keep_ratio,
                variance_threshold=args.variance_threshold,
                min_gop_frames=args.min_gop_frames,
                verbose=args.verbose,
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"  FAILED: {e}")

    print_aggregate_analysis(all_summaries)

    # 保存
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "scoring_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
