"""
GOP 结构分析脚本

对 Video-MME 和 ActivityNet 数据集中的视频进行 GOP 解析，
输出统计数据，帮助理解 GOP 结构特征，为稀疏化设计提供依据。

Usage:
    python fasteromni/analyze_gop.py [--num-videos N] [--dataset all|vmme|anet]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

# 添加项目根目录到 path
SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.modules.gop_parser import parse_gops, print_gop_table, GOPAnalysis


# 数据集视频目录
VIDEO_DIRS = {
    "vmme": "/root/autodl-tmp/data/Video-MME/videos/data",
    "anet": "/root/autodl-tmp/data/ActivityNet-QA/videos",
}


def collect_video_paths(dataset: str, num_videos: int) -> list[str]:
    """收集指定数据集的视频文件路径"""
    paths = []
    if dataset in ("all", "vmme"):
        vmme_dir = VIDEO_DIRS["vmme"]
        vmme_videos = sorted(glob.glob(os.path.join(vmme_dir, "*.mp4")))
        paths.extend(vmme_videos[:num_videos])

    if dataset in ("all", "anet"):
        anet_dir = VIDEO_DIRS["anet"]
        # ActivityNet 可能有子目录
        anet_videos = sorted(glob.glob(os.path.join(anet_dir, "*.mp4")))
        if not anet_videos:
            anet_videos = sorted(glob.glob(os.path.join(anet_dir, "**", "*.mp4"), recursive=True))
        paths.extend(anet_videos[:num_videos])

    return paths


def analyze_videos(video_paths: list[str], verbose: bool = False) -> list[GOPAnalysis]:
    """对多个视频进行 GOP 分析"""
    results = []
    for i, vpath in enumerate(video_paths):
        fname = os.path.basename(vpath)
        print(f"[{i+1}/{len(video_paths)}] Parsing: {fname} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            analysis = parse_gops(vpath)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"OK ({elapsed:.0f}ms) | GOPs={analysis.num_gops}, "
                  f"Frames={analysis.total_frames}, Duration={analysis.video_duration_sec:.1f}s")
            if verbose:
                print_gop_table(analysis, max_rows=15)
            results.append(analysis)
        except Exception as e:
            print(f"FAILED: {e}")
    return results


def print_aggregate_stats(results: list[GOPAnalysis]) -> None:
    """打印跨视频的汇总统计"""
    if not results:
        print("No results to summarize.")
        return

    import statistics

    all_gop_counts = [r.num_gops for r in results]
    all_avg_gop_frames = [r.avg_gop_frames for r in results]
    all_i_ratios = [r.i_frame_ratio for r in results]
    all_durations = [r.video_duration_sec for r in results]

    # 收集所有 I 帧码率
    all_i_sizes = []
    for r in results:
        all_i_sizes.extend(r.i_frame_sizes)

    # 收集所有 GOP 的帧数分布
    all_gop_frame_counts = []
    for r in results:
        all_gop_frame_counts.extend([g.num_frames for g in r.gops])

    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    print(f"Videos analyzed: {len(results)}")
    print(f"Total GOPs: {sum(all_gop_counts)}")
    print()

    # 视频时长
    print("--- Video Duration (sec) ---")
    print(f"  Min:  {min(all_durations):.1f}")
    print(f"  Max:  {max(all_durations):.1f}")
    print(f"  Mean: {statistics.mean(all_durations):.1f}")
    print(f"  Median: {statistics.median(all_durations):.1f}")
    print()

    # GOP 数量
    print("--- GOPs per Video ---")
    print(f"  Min:  {min(all_gop_counts)}")
    print(f"  Max:  {max(all_gop_counts)}")
    print(f"  Mean: {statistics.mean(all_gop_counts):.1f}")
    print(f"  Median: {statistics.median(all_gop_counts):.1f}")
    print()

    # GOP 帧数
    print("--- Frames per GOP ---")
    print(f"  Min:  {min(all_gop_frame_counts)}")
    print(f"  Max:  {max(all_gop_frame_counts)}")
    print(f"  Mean: {statistics.mean(all_gop_frame_counts):.1f}")
    print(f"  Median: {statistics.median(all_gop_frame_counts):.1f}")
    if len(all_gop_frame_counts) > 1:
        print(f"  Stdev: {statistics.stdev(all_gop_frame_counts):.1f}")
    print()

    # I 帧码率（KB）
    all_i_kb = [s / 1024 for s in all_i_sizes]
    print("--- I-frame Size (KB) ---")
    print(f"  Min:  {min(all_i_kb):.1f}")
    print(f"  Max:  {max(all_i_kb):.1f}")
    print(f"  Mean: {statistics.mean(all_i_kb):.1f}")
    print(f"  Median: {statistics.median(all_i_kb):.1f}")
    if len(all_i_kb) > 1:
        print(f"  Stdev: {statistics.stdev(all_i_kb):.1f}")
    print()

    # I 帧占比
    print("--- I-frame Ratio ---")
    print(f"  Min:  {min(all_i_ratios):.2%}")
    print(f"  Max:  {max(all_i_ratios):.2%}")
    print(f"  Mean: {statistics.mean(all_i_ratios):.2%}")
    print()

    # 每个视频的摘要表格
    print("--- Per-Video Summary ---")
    print(f"{'Video':>40} | {'Dur(s)':>6} | {'GOPs':>5} | {'Avg F/GOP':>9} | "
          f"{'I-ratio':>7} | {'I-size mean(KB)':>15}")
    print("-" * 100)
    for r in results:
        fname = os.path.basename(r.video_path)[:38]
        s = r.summary_dict()
        print(f"{fname:>40} | {s['duration_sec']:>6.1f} | {s['num_gops']:>5} | "
              f"{s['avg_gop_frames']:>9.1f} | {s['i_frame_ratio']:>7.2%} | "
              f"{s['i_frame_size_mean']/1024:>15.1f}")
    print()


def save_results(results: list[GOPAnalysis], out_dir: str) -> str:
    """保存分析结果到 JSON"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gop_analysis.json")
    data = {
        "num_videos": len(results),
        "summaries": [r.summary_dict() for r in results],
        "per_video_gops": {},
    }
    for r in results:
        fname = os.path.basename(r.video_path)
        data["per_video_gops"][fname] = [
            {
                "gop_index": g.gop_index,
                "start_time_sec": g.start_time_sec,
                "duration_sec": g.duration_sec,
                "num_frames": g.num_frames,
                "i_frame_size": g.i_frame_size,
                "total_packet_size": g.total_packet_size,
            }
            for g in r.gops
        ]
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="GOP Structure Analysis")
    parser.add_argument("--num-videos", type=int, default=5,
                        help="Number of videos to analyze per dataset")
    parser.add_argument("--dataset", choices=["all", "vmme", "anet"], default="all",
                        help="Which dataset to analyze")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed GOP table for each video")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/results/fasteromni/gop-analysis",
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 80)
    print("FasterOmni - GOP Structure Analysis")
    print("=" * 80)

    # 收集视频
    video_paths = collect_video_paths(args.dataset, args.num_videos)
    print(f"\nCollected {len(video_paths)} videos to analyze\n")

    if not video_paths:
        print("No videos found!")
        return

    # 分析
    results = analyze_videos(video_paths, verbose=args.verbose)

    # 汇总
    print_aggregate_stats(results)

    # 保存
    save_results(results, args.out_dir)


if __name__ == "__main__":
    main()
