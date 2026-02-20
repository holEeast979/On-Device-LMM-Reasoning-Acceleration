"""
Baseline vs Sparse 对比实验

对同一组视频分别跑 baseline（全量帧）和 sparse（GOP 稀疏化）推理，
对比 TTFT、visual token 数量、模型输出质量。

Usage:
    python fasteromni/run_comparison.py [--num-videos 3] [--keep-ratio 0.5]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.pipeline import SparseInferencePipeline, PipelineResult, print_result


VIDEO_DIRS = {
    "vmme": "/root/autodl-tmp/data/Video-MME/videos/data",
    "anet": "/root/autodl-tmp/data/ActivityNet-QA/videos",
}

# 简单的测试问题（用于验证模型输出是否合理）
DEFAULT_QUESTION = "Describe what happens in this video in detail."


def collect_videos(dataset: str, num_videos: int) -> list[str]:
    paths = []
    if dataset in ("all", "vmme"):
        vmme = sorted(glob.glob(os.path.join(VIDEO_DIRS["vmme"], "*.mp4")))
        paths.extend(vmme[:num_videos])
    if dataset in ("all", "anet"):
        anet = sorted(glob.glob(os.path.join(VIDEO_DIRS["anet"], "*.mp4")))
        if not anet:
            anet = sorted(glob.glob(os.path.join(VIDEO_DIRS["anet"], "**", "*.mp4"), recursive=True))
        paths.extend(anet[:num_videos])
    return paths


def result_to_dict(r: PipelineResult) -> dict:
    return {
        "mode": r.mode,
        "video": os.path.basename(r.video_path),
        "error": r.error,
        "preprocess_ms": round(r.preprocess_ms, 1),
        "tokenize_ms": round(r.tokenize_ms, 1),
        "audio_feature_ms": round(r.audio_feature_ms, 1),
        "generate_ms": round(r.generate_ms, 1),
        "total_ms": round(r.total_ms, 1),
        "visual_tokens": r.visual_tokens,
        "audio_tokens": r.audio_tokens,
        "total_tokens": r.total_tokens,
        "num_frames_input": r.num_frames_input,
        "total_gops": r.total_gops,
        "selected_gops": r.selected_gops,
        "keep_ratio_actual": round(r.keep_ratio_actual, 3),
        "output_text": r.output_text[:500],
        # sparse-specific timings
        "gop_parse_ms": round(r.gop_parse_ms, 1),
        "audio_extract_ms": round(r.audio_extract_ms, 1),
        "scoring_ms": round(r.scoring_ms, 1),
        "i_frame_decode_ms": round(r.i_frame_decode_ms, 1),
    }


def print_comparison_table(results: list[dict]) -> None:
    """打印 baseline vs sparse 对比表"""
    # 按视频分组
    videos = {}
    for r in results:
        v = r["video"]
        if v not in videos:
            videos[v] = {}
        videos[v][r["mode"]] = r

    print("\n" + "=" * 100)
    print("COMPARISON: Baseline vs Sparse")
    print("=" * 100)
    print(f"{'Video':>30} | {'Mode':>8} | {'TTFT(ms)':>9} | {'Vis Tok':>8} | "
          f"{'Frames':>6} | {'Total Tok':>9} | {'Total(ms)':>9}")
    print("-" * 100)

    for video, modes in videos.items():
        for mode in ["baseline", "sparse"]:
            if mode in modes:
                r = modes[mode]
                err = " ERR" if r.get("error") else ""
                print(f"{video[:30]:>30} | {mode:>8} | {r['generate_ms']:>9.1f} | "
                      f"{r['visual_tokens']:>8} | {r['num_frames_input']:>6} | "
                      f"{r['total_tokens']:>9} | {r['total_ms']:>9.1f}{err}")
        # 计算 speedup
        if "baseline" in modes and "sparse" in modes:
            b = modes["baseline"]
            s = modes["sparse"]
            if b["generate_ms"] > 0 and s["generate_ms"] > 0 and not b.get("error") and not s.get("error"):
                speedup = b["generate_ms"] / s["generate_ms"]
                token_reduction = 1 - (s["visual_tokens"] / b["visual_tokens"]) if b["visual_tokens"] > 0 else 0
                print(f"{'':>30}   {'SPEEDUP':>8} | {speedup:>8.2f}x | "
                      f"{token_reduction:>7.0%} | "
                      f"{s['num_frames_input']:>6} | {'':>9} |")
        print("-" * 100)

    # 汇总
    baselines = [r for r in results if r["mode"] == "baseline" and not r.get("error")]
    sparses = [r for r in results if r["mode"] == "sparse" and not r.get("error")]

    if baselines and sparses:
        import numpy as np
        avg_b_ttft = np.mean([r["generate_ms"] for r in baselines])
        avg_s_ttft = np.mean([r["generate_ms"] for r in sparses])
        avg_b_vtok = np.mean([r["visual_tokens"] for r in baselines])
        avg_s_vtok = np.mean([r["visual_tokens"] for r in sparses])
        print(f"\n--- Average ---")
        print(f"  Baseline TTFT: {avg_b_ttft:.0f}ms | Visual tokens: {avg_b_vtok:.0f}")
        print(f"  Sparse   TTFT: {avg_s_ttft:.0f}ms | Visual tokens: {avg_s_vtok:.0f}")
        if avg_b_ttft > 0:
            print(f"  TTFT Speedup: {avg_b_ttft / avg_s_ttft:.2f}x")
        if avg_b_vtok > 0:
            print(f"  Token Reduction: {1 - avg_s_vtok / avg_b_vtok:.0%}")


def main():
    parser = argparse.ArgumentParser(description="Baseline vs Sparse Comparison")
    parser.add_argument("--num-videos", type=int, default=3,
                        help="Videos per dataset")
    parser.add_argument("--dataset", choices=["all", "vmme", "anet"], default="vmme",
                        help="Dataset to use")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--variance-threshold", type=float, default=0.02)
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out-dir", default="/root/autodl-tmp/results/fasteromni/comparison")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline runs (if already have data)")
    args = parser.parse_args()

    print("=" * 70)
    print("FasterOmni - Baseline vs Sparse Comparison")
    print(f"keep_ratio={args.keep_ratio}, alpha={args.alpha}, "
          f"variance_threshold={args.variance_threshold}")
    print("=" * 70)

    video_paths = collect_videos(args.dataset, args.num_videos)
    print(f"\nCollected {len(video_paths)} videos\n")

    if not video_paths:
        print("No videos found!")
        return

    # 初始化 Pipeline
    pipe = SparseInferencePipeline(dtype="bf16")

    all_results = []

    for i, vpath in enumerate(video_paths):
        fname = os.path.basename(vpath)
        print(f"\n{'#'*70}")
        print(f"[{i+1}/{len(video_paths)}] {fname}")
        print(f"{'#'*70}")

        # Baseline
        if not args.skip_baseline:
            print(f"\n--- Running Baseline ---")
            baseline_result = pipe.run_baseline(vpath, args.question, args.max_new_tokens)
            print_result(baseline_result)
            all_results.append(result_to_dict(baseline_result))

        # Sparse
        print(f"\n--- Running Sparse (keep_ratio={args.keep_ratio}) ---")
        sparse_result = pipe.run_sparse(
            vpath, args.question, args.max_new_tokens,
            alpha=args.alpha, keep_ratio=args.keep_ratio,
            variance_threshold=args.variance_threshold,
        )
        print_result(sparse_result)
        all_results.append(result_to_dict(sparse_result))

    # 对比表
    print_comparison_table(all_results)

    # 保存结果
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # 也存 CSV
    csv_path = os.path.join(args.out_dir, "comparison_results.csv")
    if all_results:
        keys = all_results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
