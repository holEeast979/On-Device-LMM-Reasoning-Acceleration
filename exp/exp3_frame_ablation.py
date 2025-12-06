#!/usr/bin/env python3
"""
Experiment 3: Frame-count ablation (MSVD-QA)

Runs Qwen2-VL on the same set of videos with different frame counts N,
filtering samples that don't have enough frames for each N.

Usage:
  python scripts/exp3_frame_ablation.py \
    --qwen2-vl ./Qwen2-VL-7B-Instruct \
    --videos data/MSVD-QA_subset/manifest.csv \
    --frame-list 4,8,16 \
    --out results/exp3_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import pandas as pd
from decord import VideoReader, cpu
from tqdm import tqdm

import scripts.common as C


def load_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({k.strip(): v.strip() if v is not None else "" for k, v in r.items()})
    return rows


def enough_frames(video_path: str, n: int) -> bool:
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        return len(vr) >= n
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser("exp3 frame ablation")
    ap.add_argument("--qwen2-vl", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--frame-list", default="4,8,16")
    ap.add_argument("--short-side", type=int, default=336)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out", default="results/exp3_ablation.csv")
    args = ap.parse_args()

    frames_list = [int(x) for x in args.frame_list.split(",") if x]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = load_csv(args.videos)
    if args.max_samples:
        rows = rows[: args.max_samples]

    model, proc = C.load_qwen2_vl(args.qwen2_vl, args.dtype)
    out_rows: List[Dict[str, object]] = []

    print("\n=== Starting Experiment 3: Frame Ablation ===")
    # Pre-filter by each N to avoid empty samples
    for N in tqdm(frames_list, desc="Frame counts"):
        sel = [r for r in rows if enough_frames(r["video_path"], N)]
        print(f"\n[N={N}] Processing {len(sel)} videos...")
        for i, r in enumerate(tqdm(sel, desc=f"N={N} frames", leave=False)):
            frames, t_decode, t_pre = C.sample_video_frames(r["video_path"], N, args.short_side)
            if not frames:
                print(f"Warning: Skipping video {i} for N={N} - no frames decoded")
                continue
            ans, ttft, total, tok_s, t_pack = C.run_qwen2_vl_video(model, proc, frames, r["question"], args.max_new_tokens)
            out_rows.append(
                {
                    "frames": N,
                    "modality": "video",
                    "dataset": "MSVD-QA",
                    "sample_id": i,
                    "ttft_ms": ttft * 1000,
                    "t_total_ms": total * 1000,
                    "tok_per_s": tok_s,
                    "t_decode_ms": t_decode * 1000,
                    "t_pre_ms": t_pre * 1000,
                    "t_pack_ms": t_pack * 1000,
                    "pred": ans,
                    "gold": r.get("answer", ""),
                    "correct": 1 if ans.strip() and r.get("answer", "").strip() and (ans.strip().lower() == r.get("answer", "").strip().lower()) else 0,
                }
            )

    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print("Saved:", args.out, "rows:", len(out_rows))


if __name__ == "__main__":
    main()

