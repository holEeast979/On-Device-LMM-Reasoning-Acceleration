#!/usr/bin/env python3
"""
Experiment 2: Projection strategy comparison

Compare latency/accuracy on images + videos for:
  - Qwen2-VL-7B (linear/MLP style)
  - LLaVA 1.5 7B HF (linear/MLP style)
  - BLIP-2 flan-t5-xl (Q-Former; video uses first frame)

Usage:
  python scripts/exp2_projection_compare.py \
    --qwen2-vl ./Qwen2-VL-7B-Instruct \
    --llava ./llava-1.5-7b-hf \
    --blip2 ./blip2-flan-t5-xl \
    --images data/VQAv2_subset/manifest.csv \
    --videos data/MSVD-QA_subset/manifest.csv \
    --frames 8 --out results/exp2_projection.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

import scripts.common as C
import cv2


def norm(s: str) -> str:
    import re

    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def em(pred: str, gold: str) -> int:
    return int(norm(pred) == norm(gold))


def load_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({k.strip(): v.strip() if v is not None else "" for k, v in r.items()})
    return rows


def run_images(args) -> List[Dict[str, object]]:
    import torch
    import gc
    
    rows = load_csv(args.images)
    out_rows: List[Dict[str, object]] = []
    samples = rows[: args.max_samples or len(rows)]
    
    print(f"[Images] Processing {len(samples)} samples with 3 models (sequential loading)...")
    
    # 模型列表
    models = [
        ("qwen2_vl", args.qwen2_vl, C.load_qwen2_vl, C.run_qwen2_vl_image),
        ("llava", args.llava, C.load_llava_like, C.run_llava_image),
        ("blip2", args.blip2, C.load_blip2, C.run_blip2_image)
    ]
    
    for model_name, model_path, load_func, run_func in models:
        print(f"  Loading {model_name}...")
        model, proc = load_func(model_path, args.dtype)
        
        for i, r in enumerate(tqdm(samples, desc=f"Images-{model_name}")):
            img = cv2.imread(r["image_path"])[:, :, ::-1].copy()  # BGR->RGB, copy to avoid negative stride
            a, t1, ttot, ts, tp = run_func(model, proc, img, r["question"], args.max_new_tokens)
            out_rows.append({"tag": model_name, "modality": "image", "dataset": "VQAv2", "sample_id": i,
                             "frames": 1, "ttft_ms": t1 * 1000, "t_total_ms": ttot * 1000, "tok_per_s": ts,
                             "t_decode_ms": 0.0, "t_pre_ms": 0.0, "t_pack_ms": tp * 1000,
                             "pred": a, "gold": r.get("answer", ""), "correct": em(a, r.get("answer", ""))})
        
        # 释放显存
        print(f"  Unloading {model_name}...")
        del model, proc
        torch.cuda.empty_cache()
        gc.collect()
    
    return out_rows


def run_videos(args) -> List[Dict[str, object]]:
    import torch
    import gc
    
    rows = load_csv(args.videos)
    out_rows: List[Dict[str, object]] = []
    samples = rows[: args.max_samples or len(rows)]
    
    print(f"[Videos] Processing {len(samples)} samples with 3 models (sequential loading)...")
    
    # 模型列表 (BLIP-2 特殊处理)
    models = [
        ("qwen2_vl", args.qwen2_vl, C.load_qwen2_vl, C.run_qwen2_vl_video, False),
        ("llava", args.llava, C.load_llava_like, C.run_llava_video, False),
        ("blip2", args.blip2, C.load_blip2, C.run_blip2_image, True)  # BLIP-2 用第一帧
    ]
    
    for model_name, model_path, load_func, run_func, use_first_frame in models:
        print(f"  Loading {model_name}...")
        model, proc = load_func(model_path, args.dtype)
        
        for i, r in enumerate(tqdm(samples, desc=f"Videos-{model_name}")):
            frames, t_decode, t_pre = C.sample_video_frames(r["video_path"], args.frames, args.short_side)
            if not frames:
                print(f"Warning: Skipping video {i} - no frames decoded")
                continue
            
            if use_first_frame:
                # BLIP-2: use first frame only (Q-Former not designed for long frame lists)
                a, t1, ttot, ts, tp = run_func(model, proc, frames[0], r["question"], args.max_new_tokens)
                frame_count = 1
            else:
                a, t1, ttot, ts, tp = run_func(model, proc, frames, r["question"], args.max_new_tokens)
                frame_count = len(frames)
                
            out_rows.append({"tag": model_name, "modality": "video", "dataset": "MSVD-QA", "sample_id": i,
                             "frames": frame_count, "ttft_ms": t1 * 1000, "t_total_ms": ttot * 1000, "tok_per_s": ts,
                             "t_decode_ms": t_decode * 1000, "t_pre_ms": t_pre * 1000, "t_pack_ms": tp * 1000,
                             "pred": a, "gold": r.get("answer", ""), "correct": em(a, r.get("answer", ""))})
        
        # 释放显存
        print(f"  Unloading {model_name}...")
        del model, proc
        torch.cuda.empty_cache()
        gc.collect()
    
    return out_rows


def main():
    ap = argparse.ArgumentParser("exp2 projection compare")
    ap.add_argument("--qwen2-vl", required=True)
    ap.add_argument("--llava", required=True)
    ap.add_argument("--blip2", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--short-side", type=int, default=336)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out", default="results/exp2_projection.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("\n=== Starting Experiment 2: Projection Comparison ===")
    rows: List[Dict[str, object]] = []
    print("\n--- Phase 1/2: Images ---")
    rows += run_images(args)
    print("\n--- Phase 2/2: Videos ---")
    rows += run_videos(args)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Saved:", args.out, "rows:", len(rows))


if __name__ == "__main__":
    main()

