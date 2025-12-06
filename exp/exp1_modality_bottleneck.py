#!/usr/bin/env python3
"""
Experiment 1: Modality bottleneck

Runs latency + accuracy on:
  - Images (VQAv2 subset) with Qwen2-VL
  - Videos (MSVD-QA subset) with Qwen2-VL (single frame-count N)
  - Audio (AudioCaps subset) with Qwen2-Audio (latency only by default)

Input manifests (CSV):
  images: image_path,question,answer
  videos: video_path,question,answer
  audios: audio_path,caption

Usage:
  python scripts/exp1_modality_bottleneck.py \
    --qwen2-vl ./Qwen2-VL-7B-Instruct \
    --qwen2-audio ./Qwen2-Audio-7B-Instruct \
    --images data/VQAv2_subset/manifest.csv \
    --videos data/MSVD-QA_subset/manifest.csv \
    --audios data/AudioCaps_subset/manifest.csv \
    --out results/exp1_bottleneck.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import scripts.common as C


def norm(s: str) -> str:
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
    rows = load_csv(args.images)
    model, proc = C.load_qwen2_vl(args.qwen2_vl, args.dtype)
    out_rows: List[Dict[str, object]] = []

    import cv2

    print(f"[Image] Processing {len(rows[: args.max_samples or len(rows)])} samples...")
    for i, r in enumerate(tqdm(rows[: args.max_samples or len(rows)], desc="Images")):
        img = cv2.imread(r["image_path"])[:, :, ::-1].copy()  # BGR->RGB, copy to avoid negative stride
        ans, ttft, total, tok_s, t_pack = C.run_qwen2_vl_image(model, proc, img, r["question"], args.max_new_tokens)
        out_rows.append(
            {
                "modality": "image",
                "dataset": "VQAv2",
                "sample_id": i,
                "frames": 1,
                "ttft_ms": ttft * 1000,
                "t_total_ms": total * 1000,
                "tok_per_s": tok_s,
                "t_decode_ms": 0.0,
                "t_pre_ms": 0.0,
                "t_pack_ms": t_pack * 1000,
                "pred": ans,
                "gold": r.get("answer", ""),
                "correct": em(ans, r.get("answer", "")),
            }
        )
    return out_rows


def run_videos(args) -> List[Dict[str, object]]:
    rows = load_csv(args.videos)
    model, proc = C.load_qwen2_vl(args.qwen2_vl, args.dtype)
    out_rows: List[Dict[str, object]] = []
    print(f"[Video] Processing {len(rows[: args.max_samples or len(rows)])} samples...")
    for i, r in enumerate(tqdm(rows[: args.max_samples or len(rows)], desc="Videos")):
        frames, t_decode, t_pre = C.sample_video_frames(r["video_path"], args.frames, args.short_side)
        if not frames:
            print(f"Warning: Skipping video {i} - no frames decoded")
            continue
        ans, ttft, total, tok_s, t_pack = C.run_qwen2_vl_video(model, proc, frames, r["question"], args.max_new_tokens)
        out_rows.append(
            {
                "modality": "video",
                "dataset": "MSVD-QA",
                "sample_id": i,
                "frames": len(frames),
                "ttft_ms": ttft * 1000,
                "t_total_ms": total * 1000,
                "tok_per_s": tok_s,
                "t_decode_ms": t_decode * 1000,
                "t_pre_ms": t_pre * 1000,
                "t_pack_ms": t_pack * 1000,
                "pred": ans,
                "gold": r.get("answer", ""),
                "correct": em(ans, r.get("answer", "")),
            }
        )
    return out_rows


def run_audios(args) -> List[Dict[str, object]]:
    rows = load_csv(args.audios)
    model, proc = C.load_qwen2_audio(args.qwen2_audio, args.dtype)
    out_rows: List[Dict[str, object]] = []
    q = args.audio_prompt
    print(f"[Audio] Processing {len(rows[: args.max_samples or len(rows)])} samples...")
    for i, r in enumerate(tqdm(rows[: args.max_samples or len(rows)], desc="Audios")):
        audio_path = r["audio_path"]
        if not os.path.exists(audio_path):
            print(f"Warning: Skipping audio {i} - file not found: {audio_path}")
            continue
        try:
            wav, sr = sf.read(audio_path)
        except Exception as e:
            print(f"Warning: Skipping audio {i} - failed to read: {e}")
            continue
        ans, ttft, total, tok_s, t_pack = C.run_qwen2_audio(model, proc, wav, sr, q, args.max_new_tokens)
        out_rows.append(
            {
                "modality": "audio",
                "dataset": "AudioCaps",
                "sample_id": i,
                "frames": 0,
                "ttft_ms": ttft * 1000,
                "t_total_ms": total * 1000,
                "tok_per_s": tok_s,
                "t_decode_ms": 0.0,
                "t_pre_ms": 0.0,
                "t_pack_ms": t_pack * 1000,
                "pred": ans,
                "gold": r.get("caption", ""),
                "correct": 0,
            }
        )
    return out_rows


def main():
    ap = argparse.ArgumentParser("exp1 bottleneck")
    ap.add_argument("--qwen2-vl", required=True, help="local path to Qwen2-VL-7B-Instruct")
    ap.add_argument("--qwen2-audio", required=True, help="local path to Qwen2-Audio-7B-Instruct")
    ap.add_argument("--images", required=True, help="VQAv2 manifest.csv")
    ap.add_argument("--videos", required=True, help="MSVD-QA manifest.csv")
    ap.add_argument("--audios", required=True, help="AudioCaps manifest.csv")
    ap.add_argument("--out", default="results/exp1_bottleneck.csv")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--short-side", type=int, default=336)
    ap.add_argument("--max-samples", type=int, default=0, help="0 for all")
    ap.add_argument("--audio-prompt", default="Describe the audio briefly.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print("\n=== Starting Experiment 1: Modality Bottleneck ===")
    rows: List[Dict[str, object]] = []
    print("\n--- Phase 1/3: Images ---")
    rows += run_images(args)
    print("\n--- Phase 2/3: Videos ---")
    rows += run_videos(args)
    print("\n--- Phase 3/3: Audios ---")
    rows += run_audios(args)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Saved:", args.out, "rows:", len(rows))


if __name__ == "__main__":
    main()

