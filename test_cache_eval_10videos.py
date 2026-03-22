#!/usr/bin/env -S python3 -u
"""10-video repeated-query encoder cache evaluation.

思路：
- 选前 10 个视频（每个 3 问）
- uncached: 正常跑 30 条（每条独立调用 run_sparse）
- cached: 按视频分组，同一视频的 3 条查询共享 encoder cache
- 采用 AB/BA 分组，尽量把预热效应摊平
- 对比：逐题预测是否一致 + 平均延迟加速比
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

DEFAULT_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/root/scripts")
if DEFAULT_PROJECT_ROOT and DEFAULT_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_PROJECT_ROOT)

from fasteromni.encoder_cache import (
    EncoderCacheHook,
    patch_pipeline_run_inference,
    restore_pipeline_run_inference,
)
from fasteromni.pipeline import SparseInferencePipeline

ANNOTATION_PATH = "/root/autodl-tmp/data/Video-MME/annotations/video_mme_test.json"
VIDEO_DIR = "/root/autodl-tmp/data/Video-MME/videos"
MAX_VIDEOS = 10
MAX_FRAMES = 16
MAX_NEW_TOKENS = 16
KEEP_RATIO = 0.5
CACHE_STRATEGY = "same_video_repeated_query_sparse"


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def format_mcq_prompt(question: str, options: List[str]) -> str:
    opts = "\n".join(options)
    return (
        f"{question}\n{opts}\n"
        "Answer with the option's letter from the given choices directly."
    )


def extract_answer(text: str) -> str:
    text = text.strip()
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1).upper()
    return text[:1].upper() if text else ""


def load_samples(max_videos: int) -> Dict[str, Dict[str, Any]]:
    with open(ANNOTATION_PATH) as f:
        data = json.load(f)

    # Group by video_id
    groups = defaultdict(list)
    for item in data:
        groups[item["video_id"]].append(item)

    # Take first N videos
    selected = {}
    for vid in sorted(groups.keys()):
        if len(selected) >= max_videos:
            break
        # Check video file exists
        video_path = os.path.join(VIDEO_DIR, "data", f"{groups[vid][0]['videoID']}.mp4")
        if not os.path.exists(video_path):
            continue
        selected[vid] = {
            "video_path": video_path,
            "questions": groups[vid],
        }

    return selected


def run_uncached(pipe, videos: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """正常跑，每条独立调用 run_sparse"""
    results: List[Dict[str, Any]] = []
    total = sum(len(v["questions"]) for v in videos.values())
    idx = 0

    for vid, info in videos.items():
        for q in info["questions"]:
            idx += 1
            prompt = format_mcq_prompt(q["question"], q["options"])
            t0 = time.perf_counter()
            r = pipe.run_sparse(
                video_path=info["video_path"],
                question=prompt,
                keep_ratio=KEEP_RATIO,
                max_frames=MAX_FRAMES,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            _sync_cuda()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            pred = extract_answer(getattr(r, "output_text", ""))
            gt = q["answer"]
            correct = pred == gt

            results.append({
                "question_id": q["question_id"],
                "video_id": vid,
                "pred": pred,
                "gt": gt,
                "correct": correct,
                "elapsed_ms": elapsed_ms,
                "output_text": getattr(r, "output_text", ""),
                "error": getattr(r, "error", None),
            })

            mark = "✓" if correct else "✗"
            err = f" ERR:{r.error[:20]}" if getattr(r, "error", None) else ""
            print(f"  [uncached {idx}/{total}] {q['question_id']} {mark} pred={pred} gt={gt} {elapsed_ms:.0f}ms{err}", flush=True)

    return results


def run_cached(pipe, videos: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按视频分组，同一视频的 3 问共享 encoder cache"""
    cache = EncoderCacheHook(pipe._model, pipe._proc)
    cache.enable()

    results: List[Dict[str, Any]] = []
    total = sum(len(v["questions"]) for v in videos.values())
    idx = 0

    try:
        for vid, info in videos.items():
            cache_key = cache.make_cache_key(
                info["video_path"],
                max_frames=MAX_FRAMES,
                keep_ratio=KEEP_RATIO,
                selection_strategy=CACHE_STRATEGY,
            )
            original = patch_pipeline_run_inference(pipe, cache, cache_key)

            try:
                for q in info["questions"]:
                    idx += 1
                    prompt = format_mcq_prompt(q["question"], q["options"])
                    t0 = time.perf_counter()
                    r = pipe.run_sparse(
                        video_path=info["video_path"],
                        question=prompt,
                        keep_ratio=KEEP_RATIO,
                        max_frames=MAX_FRAMES,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    _sync_cuda()
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    pred = extract_answer(getattr(r, "output_text", ""))
                    gt = q["answer"]
                    correct = pred == gt

                    results.append({
                        "question_id": q["question_id"],
                        "video_id": vid,
                        "pred": pred,
                        "gt": gt,
                        "correct": correct,
                        "elapsed_ms": elapsed_ms,
                        "output_text": getattr(r, "output_text", ""),
                        "error": getattr(r, "error", None),
                    })

                    mark = "✓" if correct else "✗"
                    err = f" ERR:{r.error[:20]}" if getattr(r, "error", None) else ""
                    print(f"  [cached  {idx}/{total}] {q['question_id']} {mark} pred={pred} gt={gt} {elapsed_ms:.0f}ms{err}", flush=True)
            finally:
                restore_pipeline_run_inference(pipe, original)
                cache.clear_cache()

    finally:
        cache.disable()

    return results


def summarize(results: List[Dict[str, Any]], label: str) -> Dict[str, float]:
    valid = [r for r in results if not r.get("error")]
    correct = sum(r["correct"] for r in valid)
    acc = correct / len(valid) * 100 if valid else 0
    avg_ms = sum(r["elapsed_ms"] for r in valid) / len(valid) if valid else 0

    # Per-query-position breakdown (1/2/3 based on question_id suffix)
    by_turn = defaultdict(list)
    for r in valid:
        qid = r["question_id"]
        turn = int(qid.split("-")[-1]) if "-" in qid else 1
        by_turn[turn].append(r)

    print(f"\n{label}:", flush=True)
    print(f"  Accuracy: {acc:.1f}% ({correct}/{len(valid)})", flush=True)
    print(f"  Avg latency: {avg_ms:.0f}ms", flush=True)
    print(f"  Errors: {len(results) - len(valid)}", flush=True)

    for turn in sorted(by_turn.keys()):
        turn_results = by_turn[turn]
        turn_acc = sum(r["correct"] for r in turn_results) / len(turn_results) * 100
        turn_avg_ms = sum(r["elapsed_ms"] for r in turn_results) / len(turn_results)
        print(
            f"  Query {turn}: acc={turn_acc:.1f}% avg={turn_avg_ms:.0f}ms (n={len(turn_results)})",
            flush=True,
        )

    return {"acc": acc, "avg_ms": avg_ms, "correct": correct, "total": len(valid)}


def compare_predictions(
    uncached_results: List[Dict[str, Any]],
    cached_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """逐题对比预测，防止“总分一样但具体题目不一样”的假一致。"""
    uncached_map = {row["question_id"]: row for row in uncached_results if not row.get("error")}
    cached_map = {row["question_id"]: row for row in cached_results if not row.get("error")}

    mismatches: List[Dict[str, Any]] = []
    for question_id in sorted(set(uncached_map) | set(cached_map)):
        uncached = uncached_map.get(question_id)
        cached = cached_map.get(question_id)

        if uncached is None or cached is None:
            mismatches.append({
                "question_id": question_id,
                "uncached_pred": None if uncached is None else uncached["pred"],
                "cached_pred": None if cached is None else cached["pred"],
                "gt": None if uncached is None else uncached["gt"],
                "reason": "missing_result",
            })
            continue

        if uncached["pred"] != cached["pred"]:
            mismatches.append({
                "question_id": question_id,
                "uncached_pred": uncached["pred"],
                "cached_pred": cached["pred"],
                "gt": uncached["gt"],
                "reason": "prediction_mismatch",
            })

    return mismatches


def split_ab_ba_groups(videos: Dict[str, Dict[str, Any]]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """把视频随机打散后一分为二，一半先跑 uncached，一半先跑 cached。"""
    video_ids = list(videos.keys())
    random.seed(42)
    random.shuffle(video_ids)

    midpoint = (len(video_ids) + 1) // 2
    ab_group = {video_id: videos[video_id] for video_id in video_ids[:midpoint]}
    ba_group = {video_id: videos[video_id] for video_id in video_ids[midpoint:]}
    return ab_group, ba_group


def main():
    print("=" * 80, flush=True)
    print("Same-Video Repeated-Query Encoder Cache Evaluation", flush=True)
    print("=" * 80, flush=True)

    # Load data
    print("\n[1] Loading samples...", flush=True)
    videos = load_samples(MAX_VIDEOS)
    total_q = sum(len(v["questions"]) for v in videos.values())
    print(f"Selected {len(videos)} videos, {total_q} questions", flush=True)

    # Load pipeline
    print("\n[2] Loading pipeline...", flush=True)
    pipe = SparseInferencePipeline()
    pipe.load_model()
    print("Pipeline loaded.", flush=True)

    ab_videos, ba_videos = split_ab_ba_groups(videos)
    print(
        f"AB/BA split: {len(ab_videos)} videos uncached-first, {len(ba_videos)} videos cached-first",
        flush=True,
    )

    print("\n[3] Running AB group (uncached first)...", flush=True)
    ab_uncached = run_uncached(pipe, ab_videos)
    ab_cached = run_cached(pipe, ab_videos)

    print("\n[4] Running BA group (cached first)...", flush=True)
    ba_cached = run_cached(pipe, ba_videos)
    ba_uncached = run_uncached(pipe, ba_videos)

    uncached_results = ab_uncached + ba_uncached
    cached_results = ab_cached + ba_cached

    uncached_summary = summarize(uncached_results, "UNCACHED")
    cached_summary = summarize(cached_results, "CACHED")

    # Compare
    print("\n" + "=" * 80, flush=True)
    print("COMPARISON", flush=True)
    print("=" * 80, flush=True)
    print(f"  Uncached: acc={uncached_summary['acc']:.1f}%  avg_latency={uncached_summary['avg_ms']:.0f}ms", flush=True)
    print(f"  Cached:   acc={cached_summary['acc']:.1f}%  avg_latency={cached_summary['avg_ms']:.0f}ms", flush=True)

    speedup = uncached_summary["avg_ms"] / cached_summary["avg_ms"] if cached_summary["avg_ms"] > 0 else 0
    print(f"  Speedup:  {speedup:.2f}x", flush=True)

    mismatches = compare_predictions(uncached_results, cached_results)
    if mismatches:
        print(f"  Prediction match: FAIL ({len(mismatches)} mismatches)", flush=True)
        print("  First mismatches:", flush=True)
        for mismatch in mismatches[:5]:
            print(
                "    "
                f"{mismatch['question_id']}: uncached={mismatch['uncached_pred']} "
                f"cached={mismatch['cached_pred']} gt={mismatch['gt']} "
                f"reason={mismatch['reason']}",
                flush=True,
            )
        remaining = len(mismatches) - 5
        if remaining > 0:
            print(f"    ... and {remaining} more", flush=True)
    else:
        print("  Prediction match: PASS (0 mismatches)", flush=True)

    print("\n" + "=" * 80, flush=True)
    prediction_match = len(mismatches) == 0
    print("OVERALL: " + ("PASS" if prediction_match else "FAIL"), flush=True)
    print("=" * 80, flush=True)

    return 0 if prediction_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
