#!/usr/bin/env -S python3 -u
"""Full repeated-query encoder cache evaluation.

Supports four experiment combinations:
- cache_only + videomme
- cache_only + activitynet
- gop_cache + videomme
- gop_cache + activitynet

Design:
- AB/BA split at the video level to reduce warmup bias
- per-question prediction parity checks
- resumable CSV outputs for uncached/cached runs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

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


VIDEO_MME_ANNOTATIONS = "/root/autodl-tmp/data/Video-MME/annotations/video_mme_test.json"
VIDEO_MME_VIDEO_DIR = "/root/autodl-tmp/data/Video-MME/videos/data"
ACTIVITYNET_ANNOTATIONS = "/root/autodl-tmp/data/ActivityNet-QA/annotations/activitynet_qa_test.json"
ACTIVITYNET_VIDEO_DIR = "/root/autodl-tmp/data/ActivityNet-QA/videos"
RESULT_ROOT = "/root/autodl-tmp/results/fasteromni"

KEEP_RATIO = 0.5
MAX_FRAMES = 32
MAX_NEW_TOKENS = 16
MIN_FRAMES = 8
RANDOM_SEED = 42

CSV_FIELDNAMES = [
    "question_id",
    "video_id",
    "query_index",
    "pred",
    "gt",
    "correct",
    "elapsed_ms",
    "visual_tokens",
    "output_text",
    "error",
]


@dataclass
class EvalSample:
    dataset: str
    video_id: str
    video_path: str
    question_id: str
    question: str
    gt_answer: str
    query_index: int
    options: Optional[List[str]] = None
    is_mcq: bool = False


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def format_mcq_prompt(question: str, options: List[str]) -> str:
    opts = "\n".join(options)
    return (
        f"{question}\n{opts}\n"
        "Answer with the option letter only (A, B, C, or D)."
    )


def extract_answer_letter(output: str) -> str:
    text = output.strip()
    if text.upper() in ("A", "B", "C", "D"):
        return text.upper()

    match = re.match(r"^([A-Da-d])[.\s,)]", text)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()

    for char in text.upper():
        if char in ("A", "B", "C", "D"):
            return char
    return ""


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def match_activitynet_answer(pred: str, gt: str) -> bool:
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    if pred_norm == gt_norm:
        return True
    if gt_norm and gt_norm in pred_norm:
        return True
    return False


def build_prompt(sample: EvalSample) -> str:
    if sample.is_mcq:
        return format_mcq_prompt(sample.question, sample.options or [])
    return sample.question


def discover_activitynet_videos(video_root: str) -> Dict[str, str]:
    downloaded: Dict[str, str] = {}
    for root, _, files in os.walk(video_root):
        for file_name in files:
            if not file_name.endswith(".mp4"):
                continue
            stem = os.path.splitext(file_name)[0]
            if stem.startswith("v_"):
                downloaded[stem[2:]] = os.path.join(root, file_name)
            downloaded[stem] = os.path.join(root, file_name)
    return downloaded


def _sorted_video_ids(groups: Dict[str, List[EvalSample]]) -> List[str]:
    return sorted(groups.keys())


def load_videomme_samples(max_videos: int = 0) -> Dict[str, List[EvalSample]]:
    with open(VIDEO_MME_ANNOTATIONS, "r") as handle:
        data = json.load(handle)

    downloaded = {
        os.path.splitext(file_name)[0]: os.path.join(VIDEO_MME_VIDEO_DIR, file_name)
        for file_name in os.listdir(VIDEO_MME_VIDEO_DIR)
        if file_name.endswith(".mp4")
    }

    groups: Dict[str, List[EvalSample]] = defaultdict(list)
    for item in data:
        if item.get("duration") != "short":
            continue

        file_id = item["videoID"]
        video_path = downloaded.get(file_id)
        if video_path is None:
            continue

        if max_videos > 0 and file_id not in groups and len(groups) >= max_videos:
            continue

        question_id = item["question_id"]
        query_index = int(question_id.split("-")[-1]) if "-" in question_id else len(groups[file_id]) + 1
        groups[file_id].append(
            EvalSample(
                dataset="videomme",
                video_id=file_id,
                video_path=video_path,
                question_id=question_id,
                question=item["question"],
                gt_answer=item["answer"],
                query_index=query_index,
                options=item["options"],
                is_mcq=True,
            )
        )

    for video_id in list(groups.keys()):
        groups[video_id].sort(key=lambda sample: (sample.query_index, sample.question_id))
    return dict(groups)


def load_activitynet_samples(max_videos: int = 0) -> Dict[str, List[EvalSample]]:
    with open(ACTIVITYNET_ANNOTATIONS, "r") as handle:
        data = json.load(handle)

    downloaded = discover_activitynet_videos(ACTIVITYNET_VIDEO_DIR)
    groups: Dict[str, List[EvalSample]] = defaultdict(list)
    for item in data:
        video_name = item["video_name"]
        video_path = downloaded.get(video_name)
        if video_path is None:
            continue

        if max_videos > 0 and video_name not in groups and len(groups) >= max_videos:
            continue

        groups[video_name].append(
            EvalSample(
                dataset="activitynet",
                video_id=video_name,
                video_path=video_path,
                question_id=item["question_id"],
                question=item["question"],
                gt_answer=item["answer"],
                query_index=len(groups[video_name]) + 1,
                is_mcq=False,
            )
        )

    for video_id in list(groups.keys()):
        groups[video_id].sort(key=lambda sample: (sample.query_index, sample.question_id))
        for index, sample in enumerate(groups[video_id], start=1):
            sample.query_index = index
    return dict(groups)


def load_samples(dataset: str, max_videos: int = 0) -> Dict[str, List[EvalSample]]:
    if dataset == "videomme":
        return load_videomme_samples(max_videos=max_videos)
    if dataset == "activitynet":
        return load_activitynet_samples(max_videos=max_videos)
    raise ValueError(f"Unknown dataset: {dataset}")


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_existing_records(csv_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return []

    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "question_id": row.get("question_id", ""),
                    "video_id": row.get("video_id", ""),
                    "query_index": int(row.get("query_index", "0") or 0),
                    "pred": row.get("pred", ""),
                    "gt": row.get("gt", ""),
                    "correct": row.get("correct", "") == "True",
                    "elapsed_ms": float(row.get("elapsed_ms", "0") or 0.0),
                    "visual_tokens": int(row.get("visual_tokens", "0") or 0),
                    "output_text": row.get("output_text", ""),
                    "error": row.get("error", ""),
                }
            )
    return rows


def completed_question_ids(csv_path: str) -> set[str]:
    completed: set[str] = set()
    for row in load_existing_records(csv_path):
        if row["output_text"].strip() and row["elapsed_ms"] > 0:
            completed.add(row["question_id"])
    return completed


def open_csv_writer(csv_path: str) -> tuple[Any, csv.DictWriter]:
    ensure_parent_dir(csv_path)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    handle = open(csv_path, "a", newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        handle.flush()
    return handle, writer


def make_result_row(sample: EvalSample, result: Any, elapsed_ms: float) -> Dict[str, Any]:
    output_text = getattr(result, "output_text", "") or ""
    error = getattr(result, "error", None) or ""
    if sample.is_mcq:
        pred = extract_answer_letter(output_text)
        correct = pred == sample.gt_answer if pred else False
    else:
        pred = output_text.strip()
        correct = match_activitynet_answer(pred, sample.gt_answer) if pred else False

    return {
        "question_id": sample.question_id,
        "video_id": sample.video_id,
        "query_index": sample.query_index,
        "pred": pred,
        "gt": sample.gt_answer,
        "correct": correct,
        "elapsed_ms": elapsed_ms,
        "visual_tokens": int(getattr(result, "visual_tokens", 0) or 0),
        "output_text": output_text.strip(),
        "error": error,
    }


def make_error_row(sample: EvalSample, error: str) -> Dict[str, Any]:
    return {
        "question_id": sample.question_id,
        "video_id": sample.video_id,
        "query_index": sample.query_index,
        "pred": "",
        "gt": sample.gt_answer,
        "correct": False,
        "elapsed_ms": 0.0,
        "visual_tokens": 0,
        "output_text": "",
        "error": error,
    }


def print_row(prefix: str, current: int, total: int, row: Dict[str, Any]) -> None:
    mark = "✓" if row["correct"] else "✗"
    err = f" ERR:{str(row['error'])[:40]}" if row.get("error") else ""
    pred = row["pred"] if row["pred"] else row["output_text"][:24]
    print(
        f"  [{prefix} {current}/{total}] {row['question_id']} {mark} "
        f"pred={pred} gt={row['gt']} {row['elapsed_ms']:.0f}ms{err}",
        flush=True,
    )


def run_pipeline_once(
    pipe: SparseInferencePipeline,
    sample: EvalSample,
    mode: str,
) -> Dict[str, Any]:
    prompt = build_prompt(sample)
    t0 = time.perf_counter()
    try:
        if mode == "cache_only":
            # Use naive_iframe as baseline, NOT sparse (AV-LRM)
            result = pipe.run_naive(
                video_path=sample.video_path,
                question=prompt,
                strategy="iframe_uniform",
                keep_ratio=KEEP_RATIO,
                max_frames=MAX_FRAMES,
                max_new_tokens=MAX_NEW_TOKENS,
                min_frames=MIN_FRAMES,
            )
        elif mode == "gop_cache":
            result = pipe.run_naive(
                sample.video_path,
                prompt,
                strategy="iframe_uniform",
                keep_ratio=KEEP_RATIO,
                max_frames=MAX_FRAMES,
                max_new_tokens=MAX_NEW_TOKENS,
                min_frames=MIN_FRAMES,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        _sync_cuda()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return make_result_row(sample, result, elapsed_ms)
    except torch.cuda.OutOfMemoryError:
        if hasattr(pipe, "_clear_gpu"):
            pipe._clear_gpu()
        return make_error_row(sample, "OOM")
    except Exception as exc:
        return make_error_row(sample, str(exc))


def split_ab_ba_groups(groups: Dict[str, List[EvalSample]]) -> tuple[Dict[str, List[EvalSample]], Dict[str, List[EvalSample]]]:
    video_ids = list(_sorted_video_ids(groups))
    random.seed(RANDOM_SEED)
    random.shuffle(video_ids)

    midpoint = (len(video_ids) + 1) // 2
    ab_ids = video_ids[:midpoint]
    ba_ids = video_ids[midpoint:]
    ab_group = {video_id: groups[video_id] for video_id in ab_ids}
    ba_group = {video_id: groups[video_id] for video_id in ba_ids}
    return ab_group, ba_group


def run_uncached(
    pipe: SparseInferencePipeline,
    groups: Dict[str, List[EvalSample]],
    mode: str,
    csv_path: str,
) -> None:
    completed = completed_question_ids(csv_path)
    pending = [
        sample
        for video_id in _sorted_video_ids(groups)
        for sample in groups[video_id]
        if sample.question_id not in completed
    ]
    if completed:
        print(
            f"  [resume] uncached CSV already has {len(completed)} completed questions, skipping them",
            flush=True,
        )
    if not pending:
        print("  [resume] uncached run already complete", flush=True)
        return

    handle, writer = open_csv_writer(csv_path)
    try:
        total = len(pending)
        for index, sample in enumerate(pending, start=1):
            row = run_pipeline_once(pipe, sample, mode)
            writer.writerow(row)
            handle.flush()
            print_row("uncached", index, total, row)
    finally:
        handle.close()


def run_cached(
    pipe: SparseInferencePipeline,
    groups: Dict[str, List[EvalSample]],
    mode: str,
    dataset: str,
    csv_path: str,
) -> None:
    completed = completed_question_ids(csv_path)
    if completed:
        print(
            f"  [resume] cached CSV already has {len(completed)} completed questions, skipping them",
            flush=True,
        )

    pending = [
        sample
        for video_id in _sorted_video_ids(groups)
        for sample in groups[video_id]
        if sample.question_id not in completed
    ]
    if not pending:
        print("  [resume] cached run already complete", flush=True)
        return

    cache = EncoderCacheHook(pipe._model, pipe._proc)
    cache.enable()
    handle, writer = open_csv_writer(csv_path)
    total = len(pending)
    current = 0

    try:
        for video_id in _sorted_video_ids(groups):
            samples = [sample for sample in groups[video_id] if sample.question_id not in completed]
            if not samples:
                continue

            cache_key = cache.make_cache_key(
                samples[0].video_path,
                max_frames=MAX_FRAMES,
                keep_ratio=KEEP_RATIO,
                selection_strategy=f"{mode}_{dataset}",
            )
            original = patch_pipeline_run_inference(pipe, cache, cache_key)
            try:
                for sample in samples:
                    current += 1
                    row = run_pipeline_once(pipe, sample, mode)
                    writer.writerow(row)
                    handle.flush()
                    print_row("cached ", current, total, row)
            finally:
                restore_pipeline_run_inference(pipe, original)
                cache.clear_cache()
    finally:
        handle.close()
        cache.disable()


def summarize_results(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    accuracy = (sum(row["correct"] for row in valid) / len(valid) * 100.0) if valid else 0.0
    avg_latency_ms = (sum(row["elapsed_ms"] for row in valid) / len(valid)) if valid else 0.0

    by_query: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_query[int(row["query_index"])].append(row)

    print(f"\n{label}:", flush=True)
    print(f"  Accuracy: {accuracy:.1f}% ({sum(row['correct'] for row in valid)}/{len(valid)})", flush=True)
    print(f"  Avg latency: {avg_latency_ms:.0f}ms", flush=True)
    print(f"  Errors: {len(rows) - len(valid)}", flush=True)

    per_query_breakdown: Dict[str, Dict[str, float]] = {}
    for query_index in sorted(by_query.keys()):
        query_rows = by_query[query_index]
        query_accuracy = sum(row["correct"] for row in query_rows) / len(query_rows) * 100.0
        query_avg_ms = sum(row["elapsed_ms"] for row in query_rows) / len(query_rows)
        print(
            f"  Query {query_index}: acc={query_accuracy:.1f}% avg={query_avg_ms:.0f}ms (n={len(query_rows)})",
            flush=True,
        )
        per_query_breakdown[str(query_index)] = {
            "accuracy": query_accuracy,
            "avg_latency_ms": query_avg_ms,
            "count": len(query_rows),
        }

    return {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency_ms,
        "count": len(valid),
        "error_count": len(rows) - len(valid),
        "per_query_breakdown": per_query_breakdown,
    }


def compare_predictions(
    uncached_rows: List[Dict[str, Any]],
    cached_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    uncached_map = {row["question_id"]: row for row in uncached_rows if not row.get("error")}
    cached_map = {row["question_id"]: row for row in cached_rows if not row.get("error")}

    mismatches: List[Dict[str, Any]] = []
    for question_id in sorted(set(uncached_map) | set(cached_map)):
        uncached = uncached_map.get(question_id)
        cached = cached_map.get(question_id)

        if uncached is None or cached is None:
            mismatches.append(
                {
                    "question_id": question_id,
                    "uncached_pred": None if uncached is None else uncached["pred"],
                    "cached_pred": None if cached is None else cached["pred"],
                    "gt": None if uncached is None else uncached["gt"],
                    "reason": "missing_result",
                }
            )
            continue

        if uncached["pred"] != cached["pred"]:
            mismatches.append(
                {
                    "question_id": question_id,
                    "uncached_pred": uncached["pred"],
                    "cached_pred": cached["pred"],
                    "gt": uncached["gt"],
                    "reason": "prediction_mismatch",
                }
            )

    return mismatches


def save_summary(
    output_dir: str,
    mode: str,
    dataset: str,
    uncached_rows: List[Dict[str, Any]],
    cached_rows: List[Dict[str, Any]],
) -> None:
    uncached_summary = summarize_results(uncached_rows, "UNCACHED")
    cached_summary = summarize_results(cached_rows, "CACHED")
    mismatches = compare_predictions(uncached_rows, cached_rows)
    speedup = (
        uncached_summary["avg_latency_ms"] / cached_summary["avg_latency_ms"]
        if cached_summary["avg_latency_ms"] > 0
        else 0.0
    )

    print("\n" + "=" * 80, flush=True)
    print("COMPARISON", flush=True)
    print("=" * 80, flush=True)
    print(
        f"  Uncached: acc={uncached_summary['accuracy']:.1f}%  avg_latency={uncached_summary['avg_latency_ms']:.0f}ms",
        flush=True,
    )
    print(
        f"  Cached:   acc={cached_summary['accuracy']:.1f}%  avg_latency={cached_summary['avg_latency_ms']:.0f}ms",
        flush=True,
    )
    print(f"  Speedup:  {speedup:.2f}x", flush=True)

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
        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches) - 5} more", flush=True)
    else:
        print("  Prediction match: PASS (0 mismatches)", flush=True)

    summary = {
        "mode": mode,
        "dataset": dataset,
        "keep_ratio": KEEP_RATIO,
        "max_frames": MAX_FRAMES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "uncached": uncached_summary,
        "cached": cached_summary,
        "speedup": speedup,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "per_query_breakdown": {
            "uncached": uncached_summary["per_query_breakdown"],
            "cached": cached_summary["per_query_breakdown"],
        },
    }

    ensure_parent_dir(os.path.join(output_dir, "summary.json"))
    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("\n" + "=" * 80, flush=True)
    print("OVERALL: " + ("PASS" if not mismatches else "FAIL"), flush=True)
    print("=" * 80, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full encoder cache evaluation")
    parser.add_argument("--mode", choices=("cache_only", "gop_cache"), required=True)
    parser.add_argument("--dataset", choices=("videomme", "activitynet"), required=True)
    parser.add_argument("--max-videos", type=int, default=0, help="Limit number of videos for debugging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = os.path.join(RESULT_ROOT, f"cache_eval_{args.mode}_{args.dataset}")
    uncached_csv = os.path.join(output_dir, "uncached_details.csv")
    cached_csv = os.path.join(output_dir, "cached_details.csv")

    print("=" * 80, flush=True)
    print(f"Cache Evaluation: mode={args.mode} dataset={args.dataset}", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] Loading samples...", flush=True)
    groups = load_samples(args.dataset, max_videos=args.max_videos)
    video_count = len(groups)
    question_count = sum(len(samples) for samples in groups.values())
    print(f"Selected {video_count} videos, {question_count} questions", flush=True)
    print(f"Output dir: {output_dir}", flush=True)

    print("\n[2] Loading pipeline...", flush=True)
    pipe = SparseInferencePipeline()
    pipe.load_model()
    print("Pipeline loaded.", flush=True)

    ab_group, ba_group = split_ab_ba_groups(groups)
    print(
        f"AB/BA split: {len(ab_group)} videos uncached-first, {len(ba_group)} videos cached-first",
        flush=True,
    )

    print("\n[3] Running AB group (uncached first)...", flush=True)
    run_uncached(pipe, ab_group, args.mode, uncached_csv)
    run_cached(pipe, ab_group, args.mode, args.dataset, cached_csv)

    print("\n[4] Running BA group (cached first)...", flush=True)
    run_cached(pipe, ba_group, args.mode, args.dataset, cached_csv)
    run_uncached(pipe, ba_group, args.mode, uncached_csv)

    print("\n[5] Building summary...", flush=True)
    uncached_rows = load_existing_records(uncached_csv)
    cached_rows = load_existing_records(cached_csv)
    save_summary(output_dir, args.mode, args.dataset, uncached_rows, cached_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
