#!/usr/bin/env python3
"""
Bootstrap CI analysis for Video-MME details CSV files.

Outputs:
1) Question-level accuracy CI by mode x duration.
2) Paired bootstrap CI for accuracy differences:
   sparse - baseline, naive_iframe - baseline.
3) Per-video aggregated accuracy mean/std/CI by mode x duration.
4) CSV tables + terminal summary.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

REQUIRED_COLUMNS = {
    "question_id",
    "video_file_id",
    "duration",
    "mode",
    "correct",
    "error",
}

DURATION_ORDER = ["all", "short", "medium", "long"]
DEFAULT_COMPARISONS = [("sparse", "baseline"), ("naive_iframe", "baseline")]


@dataclass(frozen=True)
class QARecord:
    question_id: str
    video_file_id: str
    duration: str
    mode: str
    correct: float


def _is_nonempty(value: str) -> bool:
    s = str(value).strip().lower()
    return s not in {"", "nan", "none", "null"}


def _parse_correct(value: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    s = str(value).strip().lower()
    if s in {"true", "1", "yes"}:
        return 1.0
    if s in {"false", "0", "no"}:
        return 0.0
    raise ValueError(f"Cannot parse 'correct' value: {value!r}")


def discover_mode_csvs(input_root: str) -> Dict[str, str]:
    mode_to_csv: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(input_root, "*"))):
        if not os.path.isdir(path):
            continue
        mode = os.path.basename(path)
        preferred = [
            os.path.join(path, f"{mode}_details.csv"),
            os.path.join(path, f"videomme_{mode}_details.csv"),
        ]
        candidates = [p for p in preferred if os.path.exists(p)]
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(path, "*_details.csv")))
        if not candidates:
            continue
        mode_to_csv[mode] = candidates[0]
    if not mode_to_csv:
        raise FileNotFoundError(
            f"No '*_details.csv' found under: {input_root}"
        )
    return mode_to_csv


def load_mode_records(csv_path: str, mode_hint: str) -> Tuple[List[QARecord], Dict[str, int]]:
    records_by_qid: Dict[str, QARecord] = {}
    stats = {"rows": 0, "dropped_error": 0, "deduped_qid": 0}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{csv_path} missing required columns: {sorted(missing)}"
            )

        for row in reader:
            stats["rows"] += 1
            if _is_nonempty(row.get("error", "")):
                stats["dropped_error"] += 1
                continue

            qid = str(row["question_id"]).strip()
            rec = QARecord(
                question_id=qid,
                video_file_id=str(row["video_file_id"]).strip(),
                duration=str(row["duration"]).strip().lower(),
                mode=(str(row.get("mode", "")).strip() or mode_hint),
                correct=_parse_correct(row["correct"]),
            )
            if qid in records_by_qid:
                stats["deduped_qid"] += 1
            records_by_qid[qid] = rec

    if not records_by_qid:
        raise ValueError(f"{csv_path} has no valid rows after filtering errors.")
    return list(records_by_qid.values()), stats


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int32)
    boot_means = arr[boot_indices].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    std = arr.std(ddof=1) if n > 1 else 0.0
    return float(arr.mean()), float(std), float(ci_low), float(ci_high)


def aggregate_per_video(records: Sequence[QARecord]) -> np.ndarray:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        grouped[rec.video_file_id].append(rec.correct)
    return np.asarray([np.mean(vals) for vals in grouped.values()], dtype=np.float64)


def get_duration_filtered(records: Sequence[QARecord], duration: str) -> List[QARecord]:
    if duration == "all":
        return list(records)
    return [r for r in records if r.duration == duration]


def make_question_level_rows(
    all_mode_records: Dict[str, List[QARecord]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for mode in sorted(all_mode_records):
        records = all_mode_records[mode]
        for duration in DURATION_ORDER:
            subset = get_duration_filtered(records, duration)
            if not subset:
                continue
            values = np.asarray([r.correct for r in subset], dtype=np.float64)
            mean, std, ci_low, ci_high = bootstrap_mean_ci(values, n_bootstrap, rng)
            rows.append({
                "mode": mode,
                "duration": duration,
                "n_questions": int(values.size),
                "accuracy": mean,
                "accuracy_pct": mean * 100.0,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_low_pct": ci_low * 100.0,
                "ci_high_pct": ci_high * 100.0,
            })
    return rows


def make_per_video_rows(
    all_mode_records: Dict[str, List[QARecord]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for mode in sorted(all_mode_records):
        records = all_mode_records[mode]
        for duration in DURATION_ORDER:
            subset = get_duration_filtered(records, duration)
            if not subset:
                continue
            per_video_acc = aggregate_per_video(subset)
            mean, std, ci_low, ci_high = bootstrap_mean_ci(per_video_acc, n_bootstrap, rng)
            rows.append({
                "mode": mode,
                "duration": duration,
                "n_videos": int(per_video_acc.size),
                "mean_video_accuracy": mean,
                "mean_video_accuracy_pct": mean * 100.0,
                "std_video_accuracy": std,
                "std_video_accuracy_pct": std * 100.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_low_pct": ci_low * 100.0,
                "ci_high_pct": ci_high * 100.0,
            })
    return rows


def make_paired_rows(
    all_mode_records: Dict[str, List[QARecord]],
    comparisons: Iterable[Tuple[str, str]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for mode_a, mode_b in comparisons:
        if mode_a not in all_mode_records or mode_b not in all_mode_records:
            continue

        a_by_qid = {r.question_id: r for r in all_mode_records[mode_a]}
        b_by_qid = {r.question_id: r for r in all_mode_records[mode_b]}
        common_qids = sorted(set(a_by_qid) & set(b_by_qid))
        if not common_qids:
            continue

        for duration in DURATION_ORDER:
            qids = common_qids
            if duration != "all":
                qids = [
                    q for q in common_qids
                    if a_by_qid[q].duration == duration and b_by_qid[q].duration == duration
                ]
            if not qids:
                continue

            a_vals = np.asarray([a_by_qid[q].correct for q in qids], dtype=np.float64)
            b_vals = np.asarray([b_by_qid[q].correct for q in qids], dtype=np.float64)
            paired_diff = a_vals - b_vals
            mean, std, ci_low, ci_high = bootstrap_mean_ci(paired_diff, n_bootstrap, rng)
            rows.append({
                "mode_a": mode_a,
                "mode_b": mode_b,
                "duration": duration,
                "n_pairs": int(paired_diff.size),
                "acc_diff": mean,
                "acc_diff_pctpoint": mean * 100.0,
                "std_diff": std,
                "std_diff_pctpoint": std * 100.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_low_pctpoint": ci_low * 100.0,
                "ci_high_pctpoint": ci_high * 100.0,
                "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
            })
    return rows


def write_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: float) -> str:
    return f"{value:6.2f}%"


def _fmt_ci(low: float, high: float) -> str:
    return f"[{low:6.2f}, {high:6.2f}]"


def print_question_table(rows: Sequence[Dict[str, object]]) -> None:
    print("\n" + "=" * 95)
    print("QUESTION-LEVEL ACCURACY BOOTSTRAP CI (95%)")
    print("=" * 95)
    print(f"{'Mode':>14} | {'Duration':>8} | {'N':>4} | {'Acc':>9} | {'95% CI (acc %)':>22}")
    print("-" * 95)
    for r in rows:
        print(
            f"{r['mode']:>14} | {r['duration']:>8} | {r['n_questions']:>4} | "
            f"{_fmt_pct(r['accuracy_pct']):>9} | "
            f"{_fmt_ci(r['ci_low_pct'], r['ci_high_pct']):>22}"
        )


def print_paired_table(rows: Sequence[Dict[str, object]]) -> None:
    print("\n" + "=" * 95)
    print("PAIRED BOOTSTRAP ACCURACY DIFFERENCE (95% CI, percentage points)")
    print("=" * 95)
    print(
        f"{'Comparison':>26} | {'Duration':>8} | {'N':>4} | "
        f"{'Diff(pp)':>10} | {'95% CI(pp)':>22} | {'Sig':>5}"
    )
    print("-" * 95)
    for r in rows:
        comp = f"{r['mode_a']}-{r['mode_b']}"
        sig = "YES" if r["ci_excludes_zero"] else "NO"
        print(
            f"{comp:>26} | {r['duration']:>8} | {r['n_pairs']:>4} | "
            f"{r['acc_diff_pctpoint']:>+9.2f} | "
            f"{_fmt_ci(r['ci_low_pctpoint'], r['ci_high_pctpoint']):>22} | {sig:>5}"
        )


def print_per_video_table(rows: Sequence[Dict[str, object]]) -> None:
    print("\n" + "=" * 95)
    print("PER-VIDEO AGGREGATED ACCURACY (95% CI)")
    print("=" * 95)
    print(
        f"{'Mode':>14} | {'Duration':>8} | {'Nv':>4} | "
        f"{'Mean':>9} | {'Std':>9} | {'95% CI (mean %)':>22}"
    )
    print("-" * 95)
    for r in rows:
        print(
            f"{r['mode']:>14} | {r['duration']:>8} | {r['n_videos']:>4} | "
            f"{_fmt_pct(r['mean_video_accuracy_pct']):>9} | "
            f"{_fmt_pct(r['std_video_accuracy_pct']):>9} | "
            f"{_fmt_ci(r['ci_low_pct'], r['ci_high_pct']):>22}"
        )


def parse_comparisons(items: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid comparison '{item}', expected format modeA:modeB")
        mode_a, mode_b = item.split(":", 1)
        mode_a = mode_a.strip()
        mode_b = mode_b.strip()
        if not mode_a or not mode_b:
            raise ValueError(f"Invalid comparison '{item}', empty mode name")
        pairs.append((mode_a, mode_b))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI analysis for Video-MME details CSV files")
    parser.add_argument(
        "--input-root",
        default="/root/autodl-tmp/results/fasteromni/videomme_full",
        help="Root dir containing mode subfolders with *_details.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to save CSV summaries (default: <input-root>/bootstrap_ci)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        nargs="*",
        default=[],
        help="Optional mode whitelist, e.g. --modes baseline sparse naive_iframe",
    )
    parser.add_argument(
        "--comparisons",
        nargs="*",
        default=[f"{a}:{b}" for a, b in DEFAULT_COMPARISONS],
        help="Paired comparisons in modeA:modeB format",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_root, "bootstrap_ci")
    os.makedirs(output_dir, exist_ok=True)

    mode_to_csv = discover_mode_csvs(args.input_root)
    if args.modes:
        requested = set(args.modes)
        missing = sorted(requested - set(mode_to_csv))
        if missing:
            raise ValueError(f"Requested modes not found in input root: {missing}")
        mode_to_csv = {m: p for m, p in mode_to_csv.items() if m in requested}

    all_mode_records: Dict[str, List[QARecord]] = {}
    load_manifest: List[Dict[str, object]] = []
    for mode in sorted(mode_to_csv):
        path = mode_to_csv[mode]
        records, stats = load_mode_records(path, mode_hint=mode)
        all_mode_records[mode] = records
        load_manifest.append({
            "mode": mode,
            "csv_path": path,
            **stats,
            "valid_records": len(records),
        })

    comparisons = parse_comparisons(args.comparisons)
    rng = np.random.default_rng(args.seed)

    question_rows = make_question_level_rows(all_mode_records, args.n_bootstrap, rng)
    per_video_rows = make_per_video_rows(all_mode_records, args.n_bootstrap, rng)
    paired_rows = make_paired_rows(all_mode_records, comparisons, args.n_bootstrap, rng)

    question_csv = os.path.join(output_dir, "bootstrap_accuracy_question_level.csv")
    per_video_csv = os.path.join(output_dir, "bootstrap_accuracy_per_video.csv")
    paired_csv = os.path.join(output_dir, "bootstrap_accuracy_paired_diff.csv")
    manifest_csv = os.path.join(output_dir, "bootstrap_input_manifest.csv")
    meta_json = os.path.join(output_dir, "bootstrap_metadata.json")

    write_csv(question_rows, question_csv)
    write_csv(per_video_rows, per_video_csv)
    write_csv(paired_rows, paired_csv)
    write_csv(load_manifest, manifest_csv)

    with open(meta_json, "w") as f:
        json.dump({
            "input_root": args.input_root,
            "output_dir": output_dir,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "modes": sorted(all_mode_records.keys()),
            "comparisons": [{"mode_a": a, "mode_b": b} for a, b in comparisons],
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print("VIDEO-MME BOOTSTRAP CI SUMMARY")
    print("=" * 95)
    print(f"Input root   : {args.input_root}")
    print(f"Output dir   : {output_dir}")
    print(f"Bootstrap    : {args.n_bootstrap} resamples")
    print(f"Modes loaded : {', '.join(sorted(all_mode_records.keys()))}")

    print("\nLoaded files:")
    for row in load_manifest:
        print(
            f"  - {row['mode']:>12}: valid={row['valid_records']:>3} "
            f"(rows={row['rows']}, dropped_error={row['dropped_error']}, deduped={row['deduped_qid']}) "
            f"{row['csv_path']}"
        )

    print_question_table(question_rows)
    print_paired_table(paired_rows)
    print_per_video_table(per_video_rows)

    print("\nSaved CSV files:")
    print(f"  - {question_csv}")
    print(f"  - {per_video_csv}")
    print(f"  - {paired_csv}")
    print(f"  - {manifest_csv}")
    print(f"  - {meta_json}")


if __name__ == "__main__":
    main()
