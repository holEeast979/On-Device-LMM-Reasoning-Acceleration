#!/usr/bin/env python3
"""
Non-inferiority analysis for Video-MME paired comparisons.

Default comparisons:
- sparse vs baseline
- naive_iframe vs baseline

Durations:
- all / short / medium / long

The script can:
1) Load pre-computed paired bootstrap samples from bootstrap_paired_diff.csv
2) Or recompute paired bootstrap samples from mode *_details.csv files

Outputs:
- non_inferiority_results.csv
- non_inferiority_results.json
- bootstrap_paired_diff.csv (when recomputed)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.bootstrap_ci import (  # noqa: E402
    DURATION_ORDER,
    discover_mode_csvs,
    load_mode_records,
)

DEFAULT_DATA_DIR = "/root/autodl-tmp/results/fasteromni/videomme_full/"
DEFAULT_OUT_DIR = "/root/autodl-tmp/results/fasteromni/videomme_full/non_inferiority/"
DEFAULT_COMPARISONS = [("sparse", "baseline"), ("naive_iframe", "baseline")]
DEFAULT_DELTAS = [0.01, 0.02, 0.03, 0.05]


def _fmt_pp(x: float) -> str:
    return f"{x * 100:+.2f}pp"


def _delta_col(delta: float) -> str:
    return f"delta_{int(round(delta * 10000))}bp"


def non_inferiority_test(
    diff_samples: np.ndarray,
    delta: float = 0.03,
    alpha: float = 0.05,
) -> dict:
    """
    Non-inferiority test based on bootstrap samples.

    H0: Delta <= -delta
    H1: Delta > -delta
    Reject H0 iff one-sided lower bound > -delta.
    """
    arr = np.asarray(diff_samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("diff_samples is empty.")
    if delta < 0:
        raise ValueError("delta must be >= 0.")
    if not (0 < alpha < 0.5):
        raise ValueError("alpha must be in (0, 0.5).")

    mean_diff = float(arr.mean())
    ci_lower = float(np.percentile(arr, alpha * 100.0))
    ci_upper = float(np.percentile(arr, (1.0 - alpha) * 100.0))
    non_inferior = bool(ci_lower > -delta)
    # Bootstrap-tail estimate of one-sided p-value for H0: Delta <= -delta.
    p_value = float(np.mean(arr <= -delta))

    if non_inferior:
        conclusion = (
            f"Non-inferior: one-sided {(1 - alpha) * 100:.0f}% CI lower bound "
            f"{_fmt_pp(ci_lower)} > -{_fmt_pp(delta).lstrip('+')}."
        )
    else:
        conclusion = (
            f"Not non-inferior: one-sided {(1 - alpha) * 100:.0f}% CI lower bound "
            f"{_fmt_pp(ci_lower)} <= -{_fmt_pp(delta).lstrip('+')}."
        )

    return {
        "delta": float(delta),
        "alpha": float(alpha),
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "non_inferior": non_inferior,
        "p_value": p_value,
        "conclusion": conclusion,
    }


def parse_comparisons(items: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid comparison '{item}', expected modeA:modeB")
        mode_a, mode_b = item.split(":", 1)
        mode_a = mode_a.strip()
        mode_b = mode_b.strip()
        if not mode_a or not mode_b:
            raise ValueError(f"Invalid comparison '{item}', empty mode name")
        pairs.append((mode_a, mode_b))
    return pairs


def build_paired_diff_samples(
    data_dir: str,
    comparisons: Sequence[Tuple[str, str]],
    n_bootstrap: int,
    seed: int,
) -> Tuple[Dict[Tuple[str, str, str], np.ndarray], List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Recompute paired bootstrap diff samples from raw details.csv files.
    Pairing is by question_id (same logic as fasteromni/bootstrap_ci.py).
    """
    mode_to_csv = discover_mode_csvs(data_dir)
    needed_modes = sorted({m for pair in comparisons for m in pair})
    missing = [m for m in needed_modes if m not in mode_to_csv]
    if missing:
        raise ValueError(f"Required modes not found under data_dir: {missing}")

    mode_records = {}
    manifest: List[Dict[str, object]] = []
    for mode in needed_modes:
        path = mode_to_csv[mode]
        records, stats = load_mode_records(path, mode_hint=mode)
        mode_records[mode] = records
        manifest.append({
            "mode": mode,
            "csv_path": path,
            **stats,
            "valid_records": len(records),
        })

    rng = np.random.default_rng(seed)
    samples_map: Dict[Tuple[str, str, str], np.ndarray] = {}
    pair_info: List[Dict[str, object]] = []

    for mode_a, mode_b in comparisons:
        a_by_qid = {r.question_id: r for r in mode_records[mode_a]}
        b_by_qid = {r.question_id: r for r in mode_records[mode_b]}
        common_qids = sorted(set(a_by_qid) & set(b_by_qid))
        if not common_qids:
            continue

        for duration in DURATION_ORDER:
            if duration == "all":
                qids = common_qids
            else:
                qids = [
                    q for q in common_qids
                    if a_by_qid[q].duration == duration and b_by_qid[q].duration == duration
                ]
            if not qids:
                continue

            a_vals = np.asarray([a_by_qid[q].correct for q in qids], dtype=np.float64)
            b_vals = np.asarray([b_by_qid[q].correct for q in qids], dtype=np.float64)
            paired_diff = a_vals - b_vals
            n = paired_diff.size
            boot_indices = rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int32)
            boot_means = paired_diff[boot_indices].mean(axis=1)

            key = (mode_a, mode_b, duration)
            samples_map[key] = boot_means
            pair_info.append({
                "mode_a": mode_a,
                "mode_b": mode_b,
                "duration": duration,
                "n_pairs": int(n),
                "observed_diff": float(paired_diff.mean()),
            })

    return samples_map, manifest, pair_info


def _read_diff_value(row: Dict[str, str]) -> float:
    for key in ("diff", "acc_diff", "paired_diff", "value"):
        if key in row and str(row[key]).strip() != "":
            return float(row[key])
    raise ValueError("No diff column found in row.")


def load_precomputed_diff_samples(path: str) -> Dict[Tuple[str, str, str], np.ndarray]:
    """
    Load sample-level paired bootstrap diffs from CSV.
    Expected schema (recommended):
      mode_a, mode_b, duration, bootstrap_idx, diff
    """
    groups: Dict[Tuple[str, str, str], List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        required = {"mode_a", "mode_b", "duration"}
        if not required.issubset(cols):
            raise ValueError(
                f"{path} missing required columns {sorted(required)}; got {sorted(cols)}"
            )

        for row in reader:
            key = (
                str(row["mode_a"]).strip(),
                str(row["mode_b"]).strip(),
                str(row["duration"]).strip().lower(),
            )
            groups.setdefault(key, []).append(_read_diff_value(row))

    if not groups:
        raise ValueError(f"{path} is empty.")

    # Summary-only files (one row per key) are not sample-level bootstrap draws.
    if all(len(v) <= 1 for v in groups.values()):
        raise ValueError(
            f"{path} appears to be a summary table, not bootstrap sample draws."
        )

    return {
        k: np.asarray(v, dtype=np.float64)
        for k, v in groups.items()
    }


def save_diff_samples(
    samples_map: Dict[Tuple[str, str, str], np.ndarray],
    out_csv: str,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode_a",
                "mode_b",
                "duration",
                "bootstrap_idx",
                "diff",
                "diff_pctpoint",
            ],
        )
        writer.writeheader()
        for (mode_a, mode_b, duration), arr in sorted(samples_map.items()):
            for i, v in enumerate(arr):
                writer.writerow({
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "duration": duration,
                    "bootstrap_idx": i,
                    "diff": float(v),
                    "diff_pctpoint": float(v) * 100.0,
                })


def build_non_inferiority_rows(
    samples_map: Dict[Tuple[str, str, str], np.ndarray],
    comparisons: Sequence[Tuple[str, str]],
    deltas: Sequence[float],
    alpha: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for mode_a, mode_b in comparisons:
        for duration in DURATION_ORDER:
            key = (mode_a, mode_b, duration)
            if key not in samples_map:
                continue

            arr = samples_map[key]
            if arr.size == 0:
                continue

            # Delta-independent summary stats (using first delta call for consistency).
            base = non_inferiority_test(arr, delta=deltas[0], alpha=alpha)
            row = {
                "mode_a": mode_a,
                "mode_b": mode_b,
                "duration": duration,
                "n_samples": int(arr.size),
                "mean_diff": base["mean_diff"],
                "mean_diff_pctpoint": base["mean_diff"] * 100.0,
                "ci_lower": base["ci_lower"],
                "ci_upper": base["ci_upper"],
                "ci_lower_pctpoint": base["ci_lower"] * 100.0,
                "ci_upper_pctpoint": base["ci_upper"] * 100.0,
                "alpha": alpha,
            }

            for delta in deltas:
                r = non_inferiority_test(arr, delta=delta, alpha=alpha)
                row[_delta_col(delta)] = bool(r["non_inferior"])
                row[f"p_{_delta_col(delta)}"] = float(r["p_value"])
            rows.append(row)
    return rows


def write_csv(rows: Sequence[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_non_inferiority_tables(
    rows: Sequence[Dict[str, object]],
    comparisons: Sequence[Tuple[str, str]],
    deltas: Sequence[float],
    alpha: float,
) -> None:
    print("\n=== NON-INFERIORITY ANALYSIS (delta tolerance) ===")
    print(f"One-sided CI: {(1 - alpha) * 100:.1f}% (alpha={alpha})")

    by_key = {
        (str(r["mode_a"]), str(r["mode_b"]), str(r["duration"])): r
        for r in rows
    }

    for mode_a, mode_b in comparisons:
        print(f"\nComparison: {mode_a} - {mode_b}")
        head = ["Duration", "Mean Diff", f"{(1 - alpha) * 100:.0f}% CI_lo"] + [
            f"d={int(round(d * 100))}pp" for d in deltas
        ]
        print(" | ".join(f"{h:>10}" for h in head))
        print("-" * (13 * len(head)))

        for duration in DURATION_ORDER:
            key = (mode_a, mode_b, duration)
            row = by_key.get(key)
            if row is None:
                print(" | ".join([
                    f"{duration:>10}",
                    f"{'N/A':>10}",
                    f"{'N/A':>10}",
                    *[f"{'N/A':>10}" for _ in deltas],
                ]))
                continue

            cells = [
                f"{duration:>10}",
                f"{row['mean_diff_pctpoint']:+8.2f}pp",
                f"{row['ci_lower_pctpoint']:+8.2f}pp",
            ]
            for delta in deltas:
                mark = "PASS" if row[_delta_col(delta)] else "FAIL"
                cells.append(f"{mark:>10}")
            print(" | ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-MME non-inferiority analysis")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--bootstrap-dir",
        default=None,
        help="If provided, read pre-computed bootstrap samples; otherwise recompute",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--deltas", nargs="+", type=float, default=DEFAULT_DELTAS)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--comparisons",
        nargs="*",
        default=[f"{a}:{b}" for a, b in DEFAULT_COMPARISONS],
        help="Paired comparisons in modeA:modeB format",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    deltas = sorted(set(args.deltas))
    if any(d < 0 for d in deltas):
        raise ValueError("All deltas must be >= 0.")
    if not (0 < args.alpha < 0.5):
        raise ValueError("alpha must be in (0, 0.5).")

    comparisons = parse_comparisons(args.comparisons)
    os.makedirs(args.out_dir, exist_ok=True)

    samples_map: Dict[Tuple[str, str, str], np.ndarray]
    manifest: List[Dict[str, object]] = []
    pair_info: List[Dict[str, object]] = []
    sample_source = ""

    precomputed_path = ""
    if args.bootstrap_dir:
        candidate = args.bootstrap_dir
        if os.path.isdir(candidate):
            precomputed_path = os.path.join(candidate, "bootstrap_paired_diff.csv")
        else:
            precomputed_path = candidate

    if precomputed_path and os.path.exists(precomputed_path):
        try:
            samples_map = load_precomputed_diff_samples(precomputed_path)
            sample_source = f"precomputed:{precomputed_path}"
            print(f"[INFO] Loaded pre-computed bootstrap samples: {precomputed_path}")
        except Exception as e:
            print(
                f"[WARN] Failed to load precomputed samples ({e}), "
                f"recomputing from data-dir: {args.data_dir}"
            )
            samples_map, manifest, pair_info = build_paired_diff_samples(
                data_dir=args.data_dir,
                comparisons=comparisons,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            sample_source = "recomputed"
    else:
        if args.bootstrap_dir:
            print(
                f"[WARN] bootstrap_paired_diff.csv not found under {args.bootstrap_dir}, "
                f"recomputing from data-dir: {args.data_dir}"
            )
        samples_map, manifest, pair_info = build_paired_diff_samples(
            data_dir=args.data_dir,
            comparisons=comparisons,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        sample_source = "recomputed"

    # Persist sample-level bootstrap diffs for reproducibility/reuse.
    sample_csv = os.path.join(args.out_dir, "bootstrap_paired_diff.csv")
    save_diff_samples(samples_map, sample_csv)

    rows = build_non_inferiority_rows(
        samples_map=samples_map,
        comparisons=comparisons,
        deltas=deltas,
        alpha=args.alpha,
    )
    if not rows:
        raise RuntimeError("No non-inferiority rows generated. Check input data/comparisons.")

    csv_path = os.path.join(args.out_dir, "non_inferiority_results.csv")
    write_csv(rows, csv_path)

    # Structured JSON output.
    json_items: List[Dict[str, object]] = []
    for row in rows:
        item = {
            "mode_a": row["mode_a"],
            "mode_b": row["mode_b"],
            "duration": row["duration"],
            "n_samples": row["n_samples"],
            "mean_diff": row["mean_diff"],
            "ci_lower": row["ci_lower"],
            "ci_upper": row["ci_upper"],
            "alpha": row["alpha"],
            "deltas": [],
        }
        key = (str(row["mode_a"]), str(row["mode_b"]), str(row["duration"]))
        arr = samples_map[key]
        for d in deltas:
            r = non_inferiority_test(arr, delta=d, alpha=args.alpha)
            item["deltas"].append({
                "delta": d,
                "non_inferior": r["non_inferior"],
                "p_value": r["p_value"],
                "conclusion": r["conclusion"],
            })
        json_items.append(item)

    meta = {
        "data_dir": args.data_dir,
        "bootstrap_dir": args.bootstrap_dir,
        "sample_source": sample_source,
        "n_bootstrap": args.n_bootstrap,
        "alpha": args.alpha,
        "seed": args.seed,
        "deltas": deltas,
        "comparisons": [{"mode_a": a, "mode_b": b} for a, b in comparisons],
        "sample_csv": sample_csv,
        "manifest": manifest,
        "pair_info": pair_info,
    }
    json_path = os.path.join(args.out_dir, "non_inferiority_results.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": json_items}, f, indent=2, ensure_ascii=False)

    print_non_inferiority_tables(rows, comparisons=comparisons, deltas=deltas, alpha=args.alpha)
    print("\nSaved files:")
    print(f"  - {sample_csv}")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")


if __name__ == "__main__":
    main()
