from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from benchmark.runner import BenchmarkRunner
from benchmark.specs import register_all


def build_common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model-dir", type=str, default="/root/autodl-tmp/Qwen2.5-Omni-7B")
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--manifest", type=str, default="/root/autodl-tmp/data/MSRVTT_subset/manifest.csv")
    p.add_argument("--out-dir", type=str, default="/root/autodl-tmp/results/motivation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--video-nframes", type=int, default=8)
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--profile-mem", action="store_true", help="Record per-request GPU peak memory (allocated/reserved) with phase tags")
    p.add_argument("--mem-interval-ms", type=float, default=2.0, help="Sampling interval for --profile-mem (lower is more accurate but more overhead)")
    return p


def main() -> None:
    common = build_common_parser()

    ap = argparse.ArgumentParser(description="Unified benchmark runner (spec-based)")
    subparsers = ap.add_subparsers(dest="spec", required=True)
    register_all(subparsers, common)

    args = ap.parse_args()

    os.makedirs(str(args.out_dir), exist_ok=True)

    runner = BenchmarkRunner(model_dir=str(args.model_dir), dtype=str(args.dtype))
    out = args._spec_run(args, runner)  # type: ignore[attr-defined]
    print(out)


if __name__ == "__main__":
    main()
