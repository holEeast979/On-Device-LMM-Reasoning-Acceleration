#!/usr/bin/env python3
"""Smoke test for FasterOmni multiturn encoder cache.

This script is written as a drop-in AutoDL test runner. It uses the helper in
`fasteromni.encoder_cache` so it can work even before `pipeline.py` is patched.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List

import torch


DEFAULT_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/root/scripts")
if DEFAULT_PROJECT_ROOT and DEFAULT_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_PROJECT_ROOT)

from fasteromni.encoder_cache import run_multiturn_with_cache
from fasteromni.pipeline import SparseInferencePipeline
from utils.profiling_utils import EncoderTimer


DEFAULT_VIDEO_PATH = os.environ.get(
    "MULTITURN_VIDEO_PATH",
    "/root/autodl-tmp/data/ActivityNet-QA/videos/v_RLBfyIVpocE.mp4",
)

DEFAULT_QUESTIONS = [
    "What is happening in this video?",
    "What is happening in this video?",
]


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cleanup_counter(counter: object) -> None:
    for name in ("remove", "close", "unregister"):
        fn = getattr(counter, name, None)
        if callable(fn):
            fn()
            return


def _truncate(text: str, limit: int = 100) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _all_ok(results: Iterable[object]) -> bool:
    return all(getattr(item, "error", None) in (None, "") for item in results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for multiturn encoder cache")
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument(
        "--question",
        action="append",
        default=None,
        help="Append a custom question. Repeat for multiple turns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions: List[str] = args.question if args.question else list(DEFAULT_QUESTIONS)

    print("=" * 80)
    print("Multiturn Encoder Cache Smoke Test")
    print("=" * 80)

    pipe = SparseInferencePipeline()
    pipe.load_model()

    # Use EncoderTimer instead of ForwardCounter
    encoder_timer = EncoderTimer()
    encoder_timer.register(pipe._model)

    try:
        print("\n[Test] Running repeated questions with encoder cache hook...")
        results, cache = run_multiturn_with_cache(
            pipe,
            args.video_path,
            questions,
            max_new_tokens=args.max_new_tokens,
            max_frames=args.max_frames,
            clear_cache_after=True,
        )

        print("\n[Results]")
        for index, result in enumerate(results, start=1):
            print(f"\nTurn {index}:")
            print(f"  Question: {result.question}")
            print(f"  Output: {_truncate(result.output_text)}")
            print(f"  Time: {result.total_ms:.0f}ms")
            print(f"  Error: {result.error}")

        print("\n[Encoder Calls]")
        visual_count = len(encoder_timer.times.get("visual", []))
        audio_count = len(encoder_timer.times.get("audio", []))
        print(f"  Visual encoder calls: {visual_count}")
        print(f"  Audio encoder calls: {audio_count}")

        print("\n[Cache Stats]")
        for key, value in cache.stats().items():
            print(f"  {key}: {value}")

        print("\n[Output Consistency]")
        same_output = len(results) >= 2 and results[0].output_text == results[1].output_text
        if same_output:
            print("  PASS: Turn 1 and Turn 2 outputs match for the same question.")
        else:
            print("  FAIL: Turn 1 and Turn 2 outputs differ.")
            if len(results) >= 2:
                print(f"    Turn 1: {results[0].output_text}")
                print(f"    Turn 2: {results[1].output_text}")

        print("\n[Cache Hit Verification]")
        visual_ok = visual_count == 1
        audio_ok = audio_count == 1
        print(
            "  "
            + ("PASS" if visual_ok else "FAIL")
            + f": Visual encoder call count = {visual_count} (expected 1)."
        )
        print(
            "  "
            + ("PASS" if audio_ok else "FAIL")
            + f": Audio encoder call count = {audio_count} (expected 1)."
        )

        print("\n[Latency Comparison]")
        turn1_ms = results[0].total_ms if results else 0.0
        turn2_ms = results[1].total_ms if len(results) > 1 else 0.0
        speedup = (turn1_ms / turn2_ms) if turn2_ms > 0 else 0.0
        print(f"  Turn 1 TTFT: {turn1_ms:.0f}ms")
        print(f"  Turn 2 TTFT: {turn2_ms:.0f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        speedup_ok = speedup > 1.2
        print(
            "  "
            + ("PASS" if speedup_ok else "WARN")
            + ": Turn 2 latency "
            + ("improved enough." if speedup_ok else "did not clear the 1.2x target.")
        )

        overall_ok = _all_ok(results) and same_output and visual_ok and audio_ok and speedup_ok
        print("\n[Summary]")
        print("  PASS" if overall_ok else "  FAIL")

        print("\n" + "=" * 80)
        print("Test completed")
        print("=" * 80)

        return 0 if overall_ok else 1
    finally:
        _cleanup_counter(encoder_timer)
        _sync_cuda()


if __name__ == "__main__":
    raise SystemExit(main())
