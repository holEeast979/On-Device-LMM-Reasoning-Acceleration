#!/usr/bin/env -S python3 -u
"""最小化同视频重复查询缓存集成测试。

测试思路：
1. 直接加载 `SparseInferencePipeline`
2. 不自己调 processor，不自己拼输入
3. 两次都走 `run_sparse()`，确保 GOP 解析 + I 帧选择仍然生效
4. 只在 `_run_inference()` 外层包 `cache.active_cache_key()`
5. 用 EncoderTimer 看 visual/audio encoder 是否只跑了一次
"""

from __future__ import annotations

import os
import sys
import time

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
from utils.profiling_utils import EncoderTimer


VIDEO_PATH = "/root/autodl-tmp/data/ActivityNet-QA/videos/v_RLBfyIVpocE.mp4"
QUESTION = "What is happening in this video?"
KEEP_RATIO = 0.5
MAX_FRAMES = 16
MAX_NEW_TOKENS = 8


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_one_turn(pipe: SparseInferencePipeline, video_path: str, question: str):
    return pipe.run_sparse(
        video_path=video_path,
        question=question,
        keep_ratio=KEEP_RATIO,
        max_frames=MAX_FRAMES,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def main() -> int:
    print("=" * 80, flush=True)
    print("Repeated-Query Encoder Cache Integration Test", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] Loading pipeline...", flush=True)
    pipe = SparseInferencePipeline()
    pipe.load_model()
    print("Pipeline loaded.", flush=True)

    print("\n[2] Creating cache hook and encoder timer...", flush=True)
    cache = EncoderCacheHook(pipe._model, pipe._proc)
    cache.enable()

    encoder_timer = EncoderTimer()
    encoder_timer.register(pipe._model)

    cache_key = cache.make_cache_key(
        VIDEO_PATH,
        max_frames=MAX_FRAMES,
        keep_ratio=KEEP_RATIO,
        selection_strategy="same_video_repeated_query_sparse",
    )
    print(f"Cache key: {cache_key}", flush=True)

    print("\n[3] Patching _run_inference()...", flush=True)
    original_run_inference = patch_pipeline_run_inference(pipe, cache, cache_key)
    print("Patch installed.", flush=True)

    try:
        print("\n[4] Running Query 1...", flush=True)
        t1_start = time.perf_counter()
        result1 = _run_one_turn(pipe, VIDEO_PATH, QUESTION)
        _sync_cuda()
        turn1_ms = (time.perf_counter() - t1_start) * 1000.0
        print(f"Query 1 completed in {turn1_ms:.0f}ms", flush=True)
        print(f"  Error: {getattr(result1, 'error', None)}", flush=True)
        print(f"  Output: {getattr(result1, 'output_text', '')[:120]}", flush=True)

        print("\n[5] Running Query 2 (same video + same question)...", flush=True)
        t2_start = time.perf_counter()
        result2 = _run_one_turn(pipe, VIDEO_PATH, QUESTION)
        _sync_cuda()
        turn2_ms = (time.perf_counter() - t2_start) * 1000.0
        print(f"Query 2 completed in {turn2_ms:.0f}ms", flush=True)
        print(f"  Error: {getattr(result2, 'error', None)}", flush=True)
        print(f"  Output: {getattr(result2, 'output_text', '')[:120]}", flush=True)

        visual_count = len(encoder_timer.times.get("visual", []))
        audio_count = len(encoder_timer.times.get("audio", []))
        speedup = (turn1_ms / turn2_ms) if turn2_ms > 0 else 0.0
        outputs_match = getattr(result1, "output_text", "") == getattr(result2, "output_text", "")
        no_error = not getattr(result1, "error", None) and not getattr(result2, "error", None)
        visual_ok = visual_count == 1
        audio_ok = audio_count == 1
        speedup_ok = speedup > 1.2

        print("\n[6] Verification", flush=True)
        print(f"  Visual encoder calls: {visual_count} -> {'PASS' if visual_ok else 'FAIL'}", flush=True)
        print(f"  Audio encoder calls: {audio_count} -> {'PASS' if audio_ok else 'FAIL'}", flush=True)
        print(f"  Outputs match: {outputs_match} -> {'PASS' if outputs_match else 'FAIL'}", flush=True)
        print(f"  Query 1 latency: {turn1_ms:.0f}ms", flush=True)
        print(f"  Query 2 latency: {turn2_ms:.0f}ms", flush=True)
        print(f"  Speedup: {speedup:.2f}x -> {'PASS' if speedup_ok else 'FAIL'}", flush=True)

        print("\n[7] Cache stats", flush=True)
        for key, value in cache.stats().items():
            print(f"  {key}: {value}", flush=True)

        overall_ok = no_error and outputs_match and visual_ok and audio_ok and speedup_ok

        print("\n" + "=" * 80, flush=True)
        print("OVERALL: " + ("PASS" if overall_ok else "FAIL"), flush=True)
        print("=" * 80, flush=True)

        return 0 if overall_ok else 1

    finally:
        print("\n[Cleanup] Restoring pipeline and clearing cache...", flush=True)
        restore_pipeline_run_inference(pipe, original_run_inference)
        cache.disable()
        cache.clear_cache()
        encoder_timer.remove()
        _sync_cuda()
        print("Cleanup done.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
