#!/usr/bin/env -S python3 -u
"""最小 baseline 推理测试。

用途：
1. 不加任何 hook
2. 不加 EncoderTimer / ForwardCounter
3. 只跑一次 `SparseInferencePipeline.run_sparse()`
4. 量一下真实总耗时，判断是环境慢，还是缓存/hook 逻辑有问题
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

from fasteromni.pipeline import SparseInferencePipeline


VIDEO_PATH = "/root/autodl-tmp/data/ActivityNet-QA/videos/v_RLBfyIVpocE.mp4"
QUESTION = "What is happening in this video?"
KEEP_RATIO = 0.5
MAX_FRAMES = 16
MAX_NEW_TOKENS = 8


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    print("=" * 80, flush=True)
    print("Sparse Baseline One-Shot Test", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] Creating pipeline...", flush=True)
    pipe = SparseInferencePipeline()

    print("\n[2] Loading model...", flush=True)
    load_start = time.perf_counter()
    pipe.load_model()
    _sync_cuda()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    print(f"Model loaded in {load_ms:.0f}ms", flush=True)

    print("\n[3] Running run_sparse() once...", flush=True)
    infer_start = time.perf_counter()
    result = pipe.run_sparse(
        video_path=VIDEO_PATH,
        question=QUESTION,
        keep_ratio=KEEP_RATIO,
        max_frames=MAX_FRAMES,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    _sync_cuda()
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    print("\n[4] Result", flush=True)
    print(f"  Error: {getattr(result, 'error', None)}", flush=True)
    print(f"  Total time: {infer_ms:.0f}ms", flush=True)
    print(f"  Output: {getattr(result, 'output_text', '')[:200]}", flush=True)

    # 尽量把常见字段都打出来，方便快速定位
    for field in (
        "mode",
        "num_frames",
        "visual_tokens",
        "generate_ms",
        "total_ms",
        "visual_encoder_ms",
        "audio_encoder_ms",
        "prefill_ms",
    ):
        if hasattr(result, field):
            print(f"  {field}: {getattr(result, field)}", flush=True)

    ok = not getattr(result, "error", None)

    print("\n" + "=" * 80, flush=True)
    print("OVERALL: " + ("PASS" if ok else "FAIL"), flush=True)
    print("=" * 80, flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
