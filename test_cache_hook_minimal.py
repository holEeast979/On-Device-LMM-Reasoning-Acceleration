#!/usr/bin/env python3
"""Minimal test for encoder cache hook mechanism (no full inference)."""

import os
import sys
import torch

DEFAULT_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/root/scripts")
if DEFAULT_PROJECT_ROOT and DEFAULT_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_PROJECT_ROOT)

from fasteromni.encoder_cache import EncoderCacheHook
from fasteromni.pipeline import SparseInferencePipeline


def main():
    print("=" * 80)
    print("Minimal Encoder Cache Hook Test")
    print("=" * 80)

    # Load model
    print("\n[1] Loading model...")
    pipe = SparseInferencePipeline()
    pipe.load_model()
    print("Model loaded.")

    # Create cache hook
    print("\n[2] Creating EncoderCacheHook...")
    cache = EncoderCacheHook(pipe._model, pipe._proc)
    print(f"Cache created. Hook enabled: {cache.hook_enabled}")

    # Enable hook
    print("\n[3] Enabling hook...")
    cache.enable()
    print(f"Hook enabled: {cache.hook_enabled}")

    # Create fake inputs to test hook interception
    print("\n[4] Testing hook interception with fake tensors...")

    # Fake video input (batch=1, frames=4, channels=3, height=224, width=224)
    fake_video = torch.randn(1, 4, 3, 224, 224).to(pipe._model.device)
    fake_video_grid_thw = torch.tensor([[4, 14, 14]], dtype=torch.long).to(pipe._model.device)

    # Fake audio input
    fake_audio = torch.randn(1, 128, 80).to(pipe._model.device)
    fake_audio_lengths = torch.tensor([128], dtype=torch.long).to(pipe._model.device)

    cache_key = "test_video_key"

    # First call (cache miss)
    print("\n[5] First call (should be cache miss)...")
    with cache.active_cache_key(cache_key):
        video_out1 = pipe._model.thinker.get_video_features(
            pixel_values_videos=fake_video,
            video_grid_thw=fake_video_grid_thw
        )
        audio_out1 = pipe._model.thinker.get_audio_features(
            input_features=fake_audio,
            audio_feature_lengths=fake_audio_lengths
        )

    print(f"Video output shape: {video_out1.shape}")
    print(f"Audio output shape: {audio_out1.shape}")
    print(f"Cache stats: {cache.stats()}")

    # Second call (cache hit)
    print("\n[6] Second call (should be cache hit)...")
    with cache.active_cache_key(cache_key):
        video_out2 = pipe._model.thinker.get_video_features(
            pixel_values_videos=fake_video,
            video_grid_thw=fake_video_grid_thw
        )
        audio_out2 = pipe._model.thinker.get_audio_features(
            input_features=fake_audio,
            audio_feature_lengths=fake_audio_lengths
        )

    print(f"Video output shape: {video_out2.shape}")
    print(f"Audio output shape: {audio_out2.shape}")
    print(f"Cache stats: {cache.stats()}")

    # Verify outputs match
    print("\n[7] Verifying outputs match...")
    video_match = torch.allclose(video_out1, video_out2, rtol=1e-5, atol=1e-5)
    audio_match = torch.allclose(audio_out1, audio_out2, rtol=1e-5, atol=1e-5)

    print(f"Video outputs match: {video_match}")
    print(f"Audio outputs match: {audio_match}")

    # Check cache hit counts
    print("\n[8] Checking cache hit counts...")
    stats = cache.stats()
    video_hits_ok = stats["video_cache_hits"] == 1
    audio_hits_ok = stats["audio_cache_hits"] == 1
    video_misses_ok = stats["video_cache_misses"] == 1
    audio_misses_ok = stats["audio_cache_misses"] == 1

    print(f"Video cache hits: {stats['video_cache_hits']} (expected 1) - {'PASS' if video_hits_ok else 'FAIL'}")
    print(f"Audio cache hits: {stats['audio_cache_hits']} (expected 1) - {'PASS' if audio_hits_ok else 'FAIL'}")
    print(f"Video cache misses: {stats['video_cache_misses']} (expected 1) - {'PASS' if video_misses_ok else 'FAIL'}")
    print(f"Audio cache misses: {stats['audio_cache_misses']} (expected 1) - {'PASS' if audio_misses_ok else 'FAIL'}")

    # Cleanup
    print("\n[9] Cleaning up...")
    cache.disable()
    cache.clear_cache()
    print(f"Hook enabled: {cache.hook_enabled}")
    print(f"Cache entries: {cache.stats()['entries']}")

    # Final result
    print("\n" + "=" * 80)
    all_pass = video_match and audio_match and video_hits_ok and audio_hits_ok and video_misses_ok and audio_misses_ok
    print("OVERALL: " + ("PASS" if all_pass else "FAIL"))
    print("=" * 80)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
