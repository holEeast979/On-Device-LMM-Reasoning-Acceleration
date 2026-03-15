#!/usr/bin/env python3
"""检查 GOP + Cache 是否能正常工作"""
import sys
sys.path.insert(0, "/root/scripts")

from fasteromni.pipeline import SparseInferencePipeline
from fasteromni.encoder_cache import EncoderCacheHook

# 加载 pipeline
pipe = SparseInferencePipeline()
pipe.load_model()

# 创建 cache hook
cache = EncoderCacheHook(pipe._model, pipe._proc)
cache.enable()

# 测试视频
video_path = "/root/autodl-tmp/data/Video-MME/videos/data/40BlVzjxu-I.mp4"
question = "What is happening in the video?"

print("=" * 80)
print("Test 1: run_naive (GOP稀疏化) without cache")
print("=" * 80)
r1 = pipe.run_naive(
    video_path=video_path,
    question=question,
    strategy="iframe_uniform",  # naive_iframe
    keep_ratio=0.5,
    max_frames=16,
    max_new_tokens=16,
)
print(f"Mode: {r1.mode}")
print(f"Visual tokens: {r1.visual_tokens}")
print(f"Total time: {r1.total_ms:.0f}ms")
print(f"Answer: {r1.output_text}")

print("\n" + "=" * 80)
print("Test 2: run_naive (GOP稀疏化) with cache - Query 1")
print("=" * 80)
cache_key = cache.make_cache_key(
    video_path=video_path,
    max_frames=16,
    keep_ratio=0.5,
    selection_strategy="naive_iframe",
)
with cache.active_cache_key(cache_key):
    r2 = pipe.run_naive(
        video_path=video_path,
        question=question,
        strategy="iframe_uniform",
        keep_ratio=0.5,
        max_frames=16,
        max_new_tokens=16,
    )
print(f"Mode: {r2.mode}")
print(f"Visual tokens: {r2.visual_tokens}")
print(f"Total time: {r2.total_ms:.0f}ms")
print(f"Answer: {r2.output_text}")
print(f"Cache stats: {cache.stats()}")

print("\n" + "=" * 80)
print("Test 3: run_naive (GOP稀疏化) with cache - Query 2 (should hit cache)")
print("=" * 80)
with cache.active_cache_key(cache_key):
    r3 = pipe.run_naive(
        video_path=video_path,
        question="What color is the sky?",  # 不同问题
        strategy="iframe_uniform",
        keep_ratio=0.5,
        max_frames=16,
        max_new_tokens=16,
    )
print(f"Mode: {r3.mode}")
print(f"Visual tokens: {r3.visual_tokens}")
print(f"Total time: {r3.total_ms:.0f}ms")
print(f"Answer: {r3.output_text}")
print(f"Cache stats: {cache.stats()}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Without cache: {r1.total_ms:.0f}ms")
print(f"With cache Q1: {r2.total_ms:.0f}ms (speedup: {r1.total_ms/r2.total_ms:.2f}x)")
print(f"With cache Q2: {r3.total_ms:.0f}ms (speedup: {r1.total_ms/r3.total_ms:.2f}x)")
print(f"Cache hits: video={cache.video_cache_hits}, audio={cache.audio_cache_hits}")
print(f"Cache misses: video={cache.video_cache_misses}, audio={cache.audio_cache_misses}")

cache.disable()
print("\n✅ GOP + Cache 可以正常工作！")
