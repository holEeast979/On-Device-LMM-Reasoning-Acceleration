"""
Patch for SparseInferencePipeline to add same-video repeated-query cache support.

Add this to pipeline.py after the _run_inference method.
"""

def enable_encoder_cache(self):
    """Enable encoder cache for repeated queries on the same video."""
    if not hasattr(self, '_encoder_cache'):
        from fasteromni.encoder_cache import EncoderCacheHook
        self._encoder_cache = EncoderCacheHook(self._model, self._proc)
    self._encoder_cache.enable()
    return self._encoder_cache


def run_multiturn(
    self,
    video_path: str,
    questions: list[str],
    mode: str = "sparse",
    keep_ratio: float = 0.5,
    max_new_tokens: int = 16,
    max_frames: int = 32,
) -> tuple[list, object]:
    """
    Run multiple questions on the same video with encoder caching.

    Returns:
        (results, cache): List of PipelineResult objects and the cache hook.
    """
    self.load_model()

    # Enable cache
    cache = self.enable_encoder_cache()
    cache_key = cache.make_cache_key(
        video_path,
        max_frames=max_frames,
        keep_ratio=keep_ratio,
        selection_strategy=f"repeated_query_{mode}",
    )

    results = []

    try:
        for question in questions:
            # Use existing run_sparse or run_baseline logic
            if mode == "sparse":
                # Prepare frames using GOP parsing (existing logic)
                from fasteromni.modules.gop_parser import GOPParser
                from fasteromni.modules.frame_decoder import FrameDecoder
                from fasteromni.modules.sparse import select_frames_naive_iframe

                gop_parser = GOPParser()
                frame_decoder = FrameDecoder()

                # Parse GOP
                gop_info = gop_parser.parse(video_path)

                # Select I-frames
                selected = select_frames_naive_iframe(
                    video_path=video_path,
                    question=question,
                    gop_info=gop_info,
                    keep_ratio=keep_ratio,
                    max_frames=max_frames,
                    frame_decoder=frame_decoder,
                )

                result = PipelineResult(mode="repeated_query_sparse")
                result.video_path = video_path
                result.question = question

                # Run inference with cache
                with cache.active_cache_key(cache_key):
                    self._run_inference(selected, result, max_new_tokens=max_new_tokens)

                results.append(result)
            else:
                # Baseline mode
                result = self.run_baseline(
                    video_path=video_path,
                    question=question,
                    max_new_tokens=max_new_tokens,
                    max_frames=max_frames,
                )
                results.append(result)

    finally:
        cache.disable()

    return results, cache
