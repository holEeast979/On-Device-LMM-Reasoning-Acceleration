"""Hook-based encoder cache for FasterOmni repeated-query inference.

这个模块只做一件事：hook `model.thinker.get_video_features()` 和
`model.thinker.get_audio_features()`，缓存同一视频第一次查询的 encoder 输出，
后续重复查询直接返回缓存结果。

刻意不做的事：
- 不自己调用 processor
- 不自己调用 model.generate
- 不自己传 video_path 给 processor

原因很简单：FasterOmni 的核心价值在 GOP 解析 + I 帧选择。测试和集成都应该
继续走 `SparseInferencePipeline` 现有链路，让 `_run_inference()` 里那条
`videos=selected.videos` 的路径保持不变。
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import gc
import hashlib
import threading
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch


def _safe_detach(result: Any) -> Any:
    """安全 detach：tensor 直接 detach，BaseModelOutput 则整体缓存。"""
    if torch.is_tensor(result):
        return result.detach()
    # BaseModelOutputWithPooling 等 HF 输出对象，直接缓存整个对象
    return result


def _tensor_shape(value: Any) -> Optional[Tuple[int, ...]]:
    if torch.is_tensor(value):
        return tuple(int(dim) for dim in value.shape)
    return None


def _clone_meta(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _same_meta(lhs: Any, rhs: Any) -> bool:
    if lhs is None and rhs is None:
        return True
    if lhs is None or rhs is None:
        return False
    if torch.is_tensor(lhs) and torch.is_tensor(rhs):
        return lhs.shape == rhs.shape and torch.equal(lhs.detach().cpu(), rhs.detach().cpu())
    return lhs == rhs


def _coerce_frame_indices(frame_indices: Any) -> Optional[Tuple[int, ...]]:
    if frame_indices is None:
        return None
    if torch.is_tensor(frame_indices):
        return tuple(int(item) for item in frame_indices.detach().cpu().reshape(-1).tolist())
    try:
        return tuple(int(item) for item in frame_indices)
    except TypeError:
        return (int(frame_indices),)


def _coerce_number(value: Any, cast_type: type) -> Optional[Any]:
    if value is None:
        return None
    try:
        return cast_type(value)
    except (TypeError, ValueError):
        return None


def _first_attr(source: Any, *names: str) -> Any:
    if source is None:
        return None
    for name in names:
        if not hasattr(source, name):
            continue
        value = getattr(source, name)
        if value is not None:
            return value
    return None


@dataclass
class CachedFeatures:
    """缓存的 encoder 输出，以及命中校验用的轻量 metadata。"""

    video_embeds: Optional[torch.Tensor] = None
    audio_features: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
    audio_feature_lengths: Optional[torch.Tensor] = None
    video_input_shape: Optional[Tuple[int, ...]] = None
    audio_input_shape: Optional[Tuple[int, ...]] = None
    created_at: float = 0.0
    last_access_at: float = 0.0
    video_hits: int = 0
    audio_hits: int = 0

    def matches_video_request(self, pixel_values_videos: Any, video_grid_thw: Any) -> bool:
        return (
            self.video_embeds is not None
            and self.video_input_shape == _tensor_shape(pixel_values_videos)
            and _same_meta(self.video_grid_thw, video_grid_thw)
        )

    def matches_audio_request(self, input_features: Any, audio_feature_lengths: Any) -> bool:
        return (
            self.audio_features is not None
            and self.audio_input_shape == _tensor_shape(input_features)
            and _same_meta(self.audio_feature_lengths, audio_feature_lengths)
        )


class EncoderCacheHook:
    """只负责 hook encoder，不负责构造输入。"""

    def __init__(self, model: Any, processor: Any | None = None) -> None:
        self.model = model
        self.processor = processor

        self.cache: Dict[str, CachedFeatures] = {}
        self._cache_key_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
            f"encoder_cache_key_{id(self)}",
            default=None,
        )
        self._original_get_video: Optional[Any] = None
        self._original_get_audio: Optional[Any] = None
        self._hook_enabled = False
        self._lock = threading.RLock()

        self.video_cache_hits = 0
        self.video_cache_misses = 0
        self.audio_cache_hits = 0
        self.audio_cache_misses = 0

    @property
    def hook_enabled(self) -> bool:
        return self._hook_enabled

    def enable(self) -> None:
        """安装 hook。"""
        with self._lock:
            if self._hook_enabled:
                return

            thinker = self.model.thinker
            self._original_get_video = thinker.get_video_features
            self._original_get_audio = thinker.get_audio_features

            def cached_get_video(*args: Any, **kwargs: Any) -> Any:
                return self._dispatch_video(*args, **kwargs)

            def cached_get_audio(*args: Any, **kwargs: Any) -> Any:
                return self._dispatch_audio(*args, **kwargs)

            thinker.get_video_features = cached_get_video
            thinker.get_audio_features = cached_get_audio
            self._hook_enabled = True

    def disable(self) -> None:
        """恢复原始 encoder 方法。"""
        with self._lock:
            if not self._hook_enabled:
                return

            thinker = self.model.thinker
            if self._original_get_video is not None:
                thinker.get_video_features = self._original_get_video
            if self._original_get_audio is not None:
                thinker.get_audio_features = self._original_get_audio

            self._hook_enabled = False

    def _dispatch_video(self, *args: Any, **kwargs: Any) -> Any:
        pixel_values_videos = kwargs.get("pixel_values_videos", args[0] if args else None)
        video_grid_thw = kwargs.get("video_grid_thw", args[1] if len(args) > 1 else None)

        with self._lock:
            cache_key = self._cache_key_context.get()
            cached = self.cache.get(cache_key) if cache_key is not None else None
            if (
                cache_key is not None
                and cached is not None
                and cached.matches_video_request(pixel_values_videos, video_grid_thw)
            ):
                cached.last_access_at = time.perf_counter()
                cached.video_hits += 1
                self.video_cache_hits += 1
                return cached.video_embeds

            self.video_cache_misses += 1

        result = self._original_get_video(*args, **kwargs)

        with self._lock:
            cache_key = self._cache_key_context.get()
            if cache_key is None:
                return result

            cached = self.cache.setdefault(
                cache_key,
                CachedFeatures(created_at=time.perf_counter()),
            )
            cached.video_embeds = _safe_detach(result)
            cached.video_grid_thw = _clone_meta(video_grid_thw)
            cached.video_input_shape = _tensor_shape(pixel_values_videos)
            cached.last_access_at = time.perf_counter()

        return result

    def _dispatch_audio(self, *args: Any, **kwargs: Any) -> Any:
        input_features = kwargs.get("input_features", args[0] if args else None)
        audio_feature_lengths = kwargs.get(
            "audio_feature_lengths",
            args[2] if len(args) > 2 else None,
        )

        with self._lock:
            cache_key = self._cache_key_context.get()
            cached = self.cache.get(cache_key) if cache_key is not None else None
            if (
                cache_key is not None
                and cached is not None
                and cached.matches_audio_request(input_features, audio_feature_lengths)
            ):
                cached.last_access_at = time.perf_counter()
                cached.audio_hits += 1
                self.audio_cache_hits += 1
                return cached.audio_features

            self.audio_cache_misses += 1

        result = self._original_get_audio(*args, **kwargs)

        with self._lock:
            cache_key = self._cache_key_context.get()
            if cache_key is None:
                return result

            cached = self.cache.setdefault(
                cache_key,
                CachedFeatures(created_at=time.perf_counter()),
            )
            cached.audio_features = _safe_detach(result)
            cached.audio_feature_lengths = _clone_meta(audio_feature_lengths)
            cached.audio_input_shape = _tensor_shape(input_features)
            cached.last_access_at = time.perf_counter()

        return result

    @contextmanager
    def active_cache_key(self, cache_key: str) -> Iterable[None]:
        """在当前上下文里设置缓存键，避免并发请求互相踩 key。"""
        token = self._cache_key_context.set(cache_key)
        try:
            yield
        finally:
            self._cache_key_context.reset(token)

    def make_cache_key(
        self,
        video_path: str,
        *,
        max_frames: int,
        keep_ratio: float = 0.5,
        selection_strategy: str = "naive_iframe",
        frame_indices: Optional[Sequence[int]] = None,
        audio_start_ms: Optional[float] = None,
        audio_end_ms: Optional[float] = None,
    ) -> str:
        """生成缓存键，默认兼容旧调用，但允许把真实选择信息一起编码进去。"""
        key_parts = [
            f"video={video_path}",
            f"max_frames={int(max_frames)}",
            f"keep_ratio={float(keep_ratio):.6f}",
            f"strategy={selection_strategy}",
        ]

        normalized_frames = _coerce_frame_indices(frame_indices)
        if normalized_frames is not None:
            frame_key = ",".join(str(index) for index in normalized_frames)
            frame_fingerprint = hashlib.md5(frame_key.encode("utf-8")).hexdigest()[:8]
            key_parts.append(f"frames={frame_fingerprint}")

        if audio_start_ms is not None or audio_end_ms is not None:
            start = "none" if audio_start_ms is None else f"{float(audio_start_ms):.0f}"
            end = "none" if audio_end_ms is None else f"{float(audio_end_ms):.0f}"
            key_parts.append(f"audio={start}-{end}")

        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()

    def resolve_cache_key(
        self,
        fallback_cache_key: Optional[str],
        *,
        selected: Any = None,
        result: Any = None,
    ) -> str:
        """优先从当前选择结果里拼更细的 key，拿不到信息时退回旧 key。"""
        video_path = _first_attr(result, "video_path")
        if video_path is None:
            video_path = _first_attr(selected, "video_path")

        max_frames_value = _first_attr(selected, "max_frames")
        if max_frames_value is None:
            max_frames_value = _first_attr(result, "max_frames")
        max_frames = _coerce_number(max_frames_value, int)

        keep_ratio_value = _first_attr(selected, "keep_ratio")
        if keep_ratio_value is None:
            keep_ratio_value = _first_attr(result, "keep_ratio")
        keep_ratio = _coerce_number(keep_ratio_value, float)

        selection_strategy = _first_attr(selected, "selection_strategy", "strategy")
        if selection_strategy is None:
            selection_strategy = _first_attr(result, "selection_strategy")
        if selection_strategy is None:
            selection_strategy = _first_attr(result, "mode")
        if selection_strategy is None:
            selection_strategy = "naive_iframe"
        frame_indices = _coerce_frame_indices(
            _first_attr(
                selected,
                "frame_indices",
                "selected_frame_indices",
                "selected_indices",
                "kept_frame_indices",
                "i_frame_indices",
            )
        )
        audio_start_ms = _coerce_number(
            _first_attr(selected, "audio_start_ms", "selected_audio_start_ms"),
            float,
        )
        audio_end_ms = _coerce_number(
            _first_attr(selected, "audio_end_ms", "selected_audio_end_ms"),
            float,
        )

        if video_path is None or max_frames is None:
            if fallback_cache_key is None:
                raise ValueError("cache key metadata is incomplete and no fallback cache key was provided")
            return fallback_cache_key

        return self.make_cache_key(
            video_path,
            max_frames=max_frames,
            keep_ratio=0.5 if keep_ratio is None else keep_ratio,
            selection_strategy=str(selection_strategy),
            frame_indices=frame_indices,
            audio_start_ms=audio_start_ms,
            audio_end_ms=audio_end_ms,
        )

    def has_cache(
        self,
        video_path: str,
        *,
        max_frames: int,
        keep_ratio: float = 0.5,
        selection_strategy: str = "naive_iframe",
        frame_indices: Optional[Sequence[int]] = None,
        audio_start_ms: Optional[float] = None,
        audio_end_ms: Optional[float] = None,
    ) -> bool:
        cache_key = self.make_cache_key(
            video_path,
            max_frames=max_frames,
            keep_ratio=keep_ratio,
            selection_strategy=selection_strategy,
            frame_indices=frame_indices,
            audio_start_ms=audio_start_ms,
            audio_end_ms=audio_end_ms,
        )
        return cache_key in self.cache

    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        with self._lock:
            keys = [cache_key] if cache_key is not None else list(self.cache.keys())
            for key in keys:
                cached = self.cache.pop(key, None)
                if cached is None:
                    continue
                cached.video_embeds = None
                cached.audio_features = None
                cached.video_grid_thw = None
                cached.audio_feature_lengths = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self.cache),
                "video_cache_hits": self.video_cache_hits,
                "video_cache_misses": self.video_cache_misses,
                "audio_cache_hits": self.audio_cache_hits,
                "audio_cache_misses": self.audio_cache_misses,
            }

    def _deprecated_error(self) -> RuntimeError:
        return RuntimeError(
            "Deprecated: 不能在 EncoderCacheHook 里自己调用 processor/generate。"
            "请继续走 FasterOmni 的 GOP 解析 + I 帧选择链路，"
            "在 SparseInferencePipeline._run_inference() 外层用 "
            "`with cache.active_cache_key(cache_key): ...` 包住原始推理。"
        )

    def _build_inputs(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - 明确禁用
        raise self._deprecated_error()

    def generate_with_cache(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - 明确禁用
        raise self._deprecated_error()


def patch_pipeline_run_inference(pipeline: Any, cache: EncoderCacheHook, cache_key: str):
    """给 pipeline._run_inference 打一层轻量补丁。

    这层补丁不改 FasterOmni 原有逻辑，只是在原始 `_run_inference()` 外层
    设置当前 cache key。真正的缓存命中仍发生在 `model.generate()` 内部。
    """
    original_run_inference = pipeline._run_inference

    @functools.wraps(original_run_inference)
    def patched_run_inference(selected, result, max_new_tokens=256):
        resolved_cache_key = cache.resolve_cache_key(
            cache_key,
            selected=selected,
            result=result,
        )
        with cache.active_cache_key(resolved_cache_key):
            return original_run_inference(selected, result, max_new_tokens=max_new_tokens)

    pipeline._run_inference = patched_run_inference
    return original_run_inference


def restore_pipeline_run_inference(pipeline: Any, original_run_inference: Any) -> None:
    """恢复 `_run_inference()` 原始实现。"""
    pipeline._run_inference = original_run_inference
