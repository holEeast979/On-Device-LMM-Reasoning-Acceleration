"""CPU预处理预取缓冲区（生产者-消费者模型）

用于批量评估和多轮问答场景，在GPU推理期间后台预取下一个视频的CPU预处理结果。
"""

import threading
import hashlib
import copy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PrefetchEntry:
    """预取缓冲区条目"""
    future: Future
    submit_time: float
    video_path: str

    def is_ready(self) -> bool:
        return self.future.done()

    def get_result(self, timeout: Optional[float] = None):
        """获取结果，超时则抛出异常"""
        return self.future.result(timeout=timeout)


class PrefetchRingBuffer:
    """CPU预处理预取缓冲区

    两层缓存架构的第一层（CPU层），缓存GOP解析+I帧解码+音频提取的结果。
    与EncoderCache（GPU层）配合，形成完整的多轮问答加速链。

    Args:
        capacity: 缓冲区容量（默认2=当前+下一个视频）
        max_workers: 后台线程数（默认1，单线程够用）
        timeout: 获取结果的超时时间（秒）
    """

    def __init__(self, capacity: int = 2, max_workers: int = 2, timeout: float = 30.0):
        self._capacity = capacity
        self._timeout = timeout
        self._cache: OrderedDict[str, PrefetchEntry] = OrderedDict()
        # max_workers=2: 即使一个worker被挂死的ffmpeg占住，另一个仍可处理新预取
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="prefetch")
        self._lock = threading.Lock()

        # 统计信息
        self._stats = {
            'hits': 0,
            'misses': 0,
            'timeouts': 0,
            'evictions': 0
        }

    @staticmethod
    def _make_key(video_path: str, **kwargs) -> str:
        """生成缓存键（排除question，因为同视频不同问题复用CPU预处理）"""
        # 只保留影响帧选择的参数（video_path从显式参数取，不从kwargs取）
        key_params = {
            'video_path': video_path,
            'mode': kwargs.get('mode', ''),
            'strategy': kwargs.get('strategy', ''),
            'keep_ratio': kwargs.get('keep_ratio', 0.5),
            'max_frames': kwargs.get('max_frames', 64),
            'min_iframe_interval': kwargs.get('min_iframe_interval', 10),
            'min_gop_frames': kwargs.get('min_gop_frames', 10),
            'gop_filter_mode': kwargs.get('gop_filter_mode', 'fixed'),
            'alpha': kwargs.get('alpha', 0.5),
            'min_frames': kwargs.get('min_frames', 8),
        }
        key_str = str(sorted(key_params.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def submit_prefetch(self, video_path: str, select_fn: Callable, select_kwargs: Dict[str, Any]) -> None:
        """提交后台预取任务（生产者）

        Args:
            video_path: 视频路径
            select_fn: 帧选择函数（如pipeline._select_naive）
            select_kwargs: 帧选择参数（包含conversation等）
        """
        if self._capacity == 0:
            return  # 禁用预取

        # select_kwargs 中可能包含 video_path，排除后传给 _make_key 避免重复
        filtered = {k: v for k, v in select_kwargs.items() if k != 'video_path'}
        cache_key = self._make_key(video_path, **filtered)

        with self._lock:
            if cache_key in self._cache:
                # 已在缓冲区，刷新LRU顺序
                self._cache.move_to_end(cache_key)
                return

            # 超容量时LRU淘汰
            while len(self._cache) >= self._capacity:
                evicted_key, evicted_entry = self._cache.popitem(last=False)
                if not evicted_entry.is_ready():
                    evicted_entry.future.cancel()
                self._stats['evictions'] += 1

            # 提交后台任务
            future = self._executor.submit(select_fn, **select_kwargs)
            entry = PrefetchEntry(
                future=future,
                submit_time=datetime.now().timestamp(),
                video_path=video_path
            )
            self._cache[cache_key] = entry

    def _rebuild_conversation(self, video_path: str, question: str) -> list:
        """根据当前question重建conversation（修复Issue #1: 预取用question=''，消费时需替换）"""
        video_ele = {"type": "video", "video": str(video_path)}
        return [{"role": "user", "content": [video_ele, {"type": "text", "text": question}]}]

    def get(self, video_path: str, select_fn: Callable, select_kwargs: Dict[str, Any]):
        """获取预取结果（消费者）

        命中：等待future完成，用当前question重建conversation
        未命中：fallback同步执行

        Returns:
            SelectedFrames对象（包含frames, audio, conversation等）
        """
        if self._capacity == 0:
            # 禁用预取，直接同步执行
            return select_fn(**select_kwargs)

        filtered = {k: v for k, v in select_kwargs.items() if k != 'video_path'}
        cache_key = self._make_key(video_path, **filtered)

        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                # LRU: 刷新访问顺序
                self._cache.move_to_end(cache_key)

        if entry is None:
            # 未命中，同步执行
            self._stats['misses'] += 1
            return select_fn(**select_kwargs)

        # 命中，等待结果
        try:
            result = entry.get_result(timeout=self._timeout)
            self._stats['hits'] += 1

            # 浅拷贝 + 用当前question重建conversation
            # 预取时用 question="" 占位，消费时替换为真实question
            result_copy = copy.copy(result)
            question = select_kwargs.get('question', '')
            result_copy.conversation = self._rebuild_conversation(video_path, question)

            return result_copy

        except Exception as e:
            # 超时或执行失败，从缓存中移除坏条目，防止毒化worker
            self._stats['timeouts'] += 1
            with self._lock:
                if cache_key in self._cache:
                    bad_entry = self._cache.pop(cache_key)
                    if not bad_entry.is_ready():
                        bad_entry.future.cancel()
            print(f"[PrefetchBuffer] Timeout/Error for {video_path}: {e}, fallback to sync", flush=True)
            return select_fn(**select_kwargs)

    def stats(self) -> Dict[str, Any]:
        """返回统计信息"""
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total if total > 0 else 0.0
        return {
            **self._stats,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache)
        }

    def shutdown(self, wait: bool = True):
        """关闭线程池，清理缓冲区"""
        with self._lock:
            # 取消所有未完成的任务
            for entry in self._cache.values():
                if not entry.is_ready():
                    entry.future.cancel()
            self._cache.clear()

        self._executor.shutdown(wait=wait)
