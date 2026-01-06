import time
import threading
import json
from typing import Dict, List, Optional
import torch

# ============ Token Helpers ============

_MM_TOKEN_ID_CACHE: Dict[int, Dict[str, List[int]]] = {}

def get_mm_token_ids_from_tokenizer(tokenizer) -> Dict[str, List[int]]:
    key = id(tokenizer)
    if key in _MM_TOKEN_ID_CACHE:
        return _MM_TOKEN_ID_CACHE[key]

    tokens = []
    try:
        tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    except Exception:
        tokens = []

    def to_id(tok: str) -> int:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            return int(tid) if tid is not None else -1
        except Exception:
            return -1

    vision_ids: List[int] = []
    audio_ids: List[int] = []

    for t in tokens:
        tl = str(t).lower()
        tid = to_id(t)
        if tid < 0:
            continue
        if ("audio" in tl) and ("pad" in tl or "audio" in tl):
            audio_ids.append(tid)
        if ("image" in tl or "video" in tl or "vision" in tl) and ("pad" in tl or "image" in tl or "video" in tl):
            vision_ids.append(tid)

    mm = {
        "vision_special_token_ids": sorted(set(vision_ids)),
        "audio_special_token_ids": sorted(set(audio_ids)),
    }
    _MM_TOKEN_ID_CACHE[key] = mm
    return mm


def count_special_tokens_in_input_ids(tokenizer, input_ids: torch.Tensor) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    try:
        tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    except Exception:
        tokens = []

    if input_ids is None or not isinstance(input_ids, torch.Tensor):
        return counts
    ids = input_ids.detach()
    if ids.ndim == 2:
        ids = ids[0]
    for t in tokens:
        try:
            tid = tokenizer.convert_tokens_to_ids(t)
        except Exception:
            continue
        if tid is None:
            continue
        tid = int(tid)
        if tid < 0:
            continue
        n = int((ids == tid).sum().item())
        if n > 0:
            counts[str(t)] = n
    return counts


# ============ Resource Monitoring ============

class ResourceMonitor:
    """GPU/CPU/VRAM 资源监控器"""
    
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.records: List[Dict] = []
        self.markers: List[Dict] = []
        self._stop = threading.Event()
        self._thread = None
        self._start_time = None
    
    def start(self):
        self._start_time = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def _monitor_loop(self):
        import psutil
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            has_nvml = True
        except:
            has_nvml = False
        
        while not self._stop.is_set():
            t = time.perf_counter() - self._start_time
            record = {"time": t, "cpu_percent": psutil.cpu_percent()}
            
            if has_nvml:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    record["gpu_percent"] = util.gpu
                    record["vram_used_gb"] = mem.used / 1e9
                    record["vram_total_gb"] = mem.total / 1e9
                except:
                    pass
            
            self.records.append(record)
            time.sleep(self.interval)
    
    def mark(self, name: str):
        t = time.perf_counter() - self._start_time if self._start_time else 0
        self.markers.append({"time": t, "name": name})
    
    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        return self.records, self.markers
    
    def cleanup(self):
        self.records = []
        self.markers = []


def _bytes_to_mb(n: Optional[int]) -> Optional[float]:
    if n is None:
        return None
    try:
        return float(n) / (1024.0 * 1024.0)
    except Exception:
        return None


def _safe_get_rss_bytes() -> Optional[int]:
    try:
        import os
        import psutil

        p = psutil.Process(os.getpid())
        return int(p.memory_info().rss)
    except Exception:
        return None


class TorchCudaMemPeakMonitor:
    """
    Lightweight sampler that tracks peak torch CUDA memory (allocated/reserved) overall and per phase.

    - Uses sampling (interval) rather than reset_peak_memory_stats to avoid interfering with other profilers.
    - Phase attribution is best-effort and driven by `mark()` calls (e.g., from module hooks).
    """

    def __init__(self, *, device: Optional[torch.device] = None, interval_ms: float = 2.0, track_rss: bool = True):
        self.device = device
        self.interval_s = float(interval_ms) / 1000.0
        self.track_rss = bool(track_rss)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._phase = "unknown"

        self.peak_allocated_bytes: int = 0
        self.peak_reserved_bytes: int = 0
        self.peak_rss_bytes: int = 0
        self.phase_peaks: Dict[str, Dict[str, int]] = {}

    def mark(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)

    def reset(self) -> None:
        self.peak_allocated_bytes = 0
        self.peak_reserved_bytes = 0
        self.peak_rss_bytes = 0
        self.phase_peaks = {}

    def _get_phase(self) -> str:
        with self._lock:
            return str(self._phase)

    def start(self) -> None:
        if not torch.cuda.is_available():
            return

        if self.device is None:
            try:
                self.device = torch.device("cuda", torch.cuda.current_device())
            except Exception:
                self.device = torch.device("cuda", 0)

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self.device is not None
        dev = self.device

        while not self._stop.is_set():
            try:
                with torch.cuda.device(dev):
                    allocated = int(torch.cuda.memory_allocated(dev))
                    reserved = int(torch.cuda.memory_reserved(dev))
            except Exception:
                allocated = 0
                reserved = 0

            if allocated > self.peak_allocated_bytes:
                self.peak_allocated_bytes = allocated
            if reserved > self.peak_reserved_bytes:
                self.peak_reserved_bytes = reserved

            phase = self._get_phase()
            pp = self.phase_peaks.get(phase)
            if pp is None:
                pp = {"allocated": 0, "reserved": 0, "rss": 0}
                self.phase_peaks[phase] = pp
            if allocated > pp["allocated"]:
                pp["allocated"] = allocated
            if reserved > pp["reserved"]:
                pp["reserved"] = reserved

            if self.track_rss:
                rss = _safe_get_rss_bytes() or 0
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = int(rss)
                if rss > pp["rss"]:
                    pp["rss"] = int(rss)

            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None

    def summary_mb(self, *, prefix: str = "") -> Dict[str, Optional[float]]:
        """
        Flatten summary with stable keys for CSV rows.
        """
        out: Dict[str, Optional[float]] = {
            f"{prefix}peak_allocated_mb": _bytes_to_mb(self.peak_allocated_bytes),
            f"{prefix}peak_reserved_mb": _bytes_to_mb(self.peak_reserved_bytes),
            f"{prefix}peak_rss_mb": _bytes_to_mb(self.peak_rss_bytes) if self.track_rss else None,
        }

        # Per-phase peaks
        for phase, pp in sorted(self.phase_peaks.items(), key=lambda kv: kv[0]):
            safe = str(phase).strip().lower().replace(" ", "_").replace("-", "_")
            out[f"{prefix}phase_peak_allocated_mb__{safe}"] = _bytes_to_mb(int(pp.get("allocated", 0)))
            out[f"{prefix}phase_peak_reserved_mb__{safe}"] = _bytes_to_mb(int(pp.get("reserved", 0)))
            if self.track_rss:
                out[f"{prefix}phase_peak_rss_mb__{safe}"] = _bytes_to_mb(int(pp.get("rss", 0)))
        return out


class PhaseMarker:
    """
    Helper for tagging phases. Intended for use with module hooks.
    """

    def __init__(self, monitor: TorchCudaMemPeakMonitor, phase: str, *, fallback_phase: str = "generate"):
        self.monitor = monitor
        self.phase = str(phase)
        self.fallback_phase = str(fallback_phase)
        self._handles = []

    def register(self, module) -> None:
        def pre_hook(_m, _inp):
            try:
                self.monitor.mark(self.phase)
            except Exception:
                pass

        def post_hook(_m, _inp, _out):
            try:
                self.monitor.mark(self.fallback_phase)
            except Exception:
                pass

        h1 = module.register_forward_pre_hook(pre_hook)
        h2 = module.register_forward_hook(post_hook)
        self._handles.extend([h1, h2])

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []


# ============ Encoder Timer Hook ============

class EncoderTimer:
    """用于测量 Encoder 耗时的 Hook"""
    
    def __init__(self):
        self.times = {"visual": [], "audio": []}
        self._start = {}
        self._handles = []
    
    def register(self, model):
        """注册 Hook 到 Visual 和 Audio Encoder"""
        
        def make_hooks(name):
            def pre_hook(module, input):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._start[name] = time.perf_counter()
            
            def post_hook(module, input, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - self._start[name]) * 1000
                self.times[name].append(elapsed)
            
            return pre_hook, post_hook
        
        # Visual Encoder
        if hasattr(model, "thinker") and hasattr(model.thinker, "visual"):
            vis_pre, vis_post = make_hooks("visual")
            h1 = model.thinker.visual.register_forward_pre_hook(vis_pre)
            h2 = model.thinker.visual.register_forward_hook(vis_post)
            self._handles.extend([h1, h2])
        
        # Audio Encoder
        if hasattr(model, "thinker") and hasattr(model.thinker, "audio_tower"):
            aud_pre, aud_post = make_hooks("audio")
            h3 = model.thinker.audio_tower.register_forward_pre_hook(aud_pre)
            h4 = model.thinker.audio_tower.register_forward_hook(aud_post)
            self._handles.extend([h3, h4])
    
    def clear(self):
        self.times = {"visual": [], "audio": []}
    
    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class LLMPrefillCudaEventCapture:
    def __init__(self):
        self.prefill_forward_ms = 0.0
        self.prefill_input_ids_len = 0
        self.prefill_inputs_embeds_len = 0
        self.prefill_attention_mask_len = 0
        self.prefill_seq_len = 0
        self._handles = []
        self._start_event = None
        self._end_event = None
        self._device = None
        self._cur_is_prefill = False
        self._cur_seq_len = 0
        self._cur_input_ids_len = 0
        self._cur_inputs_embeds_len = 0
        self._cur_attention_mask_len = 0

    def register(self, llm_module):
        def get_module_device(m) -> torch.device:
            try:
                return next(m.parameters()).device
            except StopIteration:
                return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        def pre_hook(module, args, kwargs):
            input_ids = kwargs.get("input_ids", None)
            inputs_embeds = kwargs.get("inputs_embeds", None)
            attention_mask = kwargs.get("attention_mask", None)

            if (input_ids is None) or (inputs_embeds is None) or (attention_mask is None):
                for a in args:
                    if not isinstance(a, torch.Tensor):
                        continue
                    if a.ndim == 3 and inputs_embeds is None:
                        inputs_embeds = a
                        continue
                    if a.ndim == 2:
                        if (input_ids is None) and (a.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)):
                            input_ids = a
                            continue
                        if attention_mask is None:
                            attention_mask = a
                            continue

            input_ids_len = int(input_ids.shape[-1]) if input_ids is not None else 0
            inputs_embeds_len = int(inputs_embeds.shape[1]) if inputs_embeds is not None else 0
            attention_mask_len = int(attention_mask.shape[-1]) if attention_mask is not None else 0
            seq_len = max(input_ids_len, inputs_embeds_len)

            is_prefill = (seq_len > 1)

            self._cur_is_prefill = bool(is_prefill)
            self._cur_seq_len = int(seq_len)
            self._cur_input_ids_len = int(input_ids_len)
            self._cur_inputs_embeds_len = int(inputs_embeds_len)
            self._cur_attention_mask_len = int(attention_mask_len)

            if not self._cur_is_prefill:
                self._start_event = None
                self._end_event = None
                self._device = get_module_device(module)
                return None

            device = get_module_device(module)
            self._device = device
            if device.type == "cuda":
                with torch.cuda.device(device):
                    self._start_event = torch.cuda.Event(enable_timing=True)
                    self._end_event = torch.cuda.Event(enable_timing=True)
                    self._start_event.record()
            else:
                self._start_event = None
                self._end_event = None
            return None

        def post_hook(module, args, kwargs, output):
            if not self._cur_is_prefill:
                return None
            if self._cur_seq_len <= 0:
                return None

            device = self._device if self._device is not None else get_module_device(module)
            if device.type != "cuda" or self._start_event is None or self._end_event is None:
                return None

            with torch.cuda.device(device):
                self._end_event.record()
                self._end_event.synchronize()
                elapsed = float(self._start_event.elapsed_time(self._end_event))

            if self._cur_seq_len >= int(self.prefill_seq_len):
                self.prefill_forward_ms = float(elapsed)
                self.prefill_input_ids_len = int(self._cur_input_ids_len)
                self.prefill_inputs_embeds_len = int(self._cur_inputs_embeds_len)
                self.prefill_attention_mask_len = int(self._cur_attention_mask_len)
                self.prefill_seq_len = int(self._cur_seq_len)
            return None

        h1 = llm_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        h2 = llm_module.register_forward_hook(post_hook, with_kwargs=True)
        self._handles.extend([h1, h2])

    def clear(self):
        self.prefill_forward_ms = 0.0
        self.prefill_input_ids_len = 0
        self.prefill_inputs_embeds_len = 0
        self.prefill_attention_mask_len = 0
        self.prefill_seq_len = 0

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class ModuleCudaEventTimer:
    def __init__(self):
        self.times: List[float] = []
        self.last_input_shape = None
        self._device = None
        self._start_event = None
        self._end_event = None
        self._start = 0.0
        self._handles = []

    def register(self, module):
        def get_module_device(m) -> torch.device:
            try:
                return next(m.parameters()).device
            except StopIteration:
                return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        def pre_hook(m, input):
            device = get_module_device(m)
            self._device = device
            self.last_input_shape = None
            try:
                if input and isinstance(input[0], torch.Tensor):
                    self.last_input_shape = tuple(input[0].shape)
            except Exception:
                self.last_input_shape = None
            if device.type == "cuda":
                with torch.cuda.device(device):
                    self._start_event = torch.cuda.Event(enable_timing=True)
                    self._end_event = torch.cuda.Event(enable_timing=True)
                    self._start_event.record()
            else:
                self._start = time.perf_counter()

        def post_hook(m, input, output):
            device = self._device if self._device is not None else get_module_device(m)
            if device.type == "cuda" and self._start_event is not None and self._end_event is not None:
                with torch.cuda.device(device):
                    self._end_event.record()
                    self._end_event.synchronize()
                    elapsed = float(self._start_event.elapsed_time(self._end_event))
            else:
                elapsed = (time.perf_counter() - self._start) * 1000
            self.times.append(elapsed)

        h1 = module.register_forward_pre_hook(pre_hook)
        h2 = module.register_forward_hook(post_hook)
        self._handles.extend([h1, h2])

    def clear(self):
        self.times = []
        self.last_input_shape = None

    def get_last(self) -> float:
        return self.times[-1] if self.times else 0.0

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class LLMSeqLenCapture:
    """Hook to capture max input sequence length seen by LLM during generate (prefill is the longest)."""
    def __init__(self):
        self.max_input_ids_len = 0
        self.max_inputs_embeds_len = 0
        self.max_attention_mask_len = 0
        self._handles = []

    def register(self, model):
        def pre_hook(module, args, kwargs):
            if "input_ids" in kwargs and kwargs["input_ids"] is not None:
                seq_len = int(kwargs["input_ids"].shape[-1])
                if seq_len > self.max_input_ids_len:
                    self.max_input_ids_len = seq_len
            if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
                seq_len = int(kwargs["inputs_embeds"].shape[1])
                if seq_len > self.max_inputs_embeds_len:
                    self.max_inputs_embeds_len = seq_len
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                seq_len = int(kwargs["attention_mask"].shape[-1])
                if seq_len > self.max_attention_mask_len:
                    self.max_attention_mask_len = seq_len
            return None
        
        # Check if thinker.model exists (Qwen2.5-Omni)
        if hasattr(model, "thinker") and hasattr(model.thinker, "model"):
            h = model.thinker.model.register_forward_pre_hook(pre_hook, with_kwargs=True)
            self._handles.append(h)
        # Fallback or other models
        elif hasattr(model, "model"):
             h = model.model.register_forward_pre_hook(pre_hook, with_kwargs=True)
             self._handles.append(h)

    def clear(self):
        self.max_input_ids_len = 0
        self.max_inputs_embeds_len = 0
        self.max_attention_mask_len = 0

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
