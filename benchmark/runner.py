from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import common as C
import profiling_utils as P


@dataclass
class ExtractedAV:
    audios: List[np.ndarray]
    videos: Any
    extract_ms: float


@dataclass
class PackedInputs:
    inputs: Dict[str, Any]
    pack_ms: float


class ForwardCounter:
    def __init__(self):
        self.count = 0
        self._handles = []

    def register(self, module, with_kwargs: bool = False):
        if with_kwargs:
            def pre_hook(m, args, kwargs):
                self.count += 1
                return None

            h = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        else:
            def pre_hook(m, input):
                self.count += 1
                return None

            h = module.register_forward_pre_hook(pre_hook)
        self._handles.append(h)

    def clear(self):
        self.count = 0

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []


class BenchmarkRunner:
    def __init__(self, *, model_dir: str, dtype: str = "bf16"):
        self.model_dir = str(model_dir)
        self.dtype = str(dtype)
        self._model = None
        self._proc = None
        self._fe = None

    def load(self):
        if self._model is not None and self._proc is not None and self._fe is not None:
            return type("Loaded", (), {"model": self._model, "processor": self._proc, "feature_extractor": self._fe})

        model, proc = C.load_qwen25_omni(self.model_dir, self.dtype)
        from transformers import WhisperFeatureExtractor

        fe = WhisperFeatureExtractor.from_pretrained(self.model_dir)
        self._model = model
        self._proc = proc
        self._fe = fe
        return type("Loaded", (), {"model": model, "processor": proc, "feature_extractor": fe})

    def env_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "torch": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
        return info

    def write_json(self, path: str, obj: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_manifest_csv(manifest_path: str, n_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
        df = pd.read_csv(manifest_path)
        if len(df) == 0:
            return []
        if n_samples is None or int(n_samples) <= 0 or int(n_samples) >= len(df):
            sub = df
        else:
            sub = df.sample(n=int(n_samples), random_state=int(seed))
        out: List[Dict[str, Any]] = []
        for _, r in sub.iterrows():
            d = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.to_dict().items()}
            if "sample_id" not in d or d.get("sample_id") is None or str(d.get("sample_id")).strip() == "":
                d["sample_id"] = str(len(out))
            out.append(d)
        return out

    @staticmethod
    def get_qwen25_modules(model) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        visual = None
        audio = None
        llm = None
        try:
            visual = model.thinker.visual
        except Exception:
            visual = None
        try:
            audio = model.thinker.audio_tower
        except Exception:
            audio = None
        try:
            llm = model.thinker.model
        except Exception:
            llm = None
        return visual, audio, llm

    @staticmethod
    def _sync_model_devices(model) -> None:
        if not torch.cuda.is_available():
            return
        devices = set()
        try:
            d = getattr(model, "device", None)
            if isinstance(d, torch.device):
                devices.add(d)
        except Exception:
            pass
        try:
            devices.add(next(model.parameters()).device)
        except Exception:
            pass
        try:
            devices.add(next(model.thinker.audio_tower.parameters()).device)
        except Exception:
            pass
        try:
            devices.add(next(model.thinker.visual.parameters()).device)
        except Exception:
            pass

        for d in devices:
            if isinstance(d, torch.device) and d.type == "cuda":
                torch.cuda.synchronize(device=d)

    def extract_av_from_video(self, video_path: str, question: str, video_nframes: Optional[int]) -> ExtractedAV:
        from qwen_omni_utils import process_mm_info

        video_ele: Dict[str, Any] = {"type": "video", "video": str(video_path)}
        if video_nframes is not None and int(video_nframes) > 0:
            video_ele["nframes"] = int(video_nframes)
        conversation = [{"role": "user", "content": [video_ele, {"type": "text", "text": str(question)}]}]

        t0 = time.perf_counter()
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        extract_ms = (time.perf_counter() - t0) * 1000

        audios = list(audios or [])
        return ExtractedAV(audios=audios, videos=videos, extract_ms=float(extract_ms))

    def prepare_base_inputs(self, model, proc, *, videos: Any, question: str, video_path: str, video_nframes: Optional[int]) -> PackedInputs:
        video_ele: Dict[str, Any] = {"type": "video", "video": str(video_path)}
        if video_nframes is not None and int(video_nframes) > 0:
            video_ele["nframes"] = int(video_nframes)
        conversation = [{"role": "user", "content": [video_ele, {"type": "text", "text": str(question)}]}]

        t0 = time.perf_counter()
        text = proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=text, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        pack_ms = (time.perf_counter() - t0) * 1000
        return PackedInputs(inputs=inputs, pack_ms=float(pack_ms))

    @staticmethod
    def truncate_audio(audio: np.ndarray, target_seconds: float, sample_rate: int = 16000) -> np.ndarray:
        target_samples = int(float(target_seconds) * float(sample_rate))
        if target_samples <= 0:
            return audio
        if audio is None:
            return audio
        if len(audio) > target_samples:
            return audio[:target_samples]
        if len(audio) < target_samples:
            pad = np.zeros((target_samples - len(audio),), dtype=audio.dtype)
            return np.concatenate([audio, pad], axis=0)
        return audio

    def build_audio_features(self, fe, audio: np.ndarray, *, padding: str, audio_max_seconds: float) -> Tuple[Dict[str, torch.Tensor], int, float]:
        t0 = time.perf_counter()
        if str(padding) == "max_length":
            max_audio_samples = int(float(audio_max_seconds) * 16000)
            af = fe(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=max_audio_samples,
                truncation=True,
            )
        elif str(padding) == "do_not_pad":
            af = fe(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="do_not_pad",
                truncation=False,
            )
        else:
            raise ValueError(f"unknown padding: {padding}")
        mel_frames = int(af["input_features"].shape[2])
        ms = (time.perf_counter() - t0) * 1000
        return af, int(mel_frames), float(ms)

    def attach_audio_features(
        self,
        *,
        base_inputs: Dict[str, Any],
        af: Dict[str, torch.Tensor],
        mel_frames: int,
        model,
        dtype: str,
    ) -> Dict[str, Any]:
        out = {k: v for k, v in base_inputs.items()}
        model_dtype = C.get_dtype(dtype)
        out["input_features"] = af["input_features"].to(model.device, dtype=model_dtype)
        out["feature_attention_mask"] = torch.ones((1, int(mel_frames)), device=model.device, dtype=torch.long)
        return out

    def run_generate_1token_ms(self, model, inputs: Dict[str, Any]) -> float:
        self._sync_model_devices(model)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_audio=False,
            )
        self._sync_model_devices(model)
        return float((time.perf_counter() - t0) * 1000)

    def get_token_stats(self, proc, input_ids: Optional[torch.Tensor]) -> Dict[str, Any]:
        if input_ids is None or not isinstance(input_ids, torch.Tensor):
            return {"text_tokens": None, "audio_tokens": None, "visual_tokens": None, "total_tokens": None}

        ids = input_ids.detach()
        if ids.ndim == 2:
            ids = ids[0]

        tok = getattr(proc, "tokenizer", proc)
        mm = P.get_mm_token_ids_from_tokenizer(tok)
        vision_ids = set(int(x) for x in mm.get("vision_special_token_ids", []) or [])
        audio_ids = set(int(x) for x in mm.get("audio_special_token_ids", []) or [])

        total = int(ids.shape[-1])
        audio_n = int(sum(int((ids == tid).sum().item()) for tid in audio_ids)) if audio_ids else 0
        vision_n = int(sum(int((ids == tid).sum().item()) for tid in vision_ids)) if vision_ids else 0

        return {
            "text_tokens": int(ids.shape[-1]),
            "audio_tokens": int(audio_n),
            "visual_tokens": int(vision_n),
            "total_tokens": int(total),
        }
