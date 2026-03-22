#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import common as C
import profiling_utils as P



def _parse_float_list(s: Optional[str]) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            continue
    return out


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


def _truncate_audio_no_pad(audio: np.ndarray, target_seconds: float, sample_rate: int = 16000) -> np.ndarray:
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


class _ForwardCounter:
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


def _prepare_base_inputs(model, proc, videos: Any, question: str, video_path: str, video_nframes: Optional[int]) -> Dict[str, Any]:
    video_ele: Dict[str, Any] = {"type": "video", "video": video_path}
    if video_nframes is not None and int(video_nframes) > 0:
        video_ele["nframes"] = int(video_nframes)

    conversation = [{"role": "user", "content": [video_ele, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, videos=videos, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    return inputs


def _extract_av_from_video(video_path: str, question: str, video_nframes: Optional[int]):
    from qwen_omni_utils import process_mm_info

    video_ele: Dict[str, Any] = {"type": "video", "video": video_path}
    if video_nframes is not None and int(video_nframes) > 0:
        video_ele["nframes"] = int(video_nframes)
    conversation = [{"role": "user", "content": [video_ele, {"type": "text", "text": question}]}]
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    return audios, videos


def _run_generate_1token(model, inputs: Dict[str, Any]) -> float:
    _sync_model_devices(model)
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_audio=False,
        )
    _sync_model_devices(model)
    return (time.perf_counter() - t0) * 1000


def run_audio_padding_experiment(
    *,
    model,
    proc,
    fe,
    samples: List[Dict[str, Any]],
    out_dir: str,
    audio_seconds: List[float],
    audio_max_seconds: float,
    repeats: int,
    warmup: int,
    video_nframes: Optional[int],
    question_override: Optional[str],
    dtype: str,
):
    os.makedirs(out_dir, exist_ok=True)
    model_dtype = C.get_dtype(dtype)

    audio_timer = ModuleCudaEventTimer()
    audio_timer.register(model.thinker.audio_tower)

    rows: List[Dict[str, Any]] = []
    for si, s in enumerate(samples):
        video_path = str(s.get("video_path", ""))
        sample_id = str(s.get("sample_id", f"sample_{si}"))
        question = str(question_override) if question_override else str(s.get("question", "Describe what you see and hear."))

        t_ext0 = time.perf_counter()
        try:
            audios, videos = _extract_av_from_video(video_path, question, video_nframes)
        except Exception as e:
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_path": video_path,
                    "error": f"extract_failed: {type(e).__name__}: {e}",
                }
            )
            continue
        t_ext_ms = (time.perf_counter() - t_ext0) * 1000
        if not audios:
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_path": video_path,
                    "error": "no_audio_extracted",
                }
            )
            continue

        original_audio = audios[0]
        original_duration_s = float(len(original_audio) / 16000)

        base_inputs = _prepare_base_inputs(model, proc, videos, question, video_path, video_nframes)

        for sec in audio_seconds:
            audio_trim = _truncate_audio_no_pad(original_audio, sec)
            actual_duration_s = float(len(audio_trim) / 16000)

            for padding in ("max_length", "do_not_pad"):
                for r in range(int(max(1, repeats))):
                    do_record = r >= int(max(0, warmup))

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    t_fft0 = time.perf_counter()
                    if padding == "max_length":
                        max_audio_samples = int(float(audio_max_seconds) * 16000)
                        af = fe(
                            audio_trim,
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding="max_length",
                            max_length=max_audio_samples,
                            truncation=True,
                        )
                    else:
                        af = fe(
                            audio_trim,
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding="do_not_pad",
                            truncation=False,
                        )

                    mel_frames = int(af["input_features"].shape[2])
                    fft_mel_ms = (time.perf_counter() - t_fft0) * 1000

                    inputs = {k: v for k, v in base_inputs.items()}
                    inputs["input_features"] = af["input_features"].to(model.device, dtype=model_dtype)
                    inputs["feature_attention_mask"] = torch.ones(
                        (1, mel_frames),
                        device=model.device,
                        dtype=torch.long,
                    )

                    audio_timer.clear()
                    ttft_ms = _run_generate_1token(model, inputs)
                    audio_encoder_ms = float(sum(audio_timer.times))

                    audio_tower_in_frames = None
                    if audio_timer.last_input_shape is not None and len(audio_timer.last_input_shape) >= 2:
                        audio_tower_in_frames = int(audio_timer.last_input_shape[-1])

                    if not do_record:
                        continue

                    rows.append(
                        {
                            "sample_id": sample_id,
                            "video_path": video_path,
                            "video_nframes": int(video_nframes) if video_nframes is not None else None,
                            "extract_ms": float(t_ext_ms),
                            "original_duration_s": float(original_duration_s),
                            "target_duration_s": float(sec),
                            "actual_duration_s": float(actual_duration_s),
                            "padding": padding,
                            "mel_frames": int(mel_frames),
                            "audio_tower_in_frames": audio_tower_in_frames,
                            "audio_tower_input_shape": str(audio_timer.last_input_shape) if audio_timer.last_input_shape is not None else None,
                            "fft_mel_ms": float(fft_mel_ms),
                            "audio_encoder_ms": float(audio_encoder_ms),
                            "ttft_ms": float(ttft_ms),
                            "repeat": int(r),
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "audio_padding_results.csv"), index=False)

    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    if len(df_ok):
        agg = (
            df_ok.groupby(["padding", "target_duration_s"], dropna=False)
            .agg(
                mel_frames_mean=("mel_frames", "mean"),
                audio_encoder_ms_mean=("audio_encoder_ms", "mean"),
                ttft_ms_mean=("ttft_ms", "mean"),
                n=("ttft_ms", "count"),
            )
            .reset_index()
        )
        agg.to_csv(os.path.join(out_dir, "audio_padding_summary.csv"), index=False)

        for metric, ylab, fname in (
            ("mel_frames_mean", "mel_frames", "audio_padding_mel_frames.png"),
            ("audio_encoder_ms_mean", "audio_encoder_ms (ms)", "audio_padding_audio_encoder_ms.png"),
            ("ttft_ms_mean", "ttft_ms (ms)", "audio_padding_ttft_ms.png"),
        ):
            plt.figure(figsize=(7, 4))
            for padding in ["max_length", "do_not_pad"]:
                sub = agg[agg["padding"] == padding].sort_values("target_duration_s")
                if len(sub) == 0:
                    continue
                plt.plot(
                    sub["target_duration_s"],
                    sub[metric],
                    marker="o",
                    label=padding,
                )
            plt.xlabel("target_duration_s")
            plt.ylabel(ylab)
            plt.title(ylab)
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
            plt.close()

    audio_timer.remove()
    return df


def run_multiturn_experiment(
    *,
    model,
    proc,
    fe,
    sample: Dict[str, Any],
    out_dir: str,
    audio_max_seconds: float,
    repeats: int,
    warmup: int,
    video_nframes: Optional[int],
    question_override: Optional[str],
    dtype: str,
):
    os.makedirs(out_dir, exist_ok=True)
    model_dtype = C.get_dtype(dtype)

    video_path = str(sample.get("video_path", ""))
    sample_id = str(sample.get("sample_id", "sample"))
    question = str(question_override) if question_override else str(sample.get("question", "Describe what you see and hear."))

    visual_timer = P.ModuleCudaEventTimer()
    visual_timer.register(model.thinker.visual)
    audio_timer = P.ModuleCudaEventTimer()
    audio_timer.register(model.thinker.audio_tower)

    prefill_capture = P.LLMPrefillCudaEventCapture()
    prefill_capture.register(model.thinker.model)

    visual_counter = _ForwardCounter()
    visual_counter.register(model.thinker.visual, with_kwargs=False)
    audio_counter = _ForwardCounter()
    audio_counter.register(model.thinker.audio_tower, with_kwargs=False)
    llm_counter = _ForwardCounter()
    llm_counter.register(model.thinker.model, with_kwargs=True)

    rows: List[Dict[str, Any]] = []
    for r in range(int(max(1, repeats))):
        do_record = r >= int(max(0, warmup))

        t_ext0 = time.perf_counter()
        try:
            audios, videos = _extract_av_from_video(video_path, question, video_nframes)
        except Exception as e:
            if do_record:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "turn": 1,
                        "repeat": int(r),
                        "error": f"extract_failed: {type(e).__name__}: {e}",
                    }
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "turn": 2,
                        "repeat": int(r),
                        "error": f"extract_failed: {type(e).__name__}: {e}",
                    }
                )
            continue
        extract_ms = (time.perf_counter() - t_ext0) * 1000

        base_inputs = _prepare_base_inputs(model, proc, videos, question, video_path, video_nframes)

        mel_frames = None
        audio_feature_ms = None
        full_inputs = {k: v for k, v in base_inputs.items()}
        if audios:
            t_af0 = time.perf_counter()
            max_audio_samples = int(float(audio_max_seconds) * 16000)
            af = fe(
                audios[0],
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=max_audio_samples,
                truncation=True,
            )
            mel_frames = int(af["input_features"].shape[2])
            audio_feature_ms = (time.perf_counter() - t_af0) * 1000
            full_inputs["input_features"] = af["input_features"].to(model.device, dtype=model_dtype)
            full_inputs["feature_attention_mask"] = torch.ones(
                (1, mel_frames),
                device=model.device,
                dtype=torch.long,
            )

        for turn in (1, 2):
            prefill_capture.clear()
            visual_timer.clear()
            audio_timer.clear()
            visual_counter.clear()
            audio_counter.clear()
            llm_counter.clear()

            ttft_ms = _run_generate_1token(model, {k: v for k, v in full_inputs.items()})
            visual_ms = float(sum(visual_timer.times))
            audio_ms = float(sum(audio_timer.times))
            prefill_ms = float(prefill_capture.prefill_forward_ms)
            prefill_seq_len = int(prefill_capture.prefill_seq_len)

            if not do_record:
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_path": video_path,
                    "video_nframes": int(video_nframes) if video_nframes is not None else None,
                    "turn": int(turn),
                    "repeat": int(r),
                    "extract_ms": float(extract_ms),
                    "audio_feature_ms": float(audio_feature_ms) if audio_feature_ms is not None else None,
                    "mel_frames": int(mel_frames) if mel_frames is not None else None,
                    "visual_encoder_ms": float(visual_ms),
                    "audio_encoder_ms": float(audio_ms),
                    "llm_prefill_ms": float(prefill_ms),
                    "llm_prefill_seq_len": int(prefill_seq_len),
                    "ttft_ms": float(ttft_ms),
                    "visual_forward_calls": int(visual_counter.count),
                    "audio_forward_calls": int(audio_counter.count),
                    "llm_forward_calls": int(llm_counter.count),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "multiturn_results.csv"), index=False)

    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    if len(df_ok):
        agg = (
            df_ok.groupby(["turn"], dropna=False)
            .agg(
                visual_encoder_ms_mean=("visual_encoder_ms", "mean"),
                audio_encoder_ms_mean=("audio_encoder_ms", "mean"),
                llm_prefill_ms_mean=("llm_prefill_ms", "mean"),
                ttft_ms_mean=("ttft_ms", "mean"),
                visual_forward_calls_mean=("visual_forward_calls", "mean"),
                audio_forward_calls_mean=("audio_forward_calls", "mean"),
                llm_forward_calls_mean=("llm_forward_calls", "mean"),
                n=("ttft_ms", "count"),
            )
            .reset_index()
        )
        agg.to_csv(os.path.join(out_dir, "multiturn_summary.csv"), index=False)

        metrics = [
            ("visual_encoder_ms_mean", "visual_encoder_ms"),
            ("audio_encoder_ms_mean", "audio_encoder_ms"),
            ("llm_prefill_ms_mean", "llm_prefill_ms"),
            ("ttft_ms_mean", "ttft_ms"),
        ]
        plt.figure(figsize=(9, 4))
        x = np.arange(len(metrics))
        width = 0.35
        v1 = []
        v2 = []
        for k, _ in metrics:
            t1 = agg[agg["turn"] == 1][k]
            t2 = agg[agg["turn"] == 2][k]
            v1.append(float(t1.iloc[0]) if len(t1) else 0.0)
            v2.append(float(t2.iloc[0]) if len(t2) else 0.0)
        plt.bar(x - width / 2, v1, width, label="turn1")
        plt.bar(x + width / 2, v2, width, label="turn2")
        plt.xticks(x, [n for _, n in metrics], rotation=20)
        plt.ylabel("ms")
        plt.title("multi-turn: encoder/prefill/ttft")
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "multiturn_bar.png"), dpi=150, bbox_inches="tight")
        plt.close()

        ratio = {}
        for k, name in metrics:
            t1 = agg[agg["turn"] == 1][k]
            t2 = agg[agg["turn"] == 2][k]
            a = float(t1.iloc[0]) if len(t1) else 0.0
            b = float(t2.iloc[0]) if len(t2) else 0.0
            ratio[name] = None if a <= 0 else float(b / a)
        with open(os.path.join(out_dir, "multiturn_ratio.json"), "w") as f:
            json.dump(ratio, f, indent=2, ensure_ascii=False)

    visual_timer.remove()
    audio_timer.remove()
    prefill_capture.remove()
    visual_counter.remove()
    audio_counter.remove()
    llm_counter.remove()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="/root/autodl-tmp/Qwen2.5-Omni-7B")
    ap.add_argument("--manifest", type=str, default="/root/autodl-tmp/data/MSRVTT_subset/manifest.csv")
    ap.add_argument("--out-dir", type=str, default="/root/autodl-tmp/results/exp10")
    ap.add_argument("--dtype", type=str, default="bf16")

    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--question", type=str, default=None)

    ap.add_argument("--run-audio", action="store_true")
    ap.add_argument("--audio-seconds", type=str, default="2,5,10,20,30")
    ap.add_argument("--audio-max-seconds", type=float, default=30.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)

    ap.add_argument("--run-multiturn", action="store_true")
    ap.add_argument("--multiturn-index", type=int, default=0)
    ap.add_argument("--video-nframes", type=int, default=8)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.run_audio and not args.run_multiturn:
        raise SystemExit("请至少指定 --run-audio 或 --run-multiturn")

    audio_seconds = _parse_float_list(args.audio_seconds)
    if args.run_audio and not audio_seconds:
        raise SystemExit("--audio-seconds 为空")

    from transformers import WhisperFeatureExtractor

    # Import classes for type hinting if needed or just rely on profiling_utils
    LLMPrefillCudaEventCapture = P.LLMPrefillCudaEventCapture
    ModuleCudaEventTimer = P.ModuleCudaEventTimer

    print(f"model_dir: {args.model_dir}")
    print(f"manifest: {args.manifest}")
    print(f"out_dir: {args.out_dir}")
    print(f"dtype: {args.dtype}")

    model, proc = C.load_qwen25_omni(args.model_dir, args.dtype)
    fe = WhisperFeatureExtractor.from_pretrained(args.model_dir)

    if not os.path.exists(args.manifest):
        raise SystemExit(f"manifest 不存在: {args.manifest}")

    samples = C.load_dataset(args.manifest, n_samples=int(max(1, args.n_samples)))
    if not samples:
        raise SystemExit("manifest 没有可用样本")

    if args.run_audio:
        sub = os.path.join(args.out_dir, "audio_padding")
        run_audio_padding_experiment(
            model=model,
            proc=proc,
            fe=fe,
            samples=samples,
            out_dir=sub,
            audio_seconds=audio_seconds,
            audio_max_seconds=float(args.audio_max_seconds),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            video_nframes=int(args.video_nframes) if args.video_nframes else None,
            question_override=args.question,
            dtype=args.dtype,
        )
        print(f"audio_padding done: {sub}")

    if args.run_multiturn:
        idx = int(args.multiturn_index)
        if idx < 0 or idx >= len(samples):
            raise SystemExit(f"multiturn-index 越界: {idx} / {len(samples)}")
        sub = os.path.join(args.out_dir, "multiturn")
        run_multiturn_experiment(
            model=model,
            proc=proc,
            fe=fe,
            sample=samples[idx],
            out_dir=sub,
            audio_max_seconds=float(args.audio_max_seconds),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            video_nframes=int(args.video_nframes) if args.video_nframes else None,
            question_override=args.question,
            dtype=args.dtype,
        )
        print(f"multiturn done: {sub}")


if __name__ == "__main__":
    main()
