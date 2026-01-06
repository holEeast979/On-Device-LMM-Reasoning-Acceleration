from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.runner import BenchmarkRunner
from benchmark.unified_runner import UnifiedRunner
import profiling_utils as P


SPEC_NAME = "audio-padding"


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


def _select_samples(args: argparse.Namespace, runner: BenchmarkRunner) -> List[Dict[str, Any]]:
    if not os.path.exists(str(args.manifest)):
        raise SystemExit(f"manifest not found: {args.manifest}")

    all_rows = runner.load_manifest_csv(str(args.manifest), n_samples=0, seed=int(args.seed))

    usable: List[Dict[str, Any]] = []
    for r in all_rows:
        raw_v = r.get("video_path", None)
        raw_a = r.get("audio_path", None)
        v = UnifiedRunner.resolve_media_path(str(args.manifest), None if raw_v is None else str(raw_v))
        a = UnifiedRunner.resolve_media_path(str(args.manifest), None if raw_a is None else str(raw_a))

        has_v = bool(v) and os.path.exists(v)
        has_a = bool(a) and os.path.exists(a)
        if not has_v and not has_a:
            continue

        rr = dict(r)
        rr["video_path"] = v if has_v else ""
        rr["audio_path"] = a if has_a else ""
        rr["media_type"] = "video" if has_v else "audio"
        usable.append(rr)

    if not usable:
        raise SystemExit("no usable samples with existing video_path or audio_path")

    n = int(args.n_samples) if getattr(args, "n_samples", None) is not None else 1
    if n <= 0 or n >= len(usable):
        return usable

    df = pd.DataFrame(usable)
    sub = df.sample(n=n, random_state=int(args.seed))
    return [row._asdict() if hasattr(row, "_asdict") else row for row in sub.to_dict(orient="records")]


def _load_wav_mono_16k(path: str) -> np.ndarray:
    import librosa

    wav, _sr = librosa.load(path, sr=16000, mono=True)
    if wav is None:
        return np.zeros((0,), dtype=np.float32)
    if not isinstance(wav, np.ndarray):
        wav = np.asarray(wav)
    return wav.astype(np.float32)


def _build_audio_text_inputs(proc, model, *, question: str) -> Dict[str, Any]:
    msgs = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": str(question)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, return_tensors="pt", padding=True)
    return inputs.to(model.device)


def register_subcommand(subparsers, common_parser) -> None:
    p = subparsers.add_parser(SPEC_NAME, parents=[common_parser], help="Audio padding bottleneck verification")
    p.add_argument("--audio-seconds", type=str, default="2,5,10,20,30")
    p.add_argument("--audio-max-seconds", type=float, default=30.0)
    p.set_defaults(_spec_run=run)


def run(args: argparse.Namespace, runner: BenchmarkRunner) -> str:
    loaded = runner.load()
    model, proc, fe = loaded.model, loaded.processor, loaded.feature_extractor

    out_dir = os.path.join(str(args.out_dir), SPEC_NAME)
    os.makedirs(out_dir, exist_ok=True)

    audio_seconds = _parse_float_list(getattr(args, "audio_seconds", None))
    if not audio_seconds:
        raise SystemExit("--audio-seconds is empty")

    samples = _select_samples(args, runner)

    visual, audio, llm = runner.get_qwen25_modules(model)
    if audio is None:
        raise SystemExit("Qwen2.5-Omni audio_tower not found")

    visual_timer = P.ModuleCudaEventTimer()
    audio_timer = P.ModuleCudaEventTimer()
    if visual is not None:
        visual_timer.register(visual)
    audio_timer.register(audio)

    prefill_capture = P.LLMPrefillCudaEventCapture()
    if llm is not None:
        prefill_capture.register(llm)

    mem_monitor = None
    mem_markers = []
    llm_mem_handles = []
    if bool(getattr(args, "profile_mem", False)):
        mem_monitor = P.TorchCudaMemPeakMonitor(
            device=getattr(model, "device", None),
            interval_ms=float(getattr(args, "mem_interval_ms", 2.0)),
            track_rss=True,
        )
        mem_monitor.start()
        mem_monitor.mark("idle")
        if visual is not None:
            m = P.PhaseMarker(mem_monitor, "visual_encoder", fallback_phase="generate")
            m.register(visual)
            mem_markers.append(m)
        m = P.PhaseMarker(mem_monitor, "audio_encoder", fallback_phase="generate")
        m.register(audio)
        mem_markers.append(m)

        if llm is not None:
            def _llm_pre_hook(_module, _args, kwargs):
                try:
                    input_ids = kwargs.get("input_ids", None)
                    inputs_embeds = kwargs.get("inputs_embeds", None)
                    seq_len = 0
                    if input_ids is not None:
                        seq_len = int(getattr(input_ids, "shape", [0])[-1])
                    if inputs_embeds is not None:
                        seq_len = max(int(getattr(inputs_embeds, "shape", [0, 0])[1]), seq_len)
                    if seq_len > 1:
                        mem_monitor.mark("llm_prefill")
                    else:
                        mem_monitor.mark("llm_decode")
                except Exception:
                    pass
                return None

            def _llm_post_hook(_module, _args, _kwargs, _out):
                try:
                    mem_monitor.mark("generate")
                except Exception:
                    pass
                return None

            h1 = llm.register_forward_pre_hook(_llm_pre_hook, with_kwargs=True)
            h2 = llm.register_forward_hook(_llm_post_hook, with_kwargs=True)
            llm_mem_handles.extend([h1, h2])

    ur = UnifiedRunner(base=runner, spec_name=SPEC_NAME, out_dir=out_dir, args=args)

    cases: List[Dict[str, Any]] = []
    for s in samples:
        sample_id = str(s.get("sample_id", ""))
        media_type = str(s.get("media_type", "video"))
        video_path = str(s.get("video_path", ""))
        audio_path = str(s.get("audio_path", ""))
        question = str(args.question) if args.question else str(s.get("question", "Describe what you see and hear."))
        for sec in audio_seconds:
            for padding in ("max_length", "do_not_pad"):
                cases.append(
                    {
                        "sample_id": sample_id,
                        "media_type": media_type,
                        "video_path": video_path,
                        "audio_path": audio_path,
                        "video_nframes": int(args.video_nframes) if args.video_nframes else None,
                        "question": question,
                        "target_duration_s": float(sec),
                        "padding": str(padding),
                    }
                )

    def run_once(case: Dict[str, Any]) -> Dict[str, Any]:
        if mem_monitor is not None:
            mem_monitor.reset()
            mem_monitor.mark("preprocess")

        media_type = str(case.get("media_type", "video"))
        video_path = str(case.get("video_path", ""))
        audio_path = str(case.get("audio_path", ""))
        question = str(case.get("question", ""))
        video_nframes = case.get("video_nframes", None)
        target_duration_s = float(case.get("target_duration_s", 0.0))
        padding = str(case.get("padding", "max_length"))

        extract_ms = 0.0
        base_inputs: Dict[str, Any]
        base_pack_ms = 0.0
        original_audio: np.ndarray

        if media_type == "video":
            try:
                av = runner.extract_av_from_video(video_path, question, int(video_nframes) if video_nframes is not None else None)
            except Exception as e:
                return {"error": f"extract_failed: {type(e).__name__}: {e}"}

            extract_ms = float(av.extract_ms)
            if not av.audios:
                return {"error": "no_audio_extracted", "extract_ms": float(extract_ms)}
            original_audio = av.audios[0]

            base = runner.prepare_base_inputs(
                model,
                proc,
                videos=av.videos,
                question=question,
                video_path=video_path,
                video_nframes=int(video_nframes) if video_nframes is not None else None,
            )
            base_inputs = base.inputs
            base_pack_ms = float(base.pack_ms)
        elif media_type == "audio":
            if not audio_path or not os.path.exists(audio_path):
                return {"error": "audio_not_found"}
            original_audio = _load_wav_mono_16k(audio_path)
            t0 = time.perf_counter()
            base_inputs = _build_audio_text_inputs(proc, model, question=question)
            base_pack_ms = float((time.perf_counter() - t0) * 1000)
        else:
            return {"error": f"unknown_media_type: {media_type}"}

        original_duration_s = float(len(original_audio) / 16000.0)
        audio_trim = runner.truncate_audio(original_audio, target_duration_s)
        actual_duration_s = float(len(audio_trim) / 16000.0)

        af, mel_frames, audio_feature_ms = runner.build_audio_features(
            fe,
            audio_trim,
            padding=padding,
            audio_max_seconds=float(getattr(args, "audio_max_seconds", 30.0)),
        )
        full_inputs = runner.attach_audio_features(
            base_inputs=base_inputs,
            af=af,
            mel_frames=mel_frames,
            model=model,
            dtype=str(args.dtype),
        )

        token_stats = runner.get_token_stats(proc, full_inputs.get("input_ids"))

        visual_timer.clear()
        audio_timer.clear()
        prefill_capture.clear()

        if mem_monitor is not None:
            mem_monitor.mark("generate")

        ttft_ms = runner.run_generate_1token_ms(model, {k: v for k, v in full_inputs.items()})

        visual_encoder_ms = float(sum(visual_timer.times)) if visual is not None else 0.0
        audio_encoder_ms = float(sum(audio_timer.times))
        llm_prefill_ms = float(prefill_capture.prefill_forward_ms) if llm is not None else None
        llm_prefill_seq_len = int(prefill_capture.prefill_seq_len) if llm is not None else None

        audio_tower_in_frames = None
        if audio_timer.last_input_shape is not None and len(audio_timer.last_input_shape) >= 2:
            try:
                audio_tower_in_frames = int(audio_timer.last_input_shape[-1])
            except Exception:
                audio_tower_in_frames = None

        mem_row: Dict[str, Any] = {}
        if mem_monitor is not None:
            mem_monitor.mark("done")
            mem_row.update(mem_monitor.summary_mb(prefix="mem__"))
            try:
                p = mem_monitor.phase_peaks
                vis_a = int(p.get("visual_encoder", {}).get("allocated", 0))
                aud_a = int(p.get("audio_encoder", {}).get("allocated", 0))
                pre_a = int(p.get("llm_prefill", {}).get("allocated", 0))
                preproc_a = int(p.get("preprocess", {}).get("allocated", 0))
                vis_r = int(p.get("visual_encoder", {}).get("reserved", 0))
                aud_r = int(p.get("audio_encoder", {}).get("reserved", 0))
                pre_r = int(p.get("llm_prefill", {}).get("reserved", 0))
                preproc_r = int(p.get("preprocess", {}).get("reserved", 0))
                mem_row["mem__encoder_peak_allocated_mb"] = float(max(vis_a, aud_a)) / (1024.0 * 1024.0)
                mem_row["mem__prefill_peak_allocated_mb"] = float(pre_a) / (1024.0 * 1024.0) if pre_a else None
                mem_row["mem__preprocess_peak_allocated_mb"] = float(preproc_a) / (1024.0 * 1024.0) if preproc_a else None
                mem_row["mem__encoder_peak_reserved_mb"] = float(max(vis_r, aud_r)) / (1024.0 * 1024.0)
                mem_row["mem__prefill_peak_reserved_mb"] = float(pre_r) / (1024.0 * 1024.0) if pre_r else None
                mem_row["mem__preprocess_peak_reserved_mb"] = float(preproc_r) / (1024.0 * 1024.0) if preproc_r else None
            except Exception:
                pass

        return {
            "extract_ms": float(extract_ms),
            "pack_ms": float(base_pack_ms),
            "audio_feature_ms": float(audio_feature_ms),
            "preprocess_ms": float(extract_ms + base_pack_ms + audio_feature_ms),
            "original_duration_s": float(original_duration_s),
            "actual_duration_s": float(actual_duration_s),
            "mel_frames": int(mel_frames),
            "audio_tower_in_frames": audio_tower_in_frames,
            "audio_tower_input_shape": str(audio_timer.last_input_shape) if audio_timer.last_input_shape is not None else None,
            "visual_encoder_ms": float(visual_encoder_ms),
            "audio_encoder_ms": float(audio_encoder_ms),
            "llm_prefill_ms": llm_prefill_ms,
            "llm_prefill_seq_len": llm_prefill_seq_len,
            "ttft_ms": float(ttft_ms),
            **token_stats,
            **mem_row,
        }

    df = ur.run(
        cases=cases,
        repeats=int(max(1, args.repeats)),
        warmup=int(max(0, args.warmup)),
        run_once=run_once,
        clear_cache=True,
    )

    df.to_csv(os.path.join(out_dir, "audio_padding_results.csv"), index=False)

    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    if len(df_ok):
        agg_spec = {
            "mel_frames_mean": ("mel_frames", "mean"),
            "audio_encoder_ms_mean": ("audio_encoder_ms", "mean"),
            "ttft_ms_mean": ("ttft_ms", "mean"),
            "n": ("ttft_ms", "count"),
        }
        for k in (
            "mem__peak_allocated_mb",
            "mem__peak_reserved_mb",
            "mem__encoder_peak_allocated_mb",
            "mem__encoder_peak_reserved_mb",
            "mem__prefill_peak_allocated_mb",
            "mem__prefill_peak_reserved_mb",
            "mem__preprocess_peak_allocated_mb",
            "mem__preprocess_peak_reserved_mb",
        ):
            if k in df_ok.columns:
                agg_spec[f"{k}_mean"] = (k, "mean")

        agg = df_ok.groupby(["padding", "target_duration_s"], dropna=False).agg(**agg_spec).reset_index()
        agg.to_csv(os.path.join(out_dir, "audio_padding_summary.csv"), index=False)

        plot_specs = [
            ("mel_frames_mean", "mel_frames", "audio_padding_mel_frames.png"),
            ("audio_encoder_ms_mean", "audio_encoder_ms (ms)", "audio_padding_audio_encoder_ms.png"),
            ("ttft_ms_mean", "ttft_ms (ms)", "audio_padding_ttft_ms.png"),
        ]
        if "mem__peak_allocated_mb_mean" in agg.columns:
            plot_specs.append(("mem__peak_allocated_mb_mean", "peak_allocated_mb", "audio_padding_peak_allocated_mb.png"))
        if "mem__peak_reserved_mb_mean" in agg.columns:
            plot_specs.append(("mem__peak_reserved_mb_mean", "peak_reserved_mb", "audio_padding_peak_reserved_mb.png"))

        for metric, ylab, fname in plot_specs:
            plt.figure(figsize=(7, 4))
            for padding in ["max_length", "do_not_pad"]:
                sub = agg[agg["padding"] == padding].sort_values("target_duration_s")
                if len(sub) == 0:
                    continue
                plt.plot(sub["target_duration_s"], sub[metric], marker="o", label=padding)
            plt.xlabel("target_duration_s")
            plt.ylabel(ylab)
            plt.title(str(metric))
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
            plt.close()

        if "mem__peak_allocated_mb_mean" in agg.columns:
            try:
                rows = []
                for padding in ["max_length", "do_not_pad"]:
                    sub = agg[agg["padding"] == padding].sort_values("target_duration_s")
                    if len(sub) < 2:
                        rows.append({"padding": padding, "peak_allocated_mb_slope_per_s": None})
                        continue
                    x0 = float(sub["target_duration_s"].iloc[0])
                    x1 = float(sub["target_duration_s"].iloc[-1])
                    y0 = float(sub["mem__peak_allocated_mb_mean"].iloc[0])
                    y1 = float(sub["mem__peak_allocated_mb_mean"].iloc[-1])
                    slope = (y1 - y0) / max(1e-6, (x1 - x0))
                    rows.append({"padding": padding, "peak_allocated_mb_slope_per_s": float(slope)})
                pd.DataFrame(rows).to_csv(os.path.join(out_dir, "audio_padding_mem_slope.csv"), index=False)
            except Exception:
                pass

    try:
        visual_timer.remove()
    except Exception:
        pass
    try:
        audio_timer.remove()
    except Exception:
        pass
    try:
        prefill_capture.remove()
    except Exception:
        pass
    try:
        for m in mem_markers:
            m.remove()
    except Exception:
        pass
    try:
        for h in llm_mem_handles:
            h.remove()
    except Exception:
        pass
    try:
        if mem_monitor is not None:
            mem_monitor.stop()
    except Exception:
        pass

    return out_dir
