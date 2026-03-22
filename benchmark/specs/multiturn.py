from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.runner import BenchmarkRunner, ForwardCounter
from benchmark.unified_runner import UnifiedRunner
from utils import profiling_utils as P


SPEC_NAME = "multiturn"


def register_subcommand(subparsers, common_parser) -> None:
    p = subparsers.add_parser(SPEC_NAME, parents=[common_parser], help="Multi-turn encode cache missing verification")
    p.add_argument("--sample-index", type=int, default=None, help="Index into usable samples after filtering (optional)")
    p.add_argument("--audio-max-seconds", type=float, default=30.0)
    p.add_argument("--turns", type=int, default=2, help="Number of turns to repeat the same request")
    p.set_defaults(_spec_run=run)


def _load_wav_mono_16k(path: str) -> np.ndarray:
    import librosa

    wav, sr = librosa.load(path, sr=16000, mono=True)
    if wav is None:
        return np.zeros((0,), dtype=np.float32)
    if not isinstance(wav, np.ndarray):
        wav = np.asarray(wav)
    return wav.astype(np.float32)


def _select_usable_rows(args: argparse.Namespace, runner: BenchmarkRunner) -> List[Dict[str, Any]]:
    if not os.path.exists(str(args.manifest)):
        raise SystemExit(f"manifest not found: {args.manifest}")

    all_rows = runner.load_manifest_csv(str(args.manifest), n_samples=0, seed=int(args.seed))

    usable: List[Dict[str, Any]] = []
    seen = set()
    for r in all_rows:
        raw_v = r.get("video_path", None)
        raw_a = r.get("audio_path", None)

        v = UnifiedRunner.resolve_media_path(str(args.manifest), None if raw_v is None else str(raw_v))
        a = UnifiedRunner.resolve_media_path(str(args.manifest), None if raw_a is None else str(raw_a))

        has_v = bool(v) and os.path.exists(v)
        has_a = bool(a) and os.path.exists(a)
        if not has_v and not has_a:
            continue

        key = ("video", v) if has_v else ("audio", a)
        if key in seen:
            continue
        seen.add(key)

        rr = dict(r)
        rr["video_path"] = v if has_v else ""
        rr["audio_path"] = a if has_a else ""
        rr["media_type"] = "video" if has_v else "audio"
        usable.append(rr)

    if not usable:
        raise SystemExit("no usable samples with existing video_path or audio_path")
    return usable


def _build_text_inputs(proc, model, *, question: str, media_type: str) -> Tuple[Dict[str, Any], float]:
    if str(media_type) == "audio":
        content = [{"type": "audio"}, {"type": "text", "text": str(question)}]
        msgs = [{"role": "user", "content": content}]
        t0 = time.perf_counter()
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=text, return_tensors="pt", padding=True).to(model.device)
        return inputs, float((time.perf_counter() - t0) * 1000)

    raise ValueError(f"unknown media_type: {media_type}")


def run(args: argparse.Namespace, runner: BenchmarkRunner) -> str:
    loaded = runner.load()
    model, proc, fe = loaded.model, loaded.processor, loaded.feature_extractor

    out_dir = os.path.join(str(args.out_dir), SPEC_NAME)
    os.makedirs(out_dir, exist_ok=True)

    usable = _select_usable_rows(args, runner)

    selected: List[Dict[str, Any]]
    idx = getattr(args, "sample_index", None)
    if idx is not None:
        idx = int(idx)
        if idx < 0 or idx >= len(usable):
            raise SystemExit(f"sample-index out of range: {idx} / {len(usable)}")
        selected = [usable[idx]]
    else:
        n = int(getattr(args, "n_samples", 1))
        if n <= 0 or n >= len(usable):
            selected = usable
        else:
            df = pd.DataFrame(usable)
            selected = df.sample(n=n, random_state=int(args.seed)).to_dict(orient="records")

    visual, audio, llm = runner.get_qwen25_modules(model)

    visual_timer = P.ModuleCudaEventTimer()
    audio_timer = P.ModuleCudaEventTimer()
    if visual is not None:
        visual_timer.register(visual)
    if audio is not None:
        audio_timer.register(audio)

    prefill_capture = P.LLMPrefillCudaEventCapture()
    if llm is not None:
        prefill_capture.register(llm)

    visual_counter = ForwardCounter() if visual is not None else None
    audio_counter = ForwardCounter() if audio is not None else None
    llm_counter = ForwardCounter() if llm is not None else None

    if visual_counter is not None and visual is not None:
        visual_counter.register(visual, with_kwargs=False)
    if audio_counter is not None and audio is not None:
        audio_counter.register(audio, with_kwargs=False)
    if llm_counter is not None and llm is not None:
        llm_counter.register(llm, with_kwargs=True)

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
        if audio is not None:
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
    for i, sample in enumerate(selected):
        media_type = str(sample.get("media_type", "video"))
        sample_id = str(sample.get("sample_id", f"sample_{i}"))
        video_path = str(sample.get("video_path", ""))
        audio_path = str(sample.get("audio_path", ""))
        question = str(args.question) if args.question else str(sample.get("question", "Describe what you see and hear."))
        cases.append(
            {
                "sample_id": sample_id,
                "media_type": media_type,
                "video_path": video_path,
                "audio_path": audio_path,
                "video_nframes": int(args.video_nframes) if args.video_nframes else None,
                "question": question,
            }
        )

    def run_once(case: Dict[str, Any]) -> List[Dict[str, Any]]:
        turns = int(max(1, getattr(args, "turns", 2)))
        media_type_ = str(case.get("media_type", "video"))
        question_ = str(case.get("question", ""))
        video_nframes = case.get("video_nframes", None)
        video_path_ = str(case.get("video_path", ""))
        audio_path_ = str(case.get("audio_path", ""))

        extract_ms = 0.0
        pack_ms = 0.0
        audio_feature_ms = 0.0
        mel_frames = None
        preprocess_mem: Dict[str, Any] = {}

        if mem_monitor is not None:
            mem_monitor.reset()
            mem_monitor.mark("preprocess")

        if media_type_ == "video":
            t_ext0 = time.perf_counter()
            try:
                av = runner.extract_av_from_video(video_path_, question_, int(video_nframes) if video_nframes is not None else None)
            except Exception as e:
                return [{"turn": int(t), "error": f"extract_failed: {type(e).__name__}: {e}"} for t in range(1, turns + 1)]
            extract_ms = float(av.extract_ms)

            base = runner.prepare_base_inputs(
                model,
                proc,
                videos=av.videos,
                question=question_,
                video_path=video_path_,
                video_nframes=int(video_nframes) if video_nframes is not None else None,
            )
            pack_ms = float(base.pack_ms)

            full_inputs: Dict[str, Any] = {k: v for k, v in base.inputs.items()}
            if av.audios:
                af, mel_frames_, audio_feature_ms_ = runner.build_audio_features(
                    fe,
                    av.audios[0],
                    padding="max_length",
                    audio_max_seconds=float(getattr(args, "audio_max_seconds", 30.0)),
                )
                mel_frames = int(mel_frames_)
                audio_feature_ms = float(audio_feature_ms_)
                full_inputs = runner.attach_audio_features(
                    base_inputs=full_inputs,
                    af=af,
                    mel_frames=int(mel_frames_),
                    model=model,
                    dtype=str(args.dtype),
                )
        elif media_type_ == "audio":
            if not audio_path_ or not os.path.exists(audio_path_):
                return [{"turn": int(t), "error": "audio_not_found"} for t in range(1, turns + 1)]

            wav = _load_wav_mono_16k(audio_path_)
            base_inputs, pack_ms = _build_text_inputs(proc, model, question=question_, media_type="audio")
            full_inputs = {k: v for k, v in base_inputs.items()}

            af, mel_frames_, audio_feature_ms_ = runner.build_audio_features(
                fe,
                wav,
                padding="max_length",
                audio_max_seconds=float(getattr(args, "audio_max_seconds", 30.0)),
            )
            mel_frames = int(mel_frames_)
            audio_feature_ms = float(audio_feature_ms_)
            full_inputs = runner.attach_audio_features(
                base_inputs=full_inputs,
                af=af,
                mel_frames=int(mel_frames_),
                model=model,
                dtype=str(args.dtype),
            )
        else:
            return [{"turn": int(t), "error": f"unknown_media_type: {media_type_}"} for t in range(1, turns + 1)]

        token_stats = runner.get_token_stats(proc, full_inputs.get("input_ids"))

        if mem_monitor is not None:
            mem_monitor.mark("preprocess_done")
            preprocess_mem = mem_monitor.summary_mb(prefix="mem_preprocess__")

        out_rows: List[Dict[str, Any]] = []
        for turn in range(1, turns + 1):
            visual_timer.clear()
            audio_timer.clear()
            prefill_capture.clear()
            if visual_counter is not None:
                visual_counter.clear()
            if audio_counter is not None:
                audio_counter.clear()
            if llm_counter is not None:
                llm_counter.clear()

            mem_row: Dict[str, Any] = {}
            if mem_monitor is not None:
                mem_monitor.reset()
                mem_monitor.mark("generate")

            ttft_ms = runner.run_generate_1token_ms(model, {k: v for k, v in full_inputs.items()})

            visual_encoder_ms = float(sum(visual_timer.times)) if visual is not None else 0.0
            audio_encoder_ms = float(sum(audio_timer.times)) if audio is not None else 0.0
            llm_prefill_ms = float(prefill_capture.prefill_forward_ms) if llm is not None else None
            llm_prefill_seq_len = int(prefill_capture.prefill_seq_len) if llm is not None else None

            if mem_monitor is not None:
                mem_monitor.mark("done")
                mem_row.update(mem_monitor.summary_mb(prefix="mem__"))
                try:
                    p = mem_monitor.phase_peaks
                    vis_a = int(p.get("visual_encoder", {}).get("allocated", 0))
                    aud_a = int(p.get("audio_encoder", {}).get("allocated", 0))
                    pre_a = int(p.get("llm_prefill", {}).get("allocated", 0))
                    vis_r = int(p.get("visual_encoder", {}).get("reserved", 0))
                    aud_r = int(p.get("audio_encoder", {}).get("reserved", 0))
                    pre_r = int(p.get("llm_prefill", {}).get("reserved", 0))
                    mem_row["mem__encoder_peak_allocated_mb"] = float(max(vis_a, aud_a)) / (1024.0 * 1024.0)
                    mem_row["mem__prefill_peak_allocated_mb"] = float(pre_a) / (1024.0 * 1024.0) if pre_a else None
                    mem_row["mem__encoder_peak_reserved_mb"] = float(max(vis_r, aud_r)) / (1024.0 * 1024.0)
                    mem_row["mem__prefill_peak_reserved_mb"] = float(pre_r) / (1024.0 * 1024.0) if pre_r else None
                except Exception:
                    pass

            row: Dict[str, Any] = {
                "turn": int(turn),
                "extract_ms": float(extract_ms) if extract_ms is not None else None,
                "pack_ms": float(pack_ms) if pack_ms is not None else None,
                "audio_feature_ms": float(audio_feature_ms) if audio_feature_ms is not None else None,
                "preprocess_ms": float(extract_ms + pack_ms + audio_feature_ms),
                "mel_frames": int(mel_frames) if mel_frames is not None else None,
                "visual_encoder_ms": float(visual_encoder_ms),
                "audio_encoder_ms": float(audio_encoder_ms),
                "llm_prefill_ms": llm_prefill_ms,
                "llm_prefill_seq_len": llm_prefill_seq_len,
                "ttft_ms": float(ttft_ms),
                "visual_forward_calls": int(visual_counter.count) if visual_counter is not None else None,
                "audio_forward_calls": int(audio_counter.count) if audio_counter is not None else None,
                "llm_forward_calls": int(llm_counter.count) if llm_counter is not None else None,
                **token_stats,
                **preprocess_mem,
                **mem_row,
            }
            out_rows.append(row)

        return out_rows

    df = ur.run(
        cases=cases,
        repeats=int(max(1, args.repeats)),
        warmup=int(max(0, args.warmup)),
        run_once=run_once,
        clear_cache=True,
    )

    df.to_csv(os.path.join(out_dir, "multiturn_results.csv"), index=False)

    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    if len(df_ok):
        agg_spec = {
            "visual_encoder_ms_mean": ("visual_encoder_ms", "mean"),
            "audio_encoder_ms_mean": ("audio_encoder_ms", "mean"),
            "llm_prefill_ms_mean": ("llm_prefill_ms", "mean"),
            "ttft_ms_mean": ("ttft_ms", "mean"),
            "visual_forward_calls_mean": ("visual_forward_calls", "mean"),
            "audio_forward_calls_mean": ("audio_forward_calls", "mean"),
            "llm_forward_calls_mean": ("llm_forward_calls", "mean"),
            "n": ("ttft_ms", "count"),
        }
        for k in (
            "mem__peak_allocated_mb",
            "mem__peak_reserved_mb",
            "mem__encoder_peak_allocated_mb",
            "mem__encoder_peak_reserved_mb",
            "mem__prefill_peak_allocated_mb",
            "mem__prefill_peak_reserved_mb",
        ):
            if k in df_ok.columns:
                agg_spec[f"{k}_mean"] = (k, "mean")

        agg = df_ok.groupby(["turn"], dropna=False).agg(**agg_spec).reset_index()
        agg.to_csv(os.path.join(out_dir, "multiturn_summary.csv"), index=False)

        metrics = [
            ("visual_encoder_ms_mean", "visual_encoder_ms"),
            ("audio_encoder_ms_mean", "audio_encoder_ms"),
            ("llm_prefill_ms_mean", "llm_prefill_ms"),
            ("ttft_ms_mean", "ttft_ms"),
        ]

        if "mem__peak_allocated_mb_mean" in agg.columns and "mem__peak_reserved_mb_mean" in agg.columns:
            try:
                turns = [int(t) for t in agg["turn"].tolist()]
                alloc = [float(agg[agg["turn"] == t]["mem__peak_allocated_mb_mean"].iloc[0]) for t in turns]
                resv = [float(agg[agg["turn"] == t]["mem__peak_reserved_mb_mean"].iloc[0]) for t in turns]
                x = np.arange(len(turns))
                width = 0.35
                plt.figure(figsize=(7, 4))
                plt.bar(x - width / 2, alloc, width, label="allocated")
                plt.bar(x + width / 2, resv, width, label="reserved")
                plt.xticks(x, [f"turn{t}" for t in turns])
                plt.ylabel("MB")
                plt.title("GPU memory peak per turn (mean)")
                plt.grid(True, axis="y", linestyle="--", alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "multiturn_mem_peaks.png"), dpi=150, bbox_inches="tight")
                plt.close()
            except Exception:
                pass

        plt.figure(figsize=(9, 4))
        x = np.arange(len(metrics))
        turns = [int(t) for t in agg["turn"].tolist()]
        width = 0.8 / max(2, len(turns))

        for i, t in enumerate(turns):
            vals = []
            for k, _ in metrics:
                col = agg[agg["turn"] == t][k]
                vals.append(float(col.iloc[0]) if len(col) else 0.0)
            plt.bar(x - 0.4 + (i + 0.5) * width, vals, width, label=f"turn{t}")

        plt.xticks(x, [n for _, n in metrics], rotation=20)
        plt.ylabel("ms")
        plt.title("multi-turn: encoder/prefill/ttft")
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "multiturn_bar.png"), dpi=150, bbox_inches="tight")
        plt.close()

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

    if visual_counter is not None:
        try:
            visual_counter.remove()
        except Exception:
            pass
    if audio_counter is not None:
        try:
            audio_counter.remove()
        except Exception:
            pass
    if llm_counter is not None:
        try:
            llm_counter.remove()
        except Exception:
            pass

    return out_dir
