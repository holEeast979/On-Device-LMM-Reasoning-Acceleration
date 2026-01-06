from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.runner import BenchmarkRunner
from benchmark.unified_runner import UnifiedRunner
import profiling_utils as P


SPEC_NAME = "ttft-breakdown"


def register_subcommand(subparsers, common_parser) -> None:
    p = subparsers.add_parser(SPEC_NAME, parents=[common_parser], help="TTFT breakdown for a single request")
    p.add_argument("--audio-max-seconds", type=float, default=30.0)
    p.set_defaults(_spec_run=run)


def _select_video_samples(args: argparse.Namespace, runner: BenchmarkRunner) -> List[Dict[str, Any]]:
    if not os.path.exists(str(args.manifest)):
        raise SystemExit(f"manifest not found: {args.manifest}")

    all_rows = runner.load_manifest_csv(str(args.manifest), n_samples=0, seed=int(args.seed))

    usable: List[Dict[str, Any]] = []
    seen_videos = set()
    for r in all_rows:
        raw_v = r.get("video_path", None)
        v = UnifiedRunner.resolve_media_path(str(args.manifest), None if raw_v is None else str(raw_v))
        if not v or not os.path.exists(v):
            continue
        if v in seen_videos:
            continue
        seen_videos.add(v)
        rr = dict(r)
        rr["video_path"] = v
        usable.append(rr)

    if not usable:
        raise SystemExit("no usable samples with existing video_path")

    n = int(getattr(args, "n_samples", 1))
    if n <= 0 or n >= len(usable):
        return usable

    df = pd.DataFrame(usable)
    sub = df.sample(n=n, random_state=int(args.seed))
    return sub.to_dict(orient="records")


def run(args: argparse.Namespace, runner: BenchmarkRunner) -> str:
    loaded = runner.load()
    model, proc, fe = loaded.model, loaded.processor, loaded.feature_extractor

    out_dir = os.path.join(str(args.out_dir), SPEC_NAME)
    os.makedirs(out_dir, exist_ok=True)

    samples = _select_video_samples(args, runner)

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
        video_path = str(s.get("video_path", ""))
        question = str(args.question) if args.question else str(s.get("question", "Describe what you see and hear."))
        cases.append(
            {
                "sample_id": sample_id,
                "video_path": video_path,
                "video_nframes": int(args.video_nframes) if args.video_nframes else None,
                "question": question,
            }
        )

    def run_once(case: Dict[str, Any]) -> Dict[str, Any]:
        if mem_monitor is not None:
            mem_monitor.reset()
            mem_monitor.mark("preprocess")

        video_path = str(case.get("video_path", ""))
        question = str(case.get("question", ""))
        video_nframes = case.get("video_nframes", None)

        try:
            av = runner.extract_av_from_video(video_path, question, int(video_nframes) if video_nframes is not None else None)
        except Exception as e:
            return {"error": f"extract_failed: {type(e).__name__}: {e}"}

        if not av.audios:
            return {"error": "no_audio_extracted", "extract_ms": float(av.extract_ms)}

        base = runner.prepare_base_inputs(
            model,
            proc,
            videos=av.videos,
            question=question,
            video_path=video_path,
            video_nframes=int(video_nframes) if video_nframes is not None else None,
        )

        af, mel_frames, audio_feature_ms = runner.build_audio_features(
            fe,
            av.audios[0],
            padding="max_length",
            audio_max_seconds=float(getattr(args, "audio_max_seconds", 30.0)),
        )
        full_inputs = runner.attach_audio_features(
            base_inputs=base.inputs,
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

        prefill_ms_for_other = float(llm_prefill_ms) if llm_prefill_ms is not None else 0.0
        other_ms = float(max(0.0, float(ttft_ms) - float(visual_encoder_ms) - float(audio_encoder_ms) - prefill_ms_for_other))

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
            "extract_ms": float(av.extract_ms),
            "pack_ms": float(base.pack_ms),
            "audio_feature_ms": float(audio_feature_ms),
            "preprocess_ms": float(av.extract_ms + base.pack_ms + audio_feature_ms),
            "mel_frames": int(mel_frames),
            "visual_encoder_ms": float(visual_encoder_ms),
            "audio_encoder_ms": float(audio_encoder_ms),
            "llm_prefill_ms": llm_prefill_ms,
            "llm_prefill_seq_len": llm_prefill_seq_len,
            "other_ms": float(other_ms),
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

    df.to_csv(os.path.join(out_dir, "ttft_breakdown_results.csv"), index=False)

    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    if len(df_ok):
        summary = {
            "n": int(len(df_ok)),
            "extract_ms_mean": float(df_ok["extract_ms"].mean()) if "extract_ms" in df_ok.columns else None,
            "pack_ms_mean": float(df_ok["pack_ms"].mean()) if "pack_ms" in df_ok.columns else None,
            "audio_feature_ms_mean": float(df_ok["audio_feature_ms"].mean()) if "audio_feature_ms" in df_ok.columns else None,
            "preprocess_ms_mean": float(df_ok["preprocess_ms"].mean()) if "preprocess_ms" in df_ok.columns else None,
            "visual_encoder_ms_mean": float(df_ok["visual_encoder_ms"].mean()) if "visual_encoder_ms" in df_ok.columns else None,
            "audio_encoder_ms_mean": float(df_ok["audio_encoder_ms"].mean()) if "audio_encoder_ms" in df_ok.columns else None,
            "llm_prefill_ms_mean": float(df_ok["llm_prefill_ms"].mean()) if "llm_prefill_ms" in df_ok.columns else None,
            "other_ms_mean": float(df_ok["other_ms"].mean()) if "other_ms" in df_ok.columns else None,
            "ttft_ms_mean": float(df_ok["ttft_ms"].mean()) if "ttft_ms" in df_ok.columns else None,
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
                summary[f"{k}_mean"] = float(df_ok[k].mean())
        pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "ttft_breakdown_summary.csv"), index=False)

        if (
            ("mem__peak_allocated_mb_mean" in summary)
            and ("mem__encoder_peak_allocated_mb_mean" in summary)
            and ("mem__preprocess_peak_allocated_mb_mean" in summary)
        ):
            try:
                labels = ["preprocess", "encoder", "prefill", "overall"]
                alloc = [
                    float(summary.get("mem__preprocess_peak_allocated_mb_mean") or 0.0),
                    float(summary.get("mem__encoder_peak_allocated_mb_mean") or 0.0),
                    float(summary.get("mem__prefill_peak_allocated_mb_mean") or 0.0),
                    float(summary.get("mem__peak_allocated_mb_mean") or 0.0),
                ]
                resv = [
                    float(summary.get("mem__preprocess_peak_reserved_mb_mean") or 0.0),
                    float(summary.get("mem__encoder_peak_reserved_mb_mean") or 0.0),
                    float(summary.get("mem__prefill_peak_reserved_mb_mean") or 0.0),
                    float(summary.get("mem__peak_reserved_mb_mean") or 0.0),
                ]
                x = np.arange(len(labels))
                width = 0.35
                plt.figure(figsize=(7, 4))
                plt.bar(x - width / 2, alloc, width, label="allocated")
                plt.bar(x + width / 2, resv, width, label="reserved")
                plt.xticks(x, labels, rotation=15, ha="right")
                plt.ylabel("MB")
                plt.title("GPU memory peaks (mean)")
                plt.grid(True, axis="y", linestyle="--", alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "ttft_breakdown_mem_peaks.png"), dpi=150, bbox_inches="tight")
                plt.close()
            except Exception:
                pass

        extract_ms = float(summary.get("extract_ms_mean") or 0.0)
        pack_ms = float(summary.get("pack_ms_mean") or 0.0)
        audio_feature_ms = float(summary.get("audio_feature_ms_mean") or 0.0)

        plt.figure(figsize=(6, 4))
        bottom = 0.0
        for v, label, color in (
            (extract_ms, "extract", "#4C78A8"),
            (pack_ms, "pack", "#F58518"),
            (audio_feature_ms, "audio_feature", "#E45756"),
        ):
            plt.bar([0], [v], bottom=bottom, label=label, color=color)
            bottom += float(v)
        plt.xticks([0], ["preprocess"], rotation=0)
        plt.ylabel("ms")
        plt.title("preprocess breakdown (mean)")
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ttft_breakdown_preprocess_bar.png"), dpi=150, bbox_inches="tight")
        plt.close()

        visual_ms = float(summary.get("visual_encoder_ms_mean") or 0.0)
        audio_ms = float(summary.get("audio_encoder_ms_mean") or 0.0)
        prefill_ms = float(summary.get("llm_prefill_ms_mean") or 0.0)
        other_ms = float(summary.get("other_ms_mean") or 0.0)

        plt.figure(figsize=(6, 4))
        bottom = 0.0
        for v, label, color in (
            (visual_ms, "visual_encoder", "#72B7B2"),
            (audio_ms, "audio_encoder", "#54A24B"),
            (prefill_ms, "llm_prefill", "#EECA3B"),
            (other_ms, "other", "#B279A2"),
        ):
            plt.bar([0], [v], bottom=bottom, label=label, color=color)
            bottom += float(v)
        plt.xticks([0], ["ttft"], rotation=0)
        plt.ylabel("ms")
        plt.title("ttft breakdown (mean)")
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ttft_breakdown_ttft_bar.png"), dpi=150, bbox_inches="tight")
        plt.close()

        if df_ok["sample_id"].nunique() <= 20 and "ttft_ms" in df_ok.columns:
            sub = (
                df_ok.groupby(["sample_id"], dropna=False)
                .agg(
                    visual_encoder_ms_mean=("visual_encoder_ms", "mean"),
                    audio_encoder_ms_mean=("audio_encoder_ms", "mean"),
                    llm_prefill_ms_mean=("llm_prefill_ms", "mean"),
                    other_ms_mean=("other_ms", "mean"),
                    ttft_ms_mean=("ttft_ms", "mean"),
                    n=("ttft_ms", "count"),
                )
                .reset_index()
                .sort_values("ttft_ms_mean", ascending=False)
            )

            xs = np.arange(len(sub))
            plt.figure(figsize=(max(8, 0.35 * len(sub) + 3), 4))
            bottom = np.zeros((len(sub),), dtype=float)
            for k, label, color in (
                ("visual_encoder_ms_mean", "visual_encoder", "#72B7B2"),
                ("audio_encoder_ms_mean", "audio_encoder", "#54A24B"),
                ("llm_prefill_ms_mean", "llm_prefill", "#EECA3B"),
                ("other_ms_mean", "other", "#B279A2"),
            ):
                vals = sub[k].to_numpy(dtype=float)
                plt.bar(xs, vals, bottom=bottom, label=label, color=color)
                bottom += vals
            plt.xticks(xs, sub["sample_id"].astype(str).tolist(), rotation=60, ha="right")
            plt.ylabel("ms")
            plt.title("ttft breakdown by sample (mean)")
            plt.grid(True, axis="y", linestyle="--", alpha=0.3)
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "ttft_breakdown_by_sample.png"), dpi=150, bbox_inches="tight")
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

    return out_dir
