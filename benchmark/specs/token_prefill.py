from __future__ import annotations

import argparse
import gc
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmark.runner import BenchmarkRunner
import profiling_utils as P


SPEC_NAME = "token-prefill"


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out


def register_subcommand(subparsers, common_parser) -> None:
    p = subparsers.add_parser(SPEC_NAME, parents=[common_parser], help="Prefill scales with token count")
    p.add_argument("--sample-index", type=int, default=0, help="Index into the manifest (after loading ALL rows)")
    p.add_argument("--filler-words", type=str, default="0,50,100,200,500,1000")
    p.add_argument("--filler-token", type=str, default="the")
    p.add_argument("--audio-max-seconds", type=float, default=30.0)
    p.set_defaults(_spec_run=run)


def run(args: argparse.Namespace, runner: BenchmarkRunner) -> str:
    loaded = runner.load()
    model, proc, fe = loaded.model, loaded.processor, loaded.feature_extractor

    out_dir = os.path.join(str(args.out_dir), SPEC_NAME)
    os.makedirs(out_dir, exist_ok=True)
    runner.write_json(
        os.path.join(out_dir, "meta.json"),
        {"spec": SPEC_NAME, "args": vars(args), "env": runner.env_info()},
    )

    if not os.path.exists(str(args.manifest)):
        raise SystemExit(f"manifest not found: {args.manifest}")

    samples = runner.load_manifest_csv(str(args.manifest), n_samples=0, seed=int(args.seed))
    if not samples:
        raise SystemExit("manifest has no samples")

    idx = int(args.sample_index)
    if idx < 0 or idx >= len(samples):
        raise SystemExit(f"sample-index out of range: {idx} / {len(samples)}")

    s = samples[idx]
    video_path = str(s.get("video_path", ""))
    sample_id = str(s.get("sample_id", f"sample_{idx}"))
    base_question = str(args.question) if args.question else str(s.get("question", "Describe what you see and hear."))

    filler_list = _parse_int_list(str(args.filler_words))
    if not filler_list:
        raise SystemExit("--filler-words is empty")

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

    rows: List[Dict[str, Any]] = []

    for filler_n in filler_list:
        filler = "" if filler_n <= 0 else (" " + (str(args.filler_token) + " ") * int(filler_n))
        question = base_question + filler

        try:
            av = runner.extract_av_from_video(video_path, question, int(args.video_nframes) if args.video_nframes else None)
        except Exception as e:
            rows.append({"spec": SPEC_NAME, "sample_id": sample_id, "video_path": video_path, "filler_words": int(filler_n), "error": f"extract_failed: {type(e).__name__}: {e}"})
            continue

        if not av.audios:
            rows.append({"spec": SPEC_NAME, "sample_id": sample_id, "video_path": video_path, "filler_words": int(filler_n), "error": "no_audio_extracted", "extract_ms": av.extract_ms})
            continue

        base = runner.prepare_base_inputs(
            model,
            proc,
            videos=av.videos,
            question=question,
            video_path=video_path,
            video_nframes=int(args.video_nframes) if args.video_nframes else None,
        )

        af, mel_frames, audio_feature_ms = runner.build_audio_features(
            fe,
            av.audios[0],
            padding="max_length",
            audio_max_seconds=float(args.audio_max_seconds),
        )
        full_inputs = runner.attach_audio_features(
            base_inputs=base.inputs,
            af=af,
            mel_frames=mel_frames,
            model=model,
            dtype=str(args.dtype),
        )

        token_stats = runner.get_token_stats(proc, full_inputs.get("input_ids"))

        for r in range(int(max(1, args.repeats))):
            do_record = r >= int(max(0, args.warmup))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            visual_timer.clear()
            audio_timer.clear()
            prefill_capture.clear()

            ttft_ms = runner.run_generate_1token_ms(model, {k: v for k, v in full_inputs.items()})

            visual_encoder_ms = float(sum(visual_timer.times)) if visual is not None else 0.0
            audio_encoder_ms = float(sum(audio_timer.times))
            llm_prefill_ms = float(prefill_capture.prefill_forward_ms) if llm is not None else None
            llm_prefill_seq_len = int(prefill_capture.prefill_seq_len) if llm is not None else None

            if not do_record:
                continue

            rows.append(
                {
                    "spec": SPEC_NAME,
                    "sample_id": sample_id,
                    "video_path": video_path,
                    "video_nframes": int(args.video_nframes) if args.video_nframes else None,
                    "question": question,
                    "filler_words": int(filler_n),
                    "repeat": int(r),
                    "extract_ms": float(av.extract_ms),
                    "pack_ms": float(base.pack_ms),
                    "audio_feature_ms": float(audio_feature_ms),
                    "preprocess_ms": float(av.extract_ms + base.pack_ms + audio_feature_ms),
                    "mel_frames": int(mel_frames),
                    "visual_encoder_ms": float(visual_encoder_ms),
                    "audio_encoder_ms": float(audio_encoder_ms),
                    "llm_prefill_ms": llm_prefill_ms,
                    "llm_prefill_seq_len": llm_prefill_seq_len,
                    "ttft_ms": float(ttft_ms),
                    **token_stats,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "token_prefill_results.csv"), index=False)

    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    if len(df_ok) and "llm_prefill_ms" in df_ok.columns:
        agg = (
            df_ok.groupby(["filler_words"], dropna=False)
            .agg(
                total_tokens_mean=("total_tokens", "mean"),
                llm_prefill_ms_mean=("llm_prefill_ms", "mean"),
                ttft_ms_mean=("ttft_ms", "mean"),
                n=("ttft_ms", "count"),
            )
            .reset_index()
            .sort_values("filler_words")
        )
        agg.to_csv(os.path.join(out_dir, "token_prefill_summary.csv"), index=False)

        x = agg["total_tokens_mean"].to_numpy(dtype=float)
        y = agg["llm_prefill_ms_mean"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) >= 2:
            coef = np.polyfit(x[m], y[m], deg=1)
            y_hat = coef[0] * x[m] + coef[1]
            ss_res = float(((y[m] - y_hat) ** 2).sum())
            ss_tot = float(((y[m] - float(y[m].mean())) ** 2).sum())
            r2 = None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
        else:
            coef = [float("nan"), float("nan")]
            r2 = None

        plt.figure(figsize=(7, 4))
        plt.scatter(df_ok["total_tokens"], df_ok["llm_prefill_ms"], s=12, alpha=0.6, label="runs")
        plt.scatter(x, y, s=40, label="mean")
        if np.isfinite(coef[0]) and np.isfinite(coef[1]) and int(m.sum()) >= 2:
            xx = np.linspace(float(x[m].min()), float(x[m].max()), 50)
            yy = coef[0] * xx + coef[1]
            label = f"fit: y={coef[0]:.4f}x+{coef[1]:.2f}" + ("" if r2 is None else f" (R^2={r2:.3f})")
            plt.plot(xx, yy, linewidth=2, label=label)
        plt.xlabel("total_tokens")
        plt.ylabel("llm_prefill_ms")
        plt.title("prefill scales with token count")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "token_prefill_prefill_vs_tokens.png"), dpi=150, bbox_inches="tight")
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

    return out_dir
