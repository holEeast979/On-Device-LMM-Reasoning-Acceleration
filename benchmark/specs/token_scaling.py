"""
Spec: token-scaling
探究 Token 数量与各阶段时间的关系（O(n) vs O(n²)）

目标：
1. 选择 10-15 个视频，覆盖 3000-20000 tokens
2. 每视频重复 N 次，测量各阶段时间
3. 线性/二次回归分析，判断复杂度

输出：
- token_scaling_results.csv: 原始数据
- token_scaling_summary.csv: 各视频平均值
- token_scaling_plot.png: 4 子图（tokens vs 各阶段时间）
- scaling_analysis.json: 回归分析结果

使用方法:
    python benchmark/run.py token-scaling \
        --model-dir /root/autodl-tmp/Qwen2.5-Omni-7B \
        --video-dir /root/autodl-tmp/data/ActivityNet-QA/videos \
        --out-dir /root/autodl-tmp/results \
        --n-videos 15 \
        --repeats 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from benchmark.runner import BenchmarkRunner
from benchmark.unified_runner import UnifiedRunner
from utils import profiling_utils as P


SPEC_NAME = "token-scaling"


def register_subcommand(subparsers, common_parser) -> None:
    p = subparsers.add_parser(
        SPEC_NAME,
        parents=[common_parser],
        help="Token scaling analysis: Time vs Visual Tokens relationship"
    )
    p.add_argument("--video-dir", type=str, required=True, help="Directory containing videos")
    p.add_argument("--n-videos", type=int, default=15, help="Number of videos to test")
    p.add_argument("--min-duration", type=float, default=8.0, help="Min video duration (seconds)")
    p.add_argument("--max-duration", type=float, default=38.0, help="Max video duration (seconds)")
    p.add_argument("--audio-max-seconds", type=float, default=30.0, help="Max audio length")
    p.set_defaults(_spec_run=run)


def get_video_duration(video_path: str) -> float:
    """使用 ffprobe 获取视频时长"""
    try:
        cmd = f'ffprobe -v quiet -print_format json -show_format "{video_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        info = json.loads(result.stdout)
        return float(info['format'].get('duration', 0))
    except Exception:
        return 0.0


def select_videos_for_token_range(
    video_dir: str,
    n_videos: int = 15,
    min_duration: float = 8.0,
    max_duration: float = 38.0,
) -> List[Dict[str, Any]]:
    """选择视频，均匀覆盖不同时长范围"""
    print(f"Scanning videos in {video_dir}...")
    
    all_videos = []
    for f in os.listdir(video_dir):
        if not f.endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm')):
            continue
        path = os.path.join(video_dir, f)
        duration = get_video_duration(path)
        if min_duration <= duration <= max_duration:
            all_videos.append({
                'video_path': path,
                'sample_id': f,
                'duration': duration
            })
    
    print(f"Found {len(all_videos)} videos in range [{min_duration}s, {max_duration}s]")
    
    if len(all_videos) < n_videos:
        print(f"WARNING: Only found {len(all_videos)} videos, expected {n_videos}")
        return sorted(all_videos, key=lambda x: x['duration'])
    
    all_videos.sort(key=lambda x: x['duration'])
    indices = np.linspace(0, len(all_videos) - 1, n_videos, dtype=int)
    selected = [all_videos[i] for i in indices]
    
    print(f"Selected {len(selected)} videos:")
    for v in selected:
        print(f"  {v['duration']:.1f}s: {v['sample_id']}")
    
    return selected


def run(args: argparse.Namespace, runner: BenchmarkRunner) -> str:
    loaded = runner.load()
    model, proc, fe = loaded.model, loaded.processor, loaded.feature_extractor

    out_dir = os.path.join(str(args.out_dir), SPEC_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # 选择视频
    videos = select_videos_for_token_range(
        video_dir=str(args.video_dir),
        n_videos=int(args.n_videos),
        min_duration=float(args.min_duration),
        max_duration=float(args.max_duration),
    )

    if not videos:
        raise SystemExit("No videos found!")

    # 保存选中视频列表
    with open(os.path.join(out_dir, "selected_videos.json"), "w") as f:
        json.dump(videos, f, indent=2, default=str)

    # 设置 hooks
    visual, audio, llm = runner.get_qwen25_modules(model)
    if audio is None:
        raise SystemExit("Qwen2.5-Omni audio_tower not found")

    visual_timer = P.ModuleCudaEventTimer()
    audio_timer = P.ModuleCudaEventTimer()
    thinker_capture = P.ThinkerPrefillCapture()

    if visual is not None:
        visual_timer.register(visual)
    audio_timer.register(audio)
    thinker_capture.register(model.thinker)

    ur = UnifiedRunner(base=runner, spec_name=SPEC_NAME, out_dir=out_dir, args=args)

    # 构建 cases
    cases: List[Dict[str, Any]] = []
    for v in videos:
        cases.append({
            "sample_id": str(v['sample_id']),
            "video_path": str(v['video_path']),
            "duration": float(v['duration']),
            "question": str(args.question) if args.question else "Describe what you see.",
        })

    def run_once(case: Dict[str, Any]) -> Dict[str, Any]:
        video_path = str(case.get("video_path", ""))
        question = str(case.get("question", ""))
        duration = float(case.get("duration", 0))

        try:
            av = runner.extract_av_from_video(video_path, question, None)
        except Exception as e:
            return {"error": f"extract_failed: {type(e).__name__}: {e}", "duration": duration}

        if not av.audios:
            return {"error": "no_audio_extracted", "duration": duration}

        base = runner.prepare_base_inputs(
            model, proc,
            videos=av.videos,
            question=question,
            video_path=video_path,
            video_nframes=None,
        )

        af, mel_frames, audio_feature_ms = runner.build_audio_features(
            fe, av.audios[0],
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

        # 使用统一计时逻辑
        breakdown = runner.run_generate_with_breakdown(
            model,
            {k: v for k, v in full_inputs.items()},
            visual_timer=visual_timer,
            audio_timer=audio_timer,
            thinker_capture=thinker_capture,
        )
        
        return {
            "duration": duration,
            "extract_ms": float(av.extract_ms),
            "pack_ms": float(base.pack_ms),
            "audio_feature_ms": float(audio_feature_ms),
            "preprocess_ms": float(av.extract_ms + base.pack_ms + audio_feature_ms),
            "visual_encoder_ms": breakdown["visual_encoder_ms"],
            "audio_encoder_ms": breakdown["audio_encoder_ms"],
            "prefill_ms": breakdown["prefill_ms"],  # = Embedding Merge + LLM Forward
            "others_ms": breakdown["others_ms"],    # = 调度开销
            "ttft_ms": breakdown["ttft_ms"],
            "mel_frames": int(mel_frames),
            **token_stats,
        }

    df = ur.run(
        cases=cases,
        repeats=int(max(1, args.repeats)),
        warmup=int(max(0, args.warmup)),
        run_once=run_once,
        clear_cache=True,
    )

    df.to_csv(os.path.join(out_dir, "token_scaling_results.csv"), index=False)

    # 生成可视化和分析
    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    if len(df_ok):
        _plot_scaling_analysis(df_ok, out_dir)
        _run_regression_analysis(df_ok, out_dir)
        _save_summary(df_ok, out_dir)

    # 清理 hooks
    try:
        visual_timer.remove()
    except Exception:
        pass
    try:
        audio_timer.remove()
    except Exception:
        pass
    try:
        thinker_capture.remove()
    except Exception:
        pass

    return out_dir


def _plot_scaling_analysis(df: pd.DataFrame, out_dir: str) -> None:
    """绘制 4 子图：tokens vs 各阶段时间"""
    grouped = df.groupby("visual_tokens").agg({
        "visual_encoder_ms": "mean",
        "audio_encoder_ms": "mean",
        "prefill_ms": "mean",
        "ttft_ms": "mean",
    }).reset_index().sort_values("visual_tokens")

    x = grouped["visual_tokens"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    stages = [
        ("visual_encoder_ms", "Visual Encoder", "#3498db", axes[0, 0]),
        ("audio_encoder_ms", "Audio Encoder", "#e67e22", axes[0, 1]),
        ("prefill_ms", "Prefill", "#e74c3c", axes[1, 0]),
        ("ttft_ms", "TTFT (Total)", "#2ecc71", axes[1, 1]),
    ]

    for col, title, color, ax in stages:
        y = grouped[col].values

        ax.scatter(x, y, color=color, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

        if len(x) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=2, alpha=0.8,
                   label=f"Linear: R²={r_value**2:.3f}")

            if len(x) >= 4:
                coeffs = np.polyfit(x, y, 2)
                y_quad = np.polyval(coeffs, x_fit)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_quad = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                ax.plot(x_fit, y_quad, color='gray', linestyle=':', linewidth=1.5, alpha=0.6,
                       label=f"Quadratic: R²={r2_quad:.3f}")

        ax.set_xlabel("Visual Tokens", fontsize=11)
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Token Scaling Analysis\n(Time vs Visual Tokens)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(out_dir, "token_scaling_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def _run_regression_analysis(df: pd.DataFrame, out_dir: str) -> None:
    """运行回归分析，判断复杂度"""
    grouped = df.groupby("visual_tokens").agg({
        "visual_encoder_ms": "mean",
        "audio_encoder_ms": "mean",
        "prefill_ms": "mean",
        "ttft_ms": "mean",
    }).reset_index().sort_values("visual_tokens")

    x = grouped["visual_tokens"].values
    analysis = {}

    stages = ["visual_encoder_ms", "audio_encoder_ms", "prefill_ms", "ttft_ms"]

    for col in stages:
        y = grouped[col].values

        if len(x) < 3:
            analysis[col] = {"error": "insufficient data"}
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2_linear = r_value ** 2

        if len(x) >= 4:
            coeffs = np.polyfit(x, y, 2)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_quad = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r2_quad = 0
            coeffs = [0, 0, 0]

        if r2_linear >= 0.95:
            complexity = "O(n) - Linear"
        elif r2_quad >= 0.95 and r2_quad > r2_linear + 0.02:
            complexity = "O(n²) - Quadratic"
        elif r2_linear >= 0.85:
            complexity = "~O(n) - Approximately Linear"
        elif r2_quad >= 0.85:
            complexity = "~O(n²) - Approximately Quadratic"
        else:
            complexity = "Unknown"

        analysis[col] = {
            "linear_r2": float(r2_linear),
            "linear_slope": float(slope),
            "linear_intercept": float(intercept),
            "linear_p_value": float(p_value),
            "quadratic_r2": float(r2_quad),
            "quadratic_coeffs": [float(c) for c in coeffs],
            "complexity": complexity,
        }

        print(f"\n{col}:")
        print(f"  Linear R²: {r2_linear:.4f}, Quadratic R²: {r2_quad:.4f}")
        print(f"  Complexity: {complexity}")

    save_path = os.path.join(out_dir, "scaling_analysis.json")
    with open(save_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {save_path}")


def _save_summary(df: pd.DataFrame, out_dir: str) -> None:
    """保存汇总统计"""
    summary = df.groupby(["sample_id", "duration", "visual_tokens"]).agg({
        "visual_encoder_ms": ["mean", "std"],
        "audio_encoder_ms": ["mean", "std"],
        "prefill_ms": ["mean", "std"],
        "ttft_ms": ["mean", "std"],
    }).reset_index()

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary = summary.sort_values("visual_tokens")
    summary.to_csv(os.path.join(out_dir, "token_scaling_summary.csv"), index=False)
    print(f"Saved: {os.path.join(out_dir, 'token_scaling_summary.csv')}")
