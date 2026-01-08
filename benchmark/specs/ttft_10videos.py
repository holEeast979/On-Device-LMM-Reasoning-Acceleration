"""
Spec: ttft-10videos
10个不同时长视频的 TTFT Breakdown 测试
目标：生成个案百分比堆叠图，替代之前的平均值图表

视频时长分布设计：
- 短视频 (3个): 30s, 45s, 60s
- 中视频 (4个): 2min, 3min, 4min, 5min  
- 长视频 (3个): 7min, 10min, 12min

使用方法:
    python benchmark/run.py ttft-10videos \
        --model-dir /path/to/Qwen2.5-Omni-7B \
        --manifest /path/to/video_mme_manifest.csv \
        --out-dir ./results \
        --n-samples 10 \
        --profile-mem
"""
from __future__ import annotations

import argparse
import os
import subprocess
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.runner import BenchmarkRunner
from benchmark.unified_runner import UnifiedRunner
from utils import common as C
from utils import profiling_utils as P


SPEC_NAME = "ttft-10videos"

# 目标视频时长分布 (秒, 类别)
TARGET_DURATIONS = [
    (30, "short"),
    (45, "short"),
    (60, "short"),
    (120, "medium"),
    (180, "medium"),
    (240, "medium"),
    (300, "medium"),
    (420, "long"),
    (600, "long"),
    (720, "long"),
]


def register_subcommand(subparsers, common_parser) -> None:
    p = subparsers.add_parser(
        SPEC_NAME, 
        parents=[common_parser], 
        help="TTFT breakdown for 10 videos of varying durations (individual stacked bar chart)"
    )
    p.add_argument("--audio-max-seconds", type=float, default=30.0, help="Max audio length for feature extraction")
    p.add_argument("--duration-tolerance", type=float, default=60.0, help="Tolerance in seconds for video duration matching")
    p.add_argument("--max-video-duration", type=float, default=720.0, help="Max video duration in seconds (default 12min)")
    p.set_defaults(_spec_run=run)


def get_video_duration_ffprobe(video_path: str) -> float:
    """使用 ffprobe 获取视频时长"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info["format"]["duration"])
    except Exception as e:
        print(f"  ffprobe error: {e}")
    return 0.0


def select_videos_by_duration(
    args: argparse.Namespace,
    runner: BenchmarkRunner,
    tolerance_sec: float = 60.0,
    max_duration: float = 720.0,
) -> List[Dict[str, Any]]:
    """
    按目标时长分布选择 10 个视频
    """
    if not os.path.exists(str(args.manifest)):
        raise SystemExit(f"manifest not found: {args.manifest}")

    all_rows = runner.load_manifest_csv(str(args.manifest), n_samples=0, seed=int(args.seed))
    
    # 获取所有视频的时长
    print(f"Scanning {len(all_rows)} videos for duration...")
    videos_with_duration = []
    
    for i, r in enumerate(all_rows):
        raw_v = r.get("video_path", None)
        v = UnifiedRunner.resolve_media_path(str(args.manifest), None if raw_v is None else str(raw_v))
        
        if not v or not os.path.exists(v):
            continue
        
        # 获取时长
        duration = r.get("duration", 0)
        if not duration or duration <= 0:
            duration = get_video_duration_ffprobe(v)
        
        if duration <= 0 or duration > max_duration:
            continue
        
        rr = dict(r)
        rr["video_path"] = v
        rr["duration"] = float(duration)
        videos_with_duration.append(rr)
        
        if (i + 1) % 50 == 0:
            print(f"  Scanned {i+1}/{len(all_rows)} videos...")
    
    print(f"Found {len(videos_with_duration)} usable videos with duration <= {max_duration}s")
    
    if not videos_with_duration:
        raise SystemExit("no usable videos found")
    
    # 按目标时长选择
    selected = []
    used_paths = set()
    
    for target_sec, category in TARGET_DURATIONS:
        if target_sec > max_duration:
            print(f"  Skipping target {target_sec}s (exceeds max {max_duration}s)")
            continue
            
        # 找最接近目标时长的视频
        candidates = [
            v for v in videos_with_duration
            if v["video_path"] not in used_paths
            and abs(v["duration"] - target_sec) <= tolerance_sec
        ]
        
        if not candidates:
            # 放宽条件
            candidates = [
                v for v in videos_with_duration
                if v["video_path"] not in used_paths
            ]
        
        if candidates:
            best = min(candidates, key=lambda x: abs(x["duration"] - target_sec))
            best["category"] = category
            best["target_duration"] = target_sec
            selected.append(best)
            used_paths.add(best["video_path"])
            print(f"  Selected for {target_sec}s ({category}): {os.path.basename(best['video_path'])} ({best['duration']:.1f}s)")
    
    # 按实际时长排序
    selected.sort(key=lambda x: x["duration"])
    
    return selected


def run(args: argparse.Namespace, runner: BenchmarkRunner) -> str:
    loaded = runner.load()
    model, proc, fe = loaded.model, loaded.processor, loaded.feature_extractor

    out_dir = os.path.join(str(args.out_dir), SPEC_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # 选择视频
    tolerance = float(getattr(args, "duration_tolerance", 60.0))
    max_dur = float(getattr(args, "max_video_duration", 720.0))
    samples = select_videos_by_duration(args, runner, tolerance_sec=tolerance, max_duration=max_dur)
    
    if len(samples) < 10:
        print(f"WARNING: Only found {len(samples)} videos, expected 10")
    
    # 保存选中的视频列表
    selected_json = os.path.join(out_dir, "selected_videos.json")
    with open(selected_json, "w") as f:
        json.dump(samples, f, indent=2, default=str)
    print(f"Saved selected videos: {selected_json}")

    # 设置 hooks
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

    # 内存监控
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

    # 构建 cases
    cases: List[Dict[str, Any]] = []
    for s in samples:
        sample_id = str(s.get("sample_id", os.path.basename(s["video_path"])))
        video_path = str(s.get("video_path", ""))
        question = str(args.question) if args.question else str(s.get("question", "Describe what you see and hear."))
        cases.append({
            "sample_id": sample_id,
            "video_path": video_path,
            "video_nframes": int(args.video_nframes) if args.video_nframes else None,
            "question": question,
            "duration": float(s.get("duration", 0)),
            "category": str(s.get("category", "unknown")),
            "target_duration": int(s.get("target_duration", 0)),
        })

    def run_once(case: Dict[str, Any]) -> Dict[str, Any]:
        if mem_monitor is not None:
            mem_monitor.reset()
            mem_monitor.mark("preprocess")

        video_path = str(case.get("video_path", ""))
        question = str(case.get("question", ""))
        video_nframes = case.get("video_nframes", None)
        duration = float(case.get("duration", 0))
        category = str(case.get("category", "unknown"))

        try:
            av = runner.extract_av_from_video(video_path, question, int(video_nframes) if video_nframes is not None else None)
        except Exception as e:
            return {"error": f"extract_failed: {type(e).__name__}: {e}", "duration": duration, "category": category}

        if not av.audios:
            return {"error": "no_audio_extracted", "extract_ms": float(av.extract_ms), "duration": duration, "category": category}

        base = runner.prepare_base_inputs(
            model, proc,
            videos=av.videos,
            question=question,
            video_path=video_path,
            video_nframes=int(video_nframes) if video_nframes is not None else None,
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

        return {
            "duration": duration,
            "category": category,
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

    df.to_csv(os.path.join(out_dir, "ttft_10videos_results.csv"), index=False)

    # 生成可视化
    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    if len(df_ok):
        _plot_individual_stacked_bar(df_ok, out_dir)
        _plot_percentage_stacked_bar(df_ok, out_dir)
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


def _plot_individual_stacked_bar(df: pd.DataFrame, out_dir: str) -> None:
    """绝对值堆叠条形图"""
    # 按时长分组取平均
    grouped = df.groupby(["sample_id", "duration", "category"], dropna=False).agg({
        "preprocess_ms": "mean",
        "visual_encoder_ms": "mean",
        "audio_encoder_ms": "mean",
        "llm_prefill_ms": "mean",
        "other_ms": "mean",
        "ttft_ms": "mean",
    }).reset_index().sort_values("duration")

    n = len(grouped)
    if n == 0:
        return

    x = np.arange(n)
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 7))

    phases = [
        ("preprocess_ms", "Preprocess", "#2ecc71"),
        ("visual_encoder_ms", "Visual Encoder", "#3498db"),
        ("audio_encoder_ms", "Audio Encoder", "#e67e22"),
        ("llm_prefill_ms", "Prefill", "#e74c3c"),
        ("other_ms", "Other/Decode", "#9b59b6"),
    ]

    bottom = np.zeros(n)
    for col, label, color in phases:
        vals = grouped[col].fillna(0).to_numpy() / 1000  # 转换为秒
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color)
        bottom += vals

    # X 轴标签
    labels = [f"{row['duration']:.0f}s\n({row['category']})" for _, row in grouped.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Video Duration", fontsize=11)
    ax.set_ylabel("TTFT (seconds)", fontsize=11)
    ax.set_title("TTFT Breakdown by Video Duration\n(Absolute Time, Individual Cases)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "ttft_10videos_absolute_bar.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def _plot_percentage_stacked_bar(df: pd.DataFrame, out_dir: str) -> None:
    """百分比堆叠条形图（你导师要的那个图）"""
    grouped = df.groupby(["sample_id", "duration", "category"], dropna=False).agg({
        "preprocess_ms": "mean",
        "visual_encoder_ms": "mean",
        "audio_encoder_ms": "mean",
        "llm_prefill_ms": "mean",
        "other_ms": "mean",
        "ttft_ms": "mean",
    }).reset_index().sort_values("duration")

    n = len(grouped)
    if n == 0:
        return

    x = np.arange(n)
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 7))

    phases = [
        ("preprocess_ms", "Preprocess", "#2ecc71"),
        ("visual_encoder_ms", "Visual Encoder", "#3498db"),
        ("audio_encoder_ms", "Audio Encoder", "#e67e22"),
        ("llm_prefill_ms", "Prefill", "#e74c3c"),
        ("other_ms", "Other/Decode", "#9b59b6"),
    ]

    bottom = np.zeros(n)
    for col, label, color in phases:
        vals = []
        for _, row in grouped.iterrows():
            total = row["ttft_ms"]
            v = row[col] if pd.notna(row[col]) else 0
            pct = (v / total * 100) if total > 0 else 0
            vals.append(pct)
        vals = np.array(vals)
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color)
        bottom += vals

    labels = [f"{row['duration']:.0f}s\n({row['category']})" for _, row in grouped.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Video Duration", fontsize=11)
    ax.set_ylabel("Percentage of TTFT (%)", fontsize=11)
    ax.set_title("TTFT Breakdown by Video Duration\n(Percentage, Individual Cases)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "ttft_10videos_percentage_bar.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def _save_summary(df: pd.DataFrame, out_dir: str) -> None:
    """保存汇总统计"""
    grouped = df.groupby(["duration", "category"], dropna=False).agg({
        "preprocess_ms": "mean",
        "visual_encoder_ms": "mean",
        "audio_encoder_ms": "mean",
        "llm_prefill_ms": "mean",
        "other_ms": "mean",
        "ttft_ms": "mean",
    }).reset_index().sort_values("duration")

    # 计算 encoder 占比
    grouped["encoder_pct"] = (grouped["visual_encoder_ms"] + grouped["audio_encoder_ms"]) / grouped["ttft_ms"] * 100

    summary_path = os.path.join(out_dir, "ttft_10videos_summary.csv")
    grouped.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("Summary: Encoder percentage of TTFT")
    print("=" * 60)
    for _, row in grouped.iterrows():
        print(f"  {row['duration']:6.0f}s ({row['category']:6s}): TTFT={row['ttft_ms']/1000:.2f}s, Encoder={row['encoder_pct']:.1f}%")
