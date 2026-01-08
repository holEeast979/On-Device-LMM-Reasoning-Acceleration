#!/usr/bin/env python3
"""
实验7：Video + Audio 多模态编码耗时分析

目标：
1. 测量 Visual/Audio Encoder 各自的耗时
2. 分解 TTFT，分析各阶段占比
3. 监控各阶段 GPU/CPU/VRAM 资源利用率

使用 Qwen2.5-Omni 模型处理 Video with Audio 输入
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import threading
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils import common as C
from utils import profiling_utils as P


# ============ Profiling Utils from profiling_utils.py ============
# Removed duplicate classes as they are now in profiling_utils.py


# ============ 数据加载 ============

def load_dataset(manifest_path: str, n_samples: int = 50) -> List[Dict]:
    return C.load_dataset(manifest_path, n_samples)


# ============ 推理与计时 ============

def run_inference(
    model,
    proc,
    video_path: str,
    question: str,
    monitor: P.ResourceMonitor,
    fe: "WhisperFeatureExtractor" = None,
    audio_max_seconds: float = 30.0,
    use_audio_in_video: bool = True,
    video_nframes: int | None = None,
    video_fps: float | None = None,
    video_min_frames: int | None = None,
    video_max_frames: int | None = None,
) -> Dict:
    """运行推理并记录各阶段耗时（细粒度分解）"""
    from qwen_omni_utils import process_mm_info
    
    timings = {}
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===== 1. 视频+音频提取 =====
    monitor.mark("video_audio_extract_start")
    t0 = time.perf_counter()
    
    video_ele = {"type": "video", "video": video_path}
    if video_nframes is not None and int(video_nframes) > 0:
        video_ele["nframes"] = int(video_nframes)
    if video_fps is not None and float(video_fps) > 0:
        video_ele["fps"] = float(video_fps)
    if video_min_frames is not None and int(video_min_frames) > 0:
        video_ele["min_frames"] = int(video_min_frames)
    if video_max_frames is not None and int(video_max_frames) > 0:
        video_ele["max_frames"] = int(video_max_frames)

    conversation = [{"role": "user", "content": [
        video_ele,
        {"type": "text", "text": question}
    ]}]
    
    # process_mm_info: 视频解码+帧采样+音频提取
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=bool(use_audio_in_video))
    
    t1 = time.perf_counter()
    timings["video_audio_extract_ms"] = (t1 - t0) * 1000
    monitor.mark("video_audio_extract_end")
    
    # ===== 2. 视频 tokenize (processor 处理视频帧) =====
    monitor.mark("video_tokenize_start")
    t2 = time.perf_counter()
    
    text = proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, videos=videos, return_tensors="pt", padding=True)
    tokenizer = getattr(proc, "tokenizer", None)
    if tokenizer is not None and "input_ids" in inputs:
        try:
            mm_ids = P.get_mm_token_ids_from_tokenizer(tokenizer)
            ids = inputs["input_ids"]
            vision_ids = mm_ids.get("vision_special_token_ids", [])
            audio_ids = mm_ids.get("audio_special_token_ids", [])
            timings["mm_vision_slot_tokens"] = int(sum(int((ids == tid).sum().item()) for tid in vision_ids)) if vision_ids else 0
            timings["mm_audio_slot_tokens"] = int(sum(int((ids == tid).sum().item()) for tid in audio_ids)) if audio_ids else 0
            st_counts = P.count_special_tokens_in_input_ids(tokenizer, ids)
            if st_counts:
                timings["mm_special_tokens_json"] = json.dumps(st_counts, ensure_ascii=False)
        except Exception:
            pass
    try:
        n_video_frames = None
        if videos is not None and isinstance(videos, list):
            if len(videos) > 0 and isinstance(videos[0], list):
                n_video_frames = len(videos[0])
            else:
                n_video_frames = len(videos)
        if n_video_frames is not None:
            timings["n_video_frames"] = int(n_video_frames)
    except Exception:
        pass
    inputs = inputs.to(model.device)
    
    t3 = time.perf_counter()
    timings["video_tokenize_ms"] = (t3 - t2) * 1000
    monitor.mark("video_tokenize_end")
    
    # ===== 3. 音频特征提取 (FFT + Mel频谱) =====
    monitor.mark("audio_feature_start")
    t4 = time.perf_counter()
    
    if audios and fe is not None:
        # WhisperFeatureExtractor: 傅里叶变换 + Mel频谱图
        max_audio_samples = int(audio_max_seconds * 16000)
        af = fe(
            audios[0],
            sampling_rate=16000,
            return_tensors='pt',
            padding='max_length',
            max_length=max_audio_samples,
            truncation=True,
        )
        inputs['input_features'] = af['input_features'].to(model.device, dtype=torch.bfloat16)
        inputs['feature_attention_mask'] = torch.ones(
            (1, af['input_features'].shape[2]), device=model.device, dtype=torch.long
        )
        timings["mel_frames"] = int(af["input_features"].shape[2])
    
    t5 = time.perf_counter()
    timings["audio_feature_ms"] = (t5 - t4) * 1000
    monitor.mark("audio_feature_end")
    
    if "input_ids" in inputs:
        timings["proc_input_ids_len"] = int(inputs["input_ids"].shape[-1])
    if "attention_mask" in inputs:
        timings["proc_attention_mask_len"] = int(inputs["attention_mask"].shape[-1])
    
    torch.cuda.synchronize()
    timings["preprocess_ms"] = (t5 - t0) * 1000  # 总预处理时间
    monitor.mark("preprocess_end")
    
    # ===== 4. Generate (包含 Encode + Decode) =====
    monitor.mark("generate_start")
    t_gen_start = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_audio=False,
        )
    
    torch.cuda.synchronize()
    timings["ttft_ms"] = (time.perf_counter() - t_gen_start) * 1000
    monitor.mark("generate_end")
    
    timings["total_ms"] = timings["preprocess_ms"] + timings["ttft_ms"]
    
    return timings


# ===== sweep helpers =====

def _parse_int_list(s: str) -> List[int]:
    if s is None:
        return []
    items: List[int] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            items.append(int(p))
        except Exception:
            continue
    return items


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - float(y.mean())) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def _plot_nframes_scaling(merged_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = merged_df.copy()

    for c in [
        "cfg_video_nframes",
        "n_video_frames",
        "visual_encoder_ms",
        "mm_vision_slot_tokens",
        "llm_prefill_forward_event_ms",
        "llm_prefill_forward_inputs_embeds_len",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "cfg_video_nframes" not in df.columns:
        return

    # 1) Visual encoder vs nframes (mean±std)
    if "visual_encoder_ms" in df.columns:
        agg = (
            df.groupby("cfg_video_nframes", dropna=True)
            .agg(
                n=("visual_encoder_ms", "count"),
                visual_mean=("visual_encoder_ms", "mean"),
                visual_std=("visual_encoder_ms", "std"),
                vision_tokens_mean=("mm_vision_slot_tokens", "mean"),
                prefill_evt_mean=("llm_prefill_forward_event_ms", "mean"),
            )
            .reset_index()
            .sort_values("cfg_video_nframes")
        )

        x = agg["cfg_video_nframes"].to_numpy(dtype=float)
        y = agg["visual_mean"].to_numpy(dtype=float)
        e = agg["visual_std"].to_numpy(dtype=float)

        plt.figure(figsize=(7.2, 5.0))
        plt.errorbar(x, y, yerr=e, fmt="o-", capsize=3, linewidth=2.0, markersize=6)
        plt.xlabel("Configured nframes")
        plt.ylabel("Visual encoder latency (ms)")
        plt.title("Visual encoder vs nframes")
        plt.grid(True, linestyle="--", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sweep_nframes_vs_visual_encoder.png"), dpi=180, bbox_inches="tight")
        plt.close()

        try:
            agg.to_csv(os.path.join(out_dir, "sweep_nframes_summary.csv"), index=False)
        except Exception:
            pass

    # 2) LLM prefill (CUDA event) vs vision tokens
    if "mm_vision_slot_tokens" in df.columns and "llm_prefill_forward_event_ms" in df.columns:
        xv = df["mm_vision_slot_tokens"].to_numpy(dtype=float)
        yv = df["llm_prefill_forward_event_ms"].to_numpy(dtype=float)
        m = np.isfinite(xv) & np.isfinite(yv)
        xv = xv[m]
        yv = yv[m]
        if xv.size >= 3:
            slope, intercept, r2 = _linear_fit(xv, yv)

            plt.figure(figsize=(7.2, 5.0))
            for nf, sub in df.groupby("cfg_video_nframes", dropna=True):
                xs = pd.to_numeric(sub["mm_vision_slot_tokens"], errors="coerce")
                ys = pd.to_numeric(sub["llm_prefill_forward_event_ms"], errors="coerce")
                mm2 = xs.notna() & ys.notna()
                xs = xs[mm2].to_numpy(dtype=float)
                ys = ys[mm2].to_numpy(dtype=float)
                if xs.size == 0:
                    continue
                plt.scatter(xs, ys, s=18, alpha=0.75, label=f"nframes={int(nf)}")

            x_line = np.linspace(float(xv.min()), float(xv.max()), 200)
            plt.plot(x_line, slope * x_line + intercept, color="black", linewidth=2.0, label="fit")

            ann = f"slope = {slope:.4f} ms/token\nR² = {r2:.4f}\nintercept = {intercept:.2f} ms"
            plt.text(
                0.02,
                0.98,
                ann,
                transform=plt.gca().transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            )
            plt.xlabel("mm_vision_slot_tokens")
            plt.ylabel("LLM prefill time (CUDA event, ms)")
            plt.title("LLM prefill vs vision tokens")
            plt.grid(True, linestyle="--", alpha=0.25)
            plt.legend(frameon=True, fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "sweep_vision_tokens_vs_prefill_event.png"), dpi=180, bbox_inches="tight")
            plt.close()


def run_experiment(
    args,
    model,
    proc,
    fe,
    encoder_timer: P.EncoderTimer,
    audio_event_timer: P.ModuleCudaEventTimer,
    seq_len_capture: P.LLMSeqLenCapture,
    prefill_event_capture: P.LLMPrefillCudaEventCapture,
    samples: List[Dict],
    out_dir: str,
    *,
    use_audio_in_video: bool,
    video_nframes: int | None,
    video_fps: float | None,
    video_min_frames: int | None,
    video_max_frames: int | None,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    # warmup
    print(f"\n🔥 预热 ({args.warmup} 次)...")
    for i in range(min(args.warmup, len(samples))):
        monitor = P.ResourceMonitor(interval=0.05)
        monitor.start()
        try:
            q = args.question_override if args.question_override is not None else "Describe."
            if args.prompt_filler_words and args.prompt_filler_words > 0:
                q = q + (" " + args.prompt_filler_word) * int(args.prompt_filler_words)
            run_inference(
                model,
                proc,
                samples[i]["video_path"],
                q,
                monitor,
                fe,
                audio_max_seconds=args.audio_max_seconds,
                use_audio_in_video=use_audio_in_video,
                video_nframes=video_nframes,
                video_fps=video_fps,
                video_min_frames=video_min_frames,
                video_max_frames=video_max_frames,
            )
        except Exception as e:
            print(f"  预热 {i+1} 失败: {e}")
        monitor.stop()
        monitor.cleanup()

    encoder_timer.clear()
    gc.collect()
    torch.cuda.empty_cache()

    # main loop
    print(f"\n🧪 开始测试 ({args.n_samples} 个样本)...")
    results = []
    all_sample_records = []

    for i, sample in enumerate(samples[args.warmup:args.warmup + args.n_samples]):
        print(f"\n--- 样本 {i+1}/{args.n_samples}: {sample['sample_id']} ---")

        video_path = sample["video_path"]
        if not os.path.exists(video_path):
            print(f"  ⚠️ 跳过：视频不存在")
            continue

        monitor = P.ResourceMonitor(interval=0.01)
        monitor.start()

        try:
            encoder_timer.clear()
            audio_event_timer.clear()
            seq_len_capture.clear()
            prefill_event_capture.clear()

            q = args.question_override if args.question_override is not None else sample["question"]
            if args.prompt_filler_words and args.prompt_filler_words > 0:
                q = q + (" " + args.prompt_filler_word) * int(args.prompt_filler_words)

            timings = run_inference(
                model,
                proc,
                video_path,
                q,
                monitor,
                fe,
                audio_max_seconds=args.audio_max_seconds,
                use_audio_in_video=use_audio_in_video,
                video_nframes=video_nframes,
                video_fps=video_fps,
                video_min_frames=video_min_frames,
                video_max_frames=video_max_frames,
            )

            vis_time = encoder_timer.times["visual"][-1] if encoder_timer.times["visual"] else 0
            aud_time = encoder_timer.times["audio"][-1] if encoder_timer.times["audio"] else 0
            aud_time_event = audio_event_timer.get_last()

            audio_tower_in_frames = None
            if audio_event_timer.last_input_shape is not None and len(audio_event_timer.last_input_shape) >= 2:
                audio_tower_in_frames = int(audio_event_timer.last_input_shape[-1])
            audio_tower_input_shape = str(audio_event_timer.last_input_shape) if audio_event_timer.last_input_shape is not None else None

            aud_for_prefill = aud_time_event if aud_time_event > 0 else aud_time
            llm_prefill = max(0, timings["ttft_ms"] - vis_time - aud_for_prefill)

            result = {
                "sample_id": sample["sample_id"],
                "cfg_use_audio_in_video": bool(use_audio_in_video),
                "cfg_video_nframes": int(video_nframes) if video_nframes is not None else None,
                "cfg_video_fps": float(video_fps) if video_fps is not None else None,
                "cfg_video_min_frames": int(video_min_frames) if video_min_frames is not None else None,
                "cfg_video_max_frames": int(video_max_frames) if video_max_frames is not None else None,
                "video_audio_extract_ms": timings.get("video_audio_extract_ms", 0),
                "video_tokenize_ms": timings.get("video_tokenize_ms", 0),
                "audio_feature_ms": timings.get("audio_feature_ms", 0),
                "mel_frames": timings.get("mel_frames", 0),
                "audio_tower_in_frames": audio_tower_in_frames,
                "audio_tower_input_shape": audio_tower_input_shape,
                "preprocess_ms": timings["preprocess_ms"],
                "n_video_frames": timings.get("n_video_frames"),
                "mm_vision_slot_tokens": timings.get("mm_vision_slot_tokens"),
                "mm_audio_slot_tokens": timings.get("mm_audio_slot_tokens"),
                "mm_special_tokens_json": timings.get("mm_special_tokens_json"),
                "proc_input_ids_len": timings.get("proc_input_ids_len"),
                "proc_attention_mask_len": timings.get("proc_attention_mask_len"),
                "llm_max_input_ids_len": seq_len_capture.max_input_ids_len if seq_len_capture.max_input_ids_len > 0 else None,
                "llm_max_inputs_embeds_len": seq_len_capture.max_inputs_embeds_len if seq_len_capture.max_inputs_embeds_len > 0 else None,
                "llm_max_attention_mask_len": seq_len_capture.max_attention_mask_len if seq_len_capture.max_attention_mask_len > 0 else None,
                "llm_prefill_forward_event_ms": prefill_event_capture.prefill_forward_ms if prefill_event_capture.prefill_forward_ms > 0 else None,
                "llm_prefill_forward_input_ids_len": prefill_event_capture.prefill_input_ids_len if prefill_event_capture.prefill_input_ids_len > 0 else None,
                "llm_prefill_forward_inputs_embeds_len": prefill_event_capture.prefill_inputs_embeds_len if prefill_event_capture.prefill_inputs_embeds_len > 0 else None,
                "llm_prefill_forward_attention_mask_len": prefill_event_capture.prefill_attention_mask_len if prefill_event_capture.prefill_attention_mask_len > 0 else None,
                "visual_encoder_ms": vis_time,
                "audio_encoder_ms": aud_time,
                "audio_encoder_event_ms": aud_time_event,
                "llm_prefill_ms": llm_prefill,
                "ttft_ms": timings["ttft_ms"],
                "total_ms": timings["total_ms"],
            }
            results.append(result)

            print(
                f"  预处理: 视频音频提取={timings.get('video_audio_extract_ms', 0):.0f}ms, "
                f"视频tokenize={timings.get('video_tokenize_ms', 0):.0f}ms, "
                f"音频FFT+Mel={timings.get('audio_feature_ms', 0):.0f}ms"
            )
            print(f"  Encoder: Visual={vis_time:.0f}ms, Audio={aud_time:.0f}ms, LLM Prefill={llm_prefill:.0f}ms")
            print(f"  TTFT: {timings['ttft_ms']:.0f}ms")

        except Exception as e:
            print(f"  ⚠️ 失败: {e}")

        records, _markers = monitor.stop()
        all_sample_records.append(records)
        monitor.cleanup()

        gc.collect()
        torch.cuda.empty_cache()

    # save
    print("\n📝 保存结果...")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    if results:
        results_df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    all_records_flat = [r for records in all_sample_records for r in records]
    if all_records_flat:
        pd.DataFrame(all_records_flat).to_csv(os.path.join(out_dir, "resource_records.csv"), index=False)

    if results:
        encoder_stats = _compute_encoder_stats_from_df(results_df)
        with open(os.path.join(out_dir, "encoder_stats.json"), "w") as f:
            json.dump(encoder_stats, f, indent=2)

        print("\n📊 绘制图表...")
        plot_encoder_breakdown(encoder_stats, os.path.join(out_dir, "encoder_breakdown.png"))
        plot_resource_usage_averaged(all_sample_records, os.path.join(out_dir, "resource_usage.png"))
        _save_prefill_analysis(results_df, out_dir, trim_quantile=args.trim_quantile)

    print(f"\n结果已保存至: {out_dir}")
    return results_df


# ============ 可视化 ============

def plot_encoder_breakdown(encoder_stats: Dict, output_path: str):
    """绘制 Encoder 耗时分解图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：各阶段耗时条形图
    names = ["Video+Audio\nExtract", "Video\nTokenize", "Audio FFT\n+Mel", "Visual\nEncoder", "Audio\nEncoder", "LLM\nPrefill"]
    means = [
        encoder_stats.get("video_audio_extract_mean_ms", 0),
        encoder_stats.get("video_tokenize_mean_ms", 0),
        encoder_stats.get("audio_feature_mean_ms", 0),
        encoder_stats["visual_mean_ms"],
        encoder_stats["audio_mean_ms"],
        encoder_stats.get("llm_prefill_mean_ms", 0),
    ]
    
    colors = ['#FF9800', '#FFC107', '#9C27B0', '#4CAF50', '#2196F3', '#607D8B']
    bars = axes[0].bar(names, means, color=colors, edgecolor='black')
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Stage Latency Breakdown")
    axes[0].tick_params(axis='x', rotation=0)
    
    for bar, mean in zip(bars, means):
        if mean > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        f'{mean:.0f}', ha='center', fontsize=9, fontweight='bold')
    
    # 右图：TTFT 分解饼图
    labels = ["Preprocess", "Visual Enc", "Audio Enc", "LLM Prefill"]
    values = [
        encoder_stats.get("preprocess_mean_ms", 0),
        encoder_stats["visual_mean_ms"],
        encoder_stats["audio_mean_ms"],
        encoder_stats.get("llm_prefill_mean_ms", 0),
    ]
    colors2 = ['#FFC107', '#4CAF50', '#2196F3', '#607D8B']
    
    # 过滤掉 0 值
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors2) if v > 0]
    if non_zero:
        labels, values, colors2 = zip(*non_zero)
        axes[1].pie(values, labels=labels, autopct='%1.1f%%', colors=colors2, startangle=90)
    axes[1].set_title("TTFT Breakdown")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Encoder 分解图已保存: {output_path}")


def plot_resource_usage_averaged(all_sample_records: List[List[Dict]], output_path: str, n_bins: int = 100):
    """绘制平均资源利用率图（对所有样本平均）"""
    if not all_sample_records:
        print("⚠️ 无资源记录")
        return
    
    # 将每个样本的时间归一化到 [0, 1]，然后平均
    gpu_samples = []
    vram_samples = []
    cpu_samples = []
    
    for records in all_sample_records:
        if not records:
            continue
        df = pd.DataFrame(records)
        if len(df) < 2:
            continue
        
        # 归一化时间到 [0, 1]
        t_min, t_max = df["time"].min(), df["time"].max()
        if t_max - t_min < 0.01:
            continue
        df["norm_time"] = (df["time"] - t_min) / (t_max - t_min)
        
        # 分 bin 平均
        df["bin"] = (df["norm_time"] * n_bins).astype(int).clip(0, n_bins - 1)
        binned = df.groupby("bin").mean()
        
        if "gpu_percent" in binned.columns:
            gpu_samples.append(binned["gpu_percent"].values)
        if "vram_used_gb" in binned.columns:
            vram_samples.append(binned["vram_used_gb"].values)
        cpu_samples.append(binned["cpu_percent"].values)
    
    if not cpu_samples:
        print("⚠️ 无有效资源记录")
        return
    
    # 对齐长度并平均
    def pad_and_average(samples, n_bins):
        padded = []
        for s in samples:
            if len(s) < n_bins:
                s = np.pad(s, (0, n_bins - len(s)), mode='edge')
            padded.append(s[:n_bins])
        return np.mean(padded, axis=0), np.std(padded, axis=0)
    
    x = np.linspace(0, 100, n_bins)  # 百分比时间
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # GPU 利用率
    if gpu_samples:
        gpu_mean, gpu_std = pad_and_average(gpu_samples, n_bins)
        axes[0].plot(x, gpu_mean, 'b-', linewidth=1.5, label='Mean')
        axes[0].fill_between(x, gpu_mean - gpu_std, gpu_mean + gpu_std, alpha=0.3, color='blue')
        axes[0].set_ylabel("GPU %")
        axes[0].set_ylim(0, 100)
        axes[0].set_title(f"GPU Utilization (avg of {len(gpu_samples)} samples)")
        axes[0].legend()
    
    # VRAM
    if vram_samples:
        vram_mean, vram_std = pad_and_average(vram_samples, n_bins)
        axes[1].plot(x, vram_mean, 'g-', linewidth=1.5, label='Mean')
        axes[1].fill_between(x, vram_mean - vram_std, vram_mean + vram_std, alpha=0.3, color='green')
        axes[1].set_ylabel("VRAM (GB)")
        axes[1].set_title(f"VRAM Usage (avg of {len(vram_samples)} samples)")
        axes[1].legend()
    
    # CPU
    cpu_mean, cpu_std = pad_and_average(cpu_samples, n_bins)
    axes[2].plot(x, cpu_mean, 'r-', linewidth=1.5, label='Mean')
    axes[2].fill_between(x, cpu_mean - cpu_std, cpu_mean + cpu_std, alpha=0.3, color='red')
    axes[2].set_ylabel("CPU %")
    axes[2].set_xlabel("Normalized Time (%)")
    axes[2].set_title(f"CPU Utilization (avg of {len(cpu_samples)} samples)")
    axes[2].legend()
    
    # 添加阶段标注
    for ax in axes:
        ax.axvline(x=15, color='orange', linestyle='--', alpha=0.7, label='~Preprocess End')
        ax.axvline(x=50, color='purple', linestyle='--', alpha=0.7, label='~Encode')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 平均资源利用率图已保存: {output_path}")


def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _compute_encoder_stats_from_df(df: pd.DataFrame) -> Dict:
    def col_mean(c: str) -> float:
        s = _safe_numeric_series(df, c).dropna()
        return float(s.mean()) if len(s) else 0.0

    def col_std(c: str) -> float:
        s = _safe_numeric_series(df, c).dropna()
        return float(s.std()) if len(s) else 0.0

    audio_col = "audio_encoder_event_ms" if col_mean("audio_encoder_event_ms") > 0 else "audio_encoder_ms"

    return {
        "video_audio_extract_mean_ms": col_mean("video_audio_extract_ms"),
        "video_tokenize_mean_ms": col_mean("video_tokenize_ms"),
        "audio_feature_mean_ms": col_mean("audio_feature_ms"),
        "preprocess_mean_ms": col_mean("preprocess_ms"),
        "visual_mean_ms": col_mean("visual_encoder_ms"),
        "visual_std_ms": col_std("visual_encoder_ms"),
        "audio_mean_ms": col_mean(audio_col),
        "audio_std_ms": col_std(audio_col),
        "llm_prefill_mean_ms": col_mean("llm_prefill_ms"),
        "ttft_mean_ms": col_mean("ttft_ms"),
        "total_mean_ms": col_mean("total_ms"),
        "audio_mean_source": audio_col,
    }


def _save_prefill_analysis(df: pd.DataFrame, out_dir: str, trim_quantile: float = 0.99):
    os.makedirs(out_dir, exist_ok=True)

    base_cols = [
        "llm_prefill_ms",
        "llm_prefill_forward_event_ms",
        "llm_prefill_forward_inputs_embeds_len",
        "llm_prefill_forward_attention_mask_len",
        "mm_vision_slot_tokens",
        "mm_audio_slot_tokens",
        "visual_encoder_ms",
        "audio_encoder_ms",
        "audio_encoder_event_ms",
        "preprocess_ms",
        "video_audio_extract_ms",
        "video_tokenize_ms",
        "audio_feature_ms",
        "ttft_ms",
        "total_ms",
        "mel_frames",
        "audio_tower_in_frames",
        "proc_input_ids_len",
        "proc_attention_mask_len",
        "llm_max_input_ids_len",
        "llm_max_inputs_embeds_len",
        "llm_max_attention_mask_len",
    ]
    cols = [c for c in base_cols if c in df.columns]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")

    if "llm_prefill_ms" not in numeric.columns:
        return

    prefill = numeric["llm_prefill_ms"].dropna()
    if len(prefill) < 2:
        return

    q = float(prefill.quantile(trim_quantile))
    trimmed_mask = numeric["llm_prefill_ms"].le(q)

    def describe_series(s: pd.Series) -> Dict:
        s = s.dropna()
        if len(s) == 0:
            return {"n": 0}
        return {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p10": float(s.quantile(0.10)),
            "p90": float(s.quantile(0.90)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }

    summary = {
        "trim_quantile": trim_quantile,
        "trim_threshold_llm_prefill_ms": q,
        "llm_prefill_full": describe_series(prefill),
        "llm_prefill_trimmed": describe_series(numeric.loc[trimmed_mask, "llm_prefill_ms"]),
    }
    with open(os.path.join(out_dir, "prefill_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)

    corr = numeric.corr(method="pearson")
    corr.to_csv(os.path.join(out_dir, "correlations.csv"))

    corr_prefill = corr.loc[["llm_prefill_ms"], :].T.reset_index()
    corr_prefill.columns = ["metric", "pearson_r"]
    corr_prefill = corr_prefill.sort_values("pearson_r", ascending=False)
    corr_prefill.to_csv(os.path.join(out_dir, "prefill_correlations.csv"), index=False)

    if "sample_id" in df.columns:
        outliers = df[["sample_id"]].copy()
        outliers["llm_prefill_ms"] = numeric["llm_prefill_ms"].values
        outliers = outliers.dropna().sort_values("llm_prefill_ms", ascending=False).head(10)
        outliers.to_csv(os.path.join(out_dir, "prefill_outliers_top10.csv"), index=False)

    def scatter(x_col: str, y_col: str, fname: str):
        if x_col not in numeric.columns or y_col not in numeric.columns:
            return
        x = numeric[x_col]
        y = numeric[y_col]
        mask = x.notna() & y.notna()
        if mask.sum() < 2:
            return
        plt.figure(figsize=(6, 4))
        plt.scatter(x[mask], y[mask], s=18, alpha=0.75)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()

    scatter("visual_encoder_ms", "llm_prefill_ms", "prefill_vs_visual.png")
    scatter("preprocess_ms", "llm_prefill_ms", "prefill_vs_preprocess.png")
    scatter("video_audio_extract_ms", "llm_prefill_ms", "prefill_vs_video_audio_extract.png")
    scatter("ttft_ms", "llm_prefill_ms", "prefill_vs_ttft.png")
    scatter("total_ms", "llm_prefill_ms", "prefill_vs_total.png")

    plt.figure(figsize=(6, 4))
    plt.hist(prefill.values, bins=30, alpha=0.85)
    plt.xlabel("llm_prefill_ms")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prefill_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="Video + Audio 多模态编码耗时分析")
    parser.add_argument("--model", default="/root/autodl-tmp/Qwen2.5-Omni-7B", help="模型路径")
    parser.add_argument("--data", default="/root/autodl-tmp/data/MSRVTT_subset/manifest.csv", help="数据 manifest")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/results/exp7", help="输出目录")
    parser.add_argument("--n-samples", type=int, default=20, help="测试样本数")
    parser.add_argument("--warmup", type=int, default=2, help="预热次数")
    parser.add_argument("--audio-max-seconds", type=float, default=30.0, help="音频特征提取时的最大时长（秒），将 padding/truncation 到该长度")
    parser.add_argument("--question-override", type=str, default=None, help="覆盖 manifest 中 question，对所有样本使用同一个问题")
    parser.add_argument("--prompt-filler-words", type=int, default=0, help="在问题末尾追加 filler_word 的重复次数，用于拉大输入 token 范围")
    parser.add_argument("--prompt-filler-word", type=str, default="hello", help="追加用的 filler word")
    parser.add_argument("--video-nframes", type=int, default=None, help="视频抽帧数（优先级高于 fps 采样），必须是 2 的倍数或会被 round")
    parser.add_argument("--video-fps", type=float, default=None, help="视频按 fps 采样（与 min/max_frames 配合）")
    parser.add_argument("--video-min-frames", type=int, default=None, help="fps 采样时的最小帧数")
    parser.add_argument("--video-max-frames", type=int, default=None, help="fps 采样时的最大帧数")
    parser.add_argument("--use-audio-in-video", type=int, default=1, choices=[0, 1], help="是否从视频中提取音频（1/0）")
    parser.add_argument("--sweep-nframes", type=str, default=None, help="逗号分隔的 nframes 列表；启用后会在 out-dir 下逐档跑并汇总出图")
    parser.add_argument("--analysis-only", action="store_true", help="只对 out-dir 下已有 results.csv 做统计与绘图，不重新跑模型")
    parser.add_argument("--trim-quantile", type=float, default=0.99, help="prefill 统计去长尾分位数阈值（例如 0.99 表示去掉 top1%）")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 70)
    print("🎬 实验7：Video + Audio 多模态编码耗时分析")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"样本数: {args.n_samples}")
    print(f"音频 max seconds: {args.audio_max_seconds}")
    print(f"use_audio_in_video: {bool(args.use_audio_in_video)}")
    if args.sweep_nframes:
        print(f"sweep_nframes: {args.sweep_nframes}")
    else:
        if args.video_nframes is not None:
            print(f"video_nframes: {args.video_nframes}")
        if args.video_fps is not None:
            print(f"video_fps: {args.video_fps}")

    if args.analysis_only:
        results_csv = os.path.join(args.out_dir, "results.csv")
        if not os.path.exists(results_csv):
            print(f"⚠️ results.csv 不存在: {results_csv}")
            return
        df = pd.read_csv(results_csv)
        encoder_stats = _compute_encoder_stats_from_df(df)
        with open(os.path.join(args.out_dir, "encoder_stats.json"), "w") as f:
            json.dump(encoder_stats, f, indent=2)
        print("\n📊 绘制图表...")
        plot_encoder_breakdown(encoder_stats, os.path.join(args.out_dir, "encoder_breakdown.png"))
        _save_prefill_analysis(df, args.out_dir, trim_quantile=args.trim_quantile)
        print(f"\n结果已保存至: {args.out_dir}")
        return
    
    # 加载模型
    print("\n🔄 加载模型...")
    model, proc = C.load_qwen25_omni(args.model, "bf16")
    
    # 创建 WhisperFeatureExtractor（用于音频 FFT + Mel 频谱）
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    print("✅ 已加载 WhisperFeatureExtractor (音频 FFT + Mel)")
    
    # 注册 Encoder 计时 Hook
    encoder_timer = EncoderTimer()
    encoder_timer.register(model)
    print("✅ 已注册 Visual/Audio Encoder 计时 Hook")

    audio_event_timer = ModuleCudaEventTimer()
    audio_event_timer.register(model.thinker.audio_tower)
    
    seq_len_capture = LLMSeqLenCapture()
    seq_len_capture.register(model)
    print("✅ 已注册 LLM 序列长度捕获 Hook")

    prefill_event_capture = LLMPrefillCudaEventCapture()
    prefill_event_capture.register(model.thinker.model)
    print("✅ 已注册 LLM Prefill CUDA Event 计时 Hook")
    
    # 加载数据
    print("\n🔄 加载数据...")
    if not os.path.exists(args.data):
        print(f"⚠️ 数据文件不存在: {args.data}")
        return
    
    samples = load_dataset(args.data, args.n_samples + args.warmup)
    print(f"  加载了 {len(samples)} 个样本")
    
    use_audio_in_video = bool(args.use_audio_in_video)
    if args.sweep_nframes:
        nframes_list = _parse_int_list(args.sweep_nframes)
        if not nframes_list:
            print("⚠️ sweep-nframes 为空")
            return
        merged = []
        for nf in nframes_list:
            sub_out = os.path.join(args.out_dir, f"sweep_nframes_{int(nf)}")
            df = run_experiment(
                args,
                model,
                proc,
                fe,
                encoder_timer,
                audio_event_timer,
                seq_len_capture,
                prefill_event_capture,
                samples,
                sub_out,
                use_audio_in_video=use_audio_in_video,
                video_nframes=int(nf),
                video_fps=None,
                video_min_frames=None,
                video_max_frames=None,
            )
            if df is not None and len(df):
                merged.append(df)
        if merged:
            merged_df = pd.concat(merged, ignore_index=True)
            merged_df.to_csv(os.path.join(args.out_dir, "sweep_merged.csv"), index=False)
            _plot_nframes_scaling(merged_df, args.out_dir)
            print(f"\n结果已保存至: {args.out_dir}")
        else:
            print("\n⚠️ 没有有效结果")
            print(f"\n结果已保存至: {args.out_dir}")
    else:
        run_experiment(
            args,
            model,
            proc,
            fe,
            encoder_timer,
            audio_event_timer,
            seq_len_capture,
            prefill_event_capture,
            samples,
            args.out_dir,
            use_audio_in_video=use_audio_in_video,
            video_nframes=args.video_nframes,
            video_fps=args.video_fps,
            video_min_frames=args.video_min_frames,
            video_max_frames=args.video_max_frames,
        )
    
    encoder_timer.remove()
    seq_len_capture.remove()
    prefill_event_capture.remove()


if __name__ == "__main__":
    main()
