#!/usr/bin/env python3
"""
实验9：音频长度 Scaling 实验

目标：
测量不同音频长度（1s/3s/6s/10s 等）对 Audio Encoder 延迟的影响
验证假设：Audio Encoder 耗时与序列长度（接近）线性相关

基于 exp7 结构，聚焦于音频长度这一个变量
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Tuple

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

import common as C
import profiling_utils as P


# ============ Audio Encoder 计时 Hook ============
# Replaced by profiling_utils.ModuleCudaEventTimer but keeping alias for compatibility if needed, 
# or just using P.ModuleCudaEventTimer directly.
# The original AudioEncoderTimer is very similar to ModuleCudaEventTimer but specialized for Audio Tower.
# Let's use P.ModuleCudaEventTimer and adapt.

class AudioEncoderTimer(P.ModuleCudaEventTimer):
    """专门测量 Audio Encoder 耗时的 Hook (Wrapper around ModuleCudaEventTimer)"""
    def register(self, model):
        # Specific registration for audio tower
        super().register(model.thinker.audio_tower)

# ============ 音频处理 ============


# ============ 音频处理 ============

def truncate_audio(audio: np.ndarray, target_seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """截断音频到指定秒数"""
    target_samples = int(target_seconds * sample_rate)
    if len(audio) > target_samples:
        return audio[:target_samples]
    if len(audio) < target_samples:
        pad = np.zeros((target_samples - len(audio),), dtype=audio.dtype)
        return np.concatenate([audio, pad], axis=0)
    return audio


def get_audio_from_video(video_path: str) -> Tuple[np.ndarray, int]:
    """从视频中提取音频"""
    import subprocess
    import tempfile
    import soundfile as sf
    
    # 使用 ffmpeg 提取音频
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    
    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            temp_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        audio, sr = sf.read(temp_path)
        return audio.astype(np.float32), sr
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============ 推理函数 ============

def run_single_audio_test(
    model,
    proc,
    video_path: str,
    audio_seconds: float,
    fe,
    timer: AudioEncoderTimer,
) -> Dict:
    """对单个音频长度运行一次测试"""
    from qwen_omni_utils import process_mm_info
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # 1. 提取视频帧和音频
    conversation = [{"role": "user", "content": [
        {"type": "video", "video": video_path},
        {"type": "text", "text": "Describe what you see and hear."}
    ]}]
    
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    
    if not audios:
        raise ValueError("No audio extracted from video")
    
    # 2. 截断音频
    original_audio = audios[0]
    original_duration = len(original_audio) / 16000
    truncated_audio = truncate_audio(original_audio, audio_seconds)
    actual_duration = len(truncated_audio) / 16000
    
    # 3. 准备输入
    text = proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=text, videos=videos, return_tensors="pt", padding=True).to(model.device)
    
    # 4. 音频特征提取（FFT + Mel）
    t_fft_start = time.perf_counter()
    af = fe(
        truncated_audio,
        sampling_rate=16000,
        return_tensors='pt',
        padding='do_not_pad',
        truncation=False,
    )
    inputs['input_features'] = af['input_features'].to(model.device, dtype=torch.bfloat16)
    inputs['feature_attention_mask'] = torch.ones(
        (1, af['input_features'].shape[2]), device=model.device, dtype=torch.long
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=model.device)
    fft_mel_ms = (time.perf_counter() - t_fft_start) * 1000
    
    # 记录 mel 帧数
    mel_frames = af['input_features'].shape[2]
    
    # 5. 运行 generate（只生成 1 token 以测 TTFT）
    timer.clear()  # 清除之前的计时
    
    t_gen_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_audio=False,
        )
    if torch.cuda.is_available():
        devices = set()
        devices.add(model.device)
        try:
            devices.add(next(model.thinker.audio_tower.parameters()).device)
        except StopIteration:
            pass
        try:
            devices.add(next(model.thinker.visual.parameters()).device)
        except StopIteration:
            pass
        for d in devices:
            if isinstance(d, torch.device) and d.type == "cuda":
                torch.cuda.synchronize(device=d)
    ttft_ms = (time.perf_counter() - t_gen_start) * 1000
    
    audio_encoder_ms = timer.get_last()
    audio_tower_in_frames = None
    if timer.last_input_shape is not None and len(timer.last_input_shape) >= 2:
        audio_tower_in_frames = int(timer.last_input_shape[-1])
    audio_tower_input_shape = str(timer.last_input_shape) if timer.last_input_shape is not None else None
    
    return {
        "original_duration_s": original_duration,
        "target_duration_s": audio_seconds,
        "actual_duration_s": actual_duration,
        "mel_frames": mel_frames,
        "audio_tower_in_frames": audio_tower_in_frames,
        "audio_tower_input_shape": audio_tower_input_shape,
        "fft_mel_ms": fft_mel_ms,
        "audio_encoder_ms": audio_encoder_ms,
        "ttft_ms": ttft_ms,
    }


# ============ 可视化 ============

def plot_scaling_curve(results_df: pd.DataFrame, output_path: str):
    """绘制音频长度 vs encoder 延迟曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 按音频长度分组统计
    grouped = results_df.groupby("target_duration_s").agg({
        "audio_encoder_ms": ["mean", "std"],
        "fft_mel_ms": ["mean", "std"],
        "mel_frames": "mean",
        "ttft_ms": ["mean", "std"],
    }).reset_index()
    
    # 展平列名
    grouped.columns = [
        "duration_s",
        "encoder_mean", "encoder_std",
        "fft_mean", "fft_std",
        "mel_frames",
        "ttft_mean", "ttft_std",
    ]
    
    x = grouped["duration_s"].values
    
    # 图1：音频长度 vs Audio Encoder 延迟
    axes[0].errorbar(x, grouped["encoder_mean"], yerr=grouped["encoder_std"],
                     marker='o', capsize=5, linewidth=2, markersize=8, color='#2196F3')
    axes[0].set_xlabel("Audio Duration (seconds)", fontsize=12)
    axes[0].set_ylabel("Audio Encoder Time (ms)", fontsize=12)
    axes[0].set_title("Audio Duration vs Encoder Latency", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 添加线性拟合
    if len(x) >= 2:
        z = np.polyfit(x, grouped["encoder_mean"], 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(x), max(x), 100)
        axes[0].plot(x_fit, p(x_fit), '--', color='red', alpha=0.7, 
                     label=f'Linear fit: {z[0]:.1f}ms/s')
        axes[0].legend()
    
    # 图2：Mel 帧数 vs Encoder 延迟
    axes[1].scatter(grouped["mel_frames"], grouped["encoder_mean"],
                    s=100, c='#4CAF50', edgecolors='black', linewidths=1)
    axes[1].set_xlabel("Mel Frames (sequence length)", fontsize=12)
    axes[1].set_ylabel("Audio Encoder Time (ms)", fontsize=12)
    axes[1].set_title("Mel Frames vs Encoder Latency", fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # 添加标注
    for i, row in grouped.iterrows():
        axes[1].annotate(f'{row["duration_s"]:.0f}s',
                        (row["mel_frames"], row["encoder_mean"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    # 图3：各阶段时间堆叠条形图
    bar_width = 0.6
    bars1 = axes[2].bar(x, grouped["fft_mean"], bar_width, label='FFT+Mel', color='#FFC107')
    bars2 = axes[2].bar(x, grouped["encoder_mean"], bar_width, bottom=grouped["fft_mean"],
                        label='Audio Encoder', color='#2196F3')
    
    axes[2].set_xlabel("Audio Duration (seconds)", fontsize=12)
    axes[2].set_ylabel("Time (ms)", fontsize=12)
    axes[2].set_title("Audio Processing Time Breakdown", fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for i, (d, enc, fft) in enumerate(zip(x, grouped["encoder_mean"], grouped["fft_mean"])):
        total = enc + fft
        axes[2].text(d, total + 10, f'{total:.0f}ms', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Scaling 曲线图已保存: {output_path}")


def plot_encoder_vs_duration_detail(results_df: pd.DataFrame, output_path: str):
    """绘制详细的 encoder 延迟分析图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 散点图：每个样本一个点
    scatter = ax.scatter(
        results_df["actual_duration_s"],
        results_df["audio_encoder_ms"],
        c=results_df["target_duration_s"],
        cmap='viridis',
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
    )
    
    ax.set_xlabel("Actual Audio Duration (seconds)", fontsize=12)
    ax.set_ylabel("Audio Encoder Time (ms)", fontsize=12)
    ax.set_title("Audio Encoder Latency vs Duration (All Samples)", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label("Target Duration (s)")
    
    # 线性拟合
    x = results_df["actual_duration_s"].values
    y = results_df["audio_encoder_ms"].values
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(x), max(x), 100)
        ax.plot(x_fit, p(x_fit), '--', color='red', linewidth=2,
                label=f'Linear: y = {z[0]:.1f}x + {z[1]:.1f}')
        
        # R² 值
        y_pred = p(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        ax.legend(title=f'R² = {r2:.3f}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 详细分析图已保存: {output_path}")


def plot_encoder_vs_mel_frames_detail(results_df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        results_df["mel_frames"],
        results_df["audio_encoder_ms"],
        c=results_df["target_duration_s"],
        cmap='viridis',
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
    )

    ax.set_xlabel("Mel Frames (sequence length)", fontsize=12)
    ax.set_ylabel("Audio Encoder Time (ms)", fontsize=12)
    ax.set_title("Audio Encoder Latency vs Mel Frames (All Samples)", fontsize=14)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Target Duration (s)")

    x = results_df["mel_frames"].values
    y = results_df["audio_encoder_ms"].values
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(x_fit, p(x_fit), '--', color='red', linewidth=2,
                label=f'Linear: y = {z[0]:.6f}x + {z[1]:.1f}')
        y_pred = p(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        ax.legend(title=f'R² = {r2:.3f}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Mel Frames 详细分析图已保存: {output_path}")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="音频长度 Scaling 实验")
    parser.add_argument("--model", default="/root/autodl-tmp/Qwen2.5-Omni-7B", help="模型路径")
    parser.add_argument("--data", default="/root/autodl-tmp/data/MSRVTT_subset/manifest.csv", help="数据 manifest")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/results/exp9", help="输出目录")
    parser.add_argument("--audio-lengths", default="1,3,10,30,60,120,180,240,300", help="要测试的音频长度（秒），逗号分隔")
    parser.add_argument("--n-samples", type=int, default=5, help="每个长度测试的样本数")
    parser.add_argument("--warmup", type=int, default=2, help="预热次数")
    args = parser.parse_args()
    
    # 解析音频长度
    audio_lengths = [float(x.strip()) for x in args.audio_lengths.split(",")]
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 70)
    print("🔊 实验9：音频长度 Scaling 实验")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"测试音频长度: {audio_lengths} 秒")
    print(f"每个长度样本数: {args.n_samples}")
    
    # 加载模型
    print("\n🔄 加载模型...")
    model, proc = C.load_qwen25_omni(args.model, "bf16")
    
    # 创建 WhisperFeatureExtractor
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    print("✅ 已加载 WhisperFeatureExtractor")
    
    # 注册 Audio Encoder 计时 Hook
    timer = AudioEncoderTimer()
    timer.register(model)
    print("✅ 已注册 Audio Encoder 计时 Hook")

    try:
        audio_dev = next(model.thinker.audio_tower.parameters()).device
    except StopIteration:
        audio_dev = None
    try:
        visual_dev = next(model.thinker.visual.parameters()).device
    except StopIteration:
        visual_dev = None
    print(f"Device placement: model={model.device}, audio_tower={audio_dev}, visual={visual_dev}")
    
    # 加载数据
    print("\n🔄 加载数据...")
    if not os.path.exists(args.data):
        print(f"⚠️ 数据文件不存在: {args.data}")
        return
    
    df = pd.read_csv(args.data)
    video_paths = [p for p in df["video_path"].tolist() if os.path.exists(p)]
    print(f"  找到 {len(video_paths)} 个有效视频")
    
    if len(video_paths) < args.n_samples + args.warmup:
        print(f"⚠️ 视频数量不足，需要至少 {args.n_samples + args.warmup} 个")
        return
    
    # 预热
    print(f"\n🔥 预热 ({args.warmup} 次)...")
    for i in range(args.warmup):
        try:
            run_single_audio_test(model, proc, video_paths[i], 3.0, fe, timer)
            print(f"  预热 {i+1}/{args.warmup} 完成")
        except Exception as e:
            print(f"  预热 {i+1} 失败: {e}")
    
    timer.clear()
    gc.collect()
    torch.cuda.empty_cache()
    
    # 正式测试
    print(f"\n🧪 开始 Scaling 测试...")
    results = []
    test_videos = video_paths[args.warmup:args.warmup + args.n_samples]
    
    total_tests = len(audio_lengths) * len(test_videos)
    test_count = 0
    
    for audio_len in audio_lengths:
        print(f"\n--- 测试音频长度: {audio_len} 秒 ---")
        
        for i, video_path in enumerate(test_videos):
            test_count += 1
            print(f"  [{test_count}/{total_tests}] 样本 {i+1}/{len(test_videos)}", end=" ")
            
            try:
                result = run_single_audio_test(model, proc, video_path, audio_len, fe, timer)
                result["video_path"] = os.path.basename(video_path)
                results.append(result)
                
                print(f"✓ mel={result['mel_frames']}, encoder={result['audio_encoder_ms']:.0f}ms")
                
            except Exception as e:
                print(f"✗ 失败: {e}")
            
            gc.collect()
            torch.cuda.empty_cache()
    
    # 保存结果
    print("\n📝 保存结果...")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.out_dir, "scaling_results.csv"), index=False)
    
    with open(os.path.join(args.out_dir, "scaling_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # 统计与可视化
    if results:
        print("\n📊 绘制图表...")
        plot_scaling_curve(results_df, os.path.join(args.out_dir, "scaling_curve.png"))
        plot_encoder_vs_duration_detail(results_df, os.path.join(args.out_dir, "encoder_detail.png"))
        plot_encoder_vs_mel_frames_detail(results_df, os.path.join(args.out_dir, "mel_frames_detail.png"))
        
        # 打印统计表格
        print("\n" + "=" * 70)
        print("📊 Scaling 实验结果")
        print("=" * 70)
        
        agg_cfg = {
            "mel_frames": "mean",
            "fft_mel_ms": ["mean", "std"],
            "audio_encoder_ms": ["mean", "std"],
            "ttft_ms": ["mean", "std"],
        }
        if "audio_tower_in_frames" in results_df.columns:
            agg_cfg["audio_tower_in_frames"] = "mean"
        grouped = results_df.groupby("target_duration_s").agg(agg_cfg).reset_index()
        
        grouped.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                           for col in grouped.columns.values]
        
        has_audio_tower_frames = "audio_tower_in_frames_mean" in grouped.columns
        if has_audio_tower_frames:
            print(f"\n{'Duration(s)':<12} {'Mel Frames':<12} {'Tower Frames':<12} {'FFT+Mel(ms)':<15} {'Encoder(ms)':<18} {'TTFT(ms)':<15}")
            print("-" * 86)
        else:
            print(f"\n{'Duration(s)':<12} {'Mel Frames':<12} {'FFT+Mel(ms)':<15} {'Encoder(ms)':<18} {'TTFT(ms)':<15}")
            print("-" * 72)
        
        for _, row in grouped.iterrows():
            dur = row["target_duration_s"]
            mel = row["mel_frames_mean"]
            tower_frames = row["audio_tower_in_frames_mean"] if has_audio_tower_frames else None
            fft_mean = row["fft_mel_ms_mean"]
            fft_std = row["fft_mel_ms_std"]
            enc_mean = row["audio_encoder_ms_mean"]
            enc_std = row["audio_encoder_ms_std"]
            ttft_mean = row["ttft_ms_mean"]
            ttft_std = row["ttft_ms_std"]
            
            if has_audio_tower_frames:
                tf = float(tower_frames) if tower_frames is not None and not pd.isna(tower_frames) else float('nan')
                print(f"{dur:<12.0f} {mel:<12.0f} {tf:<12.0f} {fft_mean:>6.1f}±{fft_std:<6.1f} {enc_mean:>6.1f}±{enc_std:<9.1f} {ttft_mean:>6.1f}±{ttft_std:<6.1f}")
            else:
                print(f"{dur:<12.0f} {mel:<12.0f} {fft_mean:>6.1f}±{fft_std:<6.1f} {enc_mean:>6.1f}±{enc_std:<9.1f} {ttft_mean:>6.1f}±{ttft_std:<6.1f}")
        
        # 计算 scaling 系数
        x = results_df["actual_duration_s"].values
        y = results_df["audio_encoder_ms"].values
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            print(f"\n📈 Scaling 分析:")
            print(f"  线性拟合: encoder_ms = {z[0]:.1f} × duration_s + {z[1]:.1f}")
            print(f"  每增加 1 秒音频，encoder 延迟增加约 {z[0]:.1f} ms")
            
            # R² 值
            p = np.poly1d(z)
            y_pred = p(x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            print(f"  R² = {r2:.4f} (越接近 1 表示线性关系越强)")

        x_mel = results_df["mel_frames"].values
        if len(x_mel) >= 2:
            z_mel = np.polyfit(x_mel, y, 1)
            print(f"\n📈 Scaling (mel_frames) 分析:")
            print(f"  线性拟合: encoder_ms = {z_mel[0]:.6f} × mel_frames + {z_mel[1]:.1f}")
            print(f"  每增加 1000 mel frames，encoder 延迟增加约 {z_mel[0] * 1000:.1f} ms")
            p_mel = np.poly1d(z_mel)
            y_pred_mel = p_mel(x_mel)
            ss_res_mel = np.sum((y - y_pred_mel) ** 2)
            ss_tot_mel = np.sum((y - np.mean(y)) ** 2)
            r2_mel = 1 - (ss_res_mel / ss_tot_mel) if ss_tot_mel > 0 else 0
            print(f"  R² = {r2_mel:.4f} (越接近 1 表示线性关系越强)")
    
    else:
        print("\n⚠️ 没有有效结果")
    
    print(f"\n结果已保存至: {args.out_dir}")
    timer.remove()


if __name__ == "__main__":
    main()
