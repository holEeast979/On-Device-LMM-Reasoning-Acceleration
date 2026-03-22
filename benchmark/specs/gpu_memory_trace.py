"""
Spec: gpu-memory-trace
GPU 显存占用连续监测实验

目标：画出 GPU 显存随时间变化的连续曲线，并标注各阶段边界
用于证明阶段间存在内存释放→重新申请的开销

使用方法:
    python benchmark/run.py gpu-memory-trace \
        --model-dir /path/to/Qwen2.5-Omni-7B \
        --video /path/to/video.mp4 \
        --out-dir ./results
"""
from __future__ import annotations

import argparse
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmark.runner import BenchmarkRunner
from utils import common as C
from utils import profiling_utils as P


SPEC_NAME = "gpu-memory-trace"


class MemorySampler:
    """后台线程高频采样 GPU 显存"""
    
    def __init__(self, interval_ms: float = 5.0, device: int = 0):
        self.interval_ms = interval_ms
        self.device = device
        self.samples: List[Dict[str, float]] = []
        self.stage_markers: List[Dict[str, Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0
        self._lock = threading.Lock()
    
    def start(self):
        """开始采样"""
        self._running = True
        self._start_time = time.perf_counter()
        self.samples = []
        self.stage_markers = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """停止采样"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def mark_stage(self, stage_name: str):
        """标记阶段边界"""
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        with self._lock:
            self.stage_markers.append({
                "timestamp_ms": elapsed_ms,
                "stage": stage_name
            })
    
    def _sample_loop(self):
        """采样循环"""
        while self._running:
            elapsed_ms = (time.perf_counter() - self._start_time) * 1000
            
            # 采集显存数据
            allocated_mb = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved_mb = torch.cuda.memory_reserved(self.device) / 1024**2
            max_allocated_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
            
            with self._lock:
                self.samples.append({
                    "timestamp_ms": elapsed_ms,
                    "allocated_mb": allocated_mb,
                    "reserved_mb": reserved_mb,
                    "max_allocated_mb": max_allocated_mb
                })
            
            time.sleep(self.interval_ms / 1000)
    
    def get_dataframe(self) -> pd.DataFrame:
        """获取采样数据"""
        with self._lock:
            return pd.DataFrame(self.samples)
    
    def get_markers(self) -> List[Dict[str, Any]]:
        """获取阶段标记"""
        with self._lock:
            return list(self.stage_markers)


def register_subcommand(subparsers, common_parser) -> None:
    p = subparsers.add_parser(
        SPEC_NAME,
        parents=[common_parser],
        help="GPU memory trace with continuous sampling"
    )
    p.add_argument("--video", type=str, required=True, help="Path to video file")
    p.add_argument("--sample-interval-ms", type=float, default=5.0, help="Sampling interval in milliseconds")
    p.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens to generate")
    p.set_defaults(_spec_run=run)


def run_with_memory_trace(
    model,
    processor,
    feature_extractor,
    video_path: str,
    question: str,
    sampler: MemorySampler,
    max_new_tokens: int = 50,
    audio_max_seconds: float = 30.0
) -> Dict[str, Any]:
    """带内存追踪的推理（使用统一计时逻辑）
    
    阶段划分：
    - preprocess: 视频解码 + 音频提取 + tokenize
    - visual_encode: Visual Encoder (CUDA Event 计时)
    - audio_encode: Audio Encoder (CUDA Event 计时)
    - prefill: Embedding Merge + LLM Forward (CUDA Event 计时)
    - decode: 后续 token 生成
    """
    from qwen_omni_utils import process_mm_info
    
    device = model.device
    
    # 重置显存统计
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    
    # 开始采样
    sampler.start()
    sampler.mark_stage("start")
    
    # Stage 1: Preprocess (视频解码 + 音频提取)
    sampler.mark_stage("preprocess_start")
    
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question}
        ]}
    ]
    
    # 使用 process_mm_info 处理视频和音频
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    
    # 构建文本输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 处理基础输入（不含音频特征）
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt"
    )
    
    # 音频特征提取
    mel_frames = 0
    if audios and len(audios) > 0:
        audio = audios[0]
        # 截断/填充音频到目标长度
        target_samples = int(audio_max_seconds * 16000)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            pad = np.zeros((target_samples - len(audio),), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=0)
        
        # 提取音频特征
        audio_features = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=target_samples,
            truncation=True,
        )
        
        # 计算 mel_frames
        mel_frames = audio_features["input_features"].shape[-1]
        
        # 将音频特征添加到 inputs
        inputs["input_features"] = audio_features["input_features"]
        inputs["feature_attention_mask"] = torch.ones((1, mel_frames), dtype=torch.long)
    
    # Move to GPU
    inputs = inputs.to(device)
    torch.cuda.synchronize(device)
    sampler.mark_stage("preprocess_end")
    
    # ========== 使用统一计时逻辑 ==========
    # 注册 CUDA Event Timer 来精确测量各阶段
    visual_timer = P.ModuleCudaEventTimer()
    visual_timer.register(model.thinker.visual)
    
    audio_timer = P.ModuleCudaEventTimer()
    audio_timer.register(model.thinker.audio_tower)
    
    # ThinkerPrefillCapture 测量完整的 Thinker.forward 时间
    thinker_capture = P.ThinkerPrefillCapture()
    thinker_capture.register(model.thinker)
    
    # 用于标记阶段边界的 hook
    visual_encoder = model.thinker.visual
    audio_encoder = model.thinker.audio_tower
    llm_model = model.thinker.model  # LLM 部分
    
    visual_started = [False]
    visual_ended = [False]
    audio_started = [False]
    audio_ended = [False]
    prefill_started = [False]
    prefill_ended = [False]
    
    def visual_pre_hook(module, input):
        if not visual_started[0]:
            torch.cuda.synchronize(device)
            sampler.mark_stage("visual_encode_start")
            visual_started[0] = True
    
    def visual_post_hook(module, input, output):
        if not visual_ended[0]:
            torch.cuda.synchronize(device)
            sampler.mark_stage("visual_encode_end")
            visual_ended[0] = True
    
    def audio_pre_hook(module, input):
        if not audio_started[0]:
            torch.cuda.synchronize(device)
            sampler.mark_stage("audio_encode_start")
            audio_started[0] = True
    
    def audio_post_hook(module, input, output):
        if not audio_ended[0]:
            torch.cuda.synchronize(device)
            sampler.mark_stage("audio_encode_end")
            audio_ended[0] = True
    
    def llm_pre_hook(module, input):
        if not prefill_started[0]:
            torch.cuda.synchronize(device)
            sampler.mark_stage("prefill_start")
            prefill_started[0] = True
    
    def llm_post_hook(module, input, output):
        # 只标记第一次（prefill），后续是 decode
        if not prefill_ended[0]:
            torch.cuda.synchronize(device)
            sampler.mark_stage("prefill_end")
            prefill_ended[0] = True
    
    # 注册 hooks
    h1 = visual_encoder.register_forward_pre_hook(visual_pre_hook)
    h2 = visual_encoder.register_forward_hook(visual_post_hook)
    h3 = audio_encoder.register_forward_pre_hook(audio_pre_hook)
    h4 = audio_encoder.register_forward_hook(audio_post_hook)
    h5 = llm_model.register_forward_pre_hook(llm_pre_hook)
    h6 = llm_model.register_forward_hook(llm_post_hook)
    
    sampler.mark_stage("generate_start")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_audio=False
        )
    
    torch.cuda.synchronize(device)
    sampler.mark_stage("generate_end")
    
    # 移除 hooks
    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    h5.remove()
    h6.remove()
    
    # 获取精确计时（CUDA Event）
    visual_encoder_ms = sum(visual_timer.times) if visual_timer.times else 0
    audio_encoder_ms = sum(audio_timer.times) if audio_timer.times else 0
    thinker_forward_ms = thinker_capture.prefill_forward_ms
    
    # 计算 prefill 和 others（使用统一计时逻辑）
    # prefill = Embedding Merge + LLM Forward
    prefill_ms = thinker_forward_ms - visual_encoder_ms - audio_encoder_ms
    
    # 清理计时器
    visual_timer.remove()
    audio_timer.remove()
    thinker_capture.remove()
    
    sampler.mark_stage("end")
    sampler.stop()
    
    # 获取生成的文本
    generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return {
        "input_tokens": inputs['input_ids'].shape[1],
        "output_tokens": len(generated_ids[0]),
        "generated_text": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
        "mel_frames": mel_frames,
        # 精确计时结果（CUDA Event）
        "visual_encoder_ms": visual_encoder_ms,
        "audio_encoder_ms": audio_encoder_ms,
        "prefill_ms": prefill_ms,
        "thinker_forward_ms": thinker_forward_ms,
    }


def plot_memory_trace(
    samples_df: pd.DataFrame,
    markers: List[Dict[str, Any]],
    output_path: str,
    title: str = "GPU Memory Trace"
):
    """绘制内存变化曲线"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制内存曲线
    ax.plot(samples_df['timestamp_ms'], samples_df['allocated_mb'], 
            label='Allocated', color='blue', linewidth=1.5)
    ax.plot(samples_df['timestamp_ms'], samples_df['reserved_mb'], 
            label='Reserved', color='orange', linewidth=1.0, alpha=0.7)
    
    # 添加阶段标记
    stage_colors = {
        "preprocess": "#E8E8E8",
        "to_device": "#D0D0D0",
        "visual_encode": "#FFB3B3",
        "audio_encode": "#B3FFB3",
        "prefill": "#B3B3FF",
        "generate": "#E0E0FF"
    }
    
    # 找出阶段区间
    stage_regions = []
    for i, marker in enumerate(markers):
        if marker['stage'].endswith('_start'):
            stage_name = marker['stage'].replace('_start', '')
            # 找对应的 end
            for j in range(i+1, len(markers)):
                if markers[j]['stage'] == f"{stage_name}_end":
                    stage_regions.append({
                        'stage': stage_name,
                        'start': marker['timestamp_ms'],
                        'end': markers[j]['timestamp_ms']
                    })
                    break
    
    # 绘制阶段背景（不在这里添加 legend）
    for region in stage_regions:
        color = stage_colors.get(region['stage'], '#F0F0F0')
        ax.axvspan(region['start'], region['end'], alpha=0.3, color=color)
    
    # 添加阶段标签（在阶段区域顶部显示）
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    for idx, region in enumerate(stage_regions):
        mid = (region['start'] + region['end']) / 2
        duration = region['end'] - region['start']
        # 标签位置在图表顶部 90% 处，稍微错开避免重叠
        y_pos = y_max - y_range * 0.05 - (idx % 3) * y_range * 0.08
        
        ax.annotate(
            f"{region['stage']}\n({duration:.0f}ms)",
            xy=(mid, y_pos),
            fontsize=8,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray')
        )
    
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Memory (MB)', fontsize=10)
    ax.set_title(title, fontsize=12)
    
    # 自定义图例
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=1.5, label='Allocated'),
        plt.Line2D([0], [0], color='orange', linewidth=1.0, alpha=0.7, label='Reserved'),
    ]
    for stage, color in stage_colors.items():
        legend_elements.append(Patch(facecolor=color, alpha=0.3, label=stage))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_path}")


def run(args: argparse.Namespace, runner: BenchmarkRunner = None) -> None:
    """运行 GPU 内存追踪实验"""
    print("="*60)
    print(f"GPU Memory Trace Experiment")
    print("="*60)
    
    # 创建输出目录
    out_dir = os.path.join(args.out_dir, SPEC_NAME)
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载模型
    print(f"\nLoading model from {args.model_dir}...")
    runner = BenchmarkRunner(model_dir=args.model_dir, dtype=args.dtype)
    runner.load()
    model = runner._model
    processor = runner._proc
    feature_extractor = runner._fe
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Video: {args.video}")
    print(f"Sample interval: {args.sample_interval_ms}ms")
    
    # 创建采样器
    sampler = MemorySampler(interval_ms=args.sample_interval_ms)
    
    # 运行带内存追踪的推理
    question = args.question if args.question else "Describe what you see in the video."
    print(f"\nRunning inference with memory trace...")
    result = run_with_memory_trace(
        model=model,
        processor=processor,
        feature_extractor=feature_extractor,
        video_path=args.video,
        question=question,
        sampler=sampler,
        max_new_tokens=args.max_new_tokens
    )
    
    print(f"\nResult:")
    print(f"  Input tokens: {result['input_tokens']}")
    print(f"  Output tokens: {result['output_tokens']}")
    print(f"  Generated: {result['generated_text']}")
    
    # 输出精确计时结果（CUDA Event）
    print(f"\n--- Precise Timing (CUDA Event) ---")
    print(f"  Visual Encoder: {result['visual_encoder_ms']:.2f} ms")
    print(f"  Audio Encoder:  {result['audio_encoder_ms']:.2f} ms")
    print(f"  Prefill:        {result['prefill_ms']:.2f} ms (Embedding Merge + LLM Forward)")
    print(f"  Thinker Total:  {result['thinker_forward_ms']:.2f} ms")
    
    # 获取数据
    samples_df = sampler.get_dataframe()
    markers = sampler.get_markers()
    
    print(f"\nSamples collected: {len(samples_df)}")
    print(f"Stage markers: {len(markers)}")
    
    # 保存数据
    samples_csv = os.path.join(out_dir, "memory_samples.csv")
    samples_df.to_csv(samples_csv, index=False)
    print(f"Saved samples to {samples_csv}")
    
    markers_csv = os.path.join(out_dir, "stage_markers.csv")
    pd.DataFrame(markers).to_csv(markers_csv, index=False)
    print(f"Saved markers to {markers_csv}")
    
    # 绘图
    plot_path = os.path.join(out_dir, "memory_trace.png")
    video_name = os.path.basename(args.video)
    plot_memory_trace(
        samples_df=samples_df,
        markers=markers,
        output_path=plot_path,
        title=f"GPU Memory Trace - {video_name}"
    )
    
    # 分析阶段间内存变化
    print(f"\n{'='*60}")
    print("Stage Analysis")
    print("="*60)
    
    for i, marker in enumerate(markers):
        if marker['stage'].endswith('_end'):
            stage_name = marker['stage'].replace('_end', '')
            # 找对应的 start
            for j in range(i-1, -1, -1):
                if markers[j]['stage'] == f"{stage_name}_start":
                    start_time = markers[j]['timestamp_ms']
                    end_time = marker['timestamp_ms']
                    duration = end_time - start_time
                    
                    # 获取该阶段的内存范围
                    stage_samples = samples_df[
                        (samples_df['timestamp_ms'] >= start_time) & 
                        (samples_df['timestamp_ms'] <= end_time)
                    ]
                    
                    if len(stage_samples) > 0:
                        mem_start = stage_samples['allocated_mb'].iloc[0]
                        mem_end = stage_samples['allocated_mb'].iloc[-1]
                        mem_max = stage_samples['allocated_mb'].max()
                        mem_min = stage_samples['allocated_mb'].min()
                        
                        print(f"\n{stage_name}:")
                        print(f"  Duration: {duration:.1f}ms")
                        print(f"  Memory: {mem_start:.0f}MB → {mem_end:.0f}MB (max: {mem_max:.0f}MB)")
                        
                        # 检测是否有内存释放-重申请
                        if mem_min < mem_start * 0.9:  # 如果最小值低于开始值的90%
                            print(f"  ⚠️ Detected memory release during stage (min: {mem_min:.0f}MB)")
                    break
    
    print(f"\nExperiment completed. Results saved to {out_dir}")
