#!/usr/bin/env python3
"""
实验8：双 GPU 并行 Visual + Audio Encoder

目标：
1. 验证双 GPU 下 Visual + Audio Encoder 真正并行的加速效果
2. 对比两种 GPU 放置策略：
   - 配置 A: Visual + LLM 在 GPU0, Audio 在 GPU1
   - 配置 B: Audio + LLM 在 GPU0, Visual 在 GPU1
3. 测量跨 GPU 传输开销
4. 与单 GPU 串行 (exp7) 对比

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
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import common as C
import profiling_utils as P
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


# ============ 资源监控 ============

class DualGPUResourceMonitor(P.ResourceMonitor):
    """双 GPU 资源监控器 (Inherits basic structure but overrides loop)"""
    
    def __init__(self, interval: float = 0.01, n_gpus: int = 2):
        super().__init__(interval)
        self.n_gpus = n_gpus
    
    def _monitor_loop(self):
        import psutil
        try:
            import pynvml
            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.n_gpus)]
            has_nvml = True
        except:
            has_nvml = False
            handles = []
        
        while not self._stop.is_set():
            t = time.perf_counter() - self._start_time
            record = {"time": t, "cpu_percent": psutil.cpu_percent()}
            
            if has_nvml:
                for i, handle in enumerate(handles):
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        record[f"gpu{i}_percent"] = util.gpu
                        record[f"gpu{i}_vram_gb"] = mem.used / 1e9
                    except:
                        pass
            
            self.records.append(record)
            time.sleep(self.interval)
    
    def mark(self, name: str):
        t = time.perf_counter() - self._start_time if self._start_time else 0
        self.markers.append({"time": t, "name": name})
    
    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        return self.records, self.markers
    
    def cleanup(self):
        self.records = []
        self.markers = []


# ============ 模型加载与部署 ============

def load_model_dual_gpu(model_path: str, config: str = "A") -> Tuple:
    """
    加载模型并部署到双 GPU
    
    配置 A: Visual + LLM 在 GPU0, Audio 在 GPU1
    配置 B: Audio + LLM 在 GPU0, Visual 在 GPU1
    """
    from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
    
    print(f"  加载模型到双 GPU (配置 {config})...")
    
    # 先加载到 CPU
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 先加载到 CPU
    )
    
    # 根据配置分配 GPU
    if config == "A":
        # 配置 A: Visual + LLM 在 GPU0, Audio 在 GPU1
        model.thinker.visual = model.thinker.visual.to("cuda:0")
        model.thinker.audio_tower = model.thinker.audio_tower.to("cuda:1")
        model.thinker.model = model.thinker.model.to("cuda:0")  # LLM 主体
        model.thinker.audio_projection = model.thinker.audio_projection.to("cuda:0")
        model.thinker.visual_projection = model.thinker.visual_projection.to("cuda:0") if hasattr(model.thinker, 'visual_projection') else None
        llm_device = "cuda:0"
        visual_device = "cuda:0"
        audio_device = "cuda:1"
    else:
        # 配置 B: Audio + LLM 在 GPU0, Visual 在 GPU1
        model.thinker.visual = model.thinker.visual.to("cuda:1")
        model.thinker.audio_tower = model.thinker.audio_tower.to("cuda:0")
        model.thinker.model = model.thinker.model.to("cuda:0")  # LLM 主体
        model.thinker.audio_projection = model.thinker.audio_projection.to("cuda:0")
        model.thinker.visual_projection = model.thinker.visual_projection.to("cuda:0") if hasattr(model.thinker, 'visual_projection') else None
        llm_device = "cuda:0"
        visual_device = "cuda:1"
        audio_device = "cuda:0"
    
    # 其他组件放到 LLM 所在 GPU
    if hasattr(model, 'talker'):
        model.talker = model.talker.to(llm_device)
    
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    device_info = {
        "config": config,
        "llm_device": llm_device,
        "visual_device": visual_device,
        "audio_device": audio_device,
    }
    
    print(f"    Visual Encoder: {visual_device}")
    print(f"    Audio Encoder:  {audio_device}")
    print(f"    LLM:            {llm_device}")
    
    return model, processor, device_info


def load_model_single_gpu(model_path: str) -> Tuple:
    """加载模型到单 GPU（用于串行基准测试）"""
    from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
    
    print("  加载模型到单 GPU...")
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    device_info = {
        "config": "single",
        "llm_device": "cuda:0",
        "visual_device": "cuda:0",
        "audio_device": "cuda:0",
    }
    
    return model, processor, device_info


# ============ 数据加载 ============

def load_dataset(manifest_path: str, n_samples: int = 50) -> List[Dict]:
    return C.load_dataset(manifest_path, n_samples)


# ============ Encoder 并行执行 ============

def run_encoder_serial(
    model,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    input_features: torch.Tensor,
    feature_mask: torch.Tensor,
    device_info: Dict,
) -> Dict:
    """串行执行两个 Encoder"""
    timings = {}
    
    visual_device = device_info["visual_device"]
    audio_device = device_info["audio_device"]
    llm_device = device_info["llm_device"]
    
    # Visual Encoder
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.no_grad():
        visual_out = model.thinker.visual(
            pixel_values.to(visual_device),
            grid_thw.to(visual_device)
        )
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    timings["visual_ms"] = (t1 - t0) * 1000
    
    # Audio Encoder
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    
    with torch.no_grad():
        audio_out = model.thinker.audio_tower(
            input_features.to(audio_device),
            feature_mask.to(audio_device)
        )
    
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    timings["audio_ms"] = (t3 - t2) * 1000
    
    timings["total_ms"] = timings["visual_ms"] + timings["audio_ms"]
    
    # 记录输出 shape（用于验证）
    timings["visual_out_shape"] = list(visual_out.shape)
    timings["audio_out_shape"] = list(audio_out.shape)
    
    return timings


def run_encoder_parallel(
    model,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    input_features: torch.Tensor,
    feature_mask: torch.Tensor,
    device_info: Dict,
) -> Dict:
    """并行执行两个 Encoder（使用 ThreadPoolExecutor）"""
    timings = {}
    
    visual_device = device_info["visual_device"]
    audio_device = device_info["audio_device"]
    llm_device = device_info["llm_device"]
    
    visual_out = None
    audio_out = None
    visual_time = 0
    audio_time = 0
    
    def run_visual():
        nonlocal visual_out, visual_time
        torch.cuda.synchronize(device=visual_device)
        t0 = time.perf_counter()
        with torch.no_grad():
            visual_out = model.thinker.visual(
                pixel_values.to(visual_device),
                grid_thw.to(visual_device)
            )
        torch.cuda.synchronize(device=visual_device)
        visual_time = (time.perf_counter() - t0) * 1000
    
    def run_audio():
        nonlocal audio_out, audio_time
        torch.cuda.synchronize(device=audio_device)
        t0 = time.perf_counter()
        with torch.no_grad():
            audio_out = model.thinker.audio_tower(
                input_features.to(audio_device),
                feature_mask.to(audio_device)
            )
        torch.cuda.synchronize(device=audio_device)
        audio_time = (time.perf_counter() - t0) * 1000
    
    # 并行执行
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_v = executor.submit(run_visual)
        future_a = executor.submit(run_audio)
        future_v.result()
        future_a.result()
    
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    timings["visual_ms"] = visual_time
    timings["audio_ms"] = audio_time
    timings["total_ms"] = (t_end - t_start) * 1000
    
    # 跨 GPU 传输测试
    t_transfer_start = time.perf_counter()
    if visual_device != llm_device:
        visual_out_llm = visual_out.to(llm_device)
    else:
        visual_out_llm = visual_out
    
    if audio_device != llm_device:
        audio_out_llm = audio_out.to(llm_device)
    else:
        audio_out_llm = audio_out
    torch.cuda.synchronize()
    timings["transfer_ms"] = (time.perf_counter() - t_transfer_start) * 1000
    
    # 并行时间 + 传输时间
    timings["total_with_transfer_ms"] = timings["total_ms"] + timings["transfer_ms"]
    
    # 记录输出 shape
    timings["visual_out_shape"] = list(visual_out.shape)
    timings["audio_out_shape"] = list(audio_out.shape)
    
    return timings


# ============ 完整推理测试 ============

def prepare_inputs(
    model,
    processor,
    video_path: str,
    question: str,
    fe,
) -> Tuple[Dict, Dict]:
    """准备推理输入，返回预处理时间"""
    from qwen_omni_utils import process_mm_info
    
    timings = {}
    
    # 1. 视频+音频提取
    t0 = time.perf_counter()
    conversation = [{"role": "user", "content": [
        {"type": "video", "video": video_path},
        {"type": "text", "text": question}
    ]}]
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    timings["video_audio_extract_ms"] = (time.perf_counter() - t0) * 1000
    
    # 2. 视频 tokenize
    t1 = time.perf_counter()
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, videos=videos, return_tensors="pt", padding=True)
    timings["video_tokenize_ms"] = (time.perf_counter() - t1) * 1000
    
    # 3. 音频特征提取 (FFT + Mel)
    t2 = time.perf_counter()
    if audios and fe is not None:
        af = fe(audios[0], sampling_rate=16000, return_tensors='pt')
        inputs['input_features'] = af['input_features']
        inputs['feature_attention_mask'] = torch.ones(
            (1, af['input_features'].shape[2]), dtype=torch.long
        )
    timings["audio_feature_ms"] = (time.perf_counter() - t2) * 1000
    
    timings["preprocess_total_ms"] = sum([
        timings["video_audio_extract_ms"],
        timings["video_tokenize_ms"],
        timings["audio_feature_ms"]
    ])
    
    return inputs, timings


# ============ 可视化 ============

def plot_comparison(results: Dict, output_path: str):
    """绘制串行 vs 并行对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 左图：Encoder 时间对比
    configs = list(results.keys())
    serial_times = [results[c]["serial_encoder_ms"] for c in configs]
    parallel_times = [results[c]["parallel_encoder_ms"] for c in configs]
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, serial_times, width, label='Serial', color='#FF6B6B')
    bars2 = axes[0].bar(x + width/2, parallel_times, width, label='Parallel', color='#4ECDC4')
    
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Encoder Time: Serial vs Parallel')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs)
    axes[0].legend()
    
    for bar, val in zip(bars1, serial_times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, parallel_times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', ha='center', fontsize=9)
    
    # 中图：加速比
    speedups = [results[c]["speedup"] for c in configs]
    bars3 = axes[1].bar(configs, speedups, color=['#95E1D3', '#F38181'])
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('Speedup')
    axes[1].set_title('Parallel Speedup (Serial / Parallel)')
    axes[1].set_ylim(0, max(speedups) * 1.2)
    
    for bar, val in zip(bars3, speedups):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}x', ha='center', fontsize=10, fontweight='bold')
    
    # 右图：各阶段细分
    if "Config_A" in results:
        data = results["Config_A"]
        labels = ["Visual", "Audio", "Transfer"]
        values = [data["visual_ms"], data["audio_ms"], data["transfer_ms"]]
        colors = ['#4CAF50', '#2196F3', '#FF9800']
        
        axes[2].bar(labels, values, color=colors)
        axes[2].set_ylabel('Time (ms)')
        axes[2].set_title('Parallel Breakdown (Config A)')
        
        for i, v in enumerate(values):
            axes[2].text(i, v + 5, f'{v:.1f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 对比图已保存: {output_path}")


def plot_dual_gpu_resource(records: List[Dict], output_path: str, n_gpus: int = 2):
    """绘制双 GPU 资源利用率"""
    if not records:
        return
    
    df = pd.DataFrame(records)
    
    fig, axes = plt.subplots(n_gpus + 1, 1, figsize=(12, 3 * (n_gpus + 1)), sharex=True)
    
    t = df["time"]
    
    for i in range(n_gpus):
        col_gpu = f"gpu{i}_percent"
        col_vram = f"gpu{i}_vram_gb"
        
        if col_gpu in df.columns:
            ax = axes[i]
            ax.plot(t, df[col_gpu], 'b-', linewidth=1, label=f'GPU {i} Util')
            ax.set_ylabel(f"GPU {i} %")
            ax.set_ylim(0, 100)
            ax.set_title(f"GPU {i} Utilization")
            ax.legend(loc='upper right')
            
            # 显存作为第二 Y 轴
            if col_vram in df.columns:
                ax2 = ax.twinx()
                ax2.plot(t, df[col_vram], 'g--', linewidth=1, alpha=0.7, label='VRAM')
                ax2.set_ylabel("VRAM (GB)")
                ax2.legend(loc='upper left')
    
    # CPU
    axes[-1].plot(t, df["cpu_percent"], 'r-', linewidth=1)
    axes[-1].set_ylabel("CPU %")
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_title("CPU Utilization")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 双 GPU 资源图已保存: {output_path}")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="双 GPU 并行 Visual + Audio Encoder 实验")
    parser.add_argument("--model", default="/root/autodl-tmp/Qwen2.5-Omni-7B", help="模型路径")
    parser.add_argument("--data", default="/root/autodl-tmp/data/MSRVTT_subset/manifest.csv", help="数据 manifest")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/results/exp8", help="输出目录")
    parser.add_argument("--n-samples", type=int, default=50, help="测试样本数")
    parser.add_argument("--warmup", type=int, default=2, help="预热次数")
    parser.add_argument("--config", choices=["A", "B", "both"], default="both", help="GPU 配置")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 70)
    print("🚀 实验8：双 GPU 并行 Visual + Audio Encoder")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"样本数: {args.n_samples}")
    print(f"GPU 配置: {args.config}")
    
    # 检查 GPU 数量
    n_gpus = torch.cuda.device_count()
    print(f"\n检测到 {n_gpus} 个 GPU")
    
    if n_gpus < 2:
        print("⚠️ 需要至少 2 个 GPU！当前只检测到 1 个。")
        print("   将只运行单 GPU 串行测试作为基准。")
    
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 加载数据
    print("\n🔄 加载数据...")
    samples = load_dataset(args.data, args.n_samples + args.warmup)
    print(f"  加载了 {len(samples)} 个样本")
    
    # 准备 WhisperFeatureExtractor
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    
    # 确定要测试的配置
    if args.config == "both" and n_gpus >= 2:
        configs_to_test = ["A", "B"]
    elif args.config in ["A", "B"] and n_gpus >= 2:
        configs_to_test = [args.config]
    else:
        configs_to_test = []
    
    all_results = {}
    
    # ===== 测试每种配置 =====
    for config in configs_to_test:
        print(f"\n{'='*70}")
        print(f"📋 测试配置 {config}")
        print("=" * 70)
        
        # 加载模型
        print("\n🔄 加载模型...")
        model, processor, device_info = load_model_dual_gpu(args.model, config)
        
        # 启动资源监控
        monitor = DualGPUResourceMonitor(interval=0.01, n_gpus=n_gpus)
        
        # 预热
        print(f"\n🔥 预热 ({args.warmup} 次)...")
        for i in range(min(args.warmup, len(samples))):
            try:
                inputs, _ = prepare_inputs(model, processor, samples[i]["video_path"], "Describe.", fe)
                pixel_values = inputs['pixel_values_videos'].to(torch.bfloat16)
                grid_thw = inputs['video_grid_thw']
                input_features = inputs['input_features'].to(torch.bfloat16)
                feature_mask = inputs['feature_attention_mask']
                
                run_encoder_serial(model, pixel_values, grid_thw, input_features, feature_mask, device_info)
                run_encoder_parallel(model, pixel_values, grid_thw, input_features, feature_mask, device_info)
            except Exception as e:
                print(f"  预热 {i+1} 失败: {e}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # 正式测试
        print(f"\n🧪 开始测试 ({args.n_samples} 个样本)...")
        
        serial_results = []
        parallel_results = []
        
        monitor.start()
        
        for i, sample in enumerate(samples[args.warmup:args.warmup + args.n_samples]):
            print(f"\n--- 样本 {i+1}/{args.n_samples}: {sample['sample_id']} ---")
            
            try:
                # 准备输入
                inputs, prep_timings = prepare_inputs(
                    model, processor, sample["video_path"], sample["question"], fe
                )
                
                pixel_values = inputs['pixel_values_videos'].to(torch.bfloat16)
                grid_thw = inputs['video_grid_thw']
                input_features = inputs['input_features'].to(torch.bfloat16)
                feature_mask = inputs['feature_attention_mask']
                
                print(f"  预处理: {prep_timings['preprocess_total_ms']:.0f}ms")
                
                # 串行测试
                monitor.mark(f"serial_start_{i}")
                serial_timing = run_encoder_serial(
                    model, pixel_values, grid_thw, input_features, feature_mask, device_info
                )
                monitor.mark(f"serial_end_{i}")
                serial_results.append(serial_timing)
                
                print(f"  串行: Visual={serial_timing['visual_ms']:.0f}ms, Audio={serial_timing['audio_ms']:.0f}ms, 总计={serial_timing['total_ms']:.0f}ms")
                
                # 并行测试
                monitor.mark(f"parallel_start_{i}")
                parallel_timing = run_encoder_parallel(
                    model, pixel_values, grid_thw, input_features, feature_mask, device_info
                )
                monitor.mark(f"parallel_end_{i}")
                parallel_results.append(parallel_timing)
                
                print(f"  并行: Visual={parallel_timing['visual_ms']:.0f}ms, Audio={parallel_timing['audio_ms']:.0f}ms, 总计={parallel_timing['total_ms']:.0f}ms, 传输={parallel_timing['transfer_ms']:.1f}ms")
                
                # 加速比
                speedup = serial_timing['total_ms'] / parallel_timing['total_with_transfer_ms']
                print(f"  加速比: {speedup:.2f}x")
                
            except Exception as e:
                print(f"  ⚠️ 失败: {e}")
                import traceback
                traceback.print_exc()
            
            gc.collect()
            torch.cuda.empty_cache()
        
        records, markers = monitor.stop()
        
        # 汇总配置结果
        if serial_results and parallel_results:
            config_result = {
                "config": config,
                "device_info": device_info,
                "n_samples": len(serial_results),
                # 串行统计
                "serial_encoder_ms": np.mean([r["total_ms"] for r in serial_results]),
                "serial_visual_ms": np.mean([r["visual_ms"] for r in serial_results]),
                "serial_audio_ms": np.mean([r["audio_ms"] for r in serial_results]),
                # 并行统计
                "parallel_encoder_ms": np.mean([r["total_ms"] for r in parallel_results]),
                "parallel_with_transfer_ms": np.mean([r["total_with_transfer_ms"] for r in parallel_results]),
                "visual_ms": np.mean([r["visual_ms"] for r in parallel_results]),
                "audio_ms": np.mean([r["audio_ms"] for r in parallel_results]),
                "transfer_ms": np.mean([r["transfer_ms"] for r in parallel_results]),
                # 加速比
                "speedup": np.mean([r["total_ms"] for r in serial_results]) / np.mean([r["total_with_transfer_ms"] for r in parallel_results]),
            }
            all_results[f"Config_{config}"] = config_result
            
            # 保存资源记录
            if records:
                pd.DataFrame(records).to_csv(
                    os.path.join(args.out_dir, f"resource_records_config_{config}.csv"), index=False
                )
                plot_dual_gpu_resource(records, os.path.join(args.out_dir, f"resource_usage_config_{config}.png"), n_gpus)
        
        # 释放模型
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    
    # ===== 保存结果 =====
    print("\n📝 保存结果...")
    
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 绘制对比图
    if all_results:
        plot_comparison(all_results, os.path.join(args.out_dir, "comparison.png"))
    
    # ===== 打印总结 =====
    print("\n" + "=" * 70)
    print("📊 实验总结")
    print("=" * 70)
    
    for config_name, result in all_results.items():
        print(f"\n🔹 {config_name}:")
        print(f"   设备配置: Visual={result['device_info']['visual_device']}, Audio={result['device_info']['audio_device']}, LLM={result['device_info']['llm_device']}")
        print(f"   串行 Encoder: {result['serial_encoder_ms']:.1f} ms")
        print(f"   并行 Encoder: {result['parallel_encoder_ms']:.1f} ms")
        print(f"   跨 GPU 传输:  {result['transfer_ms']:.1f} ms")
        print(f"   并行+传输:    {result['parallel_with_transfer_ms']:.1f} ms")
        print(f"   🚀 加速比:    {result['speedup']:.2f}x")
    
    # 与 exp7 对比
    print(f"\n📈 与单 GPU 串行 (exp7) 对比:")
    print(f"   exp7 Encoder 串行: ~691 ms (Visual: 188ms + Audio: 503ms)")
    if all_results:
        best_config = max(all_results.items(), key=lambda x: x[1]["speedup"])
        print(f"   exp8 最佳配置:     {best_config[0]}, 加速比 {best_config[1]['speedup']:.2f}x")
    
    print(f"\n结果已保存至: {args.out_dir}")


if __name__ == "__main__":
    main()
