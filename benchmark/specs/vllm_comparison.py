"""
Spec: vllm-comparison
vLLM vs HuggingFace TTFT 对比实验

目的：证明 vLLM 主要优化 KV cache / 批处理，不优化 encoder/prefill 阶段
     在单用户单请求场景下，TTFT 与 HF 相当

限制：vLLM 对 Qwen2.5-Omni 只支持 Thinker (无音频)，所以对比只用视频输入

使用方法:
    # 仅测 HF
    python benchmark/run.py vllm-comparison \
        --model-dir /path/to/Qwen2.5-Omni-7B \
        --manifest /path/to/manifest.csv \
        --out-dir ./results \
        --backend hf

    # 仅测 vLLM
    python benchmark/run.py vllm-comparison \
        --model-dir /path/to/Qwen2.5-Omni-7B \
        --manifest /path/to/manifest.csv \
        --out-dir ./results \
        --backend vllm

    # 两者都测 (需要分两次运行，避免显存冲突)
    python benchmark/run.py vllm-comparison --backend hf ...
    python benchmark/run.py vllm-comparison --backend vllm ...
    python benchmark/specs/vllm_comparison.py --merge-results ./results/vllm-comparison
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.runner import BenchmarkRunner
from benchmark.unified_runner import UnifiedRunner


SPEC_NAME = "vllm-comparison"


def register_subcommand(subparsers, common_parser) -> None:
    p = subparsers.add_parser(
        SPEC_NAME,
        parents=[common_parser],
        help="Compare vLLM vs HuggingFace TTFT for video-only inference"
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm", "both"],
        default="hf",
        help="Which backend to use (hf, vllm, or both - runs sequentially)"
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization (default 0.85)"
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM max model length (default 32768)"
    )
    p.add_argument(
        "--max-video-duration",
        type=float,
        default=300.0,
        help="Max video duration in seconds for selection (default 5min)"
    )
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
    except Exception:
        pass
    return 0.0


def select_videos(
    args: argparse.Namespace,
    runner: BenchmarkRunner,
    max_duration: float = 300.0,
) -> List[Dict[str, Any]]:
    """选择视频样本"""
    if not os.path.exists(str(args.manifest)):
        raise SystemExit(f"manifest not found: {args.manifest}")

    all_rows = runner.load_manifest_csv(str(args.manifest), n_samples=0, seed=int(args.seed))
    
    usable = []
    for r in all_rows:
        raw_v = r.get("video_path", None)
        v = UnifiedRunner.resolve_media_path(str(args.manifest), None if raw_v is None else str(raw_v))
        
        if not v or not os.path.exists(v):
            continue
        
        duration = r.get("duration", 0)
        if not duration or duration <= 0:
            duration = get_video_duration_ffprobe(v)
        
        if duration <= 0 or duration > max_duration:
            continue
        
        rr = dict(r)
        rr["video_path"] = v
        rr["duration"] = float(duration)
        usable.append(rr)
    
    if not usable:
        raise SystemExit("no usable videos found")
    
    n = int(getattr(args, "n_samples", 5))
    if n <= 0 or n >= len(usable):
        return usable
    
    df = pd.DataFrame(usable)
    sub = df.sample(n=n, random_state=int(args.seed))
    return sub.to_dict(orient="records")


def run_hf_backend(
    args: argparse.Namespace,
    runner: BenchmarkRunner,
    samples: List[Dict[str, Any]],
    out_dir: str,
) -> pd.DataFrame:
    """运行 HuggingFace 后端测试"""
    from benchmark.vllm_backend import HFBackend
    
    print("\n" + "=" * 60)
    print("Running HuggingFace Backend")
    print("=" * 60)
    
    backend = HFBackend(args.model_dir, dtype=str(args.dtype))
    backend.load()
    
    rows = []
    for i, s in enumerate(samples):
        video_path = str(s.get("video_path", ""))
        question = str(args.question) if args.question else str(s.get("question", "Describe what you see."))
        duration = float(s.get("duration", 0))
        sample_id = str(s.get("sample_id", os.path.basename(video_path)))
        
        print(f"\n[{i+1}/{len(samples)}] {sample_id} ({duration:.1f}s)")
        
        for r in range(int(max(1, args.repeats))):
            do_record = int(r) >= int(max(0, args.warmup))
            
            try:
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                result = backend.generate_video(
                    video_path,
                    question,
                    max_tokens=int(args.max_new_tokens) if hasattr(args, "max_new_tokens") else 50,
                    video_nframes=int(args.video_nframes) if args.video_nframes else None,
                )
                
                if do_record:
                    rows.append({
                        "backend": "hf",
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "duration": duration,
                        "repeat": r,
                        "ttft_ms": result.ttft_ms,
                        "total_ms": result.total_ms,
                        "num_tokens": result.num_tokens,
                        "prompt_tokens": result.prompt_tokens,
                        "tokens_per_sec": result.tokens_per_sec,
                        "output_text": result.output_text[:100],
                    })
                    print(f"    repeat={r}: TTFT={result.ttft_ms:.1f}ms, total={result.total_ms:.1f}ms")
                    
            except Exception as e:
                print(f"    Error: {e}")
                if do_record:
                    rows.append({
                        "backend": "hf",
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "duration": duration,
                        "repeat": r,
                        "error": str(e),
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "hf_results.csv"), index=False)
    return df


def run_vllm_backend(
    args: argparse.Namespace,
    runner: BenchmarkRunner,
    samples: List[Dict[str, Any]],
    out_dir: str,
) -> pd.DataFrame:
    """运行 vLLM 后端测试"""
    from benchmark.vllm_backend import VLLMBackend
    
    print("\n" + "=" * 60)
    print("Running vLLM Backend")
    print("=" * 60)
    
    backend = VLLMBackend(
        args.model_dir,
        dtype=str(args.dtype) if hasattr(args, "dtype") else "bfloat16",
        gpu_memory_utilization=float(getattr(args, "gpu_memory_utilization", 0.85)),
        max_model_len=int(getattr(args, "max_model_len", 32768)),
    )
    backend.load()
    
    rows = []
    for i, s in enumerate(samples):
        video_path = str(s.get("video_path", ""))
        question = str(args.question) if args.question else str(s.get("question", "Describe what you see."))
        duration = float(s.get("duration", 0))
        sample_id = str(s.get("sample_id", os.path.basename(video_path)))
        
        print(f"\n[{i+1}/{len(samples)}] {sample_id} ({duration:.1f}s)")
        
        for r in range(int(max(1, args.repeats))):
            do_record = int(r) >= int(max(0, args.warmup))
            
            try:
                result = backend.generate_video(
                    video_path,
                    question,
                    max_tokens=int(args.max_new_tokens) if hasattr(args, "max_new_tokens") else 50,
                    video_nframes=int(args.video_nframes) if args.video_nframes else None,
                )
                
                if do_record:
                    rows.append({
                        "backend": "vllm",
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "duration": duration,
                        "repeat": r,
                        "ttft_ms": result.ttft_ms,
                        "total_ms": result.total_ms,
                        "num_tokens": result.num_tokens,
                        "prompt_tokens": result.prompt_tokens,
                        "tokens_per_sec": result.tokens_per_sec,
                        "output_text": result.output_text[:100],
                    })
                    print(f"    repeat={r}: TTFT={result.ttft_ms:.1f}ms, total={result.total_ms:.1f}ms")
                    
            except Exception as e:
                print(f"    Error: {e}")
                if do_record:
                    rows.append({
                        "backend": "vllm",
                        "sample_id": sample_id,
                        "video_path": video_path,
                        "duration": duration,
                        "repeat": r,
                        "error": str(e),
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "vllm_results.csv"), index=False)
    return df


def merge_and_plot(out_dir: str) -> None:
    """合并结果并生成对比图"""
    hf_path = os.path.join(out_dir, "hf_results.csv")
    vllm_path = os.path.join(out_dir, "vllm_results.csv")
    
    dfs = []
    if os.path.exists(hf_path):
        dfs.append(pd.read_csv(hf_path))
    if os.path.exists(vllm_path):
        dfs.append(pd.read_csv(vllm_path))
    
    if not dfs:
        print("No results to merge")
        return
    
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(out_dir, "combined_results.csv"), index=False)
    
    # 过滤错误
    df_ok = df[df.get("error").isna()] if "error" in df.columns else df
    
    if len(df_ok) == 0:
        print("No valid results to plot")
        return
    
    # 按 backend 和 sample_id 聚合
    summary = df_ok.groupby(["backend", "sample_id", "duration"]).agg({
        "ttft_ms": "mean",
        "total_ms": "mean",
        "tokens_per_sec": "mean",
    }).reset_index()
    
    # 保存汇总
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    
    # 检查是否有两种 backend
    backends = summary["backend"].unique()
    if len(backends) < 2:
        print(f"Only one backend found: {backends}. Skipping comparison plot.")
        _plot_single_backend(summary, out_dir)
        return
    
    # 双 backend 对比图
    _plot_comparison(summary, out_dir)


def _plot_single_backend(summary: pd.DataFrame, out_dir: str) -> None:
    """单个 backend 的结果图"""
    backend = summary["backend"].iloc[0]
    
    # 按时长排序
    summary = summary.sort_values("duration")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # TTFT
    x = np.arange(len(summary))
    ax1.bar(x, summary["ttft_ms"] / 1000, color="#3498db", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d:.0f}s" for d in summary["duration"]], rotation=45, ha="right")
    ax1.set_xlabel("Video Duration")
    ax1.set_ylabel("TTFT (seconds)")
    ax1.set_title(f"{backend.upper()} TTFT by Video Duration")
    ax1.grid(axis="y", alpha=0.3)
    
    # Throughput
    ax2.bar(x, summary["tokens_per_sec"], color="#2ecc71", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{d:.0f}s" for d in summary["duration"]], rotation=45, ha="right")
    ax2.set_xlabel("Video Duration")
    ax2.set_ylabel("Tokens/sec")
    ax2.set_title(f"{backend.upper()} Throughput by Video Duration")
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{backend}_results.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {backend}_results.png")


def _plot_comparison(summary: pd.DataFrame, out_dir: str) -> None:
    """HF vs vLLM 对比图"""
    hf_data = summary[summary["backend"] == "hf"].sort_values("duration")
    vllm_data = summary[summary["backend"] == "vllm"].sort_values("duration")
    
    # 找共同的 sample_id
    common_samples = set(hf_data["sample_id"]) & set(vllm_data["sample_id"])
    if not common_samples:
        print("No common samples between HF and vLLM")
        return
    
    hf_data = hf_data[hf_data["sample_id"].isin(common_samples)].sort_values("duration")
    vllm_data = vllm_data[vllm_data["sample_id"].isin(common_samples)]
    vllm_data = vllm_data.set_index("sample_id").loc[hf_data["sample_id"]].reset_index()
    
    n = len(hf_data)
    x = np.arange(n)
    width = 0.35
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========== 1. TTFT 对比 ==========
    ax = axes[0, 0]
    ax.bar(x - width/2, hf_data["ttft_ms"] / 1000, width, label="HuggingFace", color="#3498db")
    ax.bar(x + width/2, vllm_data["ttft_ms"] / 1000, width, label="vLLM", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0f}s" for d in hf_data["duration"]], rotation=45, ha="right")
    ax.set_xlabel("Video Duration")
    ax.set_ylabel("TTFT (seconds)")
    ax.set_title("Time To First Token (TTFT) Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # ========== 2. Total Time 对比 ==========
    ax = axes[0, 1]
    ax.bar(x - width/2, hf_data["total_ms"] / 1000, width, label="HuggingFace", color="#3498db")
    ax.bar(x + width/2, vllm_data["total_ms"] / 1000, width, label="vLLM", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0f}s" for d in hf_data["duration"]], rotation=45, ha="right")
    ax.set_xlabel("Video Duration")
    ax.set_ylabel("Total Time (seconds)")
    ax.set_title("Total Generation Time Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # ========== 3. Throughput 对比 ==========
    ax = axes[1, 0]
    ax.bar(x - width/2, hf_data["tokens_per_sec"], width, label="HuggingFace", color="#3498db")
    ax.bar(x + width/2, vllm_data["tokens_per_sec"], width, label="vLLM", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0f}s" for d in hf_data["duration"]], rotation=45, ha="right")
    ax.set_xlabel("Video Duration")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # ========== 4. TTFT 差异百分比 ==========
    ax = axes[1, 1]
    ttft_diff_pct = (vllm_data["ttft_ms"].values - hf_data["ttft_ms"].values) / hf_data["ttft_ms"].values * 100
    colors = ["#2ecc71" if d < 0 else "#e74c3c" for d in ttft_diff_pct]
    ax.bar(x, ttft_diff_pct, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0f}s" for d in hf_data["duration"]], rotation=45, ha="right")
    ax.set_xlabel("Video Duration")
    ax.set_ylabel("TTFT Difference (%)")
    ax.set_title("vLLM TTFT vs HF\n(negative = vLLM faster, positive = HF faster)")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vllm_vs_hf_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: vllm_vs_hf_comparison.png")
    
    # 打印统计
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"HF mean TTFT:   {hf_data['ttft_ms'].mean():.1f} ms")
    print(f"vLLM mean TTFT: {vllm_data['ttft_ms'].mean():.1f} ms")
    print(f"Difference:     {ttft_diff_pct.mean():.1f}%")
    
    if abs(ttft_diff_pct.mean()) < 10:
        print("\n结论：vLLM 和 HF 的 TTFT 差异 < 10%，证明 vLLM 在单请求场景下")
        print("      对 encoder/prefill 阶段没有显著优化")


def run(args: argparse.Namespace, runner: BenchmarkRunner) -> str:
    out_dir = os.path.join(str(args.out_dir), SPEC_NAME)
    os.makedirs(out_dir, exist_ok=True)
    
    # 选择视频
    max_dur = float(getattr(args, "max_video_duration", 300.0))
    samples = select_videos(args, runner, max_duration=max_dur)
    
    print(f"Selected {len(samples)} videos for comparison")
    
    # 保存样本列表
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2, default=str)
    
    backend = str(getattr(args, "backend", "hf")).lower()
    
    if backend in ("hf", "both"):
        run_hf_backend(args, runner, samples, out_dir)
    
    if backend in ("vllm", "both"):
        # 如果是 both，需要释放 HF 模型后再加载 vLLM
        if backend == "both":
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        run_vllm_backend(args, runner, samples, out_dir)
    
    # 合并结果并生成图
    merge_and_plot(out_dir)
    
    return out_dir


if __name__ == "__main__":
    # 独立运行模式：合并已有结果
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--merge-results":
        merge_and_plot(sys.argv[2])
    else:
        print("Usage: python vllm_comparison.py --merge-results <out_dir>")
