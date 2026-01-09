#!/usr/bin/env python3
"""
重绘 TTFT 实验结果图表
用现有的 CSV 数据重新生成图表，无需重跑实验

使用方法:
    # 重绘 ttft-10videos 结果（按视觉token数排序）
    python tools/replot_ttft_results.py \
        --csv results/ttft-10videos/ttft_10videos_results.csv \
        --output results/ttft-10videos/ \
        --sort-by visual_tokens

    # 重绘 vLLM 对比结果（按视觉token数排序）
    python tools/replot_ttft_results.py \
        --csv results/vllm-comparison/combined_results.csv \
        --output results/vllm-comparison/ \
        --sort-by prompt_tokens \
        --type vllm
"""

import argparse
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ttft_breakdown(
    df: pd.DataFrame,
    output_dir: str,
    sort_by: str = "visual_tokens",
    warmup_repeats: int = 0,
) -> None:
    """
    重绘 TTFT 分解图（按指定字段排序）
    
    Args:
        df: 包含 TTFT 数据的 DataFrame
        output_dir: 输出目录
        sort_by: 排序字段 (visual_tokens, duration, prompt_tokens 等)
        warmup_repeats: 要跳过的 warmup 重复次数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 过滤错误和 warmup
    df_ok = df.copy()
    if "error" in df_ok.columns:
        df_ok = df_ok[df_ok["error"].isna()]
    if "repeat" in df_ok.columns and warmup_repeats > 0:
        df_ok = df_ok[df_ok["repeat"] >= warmup_repeats]
    
    if len(df_ok) == 0:
        print("No valid data to plot")
        return
    
    # 检查排序字段
    if sort_by not in df_ok.columns:
        print(f"Warning: '{sort_by}' not in columns, falling back to 'duration'")
        sort_by = "duration" if "duration" in df_ok.columns else df_ok.columns[0]
    
    # 需要的列
    phase_cols = ["preprocess_ms", "visual_encoder_ms", "audio_encoder_ms", "llm_prefill_ms", "other_ms"]
    available_phases = [c for c in phase_cols if c in df_ok.columns]
    
    if not available_phases:
        print("No phase columns found in data")
        return
    
    # 聚合数据
    agg_dict = {c: "mean" for c in available_phases}
    agg_dict["ttft_ms"] = "mean"
    if "duration" in df_ok.columns:
        agg_dict["duration"] = "first"
    if sort_by in df_ok.columns and sort_by not in agg_dict:
        agg_dict[sort_by] = "first"
    
    # 按 sample_id 分组
    group_col = "sample_id" if "sample_id" in df_ok.columns else df_ok.index.name or "index"
    if group_col == "index":
        df_ok = df_ok.reset_index()
    
    grouped = df_ok.groupby(group_col, dropna=False).agg(agg_dict).reset_index()
    grouped = grouped.sort_values(sort_by)
    
    n = len(grouped)
    x = np.arange(n)
    width = 0.6
    
    # 颜色映射
    colors = {
        "preprocess_ms": "#2ecc71",
        "visual_encoder_ms": "#3498db",
        "audio_encoder_ms": "#e67e22",
        "llm_prefill_ms": "#e74c3c",
        "other_ms": "#9b59b6",
    }
    labels = {
        "preprocess_ms": "Preprocess",
        "visual_encoder_ms": "Visual Encoder",
        "audio_encoder_ms": "Audio Encoder",
        "llm_prefill_ms": "Prefill",
        "other_ms": "Other/Decode",
    }
    
    # ========== 绝对值图 ==========
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bottom = np.zeros(n)
    for col in available_phases:
        vals = grouped[col].fillna(0).to_numpy() / 1000  # 转换为秒
        ax.bar(x, vals, width, bottom=bottom, label=labels.get(col, col), color=colors.get(col, "#888"))
        bottom += vals
    
    # X 轴标签
    if "duration" in grouped.columns and sort_by in grouped.columns and sort_by != "duration":
        x_labels = [f"{row['duration']:.0f}s\n({int(row[sort_by])} tok)" for _, row in grouped.iterrows()]
        ax.set_xlabel(f"Video Duration ({sort_by})", fontsize=11)
    elif "duration" in grouped.columns:
        x_labels = [f"{row['duration']:.0f}s" for _, row in grouped.iterrows()]
        ax.set_xlabel("Video Duration", fontsize=11)
    else:
        x_labels = [f"{int(row[sort_by])}" for _, row in grouped.iterrows()]
        ax.set_xlabel(sort_by, fontsize=11)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("TTFT (seconds)", fontsize=11)
    ax.set_title(f"TTFT Breakdown (Sorted by {sort_by})\n(Absolute Time)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "ttft_breakdown_absolute.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # ========== 百分比图 ==========
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bottom = np.zeros(n)
    for col in available_phases:
        vals = []
        for _, row in grouped.iterrows():
            total = row["ttft_ms"]
            v = row[col] if pd.notna(row[col]) else 0
            pct = (v / total * 100) if total > 0 else 0
            vals.append(pct)
        vals = np.array(vals)
        ax.bar(x, vals, width, bottom=bottom, label=labels.get(col, col), color=colors.get(col, "#888"))
        bottom += vals
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="right")
    ax.set_xlabel(f"Video Duration ({sort_by})" if sort_by != "duration" else "Video Duration", fontsize=11)
    ax.set_ylabel("Percentage of TTFT (%)", fontsize=11)
    ax.set_title(f"TTFT Breakdown (Sorted by {sort_by})\n(Percentage)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "ttft_breakdown_percentage.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # 保存统计
    summary = grouped.copy()
    if "ttft_ms" in summary.columns:
        encoder_cols = [c for c in ["visual_encoder_ms", "audio_encoder_ms"] if c in summary.columns]
        if encoder_cols:
            summary["encoder_pct"] = summary[encoder_cols].sum(axis=1) / summary["ttft_ms"] * 100
    summary.to_csv(os.path.join(output_dir, "ttft_summary.csv"), index=False)
    print(f"Saved: {output_dir}/ttft_summary.csv")


def plot_vllm_comparison(
    df: pd.DataFrame,
    output_dir: str,
    sort_by: str = "prompt_tokens",
) -> None:
    """
    重绘 vLLM vs HF 对比图（按指定字段排序）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 过滤错误
    df_ok = df.copy()
    if "error" in df_ok.columns:
        df_ok = df_ok[df_ok["error"].isna()]
    
    if len(df_ok) == 0:
        print("No valid data to plot")
        return
    
    # 检查 backend 列
    if "backend" not in df_ok.columns:
        print("No 'backend' column found, cannot plot comparison")
        return
    
    backends = df_ok["backend"].unique()
    if len(backends) < 2:
        print(f"Only one backend found: {backends}")
        return
    
    # 检查排序字段
    if sort_by not in df_ok.columns:
        print(f"Warning: '{sort_by}' not in columns, falling back to 'duration'")
        sort_by = "duration" if "duration" in df_ok.columns else "sample_id"
    
    # 聚合
    agg_dict = {
        "ttft_ms": "mean",
        "total_ms": "mean",
        "tokens_per_sec": "mean",
    }
    if "duration" in df_ok.columns:
        agg_dict["duration"] = "first"
    if sort_by in df_ok.columns and sort_by not in agg_dict:
        agg_dict[sort_by] = "first"
    
    summary = df_ok.groupby(["backend", "sample_id"]).agg(agg_dict).reset_index()
    
    hf_data = summary[summary["backend"] == "hf"].sort_values(sort_by)
    vllm_data = summary[summary["backend"] == "vllm"].sort_values(sort_by)
    
    # 找共同样本
    common = set(hf_data["sample_id"]) & set(vllm_data["sample_id"])
    if not common:
        print("No common samples between HF and vLLM")
        return
    
    hf_data = hf_data[hf_data["sample_id"].isin(common)].sort_values(sort_by)
    vllm_data = vllm_data[vllm_data["sample_id"].isin(common)]
    vllm_data = vllm_data.set_index("sample_id").loc[hf_data["sample_id"].values].reset_index()
    
    n = len(hf_data)
    x = np.arange(n)
    width = 0.35
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # X 轴标签
    if "duration" in hf_data.columns and sort_by in hf_data.columns and sort_by != "duration":
        x_labels = [f"{row['duration']:.0f}s\n({int(row[sort_by])} tok)" for _, row in hf_data.iterrows()]
    elif "duration" in hf_data.columns:
        x_labels = [f"{row['duration']:.0f}s" for _, row in hf_data.iterrows()]
    else:
        x_labels = [f"{int(row[sort_by])}" for _, row in hf_data.iterrows()]
    
    # 1. TTFT 对比
    ax = axes[0, 0]
    ax.bar(x - width/2, hf_data["ttft_ms"] / 1000, width, label="HuggingFace", color="#3498db")
    ax.bar(x + width/2, vllm_data["ttft_ms"] / 1000, width, label="vLLM", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel(f"Video ({sort_by})")
    ax.set_ylabel("TTFT (seconds)")
    ax.set_title("Time To First Token (TTFT) Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # 2. Total Time 对比
    ax = axes[0, 1]
    ax.bar(x - width/2, hf_data["total_ms"] / 1000, width, label="HuggingFace", color="#3498db")
    ax.bar(x + width/2, vllm_data["total_ms"] / 1000, width, label="vLLM", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel(f"Video ({sort_by})")
    ax.set_ylabel("Total Time (seconds)")
    ax.set_title("Total Generation Time Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # 3. Throughput 对比
    ax = axes[1, 0]
    ax.bar(x - width/2, hf_data["tokens_per_sec"], width, label="HuggingFace", color="#3498db")
    ax.bar(x + width/2, vllm_data["tokens_per_sec"], width, label="vLLM", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel(f"Video ({sort_by})")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # 4. TTFT 差异百分比
    ax = axes[1, 1]
    ttft_diff_pct = (vllm_data["ttft_ms"].values - hf_data["ttft_ms"].values) / hf_data["ttft_ms"].values * 100
    colors = ["#2ecc71" if d < 0 else "#e74c3c" for d in ttft_diff_pct]
    ax.bar(x, ttft_diff_pct, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel(f"Video ({sort_by})")
    ax.set_ylabel("TTFT Difference (%)")
    ax.set_title("vLLM TTFT vs HF\n(negative = vLLM faster, positive = HF faster)")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "vllm_vs_hf_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    # 统计
    print("\n" + "=" * 60)
    print(f"Summary (sorted by {sort_by})")
    print("=" * 60)
    print(f"HF mean TTFT:   {hf_data['ttft_ms'].mean():.1f} ms")
    print(f"vLLM mean TTFT: {vllm_data['ttft_ms'].mean():.1f} ms")
    print(f"Mean difference: {ttft_diff_pct.mean():.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Replot TTFT experiment results")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV results file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--sort-by", type=str, default="visual_tokens", 
                        help="Column to sort by (visual_tokens, duration, prompt_tokens, etc.)")
    parser.add_argument("--type", type=str, choices=["ttft", "vllm"], default="ttft",
                        help="Type of plot: ttft (breakdown) or vllm (comparison)")
    parser.add_argument("--warmup", type=int, default=0, help="Number of warmup repeats to skip")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return
    
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Columns: {list(df.columns)}")
    
    if args.type == "ttft":
        plot_ttft_breakdown(df, args.output, sort_by=args.sort_by, warmup_repeats=args.warmup)
    else:
        plot_vllm_comparison(df, args.output, sort_by=args.sort_by)


if __name__ == "__main__":
    main()
