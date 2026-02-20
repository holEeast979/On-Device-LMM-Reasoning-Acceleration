"""
Phase 1 消融实验脚本

扫描 keep_ratio 和 alpha，输出：
1. 每个配置的准确率 + 平均 generate 延迟
2. CSV 原始数据
3. 消融曲线图（Pareto 前沿）

Usage:
    # keep_ratio 消融（固定 alpha=0.5）
    python fasteromni/run_ablation.py --sweep keep_ratio --num-samples 50

    # alpha 消融（固定 keep_ratio=最优值）
    python fasteromni/run_ablation.py --sweep alpha --keep-ratio 0.5 --num-samples 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import List, Dict, Any

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.pipeline import SparseInferencePipeline
from fasteromni.evaluator import evaluate_answer as normalized_eval
from fasteromni.eval_accuracy import load_short_qa_samples


# ── 单配置评估 ──────────────────────────────────────────────

def run_single_config(
    pipe: SparseInferencePipeline,
    samples: List[dict],
    mode: str,                  # "baseline" or "sparse"
    keep_ratio: float = 0.5,
    alpha: float = 0.5,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """运行单个配置，返回汇总结果"""

    correct = 0
    total = 0
    errors = 0
    gen_sum = 0.0
    vis_tok_sum = 0
    aud_tok_sum = 0
    details = []

    for i, sample in enumerate(samples):
        q = sample["question"]
        gt = sample["answer"]
        prompt = f"{q} Give a short answer:"
        fname = os.path.basename(sample["video_path"])

        try:
            if mode == "baseline":
                r = pipe.run_baseline(sample["video_path"], prompt, max_new_tokens)
            else:
                r = pipe.run_sparse(
                    sample["video_path"], prompt, max_new_tokens,
                    alpha=alpha, keep_ratio=keep_ratio,
                )

            if r.error:
                errors += 1
                print(f"  [{i+1}/{len(samples)}] {fname} ERROR: {r.error}")
                continue

            ev = normalized_eval(r.output_text, gt)
            correct += int(ev.correct)
            total += 1
            gen_sum += r.generate_ms
            vis_tok_sum += r.visual_tokens
            aud_tok_sum += r.audio_tokens

            details.append({
                "qid": sample["question_id"],
                "predicted": r.output_text[:200],
                "gt": gt,
                "correct": ev.correct,
                "match_type": ev.match_type,
                "generate_ms": r.generate_ms,
                "visual_tokens": r.visual_tokens,
                "audio_tokens": r.audio_tokens,
                "num_frames": getattr(r, "num_frames_input", 0),
            })

            mark = "✓" if ev.correct else "✗"
            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                acc_so_far = correct / total if total > 0 else 0
                print(f"  [{i+1}/{len(samples)}] acc={acc_so_far:.1%} "
                      f"({correct}/{total}) avg_gen={gen_sum/total:.0f}ms")

        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(samples)}] {fname} EXCEPTION: {e}")

    accuracy = correct / total if total > 0 else 0
    avg_gen = gen_sum / total if total > 0 else 0
    avg_vis = vis_tok_sum / total if total > 0 else 0
    avg_aud = aud_tok_sum / total if total > 0 else 0

    return {
        "mode": mode,
        "keep_ratio": keep_ratio,
        "alpha": alpha,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "avg_generate_ms": avg_gen,
        "avg_visual_tokens": avg_vis,
        "avg_audio_tokens": avg_aud,
        "details": details,
    }


# ── 消融扫描 ──────────────────────────────────────────────

def run_ablation(
    pipe: SparseInferencePipeline,
    samples: List[dict],
    sweep: str,                  # "keep_ratio" or "alpha"
    keep_ratios: List[float] = None,
    alphas: List[float] = None,
    fixed_keep_ratio: float = 0.5,
    fixed_alpha: float = 0.5,
    max_new_tokens: int = 32,
) -> List[Dict]:
    """运行消融实验"""

    results = []

    # 1. 先跑一次 baseline（所有配置共享同一个 baseline）
    print(f"\n{'='*60}")
    print(f"Running BASELINE ({len(samples)} samples, max_new_tokens={max_new_tokens})")
    print(f"{'='*60}")
    baseline = run_single_config(pipe, samples, "baseline", max_new_tokens=max_new_tokens)
    results.append(baseline)
    print(f"  => Baseline accuracy: {baseline['accuracy']:.1%} ({baseline['correct']}/{baseline['total']})")
    print(f"  => Avg generate: {baseline['avg_generate_ms']:.0f}ms")

    # 2. 扫描 sparse 配置
    if sweep == "keep_ratio":
        configs = [(kr, fixed_alpha) for kr in (keep_ratios or [0.2, 0.3, 0.5, 0.7, 0.9])]
    elif sweep == "alpha":
        configs = [(fixed_keep_ratio, a) for a in (alphas or [0.0, 0.3, 0.5, 0.7, 1.0])]
    else:
        raise ValueError(f"Unknown sweep type: {sweep}")

    for kr, a in configs:
        print(f"\n{'='*60}")
        print(f"Running SPARSE keep_ratio={kr}, alpha={a} ({len(samples)} samples)")
        print(f"{'='*60}")
        sparse = run_single_config(
            pipe, samples, "sparse",
            keep_ratio=kr, alpha=a, max_new_tokens=max_new_tokens,
        )
        results.append(sparse)

        speedup = baseline["avg_generate_ms"] / sparse["avg_generate_ms"] if sparse["avg_generate_ms"] > 0 else 0
        acc_drop = baseline["accuracy"] - sparse["accuracy"]
        print(f"  => Accuracy: {sparse['accuracy']:.1%} (drop: {acc_drop:+.1%})")
        print(f"  => Avg generate: {sparse['avg_generate_ms']:.0f}ms (speedup: {speedup:.2f}x)")

    return results


# ── 输出 ──────────────────────────────────────────────────

def save_results(results: List[Dict], out_dir: str, sweep: str) -> str:
    """保存结果到 CSV 和 JSON"""
    os.makedirs(out_dir, exist_ok=True)

    # CSV（不含 details）
    csv_path = os.path.join(out_dir, f"ablation_{sweep}.csv")
    fieldnames = ["mode", "keep_ratio", "alpha", "accuracy", "correct", "total",
                  "errors", "avg_generate_ms", "avg_visual_tokens", "avg_audio_tokens"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    # JSON（含 details）
    json_path = os.path.join(out_dir, f"ablation_{sweep}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {csv_path}")
    return csv_path


def print_summary_table(results: List[Dict], sweep: str):
    """打印汇总表格"""
    print(f"\n{'='*80}")
    print(f"ABLATION SUMMARY ({sweep})")
    print(f"{'='*80}")

    baseline = results[0]
    b_acc = baseline["accuracy"]
    b_gen = baseline["avg_generate_ms"]

    sweep_col = "keep_ratio" if sweep == "keep_ratio" else "alpha"
    print(f"{'Mode':>10} | {sweep_col:>10} | {'Accuracy':>10} | {'Acc Drop':>10} | "
          f"{'Avg Gen(ms)':>12} | {'Speedup':>8} | {'Avg VisTok':>10}")
    print("-" * 80)

    print(f"{'baseline':>10} | {'':>10} | {b_acc:>9.1%} | {'':>10} | "
          f"{b_gen:>11.0f} | {'1.00x':>8} | {baseline['avg_visual_tokens']:>10.0f}")

    for r in results[1:]:
        acc = r["accuracy"]
        gen = r["avg_generate_ms"]
        speedup = b_gen / gen if gen > 0 else 0
        acc_drop = b_acc - acc
        val = r["keep_ratio"] if sweep == "keep_ratio" else r["alpha"]
        print(f"{'sparse':>10} | {val:>10.2f} | {acc:>9.1%} | {acc_drop:>+9.1%} | "
              f"{gen:>11.0f} | {speedup:>7.2f}x | {r['avg_visual_tokens']:>10.0f}")


def plot_ablation(results: List[Dict], sweep: str, out_dir: str):
    """画消融曲线图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    baseline = results[0]
    sparse_results = results[1:]

    if sweep == "keep_ratio":
        x_vals = [r["keep_ratio"] for r in sparse_results]
        x_label = "keep_ratio"
    else:
        x_vals = [r["alpha"] for r in sparse_results]
        x_label = "alpha"

    accuracies = [r["accuracy"] * 100 for r in sparse_results]
    speedups = [baseline["avg_generate_ms"] / r["avg_generate_ms"]
                if r["avg_generate_ms"] > 0 else 0 for r in sparse_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 左轴：准确率
    color1 = "#2196F3"
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel("Accuracy (%)", color=color1, fontsize=12)
    line1 = ax1.plot(x_vals, accuracies, "o-", color=color1, linewidth=2, markersize=8, label="Accuracy")
    ax1.axhline(y=baseline["accuracy"] * 100, color=color1, linestyle="--", alpha=0.5, label="Baseline Acc")
    ax1.tick_params(axis="y", labelcolor=color1)

    # 右轴：加速比
    ax2 = ax1.twinx()
    color2 = "#FF5722"
    ax2.set_ylabel("Speedup (x)", color=color2, fontsize=12)
    line2 = ax2.plot(x_vals, speedups, "s--", color=color2, linewidth=2, markersize=8, label="Speedup")
    ax2.tick_params(axis="y", labelcolor=color2)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=10)

    plt.title(f"Ablation: {sweep} (N={baseline['total']} samples)", fontsize=14)
    fig.tight_layout()

    plot_path = os.path.join(out_dir, f"ablation_{sweep}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {plot_path}")


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Ablation Experiment")
    parser.add_argument("--sweep", choices=["keep_ratio", "alpha"], required=True,
                        help="Which parameter to sweep")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--keep-ratio", type=float, default=0.5,
                        help="Fixed keep_ratio (for alpha sweep)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Fixed alpha (for keep_ratio sweep)")
    parser.add_argument("--max-duration", type=float, default=35.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/root/autodl-tmp/results/fasteromni/ablation")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print(f"FasterOmni Phase 1 - Ablation: {args.sweep}")
    print(f"num_samples={args.num_samples}, max_new_tokens={args.max_new_tokens}")
    print(f"fixed keep_ratio={args.keep_ratio}, fixed alpha={args.alpha}")
    print("=" * 60)

    # 加载样本
    samples = load_short_qa_samples(max_duration=args.max_duration)
    print(f"Found {len(samples)} short QA samples")

    if len(samples) > args.num_samples:
        samples = random.sample(samples, args.num_samples)
    print(f"Using {len(samples)} samples\n")

    # 初始化 Pipeline
    pipe = SparseInferencePipeline(dtype="bf16")

    # 运行消融
    t0 = time.perf_counter()
    results = run_ablation(
        pipe, samples,
        sweep=args.sweep,
        fixed_keep_ratio=args.keep_ratio,
        fixed_alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.perf_counter() - t0

    # 输出
    print_summary_table(results, args.sweep)
    csv_path = save_results(results, args.out_dir, args.sweep)
    plot_ablation(results, args.sweep, args.out_dir)

    print(f"\nTotal elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
