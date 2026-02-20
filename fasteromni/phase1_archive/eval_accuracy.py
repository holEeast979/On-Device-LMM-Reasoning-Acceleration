"""
准确率评估脚本

在 ActivityNet-QA 上对比 baseline vs sparse 的准确率。
使用规范化 EM 评估器（对齐 VQA 学术标准）。

支持两种模式：
- TTFT 模式 (max_new_tokens=1)：测量真正的首 token 延迟
- 准确率模式 (max_new_tokens=32)：生成完整答案用于评估

Usage:
    python fasteromni/eval_accuracy.py [--num-samples 50] [--keep-ratio 0.5]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.pipeline import SparseInferencePipeline, PipelineResult
from fasteromni.evaluator import evaluate_answer as normalized_eval


ANET_ANNOTATIONS = "/root/autodl-tmp/data/ActivityNet-QA/annotations/activitynet_qa_test.json"
ANET_VIDEO_DIR = "/root/autodl-tmp/data/ActivityNet-QA/videos"


def load_short_qa_samples(max_duration: float = 35.0) -> list[dict]:
    """加载短视频的 QA 样本"""
    import av

    with open(ANET_ANNOTATIONS) as f:
        all_qa = json.load(f)

    # 匹配视频文件
    video_files = {}
    for f in os.listdir(ANET_VIDEO_DIR):
        if f.endswith(".mp4"):
            name = f[2:-4] if f.startswith("v_") else f[:-4]
            video_files[name] = os.path.join(ANET_VIDEO_DIR, f)

    # 过滤有视频 + 短时长的样本
    samples = []
    seen_videos = {}  # 缓存视频时长

    for item in all_qa:
        vid = item["video_name"]
        if vid not in video_files:
            continue

        vpath = video_files[vid]

        # 缓存时长避免重复打开
        if vid not in seen_videos:
            try:
                c = av.open(vpath)
                s = c.streams.video[0]
                dur = float(s.duration * float(s.time_base)) if s.duration else 0
                c.close()
                seen_videos[vid] = dur
            except Exception:
                seen_videos[vid] = 999
                continue

        if seen_videos[vid] <= max_duration:
            samples.append({
                "video_name": vid,
                "video_path": vpath,
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item.get("type", ""),
                "duration": seen_videos[vid],
            })

    return samples


def evaluate_answer(predicted: str, ground_truth: str) -> bool:
    """规范化 EM 评估（委托给 evaluator.py）"""
    result = normalized_eval(predicted, ground_truth)
    return result.correct


def run_evaluation(
    pipe: SparseInferencePipeline,
    samples: list[dict],
    keep_ratio: float = 0.5,
    alpha: float = 0.5,
    max_new_tokens: int = 32,
    randomize_order: bool = True,
) -> dict:
    """
    运行 baseline 和 sparse 评估。

    每个样本随机先跑 baseline 或 sparse，消除 warm cache 偏置。
    """
    results = {
        "baseline": {"correct": 0, "total": 0, "errors": 0, "details": []},
        "sparse": {"correct": 0, "total": 0, "errors": 0, "details": []},
    }
    timing = {
        "baseline_gen_sum": 0.0,
        "sparse_gen_sum": 0.0,
    }

    for i, sample in enumerate(samples):
        fname = os.path.basename(sample["video_path"])
        q = sample["question"]
        gt = sample["answer"]
        prompt = f"{q} Give a short answer:"

        print(f"[{i+1}/{len(samples)}] {fname} | Q: {q[:50]}... | GT: {gt}", flush=True)

        # 随机化运行顺序（消除 warm cache 偏置）
        run_baseline_first = random.random() < 0.5 if randomize_order else True
        modes = ["baseline", "sparse"] if run_baseline_first else ["sparse", "baseline"]

        for mode in modes:
            try:
                if mode == "baseline":
                    r = pipe.run_baseline(sample["video_path"], prompt, max_new_tokens)
                else:
                    r = pipe.run_sparse(
                        sample["video_path"], prompt, max_new_tokens,
                        alpha=alpha, keep_ratio=keep_ratio,
                    )

                if r.error:
                    print(f"  {mode:8s} ERROR: {r.error}")
                    results[mode]["errors"] += 1
                else:
                    eval_result = normalized_eval(r.output_text, gt)
                    correct = eval_result.correct
                    results[mode]["correct"] += int(correct)
                    results[mode]["total"] += 1
                    timing[f"{mode}_gen_sum"] += r.generate_ms
                    detail = {
                        "qid": sample["question_id"],
                        "predicted": r.output_text[:200],
                        "gt": gt,
                        "correct": correct,
                        "match_type": eval_result.match_type,
                        "generate_ms": r.generate_ms,
                        "visual_tokens": r.visual_tokens,
                        "audio_tokens": r.audio_tokens,
                        "total_tokens": r.total_tokens,
                    }
                    if mode == "sparse":
                        detail["num_frames"] = r.num_frames_input
                        detail["selected_gops"] = r.selected_gops
                        detail["total_gops"] = r.total_gops
                    results[mode]["details"].append(detail)
                    mark = "✓" if correct else "✗"
                    frames_info = f", {r.num_frames_input}f" if mode == "sparse" else ""
                    print(f"  {mode:8s}: {mark} [{eval_result.match_type:8s}] "
                          f"({r.output_text[:50]}...) gen={r.generate_ms:.0f}ms{frames_info}")
            except Exception as e:
                print(f"  {mode:8s} EXCEPTION: {e}")
                results[mode]["errors"] += 1

        print()

    # 汇总
    for mode in ["baseline", "sparse"]:
        r = results[mode]
        r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0
        gen_key = f"{mode}_gen_sum"
        r["avg_generate_ms"] = timing[gen_key] / r["total"] if r["total"] > 0 else 0

    return results


def print_summary(results: dict) -> None:
    """打印评估摘要"""
    print("\n" + "=" * 70)
    print("ACCURACY EVALUATION SUMMARY")
    print("=" * 70)

    for mode in ["baseline", "sparse"]:
        r = results[mode]
        print(f"\n--- {mode.upper()} ---")
        print(f"  Accuracy: {r['correct']}/{r['total']} = {r['accuracy']:.1%}")
        print(f"  Avg Generate: {r['avg_generate_ms']:.0f}ms")
        print(f"  Errors: {r['errors']}")

    # 比较
    b = results["baseline"]
    s = results["sparse"]
    if b["total"] > 0 and s["total"] > 0:
        acc_drop = b["accuracy"] - s["accuracy"]
        speedup = b["avg_generate_ms"] / s["avg_generate_ms"] if s["avg_generate_ms"] > 0 else 0
        print(f"\n--- COMPARISON ---")
        print(f"  Accuracy drop: {acc_drop:+.1%}")
        print(f"  Generate speedup: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Accuracy Evaluation")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=35.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/root/autodl-tmp/results/fasteromni/accuracy")
    args = parser.parse_args()

    print("=" * 70)
    print("FasterOmni - Accuracy Evaluation on ActivityNet-QA")
    print(f"keep_ratio={args.keep_ratio}, alpha={args.alpha}, "
          f"max_duration={args.max_duration}s, max_new_tokens={args.max_new_tokens}")
    print("=" * 70)

    # 加载样本
    samples = load_short_qa_samples(max_duration=args.max_duration)
    print(f"\nFound {len(samples)} short QA samples")

    # 采样
    import random
    random.seed(args.seed)
    if len(samples) > args.num_samples:
        samples = random.sample(samples, args.num_samples)
    print(f"Using {len(samples)} samples\n")

    # 初始化 Pipeline
    pipe = SparseInferencePipeline(dtype="bf16")

    # 评估
    results = run_evaluation(
        pipe, samples,
        keep_ratio=args.keep_ratio,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
    )

    # 打印摘要
    print_summary(results)

    # 保存
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "accuracy_results.json")
    # Remove non-serializable items
    save_results = {
        "params": {
            "keep_ratio": args.keep_ratio,
            "alpha": args.alpha,
            "num_samples": len(samples),
            "max_duration": args.max_duration,
        },
        "baseline": {k: v for k, v in results["baseline"].items()},
        "sparse": {k: v for k, v in results["sparse"].items()},
    }
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
