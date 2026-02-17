"""
Video-MME 评估脚本

选择题格式，评估零歧义（pred == gt）。
支持 baseline / sparse / sparse-no-audio 三种模式。

Usage:
    # 快速验证（5 个视频）
    python fasteromni/eval_videomme.py --max-videos 5

    # 完整评估（全部已下载视频）
    python fasteromni/eval_videomme.py

    # 消融实验
    python fasteromni/eval_videomme.py --sweep keep_ratio
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from fasteromni.pipeline import SparseInferencePipeline, PipelineResult

# ── 路径 ──────────────────────────────────────────────────

VMME_ANNOTATIONS = "/root/autodl-tmp/data/Video-MME/annotations/video_mme_test.json"
VMME_VIDEO_DIR = "/root/autodl-tmp/data/Video-MME/videos/data"


# ── 数据加载 ──────────────────────────────────────────────

@dataclass
class VideoMMESample:
    """Video-MME 单条 QA"""
    video_id: str           # e.g. "001"
    video_file_id: str      # e.g. "fFjv93ACGo8" (YouTube ID)
    video_path: str
    question_id: str
    question: str
    options: List[str]      # ["A. ...", "B. ...", "C. ...", "D. ..."]
    answer: str             # "A" / "B" / "C" / "D"
    duration: str           # "short" / "medium" / "long"
    domain: str
    sub_category: str
    task_type: str


def load_videomme_samples(max_videos: int = 0) -> List[VideoMMESample]:
    """加载 Video-MME 样本（仅已下载的视频）"""
    with open(VMME_ANNOTATIONS) as f:
        data = json.load(f)

    # 已下载的视频
    downloaded = {}
    for fname in os.listdir(VMME_VIDEO_DIR):
        if fname.endswith(".mp4"):
            vid = os.path.splitext(fname)[0]
            downloaded[vid] = os.path.join(VMME_VIDEO_DIR, fname)

    # 匹配
    samples = []
    matched_videos = set()
    for item in data:
        file_id = item["videoID"]
        if file_id not in downloaded:
            continue

        matched_videos.add(file_id)
        if max_videos > 0 and len(matched_videos) > max_videos:
            break

        samples.append(VideoMMESample(
            video_id=item["video_id"],
            video_file_id=file_id,
            video_path=downloaded[file_id],
            question_id=item["question_id"],
            question=item["question"],
            options=item["options"],
            answer=item["answer"],
            duration=item["duration"],
            domain=item["domain"],
            sub_category=item.get("sub_category", ""),
            task_type=item.get("task_type", ""),
        ))

    return samples


# ── Prompt 格式化 ─────────────────────────────────────────

def format_mcq_prompt(question: str, options: List[str]) -> str:
    """
    格式化选择题 prompt。

    输出格式：
    Question: <question>
    A. <option_a>
    B. <option_b>
    C. <option_c>
    D. <option_d>
    Answer with the option letter only (A, B, C, or D).
    """
    opts = "\n".join(options)
    return (
        f"{question}\n{opts}\n"
        f"Answer with the option letter only (A, B, C, or D)."
    )


def extract_answer_letter(output: str) -> Optional[str]:
    """
    从模型输出中提取选项字母 A/B/C/D。

    策略（按优先级）：
    1. 输出恰好是单个字母
    2. 输出以字母开头（如 "A." 或 "A. Apples"）
    3. 在输出中找第一个出现的 A/B/C/D
    """
    text = output.strip()

    # 1. 单个字母
    if text.upper() in ("A", "B", "C", "D"):
        return text.upper()

    # 2. 以字母开头
    m = re.match(r"^([A-Da-d])[.\s,)]", text)
    if m:
        return m.group(1).upper()

    # 3. 找第一个 A/B/C/D（作为独立字母出现）
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        letter = m.group(1).upper()
        if letter in ("A", "B", "C", "D"):
            return letter

    # 4. 兜底：在整个文本中找
    for ch in text.upper():
        if ch in ("A", "B", "C", "D"):
            return ch

    return None


# ── 单条评估 ──────────────────────────────────────────────

@dataclass
class EvalRecord:
    """单条评估记录"""
    question_id: str
    video_file_id: str
    duration: str
    domain: str
    task_type: str
    mode: str               # "baseline" / "sparse" / "sparse_no_audio"
    keep_ratio: float = 0.0
    alpha: float = 0.0
    gt_answer: str = ""
    pred_answer: Optional[str] = None
    pred_raw: str = ""
    correct: bool = False
    generate_ms: float = 0.0
    total_ms: float = 0.0
    visual_tokens: int = 0
    audio_tokens: int = 0
    total_tokens: int = 0
    num_frames: int = 0
    error: str = ""


def run_single(
    pipe: SparseInferencePipeline,
    sample: VideoMMESample,
    mode: str,
    keep_ratio: float = 0.5,
    alpha: float = 0.5,
    max_new_tokens: int = 16,
    max_frames: int = 64,
) -> EvalRecord:
    """运行单条 QA 评估"""
    prompt = format_mcq_prompt(sample.question, sample.options)
    record = EvalRecord(
        question_id=sample.question_id,
        video_file_id=sample.video_file_id,
        duration=sample.duration,
        domain=sample.domain,
        task_type=sample.task_type,
        mode=mode,
        keep_ratio=keep_ratio,
        alpha=alpha,
        gt_answer=sample.answer,
    )

    try:
        if mode == "baseline":
            r = pipe.run_baseline(sample.video_path, prompt, max_new_tokens, max_frames=max_frames)
        elif mode == "sparse":
            r = pipe.run_sparse(
                sample.video_path, prompt, max_new_tokens,
                alpha=alpha, keep_ratio=keep_ratio,
            )
        elif mode == "sparse_no_audio":
            r = pipe.run_sparse(
                sample.video_path, prompt, max_new_tokens,
                alpha=alpha, keep_ratio=keep_ratio,
                skip_audio=True,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if r.error:
            record.error = r.error
        else:
            record.pred_raw = r.output_text
            record.pred_answer = extract_answer_letter(r.output_text)
            record.correct = (record.pred_answer == record.gt_answer)
            record.generate_ms = r.generate_ms
            record.total_ms = r.total_ms
            record.visual_tokens = r.visual_tokens
            record.audio_tokens = r.audio_tokens
            record.total_tokens = r.total_tokens
            record.num_frames = r.num_frames_input

    except torch.cuda.OutOfMemoryError:
        record.error = "OOM"
        pipe._clear_gpu()
    except Exception as e:
        record.error = str(e)

    return record


# ── 批量评估 ──────────────────────────────────────────────

import torch

def run_evaluation(
    pipe: SparseInferencePipeline,
    samples: List[VideoMMESample],
    mode: str,
    keep_ratio: float = 0.5,
    alpha: float = 0.5,
    max_new_tokens: int = 16,
    max_frames: int = 64,
) -> List[EvalRecord]:
    """运行一组评估"""
    records = []
    correct = 0
    total = 0
    errors = 0

    for i, sample in enumerate(samples):
        rec = run_single(pipe, sample, mode, keep_ratio, alpha, max_new_tokens, max_frames)
        records.append(rec)

        if rec.error:
            errors += 1
            status = f"ERR:{rec.error[:20]}"
        else:
            total += 1
            correct += int(rec.correct)
            mark = "✓" if rec.correct else "✗"
            status = f"{mark} pred={rec.pred_answer} gt={rec.gt_answer}"

        # 每 10 条或最后一条输出进度
        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            acc = correct / total * 100 if total > 0 else 0
            print(f"  [{i+1}/{len(samples)}] acc={acc:.1f}% ({correct}/{total}) err={errors} "
                  f"| {sample.duration} {status}", flush=True)

    return records


# ── 结果汇总 ──────────────────────────────────────────────

def summarize_records(records: List[EvalRecord], label: str = "") -> Dict:
    """汇总评估记录"""
    valid = [r for r in records if not r.error]
    by_dur = defaultdict(list)
    for r in valid:
        by_dur[r.duration].append(r)

    summary = {
        "label": label,
        "total_samples": len(records),
        "valid_samples": len(valid),
        "errors": len(records) - len(valid),
        "overall_accuracy": sum(r.correct for r in valid) / len(valid) * 100 if valid else 0,
        "avg_generate_ms": sum(r.generate_ms for r in valid) / len(valid) if valid else 0,
        "avg_visual_tokens": sum(r.visual_tokens for r in valid) / len(valid) if valid else 0,
        "by_duration": {},
    }

    for dur in ["short", "medium", "long"]:
        recs = by_dur.get(dur, [])
        if recs:
            summary["by_duration"][dur] = {
                "count": len(recs),
                "accuracy": sum(r.correct for r in recs) / len(recs) * 100,
                "avg_generate_ms": sum(r.generate_ms for r in recs) / len(recs),
                "avg_visual_tokens": sum(r.visual_tokens for r in recs) / len(recs),
            }

    return summary


def print_summary(summaries: List[Dict]):
    """打印汇总表格"""
    print(f"\n{'='*90}")
    print(f"VIDEO-MME EVALUATION SUMMARY")
    print(f"{'='*90}")

    # 总表
    print(f"\n{'Mode':>20} | {'Accuracy':>10} | {'N':>5} | {'Err':>4} | "
          f"{'Gen(ms)':>10} | {'VisTok':>8}")
    print("-" * 75)
    for s in summaries:
        print(f"{s['label']:>20} | {s['overall_accuracy']:>9.1f}% | "
              f"{s['valid_samples']:>5} | {s['errors']:>4} | "
              f"{s['avg_generate_ms']:>9.0f} | {s['avg_visual_tokens']:>8.0f}")

    # 按时长分组
    print(f"\n{'Mode':>20} | {'short':>10} | {'medium':>10} | {'long':>10}")
    print("-" * 60)
    for s in summaries:
        parts = []
        for dur in ["short", "medium", "long"]:
            d = s["by_duration"].get(dur)
            if d:
                parts.append(f"{d['accuracy']:>5.1f}%({d['count']})")
            else:
                parts.append(f"{'N/A':>10}")
        print(f"{s['label']:>20} | {'  |  '.join(parts)}")


# ── 保存 ──────────────────────────────────────────────────

def save_results(records: List[EvalRecord], summaries: List[Dict], out_dir: str, tag: str = "eval"):
    """保存结果"""
    os.makedirs(out_dir, exist_ok=True)

    # CSV 详细记录
    csv_path = os.path.join(out_dir, f"videomme_{tag}_details.csv")
    fieldnames = [
        "question_id", "video_file_id", "duration", "domain", "task_type",
        "mode", "keep_ratio", "alpha", "gt_answer", "pred_answer", "correct",
        "generate_ms", "total_ms", "visual_tokens", "audio_tokens", "total_tokens",
        "num_frames", "error", "pred_raw",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: getattr(r, k) for k in fieldnames})

    # JSON 汇总
    json_path = os.path.join(out_dir, f"videomme_{tag}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_dir}")
    print(f"  Details: {csv_path}")
    print(f"  Summary: {json_path}")
    return csv_path


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Video-MME Evaluation")
    parser.add_argument("--max-videos", type=int, default=0,
                        help="Max videos to evaluate (0 = all downloaded)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=32,
                        help="Max frames for baseline (0=unlimited, 32 for 32GB GPU)")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=["baseline", "sparse"],
                        choices=["baseline", "sparse", "sparse_no_audio"],
                        help="Modes to evaluate")
    parser.add_argument("--sweep", choices=["keep_ratio", "alpha", "none"], default="none",
                        help="Run ablation sweep")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/results/fasteromni/videomme")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("FasterOmni - Video-MME Evaluation")
    print(f"modes={args.modes}, keep_ratio={args.keep_ratio}, alpha={args.alpha}")
    print("=" * 70)

    # 加载样本
    samples = load_videomme_samples(max_videos=args.max_videos)
    n_vids = len(set(s.video_file_id for s in samples))
    dur_counts = defaultdict(int)
    for s in samples:
        dur_counts[s.duration] += 1
    print(f"Loaded {len(samples)} QA from {n_vids} videos")
    print(f"  short={dur_counts['short']}, medium={dur_counts['medium']}, long={dur_counts['long']}")

    # 初始化
    pipe = SparseInferencePipeline(dtype="bf16")

    all_records = []
    all_summaries = []

    if args.sweep == "none":
        # 单配置评估
        for mode in args.modes:
            print(f"\n{'='*60}")
            print(f"Running {mode.upper()} ({len(samples)} samples)")
            print(f"{'='*60}")
            records = run_evaluation(
                pipe, samples, mode,
                keep_ratio=args.keep_ratio, alpha=args.alpha,
                max_new_tokens=args.max_new_tokens,
                max_frames=args.max_frames,
            )
            all_records.extend(records)
            summary = summarize_records(records, label=f"{mode}(kr={args.keep_ratio})")
            all_summaries.append(summary)

        print_summary(all_summaries)
        save_results(all_records, all_summaries, args.out_dir, tag="eval")

    elif args.sweep == "keep_ratio":
        # keep_ratio 消融
        keep_ratios = [0.2, 0.3, 0.5, 0.7, 0.9]

        # baseline 只跑一次
        print(f"\n{'='*60}")
        print(f"Running BASELINE ({len(samples)} samples)")
        print(f"{'='*60}")
        base_records = run_evaluation(
            pipe, samples, "baseline",
            max_new_tokens=args.max_new_tokens,
            max_frames=args.max_frames,
        )
        all_records.extend(base_records)
        base_summary = summarize_records(base_records, label="baseline")
        all_summaries.append(base_summary)

        for kr in keep_ratios:
            print(f"\n{'='*60}")
            print(f"Running SPARSE kr={kr} ({len(samples)} samples)")
            print(f"{'='*60}")
            records = run_evaluation(
                pipe, samples, "sparse",
                keep_ratio=kr, alpha=args.alpha,
                max_new_tokens=args.max_new_tokens,
                max_frames=args.max_frames,
            )
            all_records.extend(records)
            summary = summarize_records(records, label=f"sparse(kr={kr})")
            all_summaries.append(summary)

        print_summary(all_summaries)
        save_results(all_records, all_summaries, args.out_dir, tag="ablation_kr")

    elif args.sweep == "alpha":
        alphas = [0.0, 0.3, 0.5, 0.7, 1.0]

        print(f"\n{'='*60}")
        print(f"Running BASELINE ({len(samples)} samples)")
        print(f"{'='*60}")
        base_records = run_evaluation(
            pipe, samples, "baseline",
            max_new_tokens=args.max_new_tokens,
            max_frames=args.max_frames,
        )
        all_records.extend(base_records)
        all_summaries.append(summarize_records(base_records, label="baseline"))

        for a in alphas:
            print(f"\n{'='*60}")
            print(f"Running SPARSE alpha={a} ({len(samples)} samples)")
            print(f"{'='*60}")
            records = run_evaluation(
                pipe, samples, "sparse",
                keep_ratio=args.keep_ratio, alpha=a,
                max_new_tokens=args.max_new_tokens,
                max_frames=args.max_frames,
            )
            all_records.extend(records)
            all_summaries.append(summarize_records(records, label=f"sparse(a={a})"))

        print_summary(all_summaries)
        save_results(all_records, all_summaries, args.out_dir, tag="ablation_alpha")


if __name__ == "__main__":
    main()
