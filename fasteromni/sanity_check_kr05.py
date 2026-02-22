#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import pandas as pd


DEFAULT_BASELINE_CSV = "/root/autodl-tmp/results/fasteromni/videomme_full/baseline/baseline_details.csv"
DEFAULT_KR05_CSV = "/root/autodl-tmp/results/fasteromni/pareto_naive_iframe/naive_iframe_kr0.5/videomme_combined_details.csv"
DEFAULT_OUT_JSON = "/root/autodl-tmp/results/fasteromni/sanity_check_kr05_report.json"

REQUIRED_COLUMNS = [
    "question_id",
    "video_file_id",
    "duration",
    "pred_answer",
    "gt_answer",
    "correct",
    "visual_tokens",
    "num_frames",
    "generate_ms",
    "error",
]


def _normalize_error(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"nan": "", "None": "", "none": "", "null": "", "NaN": ""})
    return s


def _parse_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "yes"])


def _verdict_text(v: str) -> str:
    if v == "pass":
        return "✅"
    if v == "fail":
        return "❌"
    return "⚠️"


def _load_and_filter(path: str, duration: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV missing columns: {missing}")

    df = df.copy()
    df["duration"] = df["duration"].astype(str).str.strip().str.lower()
    df = df[df["duration"] == duration].copy()

    df["error_norm"] = _normalize_error(df["error"])
    df = df[df["error_norm"] == ""].copy()

    # question_id 去重（保留最后一条）
    df = df.drop_duplicates(subset=["question_id"], keep="last").copy()

    df["pred_answer"] = df["pred_answer"].astype(str).str.strip()
    df["gt_answer"] = df["gt_answer"].astype(str).str.strip()
    df["correct_bool"] = _parse_bool(df["correct"])
    df["visual_tokens"] = pd.to_numeric(df["visual_tokens"], errors="coerce")
    df["num_frames"] = pd.to_numeric(df["num_frames"], errors="coerce")
    df["generate_ms"] = pd.to_numeric(df["generate_ms"], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check for naive_iframe kr=0.5 vs baseline")
    parser.add_argument("--baseline-csv", default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--kr05-csv", default=DEFAULT_KR05_CSV)
    parser.add_argument("--duration", default="short")
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    duration = args.duration.strip().lower()
    bl = _load_and_filter(args.baseline_csv, duration, "baseline")
    kr = _load_and_filter(args.kr05_csv, duration, "kr0.5")
    if len(bl) == 0:
        raise ValueError(f"baseline has no valid rows after filtering duration={duration!r}")
    if len(kr) == 0:
        raise ValueError(f"kr0.5 has no valid rows after filtering duration={duration!r}")

    merged = bl.merge(
        kr,
        on="question_id",
        how="inner",
        suffixes=("_bl", "_kr"),
    )
    if len(merged) == 0:
        raise ValueError(f"No overlapping question_id between baseline and kr0.5 for duration={duration!r}")

    # A1: 逐样本预测一致性
    total = int(len(merged))
    agree = int((merged["pred_answer_bl"] == merged["pred_answer_kr"]).sum())
    disagree = int(total - agree)
    agree_pct = (agree / total * 100.0) if total > 0 else 0.0
    if agree_pct > 95.0:
        a1_verdict = "warning"
        a1_msg = "高度怀疑 pipeline bug（预测几乎完全一致）"
    elif agree_pct < 80.0:
        a1_verdict = "pass"
        a1_msg = "帧选择影响了模型行为"
    else:
        a1_verdict = "warning"
        a1_msg = "有差异但不够强，建议进一步排查"

    # A2: Visual Token 对比（按 video 去重）
    bl_vid = bl.groupby("video_file_id", as_index=False)[["visual_tokens", "num_frames"]].mean()
    kr_vid = kr.groupby("video_file_id", as_index=False)[["visual_tokens", "num_frames"]].mean()
    vid_join = bl_vid.merge(kr_vid, on="video_file_id", how="inner", suffixes=("_bl", "_kr"))

    bl_tok_mean = float(vid_join["visual_tokens_bl"].mean()) if len(vid_join) > 0 else 0.0
    kr_tok_mean = float(vid_join["visual_tokens_kr"].mean()) if len(vid_join) > 0 else 0.0
    tok_reduction_pct = ((bl_tok_mean - kr_tok_mean) / bl_tok_mean * 100.0) if bl_tok_mean > 0 else 0.0
    bl_frame_mean = float(vid_join["num_frames_bl"].mean()) if len(vid_join) > 0 else 0.0
    kr_frame_mean = float(vid_join["num_frames_kr"].mean()) if len(vid_join) > 0 else 0.0

    if tok_reduction_pct >= 20.0 and kr_tok_mean < bl_tok_mean:
        a2_verdict = "pass"
        a2_msg = "帧选择生效（visual tokens 显著下降）"
    else:
        a2_verdict = "fail"
        a2_msg = "帧选择未生效（visual tokens 未显著下降）"

    # A3: num_frames 对比
    bl_frames = vid_join["num_frames_bl"].dropna()
    kr_frames = vid_join["num_frames_kr"].dropna()
    bl_frames_min = int(bl_frames.min()) if len(bl_frames) > 0 else 0
    bl_frames_max = int(bl_frames.max()) if len(bl_frames) > 0 else 0
    kr_frames_min = int(kr_frames.min()) if len(kr_frames) > 0 else 0
    kr_frames_max = int(kr_frames.max()) if len(kr_frames) > 0 else 0

    if len(bl_frames) > 0 and len(kr_frames) > 0 and (
        abs(bl_frame_mean - kr_frame_mean) >= 1.0 or bl_frames_min != kr_frames_min or bl_frames_max != kr_frames_max
    ):
        a3_verdict = "pass"
        a3_msg = "帧数不同，kr=0.5 未被截断成 baseline 相同输入"
    else:
        a3_verdict = "fail"
        a3_msg = "帧数几乎相同，怀疑 max_frames 截断导致输入一致"

    # A4: 翻转分析
    merged["correct_bl"] = merged["correct_bool_bl"]
    merged["correct_kr"] = merged["correct_bool_kr"]
    both_correct = int((merged["correct_bl"] & merged["correct_kr"]).sum())
    both_wrong = int((~merged["correct_bl"] & ~merged["correct_kr"]).sum())
    degraded_df = merged[(merged["correct_bl"]) & (~merged["correct_kr"])].copy()
    improved_df = merged[(~merged["correct_bl"]) & (merged["correct_kr"])].copy()
    degraded = int(len(degraded_df))
    improved = int(len(improved_df))

    degraded_details = []
    for _, r in degraded_df.sort_values("question_id").iterrows():
        degraded_details.append({
            "question_id": str(r["question_id"]),
            "video_file_id": str(r["video_file_id_bl"]),
            "gt_answer": str(r["gt_answer_bl"]),
            "baseline_pred": str(r["pred_answer_bl"]),
            "kr05_pred": str(r["pred_answer_kr"]),
        })

    improved_details = []
    for _, r in improved_df.sort_values("question_id").iterrows():
        improved_details.append({
            "question_id": str(r["question_id"]),
            "video_file_id": str(r["video_file_id_bl"]),
            "gt_answer": str(r["gt_answer_bl"]),
            "baseline_pred": str(r["pred_answer_bl"]),
            "kr05_pred": str(r["pred_answer_kr"]),
        })

    if degraded == 0 and improved == 0:
        a4_verdict = "fail"
        a4_msg = "预测完全一致，高度怀疑 pipeline bug"
    elif degraded > 0 and improved > 0:
        a4_verdict = "pass"
        a4_msg = "存在双向翻转，说明帧选择改变了模型行为"
    else:
        a4_verdict = "warning"
        a4_msg = "仅单向翻转，行为有变化但建议进一步检查"

    # 综合结论
    if "fail" in [a2_verdict, a3_verdict, a4_verdict]:
        overall = "FAIL: pipeline bug（帧选择可能未生效或输入未变化）"
    elif a1_verdict == "warning":
        overall = "WARNING: 帧选择可能生效，但预测一致性偏高，建议继续排查"
    else:
        overall = "PASS: 帧选择生效且准确率巧合相等"

    print(f"=== Sanity Check: kr=0.5 vs Baseline (Video-MME {duration.capitalize()}) ===\n")

    print("[A1] 预测一致性")
    print(f"  总题数: {total}")
    print(f"  一致: {agree} ({agree_pct:.1f}%)")
    print(f"  不一致: {disagree} ({100.0 - agree_pct:.1f}%)")
    print(f"  → 判定: {_verdict_text(a1_verdict)} {a1_msg}\n")

    print("[A2] Visual Token 对比")
    print(f"  Baseline 平均: {bl_tok_mean:.1f} tokens ({bl_frame_mean:.1f} 帧)")
    print(f"  kr=0.5 平均: {kr_tok_mean:.1f} tokens ({kr_frame_mean:.1f} 帧)")
    print(f"  Token 减少: {tok_reduction_pct:.1f}%")
    print(f"  → 判定: {_verdict_text(a2_verdict)} {a2_msg}\n")

    print("[A3] num_frames 对比")
    if len(bl_frames) > 0:
        if bl_frames_min == bl_frames_max:
            print(f"  Baseline: 全部 {bl_frames_min} 帧")
        else:
            print(f"  Baseline: 平均 {bl_frame_mean:.1f} 帧, 范围 [{bl_frames_min}, {bl_frames_max}]")
    else:
        print("  Baseline: N/A")
    print(f"  kr=0.5: 平均 {kr_frame_mean:.1f} 帧, 范围 [{kr_frames_min}, {kr_frames_max}]")
    print(f"  → 判定: {_verdict_text(a3_verdict)} {a3_msg}\n")

    print("[A4] 翻转分析")
    print(f"  Both correct: {both_correct}")
    print(f"  Both wrong: {both_wrong}")
    print(f"  Degraded (BL✓→kr✗): {degraded}")
    print(f"  Improved (BL✗→kr✓): {improved}")
    print(f"  → 判定: {_verdict_text(a4_verdict)} {a4_msg}")
    if degraded_details:
        print("  Degraded details:")
        for item in degraded_details:
            print(
                f"    - {item['question_id']} | gt={item['gt_answer']} | "
                f"BL={item['baseline_pred']} -> kr={item['kr05_pred']}"
            )
    if improved_details:
        print("  Improved details:")
        for item in improved_details:
            print(
                f"    - {item['question_id']} | gt={item['gt_answer']} | "
                f"BL={item['baseline_pred']} -> kr={item['kr05_pred']}"
            )

    print("\n=== 综合结论 ===")
    print(overall)

    report = {
        "check_time": datetime.now().isoformat(timespec="seconds"),
        "data_sources": {
            "baseline": args.baseline_csv,
            "kr05": args.kr05_csv,
        },
        "duration": duration,
        "A1_prediction_agreement": {
            "total": total,
            "agree": agree,
            "agree_pct": round(agree_pct, 4),
            "verdict": a1_verdict,
        },
        "A2_visual_tokens": {
            "baseline_mean": bl_tok_mean,
            "kr05_mean": kr_tok_mean,
            "reduction_pct": tok_reduction_pct,
            "baseline_mean_frames": bl_frame_mean,
            "kr05_mean_frames": kr_frame_mean,
            "verdict": a2_verdict,
        },
        "A3_num_frames": {
            "baseline_mean": bl_frame_mean,
            "baseline_range": [bl_frames_min, bl_frames_max],
            "kr05_mean": kr_frame_mean,
            "kr05_range": [kr_frames_min, kr_frames_max],
            "verdict": a3_verdict,
        },
        "A4_flips": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "degraded": degraded,
            "improved": improved,
            "degraded_details": degraded_details,
            "improved_details": improved_details,
            "verdict": a4_verdict,
        },
        "overall_verdict": overall,
    }

    out_path = args.out_json
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
