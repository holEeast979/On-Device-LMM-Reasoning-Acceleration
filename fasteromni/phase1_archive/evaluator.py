"""
规范化评估器 (Normalized Exact Match)

对齐 VQA 学术评估标准：
1. 文本规范化：lowercase + 去标点 + 去冠词 + 数字词→数字
2. 词形还原：NLTK WordNetLemmatizer（支持不规则变位）
3. 三种匹配模式：
   - yes/no 问题：提取首个 yes/no
   - 数字问题：提取首个数字
   - 开放问题：规范化后的包含匹配 + token F1

参考：
- VQA Challenge evaluation: https://visualqa.org/evaluation.html
- Video-ChatGPT ActivityNet-QA: GPT-judge (accuracy + score 1-5)
"""
from __future__ import annotations

import os
import re
import string
from dataclasses import dataclass
from typing import Optional

# 修复 AutoDL 环境下 OMP_NUM_THREADS=0 导致 libgomp 崩溃
os.environ.setdefault("OMP_NUM_THREADS", "1")

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.stem import WordNetLemmatizer

_LEMMATIZER = WordNetLemmatizer()


# ── 文本规范化 ──────────────────────────────────────────────

# 数字词 → 数字
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


# 常见同义词/变体（ground truth → 可接受的变体）
_SYNONYMS = {
    "grassland": {"grass", "lawn", "field", "meadow"},
    "bedroom": {"room", "bed room"},
    "outdoors": {"outdoor", "outside"},
    "indoors": {"indoor", "inside"},
}

# 冠词和常见停用词（VQA 标准做法：去除冠词）
_ARTICLES = {"a", "an", "the"}


def normalize_text(text: str) -> str:
    """VQA 标准文本规范化"""
    s = text.strip().lower()
    # 去标点
    s = s.translate(str.maketrans("", "", string.punctuation))
    # 数字词 → 数字
    tokens = s.split()
    tokens = [_NUM_WORDS.get(t, t) for t in tokens]
    # 去冠词
    tokens = [t for t in tokens if t not in _ARTICLES]
    return " ".join(tokens)


def _try_lemmatize(word: str) -> str:
    """NLTK 词形还原，尝试名词和动词两种词性，返回最短结果"""
    # 尝试动词和名词两种词性，取最短的结果
    lemma_v = _LEMMATIZER.lemmatize(word, "v")
    lemma_n = _LEMMATIZER.lemmatize(word, "n")
    candidates = {word, lemma_v, lemma_n}
    return min(candidates, key=len)


def lemmatize_text(text: str) -> str:
    """对每个 token 做词形还原"""
    tokens = text.split()
    return " ".join(_try_lemmatize(t) for t in tokens)


def _extract_first_answer(text: str) -> str:
    """提取模型输出的第一个"答案"部分（句号/逗号前的内容）"""
    # 取第一句
    for sep in [".", ",", "\n", ";"]:
        if sep in text:
            text = text[:text.index(sep)]
    return text.strip()


# ── 匹配逻辑 ──────────────────────────────────────────────

@dataclass
class EvalResult:
    """单个 QA 的评估结果"""
    correct: bool
    match_type: str     # "exact", "normalized", "lemma", "synonym", "contains", "f1"
    pred_norm: str      # 规范化后的预测
    gt_norm: str        # 规范化后的 GT
    f1_score: float = 0.0


def _compute_token_f1(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    """计算 token 级 F1 score"""
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(predicted: str, ground_truth: str) -> EvalResult:
    """
    规范化评估。

    匹配优先级：
    1. yes/no 精确匹配
    2. 数字精确匹配
    3. 规范化精确匹配
    4. 词形还原后匹配
    5. 同义词匹配
    6. 包含匹配（GT 在 pred 中）
    7. Token F1 > 0.5
    """
    gt_raw = ground_truth.strip().lower()
    pred_raw = predicted.strip().lower()

    # 提取第一个答案片段
    pred_first = _extract_first_answer(pred_raw)

    # 规范化
    gt_norm = normalize_text(gt_raw)
    pred_norm = normalize_text(pred_first)

    # ── 1. yes/no 问题 ──
    if gt_norm in ("yes", "no"):
        # 从 pred 提取第一个出现的 yes 或 no
        yes_pos = pred_norm.find("yes")
        no_pos = pred_norm.find("no")

        pred_answer = None
        if yes_pos >= 0 and no_pos >= 0:
            pred_answer = "yes" if yes_pos < no_pos else "no"
        elif yes_pos >= 0:
            pred_answer = "yes"
        elif no_pos >= 0:
            pred_answer = "no"

        correct = (pred_answer == gt_norm)
        return EvalResult(correct=correct, match_type="yesno",
                          pred_norm=pred_answer or pred_norm, gt_norm=gt_norm)

    # ── 2. 数字问题 ──
    if gt_norm.isdigit():
        # 提取 pred 中的数字
        numbers = re.findall(r"\b(\d+)\b", pred_norm)
        correct = gt_norm in numbers
        return EvalResult(correct=correct, match_type="number",
                          pred_norm=pred_norm, gt_norm=gt_norm)

    # ── 3. 规范化精确匹配 ──
    if gt_norm == pred_norm:
        return EvalResult(correct=True, match_type="exact",
                          pred_norm=pred_norm, gt_norm=gt_norm)

    # ── 4. 词形还原后匹配 ──
    gt_lemma = lemmatize_text(gt_norm)
    pred_lemma = lemmatize_text(pred_norm)
    if gt_lemma == pred_lemma:
        return EvalResult(correct=True, match_type="lemma",
                          pred_norm=pred_lemma, gt_norm=gt_lemma)

    # ── 5. 同义词匹配 ──
    for canonical, synonyms in _SYNONYMS.items():
        all_forms = {canonical} | synonyms
        if gt_norm in all_forms and pred_norm in all_forms:
            return EvalResult(correct=True, match_type="synonym",
                              pred_norm=pred_norm, gt_norm=gt_norm)

    # ── 6. 包含匹配（GT 在 pred 中，或 pred 在 GT 中）──
    if gt_lemma in pred_lemma or pred_lemma in gt_lemma:
        return EvalResult(correct=True, match_type="contains",
                          pred_norm=pred_lemma, gt_norm=gt_lemma)

    # ── 7. Token F1 ──
    gt_tokens = gt_lemma.split()
    pred_tokens = pred_lemma.split()
    f1 = _compute_token_f1(pred_tokens, gt_tokens)
    if f1 >= 0.5:
        return EvalResult(correct=True, match_type="f1",
                          pred_norm=pred_lemma, gt_norm=gt_lemma, f1_score=f1)

    return EvalResult(correct=False, match_type="none",
                      pred_norm=pred_lemma, gt_norm=gt_lemma, f1_score=f1)


# ── 自测 ──────────────────────────────────────────────────

def _self_test():
    """验证评估器在已知 case 上的表现"""
    cases = [
        # (predicted, ground_truth, expected_correct)
        ("yes", "yes", True),
        ("No, it is not.", "no", True),
        ("playing tennis", "play tennis", True),        # lemma
        ("eating ice cream", "eat ice cream", True),     # lemma
        ("grass", "grassland", True),                    # synonym
        ("knit sweater", "knit sweater", True),          # exact
        ("indoor", "bedroom", False),                    # not synonym
        ("2", "2", True),                                # number
        ("there are two children", "2", True),           # num word
        ("street", "in street", True),                   # contains
        ("a woman is doing pull ups", "doing exercises", False),  # different
    ]
    passed = 0
    for pred, gt, expected in cases:
        result = evaluate_answer(pred, gt)
        status = "✓" if result.correct == expected else "✗"
        if result.correct != expected:
            print(f"  {status} pred={pred!r:30s} gt={gt!r:20s} "
                  f"expected={expected} got={result.correct} ({result.match_type})")
        passed += int(result.correct == expected)
    print(f"Self-test: {passed}/{len(cases)} passed")
    return passed == len(cases)


if __name__ == "__main__":
    _self_test()
