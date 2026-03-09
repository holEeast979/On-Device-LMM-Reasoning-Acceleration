"""
改进的答案匹配逻辑（方案 C 增强版）

不依赖外部模型，使用规则 + 词干化 + Jaccard 相似度
"""
import re
from typing import Set

def simple_stem(word: str) -> str:
    """简单的词干化（去除常见后缀）"""
    original = word
    
    # 规则1: -ing → ""
    if word.endswith("ing") and len(word) > 4:
        stemmed = word[:-3]
        # 特殊情况：tying → tie, lying → lie
        if len(stemmed) <= 2 and stemmed.endswith("y"):
            stemmed = stemmed[:-1] + "ie"
        if len(stemmed) >= 3:
            return stemmed
    
    # 规则2: -ed → ""
    if word.endswith("ed") and len(word) > 3:
        stemmed = word[:-2]
        if len(stemmed) >= 3:
            return stemmed
    
    # 规则3: -s → ""
    if word.endswith("s") and len(word) > 2 and not word.endswith("ss"):
        return word[:-1]
    
    # 规则4: -es → ""
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    
    # 规则5: -ies → y
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    
    return original

def normalize_answer(text: str) -> str:
    """标准化答案：小写 + 去标点 + 去多余空格"""
    text = text.lower().strip()
    # 去除标点符号
    text = re.sub(r"[^\w\s]", " ", text)
    # 去除多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_tokens(text: str, use_stem: bool = True) -> Set[str]:
    """提取词汇（去除停用词 + 可选词干化）"""
    # 简单的停用词列表
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "should", "could", "may", "might", "must", "can",
                  "in", "on", "at", "to", "for", "of", "with", "by", "from", "they"}
    
    tokens = normalize_answer(text).split()
    tokens = [simple_stem(t) if use_stem else t for t in tokens]
    return set(tokens) - stop_words

def match_answer(pred: str, gt: str, threshold: float = 0.4) -> bool:
    """
    改进的答案匹配
    
    策略（按优先级）：
    1. 精确匹配（标准化后）
    2. 子串匹配（gt 在 pred 中，处理 "yes, because..." → "yes"）
    3. 数字精确匹配（针对计数问题）
    4. Token Jaccard 相似度 >= threshold（带词干化）
    
    Args:
        pred: 预测答案
        gt: 真实答案
        threshold: Jaccard 相似度阈值（默认 0.4，比较宽松）
    
    Returns:
        是否匹配
    """
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    
    # 1. 精确匹配
    if pred_norm == gt_norm:
        return True
    
    # 2. 子串匹配（处理 "yes, because..." → "yes"）
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True
    
    # 3. 提取数字（针对计数问题 "how many"）
    pred_numbers = set(re.findall(r"\d+", pred))
    gt_numbers = set(re.findall(r"\d+", gt))
    if gt_numbers and pred_numbers == gt_numbers:
        return True
    
    # 4. Token Jaccard 相似度（带词干化）
    pred_tokens = extract_tokens(pred, use_stem=True)
    gt_tokens = extract_tokens(gt, use_stem=True)
    
    if not pred_tokens or not gt_tokens:
        return False
    
    intersection = len(pred_tokens & gt_tokens)
    union = len(pred_tokens | gt_tokens)
    jaccard = intersection / union if union > 0 else 0
    
    return jaccard >= threshold
