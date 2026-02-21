# GPT 代码任务 C：Non-inferiority 分析脚本

## 背景

我们已有 Bootstrap CI 脚本（`bootstrap_ci.py`）做了配对 bootstrap，证明 sparse vs baseline 的差异 CI 跨零（统计不显著）。

但 GPT Review 指出：**"CI 跨零"只是说"差异是否存在不确定"，不等于"方法等价"。** 要主张"稀疏化几乎不掉点"，应该用 **Non-inferiority testing**（非劣效检验）。

## Non-inferiority 是什么

白话：我们不需要证明 sparse 比 baseline "更好"，只需要证明 sparse "不会比 baseline 差超过 δ"。

数学：
- H₀: Δ ≤ -δ（sparse 比 baseline 差 δ 以上）
- H₁: Δ > -δ（sparse 不比 baseline 差 δ 以上）
- 如果 Δ 的 95% CI 下界 > -δ，则拒绝 H₀，接受"non-inferior"

其中 δ 是预设的容忍边界（我们设 δ = 3pp，即最多允许掉 3 个百分点）。

## 目标

在现有 `bootstrap_ci.py` 的产物基础上，新建 `non_inferiority.py`，读取 bootstrap CI 结果并执行 non-inferiority 判断。

## 输入数据

从 `/root/autodl-tmp/results/fasteromni/videomme_full/bootstrap_ci/` 读取：
- `bootstrap_paired_diff.csv`：配对 bootstrap 差异（每行一个 bootstrap 样本）
- 或直接从 `videomme_full/*/` 读取原始 `*_details.csv`，重新做配对 bootstrap

## 需要实现的内容

### 1. Non-inferiority 判断函数

```python
def non_inferiority_test(
    diff_samples: np.ndarray,  # bootstrap 差异样本（treatment - control）
    delta: float = 0.03,       # 容忍边界（3pp = 0.03）
    alpha: float = 0.05,       # 显著性水平
) -> dict:
    """
    Non-inferiority test based on bootstrap samples.
    
    Returns:
        {
            "delta": 0.03,
            "alpha": 0.05,
            "mean_diff": float,           # 平均差异
            "ci_lower": float,            # 差异的 (1-2α)% CI 下界（单侧）
            "ci_upper": float,            # 差异的 CI 上界
            "non_inferior": bool,         # ci_lower > -delta ?
            "conclusion": str,            # 白话结论
        }
    """
```

注意：Non-inferiority 是**单侧检验**，所以用 `alpha` 对应的单侧 CI。
- 单侧 95% CI 下界 = 差异分布的第 5 百分位
- 如果 `ci_lower > -delta`，则 non-inferior

### 2. 多组对比

对以下每组做 non-inferiority test：
- sparse vs baseline（all / short / medium / long）
- naive_iframe vs baseline（all / short / medium / long）

### 3. 多个 δ 值

扫描 δ ∈ {1pp, 2pp, 3pp, 5pp}，报告每个 δ 下是否 non-inferior。
这样论文可以写："sparse is non-inferior to baseline within a δ=3pp margin (p<0.05)"。

### 4. 输出

CSV + 终端打印表格：

```
=== NON-INFERIORITY ANALYSIS (δ tolerance) ===

Comparison: sparse - baseline
┌────────┬──────────┬──────────┬──────────┬──────────┐
│Duration│ Mean Diff│ 95% CI_lo│    δ=1pp │    δ=3pp │
├────────┼──────────┼──────────┼──────────┼──────────┤
│   all  │  -2.1pp  │  -5.7pp  │    ✗     │    ✗     │
│  short │  -6.5pp  │  -13.9pp │    ✗     │    ✗     │
│ medium │  +1.1pp  │  -4.4pp  │    ✗     │    ✗     │
│  long  │  +0.0pp  │  -6.0pp  │    ✗     │    ✗     │
├────────┼──────────┼──────────┼──────────┼──────────┤

Comparison: naive_iframe - baseline
│   all  │  +0.7pp  │  -2.8pp  │    ✗     │    ✓     │  ← 这个能通！
│  short │  +0.0pp  │  -4.6pp  │    ✗     │    ✗     │
│ medium │  +3.3pp  │  -2.2pp  │    ✗     │    ✓     │
│  long  │  +0.0pp  │  -6.0pp  │    ✗     │    ✗     │
└────────┴──────────┴──────────┴──────────┴──────────┘
```

注意：根据我们已有的 bootstrap CI 数据，naive_iframe vs baseline 的 all 下界是 -2.8pp，所以在 δ=3pp 时应该能通过 non-inferiority。这是**论文里最强的一句话**：

> "naive_iframe is non-inferior to baseline within a 3pp margin (one-sided 95% CI lower bound = -2.8pp > -3pp)."

### 5. 命令行

```python
parser.add_argument("--data-dir", default="/root/autodl-tmp/results/fasteromni/videomme_full/")
parser.add_argument("--bootstrap-dir", default=None, 
                    help="If provided, read pre-computed bootstrap samples; otherwise recompute")
parser.add_argument("--n-bootstrap", type=int, default=10000)
parser.add_argument("--deltas", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05])
parser.add_argument("--out-dir", default="/root/autodl-tmp/results/fasteromni/videomme_full/non_inferiority/")
```

Usage:
```bash
python non_inferiority.py
```

### 6. 技术要求

- 纯 numpy，不依赖 scipy
- 配对 bootstrap：同 question_id 同步采样（和 bootstrap_ci.py 相同逻辑）
- 输出 CSV + JSON + 终端表格
- 结果保存到 `non_inferiority/` 子目录（不覆盖已有 bootstrap_ci/ 数据）

## 参考文件

- Bootstrap CI 脚本：`/root/autodl-tmp/results/fasteromni/videomme_full/bootstrap_ci/bootstrap_ci.py`
- 原始评估数据：`/root/autodl-tmp/results/fasteromni/videomme_full/*/`（每个 mode 一个子目录）
