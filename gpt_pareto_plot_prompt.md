# GPT Codex Task: 绘制论文级 Pareto 曲线图 + MVBench 按任务分析图

## 任务概述

绘制两张论文级图表，保存到 `/root/autodl-tmp/results/figures/`。所有数据已就绪，只需 matplotlib 绘图，不需要 GPU。

---

## 图 1: Pareto 曲线图（Accuracy vs Speed/Tokens）

### 数据（硬编码）

**Video-MME Short, naive_iframe 不同 kr：**

```python
pareto_data = {
    "kr": [0.2, 0.3, 0.5, 0.7, 0.9],
    "accuracy": [68.52, 69.44, 75.93, 71.30, 70.37],
    "visual_tokens": [2192, 3190, 4939, 6658, 7794],
    "speedup": [3.6, 2.8, 2.1, 1.6, 1.4],
}

# Baseline 参考线
baseline_accuracy = 75.93
baseline_visual_tokens = 10692
baseline_speedup = 1.0

# Sparse 对比点（同 kr）
sparse_data = {
    "kr": [0.2, 0.3, 0.5, 0.7, 0.9],
    "accuracy": [70.37, 69.44, 69.44, 62.04, 64.81],
    "visual_tokens": [2259, 3299, 5041, 6735, 7824],
}
```

### 图表要求

**布局**：1 行 2 列 subplot（共享 Y 轴 Accuracy）

**左图：Accuracy vs Visual Tokens（Token 效率）**
- X 轴：Visual Tokens（降序，从多到少）
- Y 轴：Accuracy (%)
- naive_iframe：蓝色实线 + 圆形标记，每个点标注 kr 值
- sparse：红色虚线 + 三角标记
- Baseline：灰色水平虚线 + 文字标注 "Baseline 75.93%"
- **高亮 kr=0.5 点**：加大标记 + 加粗标注 "kr=0.5 (zero-loss)"
- 标注 token 减少百分比："−54% tokens"

**右图：Accuracy vs Speedup（速度效率）**
- X 轴：Speedup (×)
- Y 轴：Accuracy (%)
- 同样的数据和样式
- 高亮 kr=0.5："2.1× faster, zero-loss"

**样式要求（论文级）**：
- 字体：serif（Times New Roman 风格），fontsize 12-14
- 图例放在不遮挡数据的位置
- 网格：浅灰色虚线
- 紧凑布局（tight_layout）
- DPI: 300
- 尺寸：(12, 5) inches

### 输出

- `/root/autodl-tmp/results/figures/pareto_curve.png`
- `/root/autodl-tmp/results/figures/pareto_curve.pdf`（论文用矢量图）

---

## 图 2: MVBench 按任务分析图（退化热力图 / 柱状图）

### 数据（硬编码）

```python
mvbench_tasks = {
    "task": [
        "counterfactual_inference", "object_existence", "moving_attribute",
        "moving_direction", "action_sequence", "action_prediction",
        "moving_count", "object_interaction", "action_antonym",
        "character_order", "action_localization", "egocentric_navigation",
        "object_shuffle", "unexpected_action", "state_change",
        "scene_transition", "fine_grained_action", "action_count",
    ],
    "baseline": [68.0, 88.5, 95.0, 59.5, 75.4, 68.0, 69.0, 75.8, 79.5,
                 74.1, 44.3, 39.5, 37.9, 82.1, 60.7, 96.5, 47.5, 34.6],
    "naive_iframe": [32.0, 55.5, 62.5, 37.0, 53.0, 46.0, 47.0, 58.5, 69.3,
                     64.0, 35.0, 32.0, 35.0, 80.0, 59.5, 96.5, 48.7, 51.0],
}

# delta = naive - baseline
```

### 图表要求

**水平柱状图（sorted by delta）**：
- Y 轴：任务名称（缩短显示，如 "counterfactual_inf." → "Counterfactual"）
- X 轴：Δ Accuracy (pp)
- 颜色编码：
  - Δ < -20：红色（严重退化）
  - -20 ≤ Δ < -5：橙色（中等退化）
  - -5 ≤ Δ < 0：浅蓝（可接受）
  - Δ ≥ 0：绿色（无损/反超）
- 每个柱子右侧标注具体 Δ 值
- 零线：黑色竖线
- 标题："MVBench Per-Task Analysis: naive_iframe(kr=0.5) vs Baseline"

**样式**：同上（论文级，serif 字体，DPI 300）

### 输出

- `/root/autodl-tmp/results/figures/mvbench_per_task.png`
- `/root/autodl-tmp/results/figures/mvbench_per_task.pdf`

---

## 实现要求

1. **单个 Python 脚本**：`/root/scripts/tools/plot_figures.py`
2. **不需要 GPU**，只用 matplotlib + numpy
3. **数据硬编码在脚本中**（不读 CSV）
4. 创建输出目录：`os.makedirs("/root/autodl-tmp/results/figures", exist_ok=True)`
5. 运行命令：`python tools/plot_figures.py`
6. 两张图都在一个脚本里生成

## 不需要做的事

- ❌ 不需要读 CSV 文件
- ❌ 不需要跑模型推理
- ❌ 不需要修改其他文件
