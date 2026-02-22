# GPT Codex 任务：Sanity Check kr=0.5 分析脚本

## 背景

我们在 Video-MME Short 上发现 **naive_iframe kr=0.5 的准确率 = Baseline（75.93%）**。GPT 5.2 Review 指出这可能是 pipeline bug（帧选择没生效、max_frames 截断成了相同输入等）。需要写一个纯数据分析脚本来验证。

## 任务

写一个 Python 脚本 `fasteromni/sanity_check_kr05.py`，读取已有的 CSV 结果数据，执行以下 4 项检查，并输出报告。

## 数据源

两个 CSV 文件，格式完全相同：

1. **Baseline**: `/root/autodl-tmp/results/fasteromni/videomme_full/baseline/baseline_details.csv`
2. **kr=0.5 naive_iframe**: `/root/autodl-tmp/results/fasteromni/pareto_naive_iframe/naive_iframe_kr0.5/videomme_combined_details.csv`

### CSV 关键字段

| 字段 | 说明 |
|------|------|
| `question_id` | 唯一题目 ID（如 `005-1`） |
| `video_file_id` | 视频文件 ID |
| `duration` | short / medium / long |
| `pred_answer` | 模型预测答案（A/B/C/D） |
| `gt_answer` | 正确答案 |
| `correct` | True/False |
| `visual_tokens` | 视觉 token 数量 |
| `num_frames` | 输入帧数 |
| `generate_ms` | 推理耗时 |
| `error` | 错误信息（空=成功） |

## 检查项

### A1: 逐样本预测一致性

- 只看 `duration == "short"` 的题（108 题）
- 按 `question_id` 对齐两个 CSV
- 计算 `pred_answer` 完全一致的比例
- **判定**：>95% 一致 → 高度怀疑 pipeline bug；<80% 一致 → 帧选择确实影响了模型行为

### A2: Visual Token 数量对比

- 按 `video_file_id` 分组（去重），对比两种模式的 `visual_tokens`
- 计算：平均 token 数、token 减少比例
- **判定**：baseline 应该是 ~11520（32帧×360），kr=0.5 应该 ~5000-6000（约减半）
- 如果两者相同 → 帧选择完全没生效

### A3: num_frames 对比

- 对比两种模式每个视频的 `num_frames`
- baseline 应为 32（max_frames 截断），kr=0.5 应为 ~14-16
- 如果相同 → max_frames 把 kr=0.5 也截断成了 32 帧

### A4: 翻转分析

- 统计四种情况的题目数量：
  - **BL正确 & kr=0.5正确**（both_correct）
  - **BL错误 & kr=0.5错误**（both_wrong）
  - **BL正确 → kr=0.5错误**（degraded，稀疏化导致答错）
  - **BL错误 → kr=0.5正确**（improved，稀疏化反而答对）
- 列出所有 degraded 和 improved 的具体 question_id 和预测
- **判定**：如果 degraded=0 且 improved=0 → 预测完全一样 → pipeline bug

## 输出要求

### 1. 终端输出（结构化 print）

```
=== Sanity Check: kr=0.5 vs Baseline (Video-MME Short) ===

[A1] 预测一致性
  总题数: 108
  一致: XX (XX.X%)
  不一致: XX (XX.X%)
  → 判定: ✅ 帧选择影响了模型行为 / ⚠️ 高度怀疑 pipeline bug

[A2] Visual Token 对比
  Baseline 平均: XXXX tokens (XX 帧)
  kr=0.5 平均: XXXX tokens (XX 帧)
  Token 减少: XX.X%
  → 判定: ✅ 帧选择生效 / ❌ 帧选择未生效

[A3] num_frames 对比
  Baseline: 全部 32 帧
  kr=0.5: 平均 XX 帧, 范围 [XX, XX]
  → 判定: ✅ 帧数不同 / ❌ 帧数相同

[A4] 翻转分析
  Both correct: XX
  Both wrong: XX
  Degraded (BL✓→kr✗): XX
  Improved (BL✗→kr✓): XX
  → 判定: ...

=== 综合结论 ===
[结论文字]
```

### 2. JSON 报告

保存到 `/root/autodl-tmp/results/fasteromni/sanity_check_kr05_report.json`

```json
{
  "check_time": "ISO timestamp",
  "data_sources": {
    "baseline": "path",
    "kr05": "path"
  },
  "A1_prediction_agreement": {
    "total": 108,
    "agree": N,
    "agree_pct": X.X,
    "verdict": "pass/fail/warning"
  },
  "A2_visual_tokens": {
    "baseline_mean": X,
    "kr05_mean": X,
    "reduction_pct": X.X,
    "verdict": "pass/fail"
  },
  "A3_num_frames": {
    "baseline_mean": X,
    "kr05_mean": X,
    "kr05_range": [min, max],
    "verdict": "pass/fail"
  },
  "A4_flips": {
    "both_correct": N,
    "both_wrong": N,
    "degraded": N,
    "improved": N,
    "degraded_details": [...],
    "improved_details": [...],
    "verdict": "pass/fail/warning"
  },
  "overall_verdict": "PASS: 帧选择生效且准确率巧合相等 / FAIL: pipeline bug"
}
```

## 编码要求

- 纯 Python，只用 pandas + json + datetime，无 GPU 依赖
- 脚本可直接 `python fasteromni/sanity_check_kr05.py` 运行
- 用 argparse 支持自定义路径（但默认值用上面的路径）
- 不要 hardcode duration 过滤，用 `--duration short` 参数（默认 short）
- 代码风格简洁，不要过度封装
