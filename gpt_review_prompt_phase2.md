# GPT Review Prompt — FasterOmni Phase 2 综合评审

## 角色

你是一位多模态 LLM 推理优化领域的资深审稿人。请对以下实验结果进行全面评审，指出论文故事线中的问题、缺失的实验、以及论文写作建议。

## 项目概述

**FasterOmni** 是一个面向 Qwen2.5-Omni-7B（7B 参数，视觉+音频+文本三模态）的视频稀疏推理框架。核心思路：通过 GOP（Group of Pictures）级别的帧选择，减少 visual tokens 数量，实现推理加速。

- **设备**：单 GPU 32GB VRAM（边缘服务器 / 轻量部署场景）
- **模型**：Qwen2.5-Omni-7B（非量化）
- **核心机制**：AV-LRM（Audio-Visual Locality-aware Relevance Metric），对每个 GOP 打分决定保留/丢弃
- **帧选择参数**：kr（keep ratio）控制保留比例

## 已完成的实验及关键数据

### 实验 1：Naive Baselines 对比（Video-MME Short, 108 题）

| kr | AV-LRM | naive_iframe | naive_uniform | naive_random |
|----|--------|-------------|---------------|-------------|
| 0.5 | 69.44% | **75.93%** | 74.07% | 73.15% |
| 0.2 | **70.37%** | 68.52% | 67.59% | 63.89% |

结论：AV-LRM 在 kr=0.5 时最差，kr=0.2 时最优。价值在跨稀疏度鲁棒性。

### 实验 2：Modality Baselines（6 模式 × 300 题，全 duration）

| 模式 | Overall | Short | Medium | Long | VisTok |
|------|---------|-------|--------|------|--------|
| text_only | 42.0% | 40.7% | 40.0% | 45.1% | 391 |
| audio_only | 51.3% | 55.6% | 51.1% | 47.1% | 391 |
| video_only | 62.2% | 73.1% | 61.1% | 49.4% | 10,800 |
| baseline | 61.9% | 75.9% | 56.7% | 49.4% | 10,795 |
| sparse(kr=0.5) | 59.0% | 69.4% | 57.8% | 49.0% | 8,744 |
| naive_iframe(kr=0.5) | 61.3% | 75.9% | 60.0% | 47.1% | 8,744 |

语言先验占 baseline 准确率的比例：Short 54%, Medium 71%, Long 91%。

### 实验 3：Sparse@64 vs Baseline@64

Baseline@64: 96/108 OOM (89%)
Sparse@64: 0/108 OOM (0%)

### 实验 4：Bootstrap CI（10,000 次，question-level 配对）

| 对比 | Duration | Diff | 95% CI | 跨零？ |
|------|----------|------|--------|:------:|
| sparse - baseline | all | -2.1pp | [-5.7, +1.4] | ✅ |
| sparse - baseline | short | -6.5pp | [-13.9, 0.0] | ✅(边界) |
| naive_iframe - baseline | all | +0.7pp | [-2.8, +4.3] | ✅ |
| naive_iframe - baseline | short | 0.0pp | [-4.6, +5.6] | ✅ |

## 已知问题

1. M/L 视频被 max_frames=32 限制，sparse 与 baseline token 数几乎相同，稀疏化无效
2. AV-LRM 在 LP-Unsolvable 题上比 naive_iframe 差（43.7% vs 47.7%）
3. Medium 上 video_only > baseline (+4.4pp)，音频可能引入干扰
4. 只在 Video-MME 一个 benchmark 上测试（MVBench 已下载待接入）

## 论文定位

- 边缘服务器 / 轻量部署（单 GPU 32GB），不是手机端侧
- 主打 Short 视频（<60s）稀疏化加速
- 双 benchmark：Video-MME Short + MVBench（~16s）

## 请评审以下方面

1. **论文故事线**：基于以上数据，最佳的论文 narrative 是什么？AV-LRM 在 kr=0.5 不如 naive，怎么讲故事？
2. **缺失实验**：还需要补什么实验才能让论文成立？（考虑 MVBench 已在计划中）
3. **统计严谨性**：Bootstrap CI 的结论足够强吗？样本量（100 视频/300 题）够吗？
4. **AV-LRM 定位**：鉴于 naive_iframe 表现出奇的好，AV-LRM 的 contribution 如何突出？
5. **论文写作建议**：哪些发现应该强调，哪些应该淡化？
6. **自适应策略**：是否应该实现 content-adaptive kr（根据视频复杂度动态调整）？会增强论文吗？
7. **与相关工作对比**：需要和哪些方法做量化对比？（Mobile-VideoGPT, FastV, TokenPacker 等）
