# GPT Code Review Prompt - FasterOmni Phase 1 完整审查

## 角色

你是一位多模态大模型推理优化领域的资深研究员，擅长视频理解、模型加速和实验方法论。请对以下工作进行全面 Review，指出潜在问题、逻辑漏洞和改进建议。

## 项目概述

**FasterOmni** 是一个针对 **Qwen2.5-Omni-7B** 多模态大模型的推理加速框架，核心方法是 **GOP（Group of Pictures）级视频 token 稀疏化**。

### 技术方案

1. **GOP 解析**：用 ffprobe 解析视频的 GOP 结构（I/P/B 帧分组）
2. **AV-LRM 打分**：对每个 GOP 计算视觉复杂度（帧间差异）和音频能量，加权打分
3. **GOP 选择**：按 keep_ratio 保留 Top-K 或 Uniform 采样的 GOP
4. **I 帧解码**：只解码选中 GOP 的 I 帧（关键帧）
5. **模型推理**：将稀疏帧作为 video tensor 送入 Qwen2.5-Omni，保留完整音频

### 评估方法

- **Benchmark**: Video-MME（100 视频 300 题，选择题 A/B/C/D）
- **划分**: Short (52-111s, 36 视频 108 题) / Medium / Long
- **指标**: Accuracy（选择题正确率）、Generate Time（推理延迟）、Visual Tokens（视觉 token 数）
- **帧数限制**: max_frames=32（32GB 显存约束）

## Phase 1 实验结果

### 实验 1: Baseline vs Sparse 完整评估（300 题）

| Duration | N | Baseline Acc | Sparse Acc | Drop | Speedup | Vis Token 减少 |
|----------|---|-------------|-----------|------|---------|---------------|
| Short | 108 | 75.0% | 69.4% | -5.6pp | 1.99x | 54.0% |
| Medium | 90 | 62.2% | 57.8% | -4.4pp | 1.02x | 0.3% |
| Long | 102 | 48.0% | 49.0% | +1.0pp | 1.01x | 0.0% |

**问题**: M/L 视频 sparse 完全无效——因为 GOP 数多（100+），kr=0.5 选完后 I 帧数仍超 max_frames=32，被截断后与 baseline 帧数完全一致。

### 实验 2: keep_ratio 消融（Short 视频 108 题）

| kr | Accuracy | Speedup | Vis Token |
|----|----------|---------|-----------|
| baseline | 75.9% | 1.0x | 10,737 |
| 0.9 | 70.4% | 1.3x | 7,794 |
| 0.7 | 68.5% | 1.6x | 6,658 |
| 0.5 | 69.4% | 2.0x | 4,939 |
| 0.3 | 69.4% | 2.8x | 3,190 |
| 0.2 | 70.4% | 3.7x | 2,192 |

**发现**: 准确率对 kr 极不敏感（68.5%~70.4% 范围波动），延迟与 visual_tokens 近似线性。

### 实验 3: 去音频消融（Short 视频 108 题，kr=0.5）

| 模式 | Accuracy | Speedup | Audio Token |
|------|----------|---------|-------------|
| baseline | 75.9% | 1.0x | ~2,900 |
| sparse (有音频) | 69.4% | 2.0x | ~2,400 |
| sparse_no_audio | 67.6% | 2.5x | 0 |

**发现**: 去音频后仅额外降 1.8pp（69.4% → 67.6%），音频"兜底"效应很小。

### 实验 4: 按 Task Type 分析（Short, baseline vs sparse kr=0.5）

| Task Type | Baseline | Sparse | Diff | N |
|-----------|----------|--------|------|---|
| Counting Problem | 50.0% | 25.0% | -25.0pp | 12 |
| Attribute Perception | 92.9% | 78.6% | -14.3pp | 14 |
| Information Synopsis | 100.0% | 85.7% | -14.3pp | 7 |
| Object Reasoning | 71.4% | 64.3% | -7.1pp | 14 |
| Action Recognition | 70.6% | 70.6% | 0.0pp | 17 |
| Action Reasoning | 80.0% | 80.0% | 0.0pp | 5 |
| Object Recognition | 65.0% | 70.0% | +5.0pp | 20 |
| Temporal Perception | 60.0% | 60.0% | 0.0pp | 5 |
| Spatial Perception | 100.0% | 100.0% | 0.0pp | 5 |

## 已知限制和问题

1. **M/L 视频 sparse 无效**：max_frames=32 截断使稀疏化在 M/L 上无意义
2. **GOP 粒度粗**：Short 视频 GOP 中位数仅 5，alpha 参数和打分公式无区分度
3. **I 帧解码是全解码**：`container.decode()` 遍历全帧再过滤 keyframe，CPU 侧无加速
4. **逐视频波动大**：部分视频准确率暴跌 66pp，部分提升 33pp
5. **仅 Short 视频有效**：论文需明确说明适用范围
6. **打分公式价值存疑**：既然 kr 不敏感，TopK 和 Uniform 选择可能无差异

## 请 Review 以下方面

1. **实验方法论**：
   - 108 题的样本量是否足够支撑统计结论？
   - 只在 Short 视频上有效是否能支撑一篇论文？
   - kr 消融中准确率不敏感，是真的鲁棒还是统计噪声（108 题 MCQ 有 25% 随机基线）？

2. **技术方案**：
   - AV-LRM 打分在 GOP 中位数仅 5 时是否有意义？是否退化为随机选择？
   - 完整音频保留的合理性——音频兜底效应仅 1.8pp，是否应该也稀疏化音频？
   - I 帧选择 vs 任意帧选择——是否做过对比？随机选帧是否效果一样？

3. **结果解读**：
   - "3.7x 加速 -5.5pp" 这个 trade-off 在领域内是否有竞争力？
   - Counting Problem 暴跌 25pp 但 Object Recognition 提升 5pp，如何解释？
   - sparse_no_audio 的 2.5x 加速是否比 sparse 2.0x 更有吸引力（多 0.5x 仅多掉 1.8pp）？

4. **论文定位**：
   - "GOP 感知推理加速" vs "首个面向端侧的音视频联合稀疏化框架"，哪个故事线更合适？
   - 与现有视频 token 压缩方法（FastV, LLaVA-PruMerge, TokenPacker 等）的差异化在哪？
   - 只在 Short 视频 + 单模型（Qwen2.5-Omni）上验证，泛化性如何论证？

5. **潜在遗漏**：
   - 是否需要做 naive baseline 对比（uniform 随机采帧、等间隔采帧）？
   - 是否需要报告 per-video 的方差/置信区间？
   - 是否需要在其他 benchmark（如 MVBench, LongVideoBench）上验证？

## 代码架构

```
fasteromni/
├── pipeline.py          # 推理管道（baseline + sparse + sparse_no_audio）
├── eval_videomme.py     # Video-MME 评估脚本（含 monkey-patch 防死锁）
├── modules/
│   ├── gop_parser.py    # GOP 解析（ffprobe）
│   ├── audio_energy.py  # 音频能量提取
│   ├── sparse.py        # AV-LRM 打分 + GOP 选择
│   └── frame_decoder.py # I 帧解码
```

请给出结构化的 Review 意见，按严重程度分为：
- **Critical**：可能推翻核心结论的问题
- **Major**：需要额外实验或修改才能发论文的问题
- **Minor**：改进建议，不影响核心结论
- **Positive**：做得好的地方
