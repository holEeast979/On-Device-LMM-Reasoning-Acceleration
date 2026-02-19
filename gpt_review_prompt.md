# GPT Code Review Prompt - FasterOmni Phase 2 P0 #1 Review（Naive Baselines）

## 角色

你是一位多模态大模型推理优化领域的资深研究员，擅长视频理解、模型加速和实验方法论。请对以下工作进行 Review，重点审查 Naive Baselines 对比实验的设计和结果解读。

## 项目概述

**FasterOmni** 是一个针对 **Qwen2.5-Omni-7B** 多模态大模型的推理加速框架，核心方法是 **GOP（Group of Pictures）级视频 token 稀疏化**。

### 技术方案

1. **GOP 解析**：用 PyAV 解析视频的 GOP 结构（I/P/B 帧分组）
2. **AV-LRM 打分**：α·V̂（I 帧码率归一化）+ (1-α)·Â（音频 RMS 归一化）
3. **GOP 选择**：方差大 → Top-K，方差小 → Uniform 采样，保留 keep_ratio 比例
4. **I 帧解码**：只解码选中 GOP 的 I 帧（关键帧）
5. **模型推理**：将稀疏帧作为 video tensor 送入 Qwen2.5-Omni，保留完整音频

### 评估方法

- **Benchmark**: Video-MME（100 视频 300 题，选择题 A/B/C/D）
- **主实验**: Short 视频（52-111s，36 视频 108 题）
- **指标**: Accuracy、Generate Time、Visual Tokens
- **帧数限制**: max_frames=32（32GB 显存约束）

---

## Phase 1 关键结果（背景）

| 模式 | Accuracy | Speedup | Vis Token |
|------|----------|---------|-----------|
| baseline (全帧) | 75.9% | 1.0x | 10,737 |
| sparse kr=0.5 (AV-LRM) | 69.4% | 2.0x | 4,939 |
| sparse kr=0.2 (AV-LRM) | 70.4% | 3.7x | 2,192 |
| sparse_no_audio | 67.6% | 2.5x | 4,939 |

**Phase 1 Review 发现的 Critical #1**：kr 不敏感（68.5%~70.4%）可能意味着"选哪几帧不重要"，必须补 naive baselines 否则 AV-LRM 贡献点站不住。

---

## Phase 2 P0 #1: Naive Baselines 对比实验

### 实验设计

**目标**：验证 AV-LRM 打分是否优于 naive 帧选择策略。

**关键约束**：所有方法使用**相同帧数** K = min(ceil(N_valid_gops × kr), max_frames)，消除帧数差异的影响。

**对比的 5 种帧选择策略**：

| 策略 | 帧来源 | 选择方法 | 音频处理 |
|------|--------|----------|----------|
| **baseline** | 全视频 | Qwen 原生采样（max_frames=32）| 全视频音频 |
| **sparse** (AV-LRM) | 选中 GOP 的 I 帧 | α·V̂ + (1-α)·Â 打分 → Top-K/Uniform | 截断到选中 GOP 时间范围 |
| **naive_uniform** | 全视频 | 等间隔采 K 帧 | 全视频音频 |
| **naive_random** | 全视频 | 随机采 K 帧 (seed=42) | 全视频音频 |
| **naive_iframe** | 所有 GOP 的 I 帧 | 等间隔选 K 个 GOP（不打分）| 截断到选中 GOP 时间范围 |

**注意的公平性问题**：
- naive_uniform / naive_random 使用全视频音频（因为帧覆盖全时间范围）
- sparse / naive_iframe 音频截断到选中 GOP 时间范围
- 这意味着 sparse 和 naive_iframe 的音频 token 可能少于 naive_uniform/random

### Smoke Test 结果（3 视频 9 题，仅验证正确性）

| 策略 | Accuracy | VisTok | Gen(ms) | Errors |
|------|----------|--------|---------|--------|
| naive_uniform | 55.6% | 4560 | 1138 | 0 |
| naive_random | 66.7% | 4560 | 1077 | 0 |
| naive_iframe | 66.7% | 4560 | 1141 | 0 |
| sparse (AV-LRM) | 55.6% | 4560 | 1064 | 0 |

✅ VisTok 全部一致（4560），确认帧数匹配正确。

### 全量结果（Short 108 题）

> ⚠️ **此处需要填入全量实验数据**。请在运行以下命令后填入：
> ```
> python fasteromni/eval_videomme.py \
>     --duration short --keep-ratio 0.5 \
>     --modes baseline sparse naive_uniform naive_random naive_iframe \
>     --out-dir /root/autodl-tmp/results/fasteromni/naive_comparison
> ```

| 策略 | Accuracy | Acc vs Baseline | Acc vs Sparse | VisTok | Gen(ms) |
|------|----------|----------------|---------------|--------|---------|
| baseline | TODO | — | — | TODO | TODO |
| sparse (AV-LRM) | TODO | TODO | — | TODO | TODO |
| naive_uniform | TODO | TODO | TODO | TODO | TODO |
| naive_random | TODO | TODO | TODO | TODO | TODO |
| naive_iframe | TODO | TODO | TODO | TODO | TODO |

---

## 请 Review 以下方面

### 1. 实验设计公平性

- **帧数匹配**：K 的计算方式（基于 GOP 结构确定）是否合理？是否应该直接用固定帧数（如 K=16）更简洁？
- **音频不对称**：naive_uniform/random 用全视频音频，sparse/naive_iframe 截断音频。这是否混淆了"帧选择效果"和"音频 token 差异"的影响？
- **I 帧 vs 任意帧**：naive_iframe 和 sparse 都只用 I 帧，naive_uniform/random 可以用任意帧（包括 P/B 帧解码后的内容）。这个差异是否影响结论？

### 2. 结果解读（全量数据出来后）

- 如果 sparse ≈ naive_uniform ≈ naive_random → AV-LRM 无贡献，"选哪几帧不重要"
- 如果 sparse > naive_uniform/random → AV-LRM 有效，打分公式确实选出了更好的帧
- 如果 naive_iframe ≈ sparse → 只要是 I 帧就行，不需要打分
- 如果 naive_iframe > naive_uniform → I 帧比任意帧更有信息量（GOP 结构有价值）

### 3. 论文叙事影响

- 如果 AV-LRM 无贡献，论文应如何调整故事线？（比如转向"GOP 感知的 I 帧采样本身就是一种高效策略"）
- 5 种策略的准确率差异如果在统计噪声范围内（±2-3pp），如何论证？是否需要 bootstrap 置信区间？

### 4. 补充建议

- 是否需要在不同 kr 值下重复 naive baseline 对比？（如 kr=0.2 和 kr=0.5 都做）
- 是否需要 per-video 配对比较（如 McNemar's test）而非整体准确率比较？

---

## 代码架构

```
fasteromni/
├── pipeline.py          # 推理管道（6 种模式：baseline + sparse + sparse_no_audio + 3 naive）
│   ├── run_baseline()   # 原生 Qwen2.5-Omni 全量帧
│   ├── run_sparse()     # AV-LRM 打分 + I 帧选择
│   └── run_naive()      # 3 种 naive 策略（uniform/random/iframe_uniform）
├── eval_videomme.py     # Video-MME 评估脚本
├── modules/
│   ├── gop_parser.py    # GOP 解析
│   ├── audio_energy.py  # 音频能量提取
│   ├── sparse.py        # AV-LRM 打分 + GOP 选择
│   └── frame_decoder.py # I 帧解码
```

请给出结构化的 Review 意见，按严重程度分为：
- **Critical**：可能推翻核心结论的问题
- **Major**：需要额外实验或修改才能发论文的问题
- **Minor**：改进建议，不影响核心结论
- **Positive**：做得好的地方
