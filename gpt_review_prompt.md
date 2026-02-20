# GPT Code Review Prompt - FasterOmni Phase 2 综合 Review

## 角色

你是一位多模态大模型推理优化领域的资深研究员，擅长视频理解、模型加速、统计方法和实验方法论。请对以下工作进行深度 Review，重点审查：
1. **三组实验的数据是否能支撑结论**（统计置信度）
2. **实验设计的公平性问题**（尤其是 max_frames 语义差异和音频不对称）
3. **论文应该怎么写**（给定实验发现的"排名反转"现象）

## 项目概述

**FasterOmni** 是一个针对 **Qwen2.5-Omni-7B** 多模态大模型的推理加速框架，核心方法是 **GOP（Group of Pictures）级视频 token 稀疏化**。

### 技术方案

1. **GOP 解析**：用 PyAV 解析视频的 GOP 结构（I/P/B 帧分组）
2. **AV-LRM 打分**：α·V̂（I 帧码率归一化）+ (1-α)·Â（音频 RMS 归一化）
3. **GOP 选择**：方差大 → Top-K，方差小 → Uniform 采样，保留 keep_ratio (kr) 比例
4. **I 帧解码**：只解码选中 GOP 的 I 帧（关键帧）
5. **模型推理**：将稀疏帧作为 video tensor 送入 Qwen2.5-Omni，保留音频（截断到选中 GOP 时间范围）

### 评估方法

- **Benchmark**: Video-MME（100 视频 300 题，选择题 A/B/C/D）
- **主实验**: Short 视频（52-111s，36 视频 108 题，每视频 3 题）
- **指标**: Accuracy、Generate Time (ms)、Visual Tokens
- **帧数限制**: max_frames=32（32GB 显存约束下 baseline 可运行的上限）

---

## Phase 1 关键结果（背景）

| 模式 | Accuracy | Speedup | Vis Token |
|------|----------|---------|-----------|
| baseline (全帧, max_frames=32) | 75.9% | 1.0x | 10,737 |
| sparse kr=0.9 | 70.4% | 1.3x | 7,794 |
| sparse kr=0.7 | 68.5% | 1.6x | 6,658 |
| sparse kr=0.5 | 69.4% | 2.0x | 4,939 |
| sparse kr=0.3 | 69.4% | 2.8x | 3,190 |
| sparse kr=0.2 | 70.4% | 3.7x | 2,192 |
| sparse_no_audio (kr=0.5) | 67.6% | 2.5x | 4,939 |

**Phase 1 核心发现**：
- 准确率对 kr 不敏感（68.5%~70.4% 范围），延迟与 visual_tokens 近似线性
- 去音频仅多降 1.8pp（67.6% vs 69.4%），音频兜底效应小

**Phase 1 Review (GPT-5.2) 的 Critical #1**：kr 不敏感可能意味着"选哪几帧不重要"，必须补 naive baselines 否则 AV-LRM 贡献点站不住。

---

## Phase 2 三组实验结果

### 实验 1：Naive Baselines 全量对比（kr=0.5，Short 108 题）

**实验设计**：5 种帧选择策略，相同帧数 K（由 GOP 结构决定），比较选帧策略对准确率的影响。

| 策略 | 帧来源 | 选择方法 | 音频处理 |
|------|--------|----------|----------|
| **baseline** | 全视频 | Qwen 原生均匀采样（max_frames=32） | 全视频音频 |
| **sparse** (AV-LRM) | 选中 GOP 的 I 帧 | α·V̂ + (1-α)·Â 打分 → Top-K/Uniform | 截断到选中 GOP 时间范围 |
| **naive_uniform** | 全视频 | 等间隔采 K 帧 | 全视频音频 |
| **naive_random** | 全视频 | 随机采 K 帧 (seed=42) | 全视频音频 |
| **naive_iframe** | 所有 GOP 的 I 帧 | 等间隔选 K 个 GOP（不打分） | 截断到选中 GOP 时间范围 |

**结果**：

| 策略 | Accuracy | Drop vs BL | Avg Gen(ms) | Speedup | VisTok |
|------|----------|-----------|-------------|---------|--------|
| baseline | 75.93% | — | 2,144 | 1.0x | 10,737 |
| **naive_iframe** | **75.93%** | **0.0pp** | 1,096 | **2.0x** | 4,939 |
| naive_uniform | 74.07% | -1.9pp | 1,105 | 1.9x | 4,939 |
| naive_random | 73.15% | -2.8pp | 1,102 | 1.9x | 4,939 |
| sparse (AV-LRM) | 69.44% | **-6.5pp** | 1,085 | 2.0x | 4,939 |

**观察**：
- VisTok 完全一致（4,939），确认帧数匹配正确
- naive_iframe 与 baseline 准确率完全相同（75.93%），获得 2x 加速
- **AV-LRM 是最差策略**，比 naive_iframe 低 6.5pp，比所有 naive 策略都差

### 实验 2：Sparse@64 vs Baseline@64（扩展能力验证）

**实验设计**：将 max_frames 从 32 提升到 64，验证稀疏化是否能扩展帧预算边界。

| 模式 | Valid Samples | OOM Errors | Accuracy* | Avg Gen(ms) | VisTok |
|------|-------------|------------|----------|-------------|--------|
| baseline@64 | 12 | **96 OOM (89%)** | 83.3%* | 2,017 | 8,944 |
| sparse@64 (kr=0.5) | 108 | **0** | 70.37% | 1,083 | 4,959 |

*baseline 准确率基于仅存的 12 个样本（存活偏差——跑通的恰好是短/简单/低帧率视频），不可直接对比。

**⚠️ 重要设计说明 — max_frames 语义差异**：

这个实验存在一个需要说明的设计细节：
- **Baseline@64**：`max_frames=64` 作为**采样目标**，`process_mm_info` 从全视频均匀采样 64 帧 → ~9,600 tokens → OOM
- **Sparse@64**：`max_frames=64` 只是**上限截断**。实际帧数由 `kr × num_GOPs` 决定。短视频 ~5-10 个 GOP，kr=0.5 选出 ~3-5 个 GOP 的 I 帧（约 15-32 帧），远低于 64 帧上限

所以 **Sparse@64 实际等价于 Sparse@32**（sparse 的帧数由 kr 决定，不受 max_frames 影响）。实验证明的是：**相同 max_frames=64 设置下，baseline 因全量采样 OOM，而 sparse 因 kr 筛选后帧数远少于上限而安全运行**。

### 实验 3：Naive Baselines kr=0.2（极端稀疏验证）

**实验设计**：将 kr 降到 0.2（每视频只保留 ~20% 的 GOP），验证极端稀疏下各策略表现。无 baseline（baseline 不做稀疏化）。

| 策略 | Accuracy | Drop vs BL | Avg Gen(ms) | Speedup | VisTok |
|------|----------|-----------|-------------|---------|--------|
| **sparse (AV-LRM)** | **70.37%** | -5.6pp | **586** | **3.7x** | 2,192 |
| naive_iframe | 68.52% | -7.4pp | 620 | 3.5x | 2,192 |
| naive_uniform | 67.59% | -8.3pp | 622 | 3.5x | 2,192 |
| naive_random | 63.89% | -12.0pp | 622 | 3.5x | 2,192 |

**观察**：
- VisTok 一致（2,192），帧数匹配正确
- **排名反转！AV-LRM 在 kr=0.2 时成为最优策略**
- naive_random 在极端稀疏下暴跌至 63.89%（-12pp）

---

## 交叉分析：核心发现

### 发现 1：AV-LRM 排名随 kr 反转

| kr | AV-LRM | naive_iframe | naive_uniform | naive_random | AV-LRM 排名 |
|----|--------|-------------|---------------|-------------|------------|
| 0.5 | 69.44% | **75.93%** | 74.07% | 73.15% | **最差 (4/4)** |
| 0.2 | **70.37%** | 68.52% | 67.59% | 63.89% | **最优 (1/4)** |

### 发现 2：AV-LRM 准确率跨 kr 高度稳定

| kr | AV-LRM | naive_iframe | naive_uniform | naive_random |
|----|--------|-------------|---------------|-------------|
| 0.5 | 69.44% | 75.93% | 74.07% | 73.15% |
| 0.2 | 70.37% | 68.52% | 67.59% | 63.89% |
| **变化** | **+0.93pp** | **-7.41pp** | **-6.48pp** | **-9.26pp** |

AV-LRM 从 kr=0.5→0.2 几乎不变（+0.93pp），而所有 naive 策略均大幅下降（-6~-9pp）。

### 我们的初步结论

AV-LRM 的真正价值不是"在某个 kr 上准确率最高"，而是**跨稀疏度的鲁棒性**——帧预算越紧张（kr 越低），AV-LRM 的选帧优势越明显。

---

## ⚠️ 请重点审查的问题

### 问题 1：这些结论的统计置信度够吗？

我们的数据规模：
- 108 题来自 36 个视频（每视频 3 题），题级**不独立**
- 有效独立样本 = 36 个视频
- 每个 mode 只跑了 **1 次**（naive_random 有随机性但 seed=42 固定）
- 只比了 kr=0.5 和 kr=0.2 两个点

具体疑问：
- 36 个视频是否足够支撑"排名反转"的结论？还是说这只是统计噪声？
- kr=0.5 下 AV-LRM 比 naive_iframe 低 6.5pp（75.93% vs 69.44%），这个差距有统计显著性吗？
- 需要什么样的额外实验来提高置信度？（增加视频数量？增加 kr 点？重复多次？）

### 问题 2：实验 2 (Sparse@64 vs Baseline@64) 的设计是否有效？

如上所述，max_frames 在两种模式中语义不同：
- Baseline: 采样目标（直接采 64 帧）
- Sparse: 上限截断（实际帧数由 kr 决定，远低于 64）

这意味着实验并非"相同 max_frames 下的公平对比"，而是"baseline 无法处理高 max_frames，sparse 可以"。这个结论有价值吗？应该如何在论文中表述？

### 问题 3：kr=0.5 时 AV-LRM 为什么是最差的？

这是最令人意外的结果。可能的解释：
1. AV-LRM 的打分公式（码率+音频能量）在 kr=0.5 时引入了偏差，选出了"信息密度高但语义不重要"的帧
2. naive_iframe 的等间隔采样保留了时间覆盖度，而 AV-LRM 的 Top-K 可能聚集在某些时间段
3. 可能是统计噪声（但 6.5pp 的差距似乎不小）

是否需要做 per-video 分析（看 AV-LRM 在哪些视频上输/赢）来理解原因？

### 问题 4：论文应该怎么写？

给定实验发现：
- kr=0.5: AV-LRM **最差**，naive_iframe = baseline
- kr=0.2: AV-LRM **最优**，naive_random 暴跌
- AV-LRM 跨 kr 鲁棒（69-70% 不动）

几种可能的论文叙事：
1. **"AV-LRM 在极端稀疏下有效"** — 但温和稀疏下反而有害，难以说服审稿人
2. **"GOP 感知 I 帧采样即高效策略"** — naive_iframe=baseline 说明只要用 I 帧+等间隔就行，AV-LRM 的打分公式是多余的
3. **"跨稀疏度鲁棒性"** — AV-LRM 不追求最优但求稳定，适合不知道最佳 kr 的场景
4. **"稀疏化扩展帧预算边界"** — 独立于选帧策略的工程价值（Baseline@64 OOM，Sparse@64 通过）

你认为哪种叙事最站得住？需要什么额外实验来支撑？

---

## 公平性混淆因素

请也评估以下已知的公平性问题：

1. **音频不对称**：naive_uniform/random 使用全视频音频，sparse/naive_iframe 截断到选中 GOP 时间范围。Phase 1 实测音频 token 差异约 7.9%，去音频仅影响 1.8pp，但仍是 confound。
2. **I 帧 vs 任意帧**：naive_iframe 和 sparse 只用 I 帧（关键帧），naive_uniform/random 可用任意帧（Qwen 原生 uniform 采样）。I 帧通常画质最高、信息最完整。
3. **帧数计算方式**：K 基于 GOP 结构（`ceil(N_valid_gops × kr)`），不是固定值。不同视频的 K 不同。是否应该用固定帧数（如 K=16）来消除变异？

---

## 代码架构（供检查逻辑正确性）

```
fasteromni/
├── pipeline.py          # 推理管道
│   ├── run_baseline()   # 原生 Qwen2.5-Omni：process_mm_info(nframes=max_frames) → 均匀采帧
│   ├── run_sparse()     # AV-LRM：GOP解析 → 打分 → TopK/Uniform选GOP → I帧解码 → 推理
│   │   └── max_frames 只在 len(i_frames) > max_frames 时截断
│   └── run_naive()      # Naive 策略：
│       ├── uniform:     等间隔从全视频采 K 帧（process_mm_info nframes=K）
│       ├── random:      随机从全帧中采 K 帧
│       └── iframe_uniform: 解析全部 GOP → 等间隔选 K 个 GOP 的 I 帧（不打分）
├── eval_videomme.py     # Video-MME 评估脚本（增量 CSV + 超时保护 + 自动恢复）
├── modules/
│   ├── gop_parser.py    # PyAV 解析 GOP 结构
│   ├── audio_energy.py  # 音频 RMS 能量按 GOP 时间窗口
│   ├── sparse.py        # AV-LRM 打分（score_gops）+ GOP 选择（select_gops）
│   │   └── select_gops: K = ceil(N_valid × kr), variance > 0.02 → TopK, else → Uniform
│   └── frame_decoder.py # I 帧解码（container.decode 遍历 + keyframe 过滤）
```

关键代码逻辑：
- **Baseline max_frames**：`video_ele["nframes"] = max_frames`（采样目标，告诉 process_mm_info 采多少帧）
- **Sparse max_frames**：`if len(i_frames) > max_frames: i_frames = linspace downsample`（上限截断）
- **帧数 K**：`K = ceil(len(valid_gop_indices) × kr)`，不同视频 GOP 数不同所以 K 不同

---

## 请给出结构化 Review

按严重程度分为：
- **Critical**：可能推翻核心结论的问题
- **Major**：需要额外实验或修改才能发论文的问题
- **Minor**：改进建议，不影响核心结论
- **Positive**：做得好的地方

**特别关注**：
1. 数据置信度是否足够支撑"排名反转"结论
2. 实验 2 的 max_frames 语义差异是否使结论无效
3. 最推荐的论文叙事方向
4. 最小代价的补充实验建议（我们的硬件是 32GB 单卡，跑一轮 108 题约 2h）
