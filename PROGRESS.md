# FasterOmni 开发进度

> **本文件用途**：
> - **当前状态快照**：最新的实验进展、待做工作、核心发现
> - **数据保护索引**：已有实验数据目录清单（禁止覆盖）
> - **代码结构概览**：当前在用的文件和模块
> - **适用对象**：Windsurf Agent（每次对话读取）、Jarvis（手机端监控）、用户（快速了解进度）
>
> **历史细节**：完整的实验数据、变更日志、外部反馈见 `PROGRESS_ARCHIVE.md`（728 行）
>
> **GitHub 地址**：`https://github.com/holEeast979/On-Device-LMM-Reasoning-Acceleration`
> - PROGRESS.md: `https://raw.githubusercontent.com/holEeast979/On-Device-LMM-Reasoning-Acceleration/master/PROGRESS.md`
> - PROGRESS_ARCHIVE.md: `https://raw.githubusercontent.com/holEeast979/On-Device-LMM-Reasoning-Acceleration/master/PROGRESS_ARCHIVE.md`

---

## 当前状态（2.24 下午）

**Layer 2 Adaptive kr 全集实验已完成，发现 M/L 视频稀疏化失效问题（详见变更日志 2.24）。待讨论 Layer 2 重新设计方向。**

### ✅ 已完成

- ✅ **Layer 1 I 帧均匀选取** — kr=0.5 零损失（75.93% = Baseline），2.1x 加速
- ✅ **Layer 2 Adaptive kr 代码** — `pipeline.py` 已实现 `kr_adaptive = min(kr, max_frames/n_valid)`，smoke test 通过（`3bb736c`）
- ⚠️ **Layer 2 M/L 失效** — 详见变更日志 [2.24 PM]
- ✅ **Video-MME Short 全部实验** — 6 模式 × 300 题 + Bootstrap CI + Non-inferiority + Pareto sweep
- ✅ **MVBench 全量** — 3 mode × 3600 题（1-GOP 修复后重跑）
- ✅ **论文图表** — Pareto 曲线图 + MVBench 按任务分析图（`tools/plot_figures.py`，输出在 `/root/autodl-tmp/results/figures/`）
- ✅ **MV 提取 PoC** — PyAV 成功提取 8766 帧 MV，1.14ms/帧，运动 profile 清晰（`tools/mv_extraction_poc.py`）
- ✅ **AV-LRM 坦诚评价** — 已写入 PROGRESS.md，Two-Regime 发现是独立贡献
- ✅ **核心技术路线确认** — naive I帧主力 / Adaptive kr 解锁 M/L / Motion-Aware L3 补时序任务

### 🔄 正在进行

- 🔄 **Layer 2 M/L 策略重新设计** — 待讨论方向（详见变更日志 [2.24 PM]）

### ⬇️ 下一步

1. **讨论 Layer 2 M/L 策略** — 提高 max_frames vs 降低 K vs 混合策略
2. **模型迁移到 FS** — 省 ~¥4/月
3. **Layer 3 Motion-Aware 设计** — P/B 帧补偿时序敏感任务

### Sanity Check 结论（2.22 午后）✅ 已关闭

**结论：pipeline 无 bug，kr=0.5 零损失是真实现象。**

脚本：`fasteromni/sanity_check_kr05.py` | 报告：`/root/autodl-tmp/results/fasteromni/sanity_check_kr05_report.json`

| 检查项 | 结果 | 判定 |
|--------|------|------|
| A1 预测一致性 | 93/108 = 86.1% | ✅ 14%的题预测不同，帧选择影响了模型行为 |
| A2 Visual Tokens | 10737→4939（-54%） | ✅ 帧选择完全生效，token 减半 |
| A3 帧数 | 32→平均14.4（范围4-32） | ✅ 帧数确实不同 |
| A4 翻转 | degraded=4, improved=4 | ✅ 8题翻转恰好对称抵消 |

**通俗解释**：模型看到的画面确实少了一半（14帧 vs 32帧），但 Short 视频大部分题靠少量关键帧就能答对。15 题答案变了，4 题变好 4 题变差恰好抵消，所以总准确率相同。

### GPT 5.2 Review 结论（2.22）

**Sanity Check**（已关闭 ✅）：
- [x] kr=0.5 和 baseline 的逐样本预测是否完全一致？→ **86.1% 一致，14% 不同**
- [x] 记录实际选帧索引 + visual token 数 → **token 减少 54%，帧数 32→14.4**
- [x] 测试 kr=0.4 / 0.6 → **已有 kr=0.3(69.44%) 和 kr=0.7(71.30%) 数据，均≠75.93%，无需额外跑**

**AV-LRM 优化建议**：
- 当前 top-K 缺多样性约束，帧可能集中在同一段
- 低成本改动：“分段 top-1”或 temporal NMS（强制最小时间间隔）
- 如果加了覆盖约束仍不如 naive → 打分就不再投入

**论文贡献度评伋**：
- 如果方法 = “均匀抽 I 帧”，单独偏弱
- 需要站稳：多数据集 tradeoff 曲线 + 端到端收益 + GOP 视角独特性
- **建议**：短期收尾（1-2天）→ 定型稀疏化 → move on 到下一技术点

---

## 项目背景

研究 **Qwen2.5-Omni-7B** 推理加速，构建**端侧多模态推理加速架构**。

**三大技术支柱**：
1. ✅ **GOP 级 token 稀疏化**（AV-LRM 打分 + 帧选择）← 已完成
2. ⬜ **显存碎片优化**（ViT 后清理激活值 + 分块 Encoding）
3. ⬜ **Ring Buffer 流水线**（CPU/GPU 异步预取）

**硬件**：RTX 5090 32GB，OOM 边界 ~25-30k tokens（~34s 视频）。
**论文定位**：边缘服务器 / 轻量部署（单 GPU 32GB），非手机端侧。

---

## 核心发现

### 实验结论

| 发现 | 数据支撑 |
|------|---------|
| **naive_iframe kr=0.5 零损失** | Video-MME Short: 75.93% = Baseline 75.93%，tokens 减 54%，2.1x 加速 |
| **Pareto 曲线非单调** | kr=0.5 是峰值，kr=0.7/0.9 反而下降 → 存在最优稀疏度 |
| **naive_iframe 在 kr=0.5 碾压 sparse** | 75.93% vs 69.44%（+6.49pp），覆盖度 > 精准打分 |
| **sparse 仅在极低预算占优** | kr=0.2: sparse 70.37% vs naive 68.52%（+1.85pp） |
| **MVBench 短视频不兼容** | 7/18 任务类别完全失败（视频仅 3-6 帧，GOP 解析退化） |
| **MVBench 兼容任务掉 ~6pp** | baseline 63.8% → naive 57.8%，但 3.9x 加速 + 83% token 减少 |
| **音频兜底效应很小** | 去音频仅多降 1.8pp，作为 free lunch 保留 |
| **Baseline@64 不可用** | 89% OOM，Sparse@64 零 OOM |
| **语言先验显著** | text_only=42%（远 > 随机 25%），Long 视频 91% 靠语言 |

### Two-Regime 理论（已量化验证）

- **Coverage-dominant（kr≈0.5）**：naive_iframe 75.93% vs sparse 69.44%（**+6.49pp**）
- **Relevance-dominant（kr≤0.2）**：sparse 70.37% vs naive 68.52%（+1.85pp）
- **交叉点在 kr≈0.3**：两种方法持平（69.44%）
- 这解释了为什么 kr=0.5 时 AV-LRM 最差、kr=0.2 时最优

### AV-LRM 坦诚评价

**结论：AV-LRM（视觉方差+音频能量打分）在绝大多数场景下不如 naive I 帧均匀选取。**

**为什么有限制？**
- AV-LRM 衡量的是"GOP 内容变化大不大"（视觉方差）和"声音活跃程度"（音频能量）
- 但"内容变化大" ≠ "对回答问题有用"——它 **不理解问题**，无法做 question-aware 的帧选择
- 要理解问题需要用模型做前向推理（类似 FastV 的 attention pruning），与 training-free 预处理定位矛盾
- 短视频 QA 任务中，时间覆盖度 > 内容选择性，均匀采样已经是最优策略

**从 AV-LRM 中得到的结论（有学术价值）**：
1. **Two-Regime 发现**：覆盖度优先 vs 精准度优先的交叉点在 kr≈0.3，为部署策略提供量化依据
2. **简单 > 复杂**：对短视频 QA，不需要复杂打分，均匀 I 帧即可
3. **演进方向**：从统计特征（AV-LRM）→ 物理运动信号（Motion Vector）是更有效的 training-free 信号

**论文定位**：AV-LRM 作为早期探索如实报告，Two-Regime 发现是独立贡献，核心方法是 GOP-aware I 帧选取 + Adaptive kr + Motion-Aware 采样

### 核心技术路线（2.22 晚确认）

```
短视频 / 大部分场景     → naive I 帧均匀选取（kr=0.5 零损失，2.1x 加速）
中长视频               → Adaptive kr 防截断 + 显存优化提上限
时序敏感任务           → Motion-Aware P/B 帧补偿（Layer 3）
极端低预算 (kr≤0.2)   → AV-LRM 有微弱优势，可作为可选模式保留
```

**一句话**：大部分场景 naive I 帧就够了，只有时序敏感任务才需要在 P/B 帧上做文章。

### Pareto 曲线（naive_iframe × Video-MME Short）

| kr | Accuracy | vs Baseline | Visual Tokens | Speedup |
|----|----------|-------------|---------------|--------|
| 0.2 | 68.52% | -7.41pp | 2192 (20%) | 3.6x |
| 0.3 | 69.44% | -6.49pp | 3190 (30%) | 2.8x |
| **0.5** | **75.93%** | **0.00pp** | 4939 (46%) | **2.1x** |
| 0.7 | 71.30% | -4.63pp | 6658 (62%) | 1.6x |
| 0.9 | 70.37% | -5.56pp | 7794 (73%) | 1.4x |

### MVBench 全量结果（1-GOP 修复后重跑）

| Mode | Valid/Total | Accuracy | Errors | Avg Generate ms | Avg Visual Tokens | Avg Frames |
|------|-----------|----------|--------|----------------|-------------------|------------|
| **baseline** | 3318/3600 | **66.94%** | 282 (OOM) | 1054ms | 5570 | 29.8 |
| naive_iframe(0.5) | 3579/3600 | 53.59% | 21 | 196ms | 628 | 3.0 |
| sparse(0.5) | 3579/3600 | 53.34% | 21 | 198ms | 628 | 3.0 |

> **vs 修复前**：错误数从 1501 降至 21（naive/sparse），新增处理了 1480 个之前崩溃的 1-GOP 视频
> **Baseline 的 282 个 OOM**：全是长视频内存溢出，即使按 OOM=答错 校正，准确率仍有 61.7%（> naive 53.6%）
> **关键发现**：MVBench 帧预算极低（kr=0.5 后仅 ~3 帧 vs baseline ~30 帧），时序密集型任务严重退化
> **naive ≈ sparse**：视觉 token 数完全相同（628），kr=0.5 处于 Coverage-dominant 区间，AV-LRM 打分无额外优势

#### MVBench 按任务分析

| 任务 | Baseline | naive_iframe | Δ | 类别 |
|------|----------|-------------|---|------|
| counterfactual_inference | 68.0% | 32.0% | **-36.0pp** | 🔴 严重退化 |
| object_existence | 88.5% | 55.5% | **-33.0pp** | 🔴 严重退化 |
| moving_attribute | 95.0% | 62.5% | **-32.5pp** | 🔴 严重退化 |
| moving_direction | 59.5% | 37.0% | -22.5pp | 🔴 严重退化 |
| action_sequence | 75.4% | 53.0% | -22.4pp | 🔴 严重退化 |
| action_prediction | 68.0% | 46.0% | -22.0pp | 🔴 严重退化 |
| moving_count | 69.0% | 47.0% | -22.0pp | 🔴 严重退化 |
| object_interaction | 75.8% | 58.5% | -17.3pp | 🟡 中等退化 |
| action_antonym | 79.5% | 69.3% | -10.2pp | 🟡 中等退化 |
| character_order | 74.1% | 64.0% | -10.1pp | 🟡 中等退化 |
| action_localization | 44.3% | 35.0% | -9.3pp | 🟡 中等退化 |
| egocentric_navigation | 39.5% | 32.0% | -7.5pp | 🟡 中等退化 |
| object_shuffle | 37.9% | 35.0% | -2.9pp | 🟢 可接受 |
| unexpected_action | 82.1% | 80.0% | -2.1pp | 🟢 可接受 |
| state_change | 60.7% | 59.5% | -1.2pp | 🟢 可接受 |
| scene_transition | 96.5% | 96.5% | +0.0pp | 🟢 无损 |
| fine_grained_action | 47.5% | 48.7% | +1.2pp | 🟢 反超 |
| action_count | 34.6% | 51.0% | **+16.4pp** | 🟢 大幅反超 |

**分析**：
- 🔴 **严重退化（7 个任务，Δ > -20pp）**：全是短视频（BL visual tokens ~935，即 ~2 帧原始视频），kr=0.5 后只剩 ~1 帧，时序信息完全丧失
- 🟢 **action_count 反超 +16.4pp**：baseline 只有 162/200 valid（38 个 OOM），naive 处理全部 200 个。naive 能处理 baseline OOM 的长视频，且 baseline 本身在这个任务上就很差（34.6%，接近随机 25%）
- 🟢 **scene_transition 完全无损**：场景转换只需少量关键帧即可判断
- **结论**：帧削减在"需要精细时序"的短视频任务上代价大，在"场景级理解"任务上几乎无损。**支持 Adaptive kr 的必要性——不同视频/任务需要不同的帧预算**

### 音频角色

- **单独有用**：audio_only(51.3%) >> text_only(42.0%)
- **有视频时无额外贡献**：video_only(62.2%) ≈ baseline(61.9%)
- **Medium 甚至干扰**：video_only(61.1%) > baseline(56.7%)
- **结论**：保留全量音频作为 free lunch（encoder 仅 ~22ms），不需特别优化也不需去掉

---

## 待做工作

### 当前（Layer 2 实验 + 论文输出）

| # | 任务 | 状态 | 说明 |
|---|------|:----:|------|
| 1 | **Layer 2 Adaptive kr 代码** | ✅ 完成 | `3bb736c` smoke test 通过，sparse+naive 两路径一致 |
| 2 | **Video-MME 全集 Adaptive kr** | 🔄 跑中 | naive_iframe+sparse × 292 题，tmux eval |
| 3 | **论文图表** | ✅ 完成 | Pareto 曲线+MVBench 按任务图（`tools/plot_figures.py`） |
| 4 | **MV 提取 PoC** | ✅ 完成 | PyAV 8766帧 MV，1.14ms/帧（`tools/mv_extraction_poc.py`） |
| 5 | **分析 Adaptive kr 实验结果** | ⬜ 等数据 | 按 S/M/L 分 accuracy，对比旧数据看 M/L 提升 |
| 6 | **模型迁移到 autodl-fs** | ⬜ 待做 | 省 ~¥4/月，实验跑完后执行 |
| 7 | **Layer 3 Motion-Aware 设计** | ⬜ 下一步 | MV PoC 已验证，需设计 motion-weighted 采样公式 |

### Phase 3（架构扩展）— 整体加速流水线

**核心目标**：解决中长视频的 max_frames=32 瓶颈，构建端到端加速系统。

**已定位的关键问题**：kr>0.5 准确率下降的根因是 max_frames=32 截断。同一批视频，kr=0.5 给 26 帧得 83.3%，kr=0.7 截断到 32 个 I 帧只有 77.8%（-5.5pp），baseline 均匀 32 帧是 83.3%。I 帧时间聚类（集中在场景切换处）导致截断后覆盖度不如均匀采样。

**解法**：提高 OOM 边界 → 放开 max_frames → 稀疏化不再被截断 → 中长视频可用。

```
输入视频
  │
  ├─[CPU] GOP 解析 + I 帧定位
  │
  ├─[CPU] Content-Adaptive kr（动态调 kr，保证帧数不超 max_frames）
  │
  ├─[CPU] I 帧解码（按批次）──┐
  │                             │  Frame-level CPU/GPU 解耦
  │          ┌──────────────────┘
  │          ▼
  │   [GPU] ViT 分块编码（编码一批 → 释放激活 → 下一批）
  │
  │   [GPU] Audio Encoder（~22ms 固定开销）
  │
  └─► [GPU] LLM Prefill → 输出 Token
```

#### 稀疏化模块三层架构

| 层 | 功能 | 状态 | 说明 |
|----|------|:----:|------|
| **L1: I 帧均匀选取** | naive_iframe 覆盖度优先 | ✅ 完成 | kr=0.5 零损失，Two-Regime 理论验证 |
| **L2: Adaptive kr** | max_frames 硬约束，防截断 | ✅ 完成 | `3bb736c` 两路径一致，全集实验跑中 |
| **L3: Motion-Aware 补偿** | 时序敏感任务补充 P/B 帧信息 | 🔄 PoC 完成 | MV 提取验证可行（1.14ms/帧），需设计采样公式 |

> **L3 设计思路（P/B 帧运动向量）**：
> - P/B 帧不存完整像素，存储 Motion Vector (MV) = 每个宏块相对参考帧的位移 (dx, dy)
> - ffmpeg 可零成本从码流中导出 MV，不需解码像素
> - MV 幅度大 = 运动剧烈 → 该区域密采样；MV 幅度小 = 静态 → 稀采样
> - 将 naive 的"均匀采样"升级为 motion-aware 非均匀采样，仍是 training-free
> - MVBench 按任务分析直接支持：7 个严重退化任务（moving_attribute -32.5pp 等）本质需要帧间运动信息
> - ⚠️ 注意：GOP 数量受编码设置影响（keyframe_interval），不完全反映内容复杂度。L3 用 MV 而非 GOP 密度更鲁棒

| # | 任务 | 优先级 | 说明 |
|---|------|:------:|------|
| P0 | **Adaptive kr（Layer 2）** | ⭐⭐⭐ | `kr = min(kr_base, max_frames / n_valid_gops)`，消除二次截断。Prompt: `gpt_adaptive_kr_prompt.md` |
| P0 | **显存碎片优化** | ⭐⭐⭐ | 三层递进：ViT 后清理 → 输入规整化 → 分块编码。初版方案见 `docs/2.15-2.16 日.md` |
| P1 | **Motion-Aware 补偿（Layer 3）** | ⭐⭐ | MV 提取 + motion-aware 非均匀采样，解决时序敏感任务退化 |
| P1 | **帧级 CPU/GPU 解耦** | ⭐⭐ | CPU 解码 batch N+1 的同时 GPU 编码 batch N，降峰值+提吞吐 |
| P2 | **Ring Buffer 流水线** | ⭐ | 完整异步预取，作为论文 future work 或加分项 |

### 论文输出

| # | 任务 | 状态 | 说明 |
|---|------|:----:|------|
| 10 | Pareto 曲线图 | ✅ 完成 | `tools/plot_figures.py` → `/root/autodl-tmp/results/figures/pareto_curve.{png,pdf}` |
| 11 | MVBench 按任务分析图 | ✅ 完成 | 同上 → `mvbench_per_task.{png,pdf}` |
| 12 | MV Profile 可视化 | ✅ 完成 | `tools/mv_extraction_poc.py` → `mv_profile_323v_FtWqvo.png` |
| 13 | Video-MME 全集结果表 | ⬜ 等实验 | Adaptive kr 实验跑完后按 S/M/L 汇总 |
| 14 | 论文初稿 | ⬜ 待启动 | 故事线已定（见下方），技术点 L1/L2 完成后可开始 |

---

## 代码结构

```
fasteromni/
├── pipeline.py          # 推理引擎（9 种 mode，帧选择+推理解耦，含 Adaptive kr）
├── eval_videomme.py     # Video-MME 评估（300 题，增量 CSV）
├── eval_mvbench.py      # MVBench 评估（3600 题，增量 CSV）
├── bootstrap_ci.py      # Bootstrap CI 统计分析
├── modules/
│   ├── gop_parser.py    # GOP 结构解析
│   ├── sparse.py        # AV-LRM 打分 + 帧选择
│   ├── audio_energy.py  # 音频能量提取
│   └── frame_decoder.py # I 帧解码
└── phase1_archive/      # 废弃脚本

scripts/
├── migrate_to_fs.sh     # 模型迁移到 autodl-fs（省 ~¥4/月）
├── plot_figures.py      # 论文图表生成（Pareto + MVBench 按任务图）
└── mv_extraction_poc.py # Motion Vector 提取 PoC（Layer 3 验证）

non_inferiority.py       # Non-inferiority 统计检验
run_all_experiments.sh   # 一键实验脚本

gpt_adaptive_kr_prompt.md  # L2 Adaptive kr Codex Prompt
gpt_pareto_plot_prompt.md  # 论文图表 Codex Prompt
gpt_mv_extraction_prompt.md # MV 提取 Codex Prompt
```

---

## 数据保护（⚠️ 必须遵守）

> **已有实验数据绝不覆盖。新实验必须写入新目录。**

**结果根目录**：`/root/autodl-tmp/results/fasteromni/`

| 目录 | 内容 | 状态 |
|------|------|:----:|
| `videomme_full/` | 6 模式 × 全量 Video-MME（300×6 题） | 🔒 |
| `videomme_full/bootstrap_ci/` | Bootstrap CI（10,000 次） | 🔒 |
| `videomme_full/non_inferiority/` | Non-inferiority 结果 | 🔒 |
| `naive_comparison/` | Naive baselines kr=0.5（108×5 题） | 🔒 |
| `naive_comparison_kr02/` | Naive baselines kr=0.2（108×4 题） | 🔒 |
| `sparse64/` | Sparse@64 vs Baseline@64（108×2 题） | 🔒 |
| `videomme/ablation_kr_short/` | kr sweep sparse only（108×6 题） | 🔒 |
| `mvbench/` | MVBench 全量（3 mode × 3600 题） | 🔒 |
| `pareto_naive_iframe/` | naive_iframe kr sweep（5 kr × 108 题） | 🔒 |
| `videomme_adaptive_kr/` | Adaptive kr 全集实验（naive+sparse × 292 题） | 🔄 写入中 |

**数据集位置**：
- Video-MME: `/root/autodl-tmp/data/Video-MME/`
- MVBench: `/root/autodl-tmp/data/MVBench/`（3,333 视频 × 20 JSON）

---

## 关键设计约束

1. ⚠️ TTFT 只能用 `max_new_tokens=1` 测量
2. ⚠️ 音频必须通过 `proc(audio=, use_audio_in_video=True)` 传入
3. ⚠️ Sparse 必须走 video tensor 路径（~150 tokens/frame）
4. Video-MME baseline 用 max_frames=32（64 OOM）
5. MVBench 不加 max_frames（视频 <30s，无 OOM）

---

## 编码规范

1. 不批量修改 generate 调用参数（逐条改，逐条验证）
2. 新增实验不动旧代码
3. 超时保护：所有外部 I/O 用 `subprocess.run(timeout=N)`
4. 增量 CSV 实时写入，支持断点恢复
5. 不改第三方库，用 monkey-patch

---

## GPT 工具分工

| GPT 版本 | 角色 | 适用场景 |
|----------|------|---------|
| **GPT 5.2** | 军师 | Review、论文故事线、实验方案、负结果解释 |
| **GPT 5.3 Codex** | 执行 | 写代码、数据处理、评估脚本 |

---

## 相关工作

| 方法 | 核心思路 | 与我们的差异 | 状态 |
|------|----------|------------|------|
| FastV | Attention-based token pruning | 帧内 patch 级 vs 我们帧级 | 待调研 |
| LLaVA-PruMerge | 融合+剪枝 visual tokens | 模型内部 vs 预处理阶段 | 待调研 |
| **Mobile-VideoGPT** | 注意力关键帧+token剪枝 | 端侧，思路类似 | ✅ 已调研 |
| MiniCPM-o | 端侧全模态 | 功能更全，未专注稀疏化 | 待调研 |
| **CoPE-VideoLM** | Codec primitives (MV+残差) 替代 RGB 帧，训练 lightweight encoder | 需训练 vs 我们 training-free；单模态 vs 多模态；Stanford 2026.02 | ✅ 已调研 |
| **Motion-Aware GOP Encoder** | GOP 内融合空间+运动信息 (CVPR 2025) | 需训练 vs training-free | ✅ 已调研 |

---

## 工作流程规范

### Review 规范
- **每个技术点最多 1 轮 Review**，聚焦 go/no-go（数据有没有问题）
- **实验数据 Review ≠ 论文 Review**：当前阶段只审查实验链路和数据可信度
- 论文结构 Review 等所有技术点完成后再做
- Review 无致命问题 → 锁定数据，推进下一个技术点

### 推进节奏
- 稀疏化模块未完成项：Content-adaptive kr / Hybrid 策略
- 完成顺序：修 bug → 重跑实验 → Review 数据 → 开发自适应 → 推进下一技术点

---

## 待探讨问题（供 Agent 讨论）

- [ ] **MVBench 41.7% 失败率根因**：1-GOP 编码视频（clevrer/ssv2）→ sparse 选 0 帧 → processor 崩溃。修复：fallback 保留至少 1 帧
- [ ] **Pareto 非单调（已解决）**：根因是 **max_frames=32 截断 + I 帧时间聚类**。kr=0.7 时 17% 视频触及上限，32 个 I 帧的时间覆盖度不如 32 个均匀帧（同视频同帧数：I 帧 77.8% vs 均匀 83.3%，差 5.5pp）。kr=0.5 是 sweet spot：帧数不触发截断（平均 14.4 帧）且覆盖度最大化。论文表述：存在最优稀疏度，超过后 I 帧聚类反而损害覆盖度
- [ ] **Medium 音频干扰**：video_only > baseline (+4.4pp)，统计显著性待验证
- [x] **M/L sparse 无效（方案已定）**：max_frames=32 限制了中长视频。解法：显存优化提高 OOM 边界 → 放开 max_frames + Content-adaptive kr 避免截断
- [ ] **CoPE-VideoLM 差异化定位**：Stanford 2026.02 发了 codec-aware VideoLM（arXiv 2602.13191，训练 encoder，-93% tokens，14 benchmarks 持平/超越），验证了 codec-aware 方向的可行性。我们的差异化：training-free + 端侧部署 + 多模态（视频+音频）。论文需引用并明确区分

### 📱 手机讨论方向（供 Agent 探索）

以下问题适合在手机上与 Agent 探讨，产出可直接用于后续开发：

1. **显存优化方案设计**：初版三层递进（ViT 后清理/输入规整化/分块编码）是否足够？老师之前的顾虑（VLM 已做过类似优化）如何回应？我们的差异化：GOP 稀疏化 + 显存优化的联合系统，不是通用 VLM 优化
2. **中长视频优化具体解法**：Content-adaptive kr 的公式设计？帧级 CPU/GPU 解耦的实现路线？如何验证效果？
3. **论文叙事**：整体加速流水线的故事线怎么讲？四个技术点（稀疏化/显存/自适应/流水线）如何组织成连贯的贡献？Pareto 非单调的论文解释？
4. **老师建议整合**：之前老师对显存优化的具体反馈是什么？如何把"已有工作做过"转化为"我们的系统性整合是新的"？与 Mobile-VideoGPT 的差异化？

### 论文故事线（2.22 Jarvis + East Hole 讨论）

**一句话定位**：面向端侧设备的 training-free 视频推理加速框架，核心洞察是利用视频编码格式（GOP 结构）作为免费的冗余信号。

**故事线（四段式）**：

1. **问题**：多模态大模型处理视频时，visual tokens 占计算量的大头。现有加速方法要么需要额外训练（CoPE-VideoLM、Motion-Aware GOP Encoder），要么需要额外前向传播计算 attention score（FastV、LLaVA-PruMerge）。端侧设备算力有限，这些额外开销本身就是负担。

2. **洞察**：视频编码格式（H.264/H.265）的 GOP 结构天然标记了帧间冗余——I 帧是信息密度最高的关键帧，这个信息解码时就能拿到，零额外计算。

3. **方法**：围绕 GOP 结构设计一套完整的加速流水线：
   - GOP 级帧选择（稀疏化）— 利用 I 帧做 token 压缩，kr=0.5 零损失
   - AV-LRM 智能选帧 — 低预算时补充精准度，和 naive 形成互补（Two-Regime）
   - GOP 边界分块编码 — 显存优化，解锁中长视频
   - CPU/GPU 异步流水线 — 端到端延迟优化

4. **贡献**：
   - 首个 training-free 的 codec-aware 视频 LMM 加速方案
   - 发现帧选择的 Two-Regime 现象（coverage-dominant vs relevance-dominant），为部署策略提供理论指导
   - 双 benchmark（Video-MME + MVBench）验证，Pareto 曲线展示 accuracy-efficiency tradeoff

**核心卖点**：codec-aware、training-free、端侧多模态

**与 CoPE-VideoLM 的区分**：CoPE 训练 encoder 走学术路线（-93% tokens 但需要预训练+微调），我们不改模型走部署路线（即插即用，适合端侧）

---

### M/L 视频 Adaptive kr 方案（2.22 晚讨论）

**问题**：中长视频 I 帧数量多，固定 kr 稀疏化后帧数仍超过 max_frames=32，被 processor 均匀截断，稀疏化策略被覆盖，等于白做。

**核心解法：Adaptive kr**
- 公式：`kr = min(kr_base, max_frames / n_iframes)`
- 效果：保证稀疏化输出 ≤ max_frames，不触发二次截断
- 选帧控制权留在稀疏化策略手里（AV-LRM / naive I 帧），不被 processor 均匀采样覆盖

**两层动态 kr**：
1. **基于 max_frames 的 adaptive kr**（必做）：解决 M/L 截断问题，实现简单效果直接
2. **基于视频内容/Task Type 的 adaptive kr**（学术亮点）：运动剧烈视频需要更多帧（低 kr 丢关键动作），对话类视频帧冗余大（高 kr 也没事）

**与显存优化的配合**：显存优化提高 max_frames 上限（32→64+），adaptive kr 自动适配新上限，M/L 视频保留更多高分帧。两者联合：上限更高 + 分配更智能。

### 项目定位更新（2.22 晚讨论）

**定位微调**：面向 HuggingFace Transformers 原生推理栈的 training-free 视频 token 稀疏化框架，适用于资源受限设备（消费级 GPU / 边缘服务器）

**关键特性**：
- **插件式优化**：不动模型源码，纯预处理层插入，monkey-patch only
- **即插即用**：目标封装为 Python 包，pip install 开箱即用
- **受众广**：HF Transformers 是开源 LMM 事实标准，几乎所有开源视频模型都走这套栈
- **与 CoPE 的区分**：CoPE 需要改架构+重新训练，我们不动模型走部署路线

**目标硬件**：消费级 GPU（RTX 3090/4090/5090）、边缘服务器，不局限于纯端侧设备

---

## 变更日志

### [2.24 PM] Adaptive kr 全集实验完成 + M/L 稀疏化失效问题（Jarvis 排查）

- **实验结果**（`videomme_adaptive_kr/`，292 题 × 2 模式）：
  - naive_iframe: overall 61.3% | Short 77.0% | Medium 60.0% | Long 47.1%
  - sparse: overall 59.7% | Short 70.4% | Medium 61.1% | Long 47.1%
- **关键发现**：M/L 视频 vistok ≈ 10737（与 baseline 完全相同），adaptive kr 未减少 token
- **根因**（Jarvis 诊断确认）：
  - M/L 视频 n_valid=100~1000（GOP 数量远超 64）
  - `kr_adaptive = min(0.5, 32/n_valid)` → K = ceil(n_valid * kr_adaptive) = 32
  - 选出 32 个 I 帧 = baseline 的 32 帧，稀疏化无优势
  - 诊断验证：152 GOPs→K=32, 23 GOPs→K=12, 983 GOPs→K=32
  - 代码逻辑本身无 bug，K ≤ max_frames 是数学保证的，硬截断（L377）未触发
- **结论**：adaptive kr 的 "防截断" 设计在 max_frames=32 下无法让 M/L 视频受益于稀疏化
- **待讨论方向**：
  1. 提高 max_frames（64/128）+ 显存优化 → 稀疏化在更大帧预算下发挥作用
  2. 降低 K + 提升选帧质量 → 用更少帧但更精准的 I 帧
  3. 混合策略 → Short 用 Layer 1，M/L 用不同帧预算策略


- **[2.22 深夜]** PROGRESS.md 全面更新（达到新对话可直接接续）。创建对话交接 Skills（`/handoff`, `/pickup`）
- **[2.22 晚-8]** Pareto 图修复三轮：图例移到图外、白底标签、kr=0.3 不遮红三角（`f4e19ec`）
- **[2.22 晚-7]** 论文图表完成（`tools/plot_figures.py`）+ MV 提取 PoC 完成（`tools/mv_extraction_poc.py`）。commit `c0f688b`
- **[2.22 晚-6]** Adaptive kr 代码 Review + Smoke test 通过（`3bb736c`）。Video-MME 全集实验启动（tmux eval）。AutoDL FS vs TMP 对比分析
- **[2.22 晚-5]** AV-LRM 坦诚评价写入 PROGRESS.md。核心技术路线确认：naive I帧为主力，Motion-Aware L3 解决时序敏感任务，AV-LRM 作为 Two-Regime 发现保留
- **[2.22 晚-4]** Adaptive kr GPT Prompt 完成（`gpt_adaptive_kr_prompt.md`）。稀疏化三层架构规划：L1 I帧选取(✅) → L2 Adaptive kr(🔄) → L3 Motion-Aware P/B帧补偿(⬜)
- **[2.22 晚-3]** MVBench 重跑结果分析完成。按任务分析：7个任务严重退化（Δ>-20pp），4个可接受/反超。action_count 反超+16.4pp（baseline OOM偏差）。支持 Adaptive kr 必要性
- **[2.22 晚-2]** M/L 视频 Adaptive kr 方案讨论完成 + 项目定位更新为 HF 原生栈 training-free 插件式优化
- **[2.22 晚]** 论文故事线讨论完成（Jarvis + East Hole）。核心定位：training-free codec-aware 端侧加速。CoPE-VideoLM 差异化明确
- **[2.22 午后-2]** Pareto 非单调根因分析完成（max_frames 截断+I 帧时间聚类）。Phase 3 架构规划完成。手机讨论方向整理
- **[2.22 午后]** Sanity Check 通过（86.1%一致率+token减54%+8题翻转对称抵消）。pipeline 无 bug，kr=0.5 零损失确认为真实现象。GPT Review concern 全部关闭。MVBench 重跑中
- **[2.22 午前-2]** GPT 5.2 Review 完成。核心结论：(1) kr=0.5=BL 需 Sanity Check 排除 pipeline bug; (2) AV-LRM 加覆盖约束后再判断; (3) 短期收尾 move on
- **[2.22 午前]** 1-GOP fallback 修复(c0069fc)+工作流程规范+MVBench重跑准备
- **[2.22 AM-2]** MVBench 失败根因定位：1-GOP 编码视频→sparse 选 1 帧+音频→processor StopIteration
- **[2.22 AM]** MVBench 全量 + Pareto naive_iframe kr sweep 完成。核心发现：kr=0.5 零损失（=BL 75.93%）
- **[2.21 PM-5]** Non-inferiority 代码验证+commit。PROGRESS.md 精简（728行→~200行），历史归档到 PROGRESS_ARCHIVE.md
- **[2.21 PM-4]** MVBench 代码就绪+数据保护约定+Non-inferiority prompt。Sparse@64 闭环分析完成
- **[2.21 PM-3]** GPT Review 完成+MVBench 解压。论文定位转型确定
- **[2.21 PM-2]** Bootstrap CI 完成（10,000 次）+MVBench 下载
- **[2.21 PM]** Benchmark 决策（Video-MME Short + MVBench）
- **[2.21 AM]** Modality Baselines 全量+架构重构完成
- **[2.20]** 代码污染事件+回退+清理归档
- **[2.19]** Phase 2 实验（Naive/Sparse@64/kr=0.2）+GPT Review
- **[2.18]** Phase 1 完整评估+kr消融+死锁修复
- **[2.15-17]** 项目启动、Pipeline 实现、Video-MME 接入

> 详细历史见 `PROGRESS_ARCHIVE.md`
