# FasterOmni 开发进度

> **本文件是跨对话、跨平台的唯一进度真实来源。**
> 历史细节见 `PROGRESS_ARCHIVE.md`。

---

## 当前状态（2.21 PM）

**补充实验阶段**：稀疏化 Pipeline 已完成，正在补实验数据提高置信度。

- 🔄 **MVBench 全量**正在 tmux `eval` 中运行（baseline 进行中，3 mode × 3600 题，~4h）
- 🔄 **Pareto 补数据**排队中（naive_iframe kr sweep，~1.5h）
- ✅ **Non-inferiority** 已完成（naive_iframe δ=3pp PASS）
- ✅ **Video-MME** 全部实验完成（6 模式 × 300 题 + Bootstrap CI）

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
| **naive_iframe 砍 54% tokens 几乎不掉点** | Non-inferiority δ=3pp PASS（ci_lower=-2.1pp > -3pp） |
| **AV-LRM 价值在低预算鲁棒性** | kr=0.5 最差(69.4%), kr=0.2 最优(70.4%)，排名反转 |
| **准确率对 kr 不敏感** | 68.5%~70.4% 范围内波动 |
| **音频兜底效应很小** | 去音频仅多降 1.8pp，作为 free lunch 保留 |
| **Baseline@64 不可用** | 89% OOM，Sparse@64 零 OOM |
| **Short 是主战场** | 2x 加速 -5.6pp；M/L 因 max_frames=32 稀疏化无效 |
| **语言先验显著** | text_only=42%（远 > 随机 25%），Long 视频 91% 靠语言 |

### Two-Regime 理论（GPT 5.2）

- **Coverage-dominant（kr≈0.5）**：覆盖度 > 精准打分 → naive_iframe 强
- **Relevance-dominant（kr≤0.2）**：预算极紧 → AV-LRM 精准投放占优
- 这解释了为什么 kr=0.5 时 AV-LRM 最差、kr=0.2 时最优

### 音频角色

- **单独有用**：audio_only(51.3%) >> text_only(42.0%)
- **有视频时无额外贡献**：video_only(62.2%) ≈ baseline(61.9%)
- **Medium 甚至干扰**：video_only(61.1%) > baseline(56.7%)
- **结论**：保留全量音频作为 free lunch（encoder 仅 ~22ms），不需特别优化也不需去掉

---

## 待做工作

### 当前（补实验 + 置信度）

| # | 任务 | 状态 | 说明 |
|---|------|:----:|------|
| 1 | **MVBench 全量** | 🔄 运行中 | 3 mode × 3600 题，第二个 benchmark |
| 2 | **Pareto naive_iframe kr sweep** | 🔄 排队 | 5 个 kr × Video-MME Short 108 题 |
| 3 | **Non-inferiority** | ✅ 完成 | naive_iframe δ=3pp PASS |
| 4 | **Sparse@64 闭环** | ✅ 数据已有 | 70.4% vs BL@32 75.9%，tokens 少 54% |
| 5 | **Hybrid 策略** | ⬜ 待设计 | naive_iframe 覆盖 + AV-LRM 分配剩余预算，~30 行 |

### Phase 3（架构扩展）

| # | 任务 | 说明 |
|---|------|------|
| 6 | **显存碎片优化** | ViT 后 hook 清理激活值 → 降低峰值 → 支持更长视频 |
| 7 | **Ring Buffer 流水线** | CPU/GPU 异步预取，隐藏预处理延迟 |
| 8 | **Content-adaptive kr** | 根据视频内容动态调 kr |
| 9 | **P/B 帧选择** | GOP 内选帧，更细粒度优化 |

### 论文输出

| # | 任务 | 依赖 |
|---|------|------|
| 10 | Pareto 曲线图 | #2 完成 |
| 11 | MVBench 结果表 | #1 完成 |
| 12 | 论文初稿 | 全部实验 |

---

## 代码结构

```
fasteromni/
├── pipeline.py          # 推理引擎（9 种 mode，帧选择+推理解耦）
├── eval_videomme.py     # Video-MME 评估（300 题，增量 CSV）
├── eval_mvbench.py      # MVBench 评估（3600 题，增量 CSV）
├── bootstrap_ci.py      # Bootstrap CI 统计分析
├── modules/
│   ├── gop_parser.py    # GOP 结构解析
│   ├── sparse.py        # AV-LRM 打分 + 帧选择
│   ├── audio_energy.py  # 音频能量提取
│   └── frame_decoder.py # I 帧解码
└── phase1_archive/      # 废弃脚本

non_inferiority.py       # Non-inferiority 统计检验
run_all_experiments.sh   # 一键实验脚本
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
| `mvbench/` | MVBench 全量（🔄 运行中） | 新 |
| `pareto_naive_iframe/` | Pareto 补数据（🔄 排队） | 新 |

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

---

## 待探讨问题（供 Agent 讨论）

- [ ] **Medium 音频干扰**：video_only > baseline (+4.4pp)，统计显著性待验证
- [ ] **Short 逐视频波动大**：部分视频准确率暴跌 66pp，Content-adaptive kr 能否解决？
- [ ] **M/L sparse 无效**：max_frames=32 卡死，需要帧级选择或提高 max_frames

---

## 变更日志

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
