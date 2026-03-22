# FasterOmni 开发进度
> **本文件是跨对话,跨平台的唯一进度真实来源.**
> Windsurf / Claude Code 每次对话开始时读取此文件获取上下文,结束时更新此文件.
> Obsidian Agent 关注 `待探讨问题` 和 `外部反馈` 区域.

## 文档体系说明

1. **PROGRESS.md**（本文件）— 实时看板：当前状态、待办、阻塞点
2. **PROGRESS_ARCHIVE.md** — 完整历史：所有实验数据、变更日志、编码规范
3. **STORY.md** — 研究叙事线：问题→动作→证据→决策的主线记录

---

## 当前阶段
**技术点 1（GOP 级 token 稀疏化）基本完成，准备论文写作**

核心实验已全部完成。经过 adaptive v2/v3 探索，确认 **naive_iframe（均匀选 I 帧）** 为最优方案，AV-LRM 打分在当前实验设置下未能超越简单策略。

**研究定位**：聚焦 **10s-180s 短视频**（Video-MME Short 11s-2min + ActivityNet-QA 30-180s），对齐端侧多模态推理加速的典型场景。

**论文策略**：
- **毕业论文**：用 naive_iframe 作为核心方法（training-free + codec-aware + 零损失加速），材料充足
- **会议论文**：需补充技术点 2/3（RingBuffer + 显存优化）形成完整系统，或探索 content-adaptive 扩展

## 项目背景
研究 **Qwen2.5-Omni-7B** 多模态大模型的推理加速，构建**端侧多模态推理加速系统**。

**三大技术支柱**（独立可解耦）：
1. ✅ **GOP 级 token 稀疏化**（naive_iframe 为主）← 实验完成 ~98%
2. ⬜ **Ring Buffer 流水线**（CPU/GPU 异步预取）← 待开发，预计 3-5 天
3. ⬜ **显存碎片优化**（ViT 后清理激活值）← 待开发，预计 1-2 天

**硬件**：RTX 5090 32GB，OOM 边界 ~25-30k tokens（约 34s 视频全量帧）。
**论文定位**：边缘服务器/轻量部署（单 GPU 32GB），training-free codec-aware 加速系统。

## 研究 Scope 确认

### 主实验数据集（10s-180s 短视频）

| 数据集 | 视频时长 | 样本数 | 用途 |
|--------|---------|--------|------|
| **Video-MME Short** | 11s - 2min (平均 80.8s) | 300 题 | 主实验，零损失加速 + OOM 解决 |
| **ActivityNet-QA** | 30s - 180s (平均 2-3min) | 8000+ 样本 | 可选扩展，大样本验证 |

**对齐同方向论文**：
- 端侧多模态推理加速论文（如 FreeVA, MMEdge）普遍使用 ActivityNet-QA, MSVD, MSRVTT
- 视频时长范围：10s - 180s
- 不评估超长视频（>15min）

### Limitation 数据集（方法边界）

| 数据集 | 视频时长 | 样本数 | 失效原因 | 论文用途 |
|--------|---------|--------|---------|---------|
| **MVBench** | 2s - 10s | 3600 题 | GOP 太少，稀疏化后时序信息丧失 | Limitation: 极短视频 |
| **Video-MME Medium** | 4min - 15min (平均 8.7min) | 300 题 | max_frames=32 锁死，稀疏化被截断 | Limitation: max_frames 限制 |
| **Video-MME Long** | 30min - 60min (平均 41.2min) | 300 题 | max_frames=32 锁死，稀疏化被截断 | Limitation: 长上下文不在 scope |

## 模块状态

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| GOP 解析 | `fasteromni/modules/gop_parser.py` | ✅ | 16 视频验证，GOP 中位数 5 |
| 音频能量 | `fasteromni/modules/audio_energy.py` | ✅ | RMS 能量按 GOP 时间窗口切分 |
| AV-LRM 打分 | `fasteromni/modules/sparse.py` | ✅ | 包含 stratified_top_k 实现 |
| I 帧解码 | `fasteromni/modules/frame_decoder.py` | ✅ | 全解码后过滤（已知限制） |
| Pipeline | `fasteromni/pipeline.py` | ✅ | baseline/sparse/naive_iframe 等模式 + 三层保护机制 |
| Video-MME 评估 | `fasteromni/eval_videomme.py` | ✅ | 300 题完整评估 + 增量 CSV + 断点恢复 |
| Bootstrap CI | `fasteromni/bootstrap_ci.py` | ✅ | 10,000 次 bootstrap + non-inferiority |

## 已完成实验汇总

### Video-MME (300 题)
| 方法 | Short | Medium | Long | Overall |
|------|-------|--------|------|---------|
| Baseline (32帧) | 75.93% | 56.67% | 49.41% | 62.00% |
| naive_iframe (kr=0.5) | **75.93%** | 60.00% | 47.06% | 62.33% |
| sparse (kr=0.5) | 69.44% | 61.11% | 47.06% | 60.33% |
| adaptive_v2 (top_k) | 72.22% | — | — | — |
| adaptive_v3 (stratified) | 71.30% | — | — | — |

### Adaptive 策略探索结果（3.10-3.11）

**v2（纯 Top-K + 方差门控）**：
- Video-MME Short: 72.22%（比 naive_iframe 75.93% 低 3.7pp）
- ActivityNet: 41.1%（比 naive_iframe 39.9% 高 1.2pp）
- 问题：Top-K 按分数排序导致高分帧时间聚类，破坏覆盖度

**v3（Stratified Top-K）**：
- Video-MME Short: 71.30%（比 v2 还低 0.9pp）
- ActivityNet: 40.6%（介于 naive 和 v2 之间）
- 问题：强制覆盖度反而稀释了高分 GOP，两头不讨好

**结论**：AV-LRM 打分逻辑可能存在问题，高分 GOP 不一定是真正重要的帧。

### Naive Baselines 对比 (Short 108 题, kr=0.5)
| naive_iframe | 75.93% | = Baseline | 4939 tokens (-54%) | 2.0x |
| naive_uniform | 74.07% | -1.9pp | 4939 tokens | 2.0x |
| naive_random | 73.15% | -2.8pp | 4939 tokens | 2.0x |
| sparse (AV-LRM) | 69.44% | -6.5pp | 4939 tokens | 2.0x |

### Modality Baselines (300 题)
| text_only=42.0% | audio_only=51.3% | video_only=62.2% | baseline=61.9% |

### Sparse@64 闭环 (Short 108 题)
- Baseline@64: 12 valid, **96 OOM (89%)**
- Sparse@64: 108 valid, **0 OOM**
- **OOM 根因**：Short@64 平均 21474 tokens，接近 OOM 边界 25-30k tokens

### Bootstrap CI
- sparse vs baseline: 95% CI [-5.7, +1.4]（跨零，无显著差异）
- naive_iframe vs baseline: 95% CI [-2.8, +4.3]（跨零，无显著差异）

### MVBench (3600 题)
- Baseline: 66.94% (3318 valid, 282 OOM)
- naive_iframe(0.5): 53.59% (-13.35pp)
- **7/18 任务严重退化**（Δ > -20pp），根因：极短视频 kr=0.5 后仅 ~3 帧

### Pareto naive_iframe kr sweep (Short 108 题)
| kr=0.5 | **75.93%** (峰值) | 4939 tokens | 2.0x |
| kr=0.3 | 69.44% | — | — |
| kr=0.7 | 71.30% | — | — |
| kr=0.9 | 70.37% | — | — |

## 关键发现

- **kr=0.5 零损失**：naive_iframe 75.93% = Baseline，2.0x 加速，54% token 减少
- **Two-Regime 理论**：kr≈0.5 coverage-dominant（naive > sparse +6.49pp），kr≤0.2 relevance-dominant（sparse > naive +1.85pp），交叉点 kr≈0.3
- **Pareto 非单调**：kr=0.5 是 sweet spot，kr>0.5 I 帧聚类反而损害覆盖度
- **语言先验**：text_only=42%（远 > 随机 25%），视觉贡献 +20.2pp
- **音频 free lunch**：去音频仅多降 1.8pp
- **M/L 稀疏化失效**：max_frames=32 锁死，Baseline@32 和 Sparse@32 token 数几乎相同
- **MVBench 退化**：极短视频 GOP 少，kr=0.5 后帧数过少，时序信息丧失
- **Sparse@64 价值**：Short@64 Baseline 89% OOM，Sparse 0 OOM，证明稀疏化在高帧数场景的价值
- **AV-LRM 打分问题**：adaptive v2/v3 均未超越 naive_iframe，怀疑 I 帧码率和音频能量不能准确反映信息密度

## 稀疏化链路三层保护机制

**完整梳理**：确保最终帧数不超过 max_frames，避免 OOM

1. **动态 GOP 过滤**：
   - `adaptive_min_gop = max(2, int(median_gop_frames × 0.5))`
   - 根据视频 GOP 特征自适应调整过滤阈值

2. **kr 自适应调整**：
   - `kr_adaptive = min(keep_ratio, max_frames / n_valid)`
   - 提前压缩 kr，避免后续 I 帧超过 max_frames

3. **兜底截断**：
   - `if len(i_frames) > max_frames: 等间隔降采样`
   - 最终保证输入模型的帧数 ≤ max_frames

## 关键设计决策
1. ⚠️ TTFT 只能用 `max_new_tokens=1` 测量
2. ⚠️ 音频必须通过 processor 传入 (`proc(audio=, use_audio_in_video=True)`)
3. ⚠️ Sparse 必须走 video tensor 路径（不是 image）
4. Video-MME baseline 用 max_frames=32（64 帧 OOM）
5. 评估主实验用 Video-MME 选择题（零歧义）
6. alpha 默认 0.3（区间稳定，不追求单点最优）
7. **Baseline 不走 GOP 逻辑**：直接用 qwen_omni_utils.process_mm_info() 均匀采样

## 待办事项

### P0（论文写作准备，3.11-3.25）
- [ ] **代码全面检查**：用 GPT 检查 AV-LRM 打分逻辑（sparse.py / gop_parser.py / audio_energy.py），确认是否有根本性问题
- [ ] **补齐 latency/token 对比表**：整理已有实验数据中的 TTFT/VisTok，输出论文级对比表
- [ ] **论文大纲更新**：基于 naive_iframe 为核心方法更新章节结构
- [ ] **图表制作**：架构图 + 流程图 + 三层保护机制示意图 + 实验曲线

### P1（技术点 2/3 开发，3.13-3.21）
- [ ] **技术点 2：RingBuffer 流水线**（3-5 天）
  - 实现 RingBuffer 类（读写指针、线程安全）
  - 改造 pipeline.py（CPU 解码线程 + GPU 推理线程）
  - 实验验证（TTFT、吞吐量、CPU/GPU 利用率）
- [ ] **技术点 3：显存碎片优化**（1-2 天）
  - ViT forward 后清理中间激活值
  - 监控峰值显存
  - 实验验证（峰值显存、OOM 率）
- [ ] **联合实验**（2-3 天）
  - 三个技术点组合测试
  - 完整系统性能数据

### P2（扩展实验，可选）
- [ ] **ActivityNet-QA 评估**：8000+ 样本，30-180s 视频，大样本验证
- [ ] **Content-adaptive 扩展**：task-aware / content-aware 动态调参
- [ ] **跨模型验证**：Qwen3.5-4B（必做）+ Qwen3.5-9B（可选）

### 已完成（从旧待办迁移）
- [x] Video-MME 完整评估 (300 题, baseline+sparse+naive)
- [x] Video-MME keep_ratio 消融 (Pareto sweep)
- [x] 去音频消融 (sparse vs sparse_no_audio)
- [x] Naive baselines 全量对比
- [x] Modality baselines (6 模式 × 300 题)
- [x] Sparse@64 闭环
- [x] Bootstrap CI + Non-inferiority
- [x] MVBench 全量 (3600 题)
- [x] Adaptive kr + M/L 边界实验
- [x] Pareto naive_iframe kr sweep
- [x] MVBench 少 GOP 退化修复
- [x] Adaptive v2 单路径重设计
- [x] Adaptive v3 stratified_top_k 实验

## 外部反馈
- **[3.11] 论文策略确认**：毕业论文用 naive_iframe（材料充足），会议论文需补充技术点 2/3 或 content-adaptive
- **[3.11] 技术点 2/3 可行性评估**：RingBuffer 3-5 天，显存优化 1-2 天，时间可控
- **[3.11] 与开题报告对齐**：RingBuffer = 推流中间件，逻辑自洽
- **[3.10] Adaptive v3 结果**：stratified_top_k 未超越 naive_iframe，AV-LRM 打分逻辑可能有问题
- **[3.9] 研究 Scope 确认**：10s-180s 短视频（Video-MME Short + ActivityNet-QA），对齐同方向论文
- **[3.2] M/L 放弃**：稀疏化在 M/L 基本失效，论文不再 claim 15min+ 有效性

## 待探讨问题
- [ ] **AV-LRM 打分逻辑根因分析**：I 帧码率是否能代表信息密度？音频能量是否有效？归一化是否正确？
- [ ] **方差门控阈值**：0.05 是否合理？是否应该动态调整？
- [ ] **Top-K 时间聚类问题**：是否需要加入时间约束（如最小间隔）？
- [ ] **Content-adaptive sparsification**：根据视频类型动态调整 kr 和 alpha
- [ ] **P/B 帧利用**：是否可以在特定任务（动作识别）中利用 P/B 帧？

## 结果文件位置
- `videomme_full/` — Video-MME 完整评估 + Modality baselines
- `naive_comparison/` — Naive baselines 对比 (kr=0.5)
- `naive_comparison_kr02/` — Naive baselines 对比 (kr=0.2)
- `sparse64/` — Sparse@64 闭环
- `pareto_naive_iframe/` — Pareto kr sweep
- `mvbench/` — MVBench 全量
- `videomme_adaptive_kr/` — Adaptive kr + M/L 边界
- `videomme/adaptive_v2/` — Adaptive v2 (top_k)
- `videomme/adaptive_v3/` — Adaptive v3 (stratified_top_k)
- `activitynet/adaptive_v2/` — ActivityNet adaptive v2
- `activitynet/adaptive_v3/` — ActivityNet adaptive v3

## 变更日志
- **[3.11]** 论文策略讨论：确认毕业论文用 naive_iframe，会议论文需补充技术点 2/3；评估技术点 2/3 工作量（RingBuffer 3-5天，显存优化 1-2天）；确认与开题报告对齐（RingBuffer = 推流中间件）
- **[3.10-3.11]** Adaptive v3 stratified_top_k 实验：Video-MME Short 71.3%，ActivityNet 40.6%，均未超越 naive_iframe；per-task-type 分析显示 0 wins / 8 ties / 4 losses；确认 AV-LRM 打分逻辑可能存在问题
- **[3.10]** Adaptive v2 单路径重设计：发现旧 run_adaptive() 恒走 naive_iframe，重写为单路径 AV-LRM 打分 + 方差门控；修复 _safe_fetch_video 分辨率 bug
- **[3.9]** 研究 Scope 最终确认（10s-180s 短视频）+ 稀疏化链路三层保护机制梳理 + MVBench 修复验证完成
- **[3.8]** PROGRESS.md 全面同步至真实状态
- **[2.24]** Adaptive kr 实现 + M/L 边界实验
- **[2.22]** MVBench 全量 + Pareto kr sweep
- **[2.21]** Bootstrap CI + Sparse@64
- **[2.20]** Modality baselines + 架构重构
- **[2.19]** Naive baselines 对比 → Two-Regime 理论形成
- **[2.18]** Video-MME 完整评估 + kr 消融
- **[2.16]** Video-MME Pipeline + GPT Code Review 修复
- **[2.15]** 串行 Pipeline 跑通
