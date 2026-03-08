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
**技术点 1 收尾 → 论文写作准备**

核心实验已全部完成（Video-MME 300题 + MVBench 3600题 + Pareto sweep + Bootstrap CI + Modality baselines + Adaptive kr 边界实验）。剩余 P0 收尾后进入论文写作，同时推进 content-adaptive 参数调整。

**研究定位**：聚焦 Short 视频（10-120s），M/L 不在 scope（max_frames=32 硬瓶颈，稀疏化无效）。

## 项目背景
研究 **Qwen2.5-Omni-7B** 多模态大模型的推理加速，构建**端侧多模态推理加速系统**。

**三大技术支柱**（独立可解耦）：
1. ✅ **GOP 级 token 稀疏化**（AV-LRM 打分 + 帧选择）← 实验完成 ~95%，收尾中
2. ⬜ **显存碎片优化**（ViT 后清理激活值 + 分块 Encoding）← 待开发
3. ⬜ **Ring Buffer 流水线**（CPU/GPU 异步预取）← 待开发

**硬件**：RTX 5090 32GB，OOM 边界 ~25-30k tokens（约 34s 视频全量帧）。
**论文定位**：边缘服务器/轻量部署（单 GPU 32GB），training-free codec-aware 加速系统。

## 模块状态

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| GOP 解析 | `fasteromni/modules/gop_parser.py` | ✅ | 16 视频验证，GOP 中位数 5 |
| 音频能量 | `fasteromni/modules/audio_energy.py` | ✅ | RMS 能量按 GOP 时间窗口切分 |
| AV-LRM 打分 | `fasteromni/modules/sparse.py` | ✅ | 归一化+Uniform 偏置已修 |
| I 帧解码 | `fasteromni/modules/frame_decoder.py` | ✅ | 全解码后过滤（已知限制） |
| Pipeline | `fasteromni/pipeline.py` | ✅ | baseline/sparse/sparse_no_audio/naive_{uniform,random,iframe} 六模式 |
| Video-MME 评估 | `fasteromni/eval_videomme.py` | ✅ | 300 题完整评估 + 增量 CSV + 断点恢复 |
| Bootstrap CI | `fasteromni/bootstrap_ci.py` | ✅ | 10,000 次 bootstrap + non-inferiority |

## 已完成实验汇总

### Video-MME (300 题)
| 方法 | Short | Medium | Long | Overall |
|------|-------|--------|------|---------|
| Baseline (32帧) | 75.93% | 56.67% | 49.41% | 62.00% |
| naive_iframe (kr=0.5) | **75.93%** | 60.00% | 47.06% | 62.33% |
| sparse (kr=0.5) | 69.44% | 61.11% | 47.06% | 60.33% |
| sparse (kr=0.2) | 70.37% | — | — | — |

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

- **kr=0.5 零损失**：naive_iframe 75.93% = Baseline，2.1x 加速，54% token 减少
- **Two-Regime 理论**：kr≈0.5 coverage-dominant（naive > sparse +6.49pp），kr≤0.2 relevance-dominant（sparse > naive +1.85pp），交叉点 kr≈0.3
- **Pareto 非单调**：kr=0.5 是 sweet spot，kr>0.5 I 帧聚类反而损害覆盖度
- **语言先验**：text_only=42%（远 > 随机 25%），视觉贡献 +20.2pp
- **音频 free lunch**：去音频仅多降 1.8pp
- **M/L 失效**：max_frames=32 锁死，Adaptive kr 数学上等价 baseline（已放弃，不在 scope）
- **MVBench 退化**：极短视频 GOP 少，kr=0.5 后帧数过少，时序信息丧失

## 关键设计决策
1. ⚠️ TTFT 只能用 `max_new_tokens=1` 测量
2. ⚠️ 音频必须通过 processor 传入 (`proc(audio=, use_audio_in_video=True)`)
3. ⚠️ Sparse 必须走 video tensor 路径（不是 image）
4. Video-MME baseline 用 max_frames=32（64 帧 OOM）
5. 评估主实验用 Video-MME 选择题（零歧义）
6. alpha 默认 0.3（区间稳定，不追求单点最优）

## 待办事项

### P0（技术点 1 收尾）
- [ ] **MVBench 少 GOP 退化修复**：增加低 `n_valid` 边界策略，避免 kr=0.5 过度稀疏到 ~3 帧（7/18 任务退化 >20pp）
- [ ] **补齐 latency/token 对比表**：整理已有实验数据中的 TTFT/VisTok，输出论文级对比表

### P1（论文准备 + 扩展）
- [ ] **Content-adaptive 参数调整**：根据视频内容动态调整 kr/alpha（与 Two-Regime 理论衔接）
- [ ] **Short 场景误差分析**：定位 sparse 70.4% vs naive 77.0% 的掉点题型
- [ ] **跨模型验证**：Qwen3.5-4B（必做）+ Qwen3.5-9B（可选），验证 training-free 可迁移性
- [ ] **论文写作**：整理实验数据，开始撰写

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

## 外部反馈
- **[3.2] M/L 放弃**：稀疏化在 M/L 基本失效，论文不再 claim 15min+ 有效性，M/L 作为 scope/limitation
- **[3.2] 参数策略**：alpha 采用稳健区间策略，默认 0.3，强调区间稳定性
- **[3.2] 方法边界**：I-only 为正文主线，PTBG 作为 Future Work
- **[3.2] 执行优先级**：P0 = MVBench 少 GOP 修复 + latency/token 对比
- **[2.16] GPT Code Review**：8 个问题修了 6 个，音频链路+TTFT 口径修复后加速比 2.39x→3.77x

## 待探讨问题
- [ ] **content-adaptive sparsification**：根据视频类型动态调整 kr 和 alpha，作为技术点 1 的自然延伸
- [ ] **音频兜底假说**：去音频仅降 1.8pp，说明视觉稀疏化本身鲁棒，但极端 kr 下是否不同？
- [ ] **GOP 粒度上限**：H.264 短视频 GOP 中位数仅 5，帧级选择能否让 alpha 发挥价值？
- [ ] **Prefill 53% "Other" 开销**：ViT 17% + Audio 13% + LLM 17% + Other 53%，能否进一步分解？

## 结果文件位置
- `videomme_full/` — Video-MME 完整评估 + Modality baselines
- `naive_comparison/` — Naive baselines 对比 (kr=0.5)
- `naive_comparison_kr02/` — Naive baselines 对比 (kr=0.2)
- `sparse64/` — Sparse@64 闭环
- `pareto_naive_iframe/` — Pareto kr sweep
- `mvbench/` — MVBench 全量
- `videomme_adaptive_kr/` — Adaptive kr + M/L 边界

## 变更日志
- **[3.8]** PROGRESS.md 全面同步至真实状态（之前滞后在 Phase 1，实际已完成 Phase 1-3 全部实验）
- **[2.24]** Adaptive kr 实现 + M/L 边界实验，确认 M/L 不在 scope
- **[2.22]** MVBench 全量 + Pareto kr sweep + 1-GOP 修复
- **[2.21]** Bootstrap CI + Sparse@64 + Non-inferiority
- **[2.20]** Modality baselines（经历代码污染事件，已恢复）+ 架构重构（帧选择与推理解耦）
- **[2.19]** Naive baselines 对比 → Two-Regime 理论形成
- **[2.18]** Video-MME 完整评估 + kr 消融 + 去音频消融
- **[2.16]** Video-MME Pipeline + GPT Code Review 修复
- **[2.15]** 串行 Pipeline 跑通，首次 TTFT 对比
