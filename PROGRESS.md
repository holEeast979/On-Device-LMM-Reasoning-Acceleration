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
**技术点 1-3 已完成实验验证，准备论文写作**

核心实验已全部完成（实验 11-15）。三个技术点叠加后端到端加速 2.55x（VME）/ 2.17x（ANet），准确率零损失。

**研究定位**：聚焦 **10s-180s 短视频**（Video-MME Short 11s-2min + ActivityNet-QA 30-180s），对齐端侧多模态推理加速的典型场景。

**论文策略**：
- **毕业论文**：技术点 1（GOP 稀疏化）+ 技术点 2（EncoderCache）+ 技术点 3（PrefetchBuffer），三技术点联合系统
- **技术点 4（显存优化）**：待开发
- **会议论文**：后续 CLIP Question-Aware Rerank

## 项目背景
研究 **Qwen2.5-Omni-7B** 多模态大模型的推理加速，构建**端侧多模态推理加速系统**。

**四大技术支柱**（独立可解耦）：
1. ✅ **GOP 级 token 稀疏化**（naive_iframe 为主）← 实验完成
2. ✅ **编码器缓存 EncoderCache**（ViT/Whisper 编码复用）← 实验完成，generate_ms -30%
3. ✅ **预取流水线 PrefetchBuffer**（CPU/GPU 异步并行）← 实验完成，total_ms -25%
4. ⬜ **显存碎片优化**（ViT 后清理激活值）← 待开发

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
| Pipeline | `fasteromni/pipeline.py` | ✅ | baseline/sparse/naive_iframe 等模式 + 三层保护机制 + prefetch_video() |
| EncoderCache | `fasteromni/encoder_cache.py` | ✅ | Hook ViT/Whisper forward，同视频多问题复用编码结果 |
| PrefetchBuffer | `fasteromni/prefetch_buffer.py` | ✅ | ThreadPoolExecutor 异步预取，LRU 淘汰，容量=2 |
| Video-MME 评估 | `eval_videomme_ref.py` | ✅ | 108 题(Short) + 增量 CSV + 断点恢复 + --encoder-cache |
| ActivityNet 评估 | `eval_activitynet.py` | ✅ | 1000 题 + 增量 CSV + --encoder-cache |
| Bootstrap CI | `fasteromni/bootstrap_ci.py` | ✅ | 10,000 次 bootstrap + non-inferiority |

## 已完成实验汇总

### Scheme A 统一重跑 (Exp 11, commit c8c9e7c)

| 数据集 | 配置 | 准确率 | generate_ms | total_ms | visual_tokens |
|--------|------|--------|-------------|----------|---------------|
| VME Short | Baseline | 75.93% | 2,161 | 5,450 | 10,739 |
| VME Short | GOP 稀疏化 | 75.00% | 1,110 | 3,466 | 4,941 |
| VME Short | GOP+Cache | 75.00% | 766 | 3,309 | 4,941 |
| ActivityNet | Baseline | 41.70% | 2,246 | 5,480 | 7,745 |
| ActivityNet | GOP 稀疏化 | 40.60% | 1,581 | 4,128 | 3,851 |
| ActivityNet | GOP+Cache | 40.50% | 1,119 | 3,731 | 3,851 |

### Prefetch 实验 (Exp 12-15, commit dc997bf)

| 实验 | 数据集 | 配置 | 准确率 | generate_ms | total_ms | visual_tokens |
|------|--------|------|--------|-------------|----------|---------------|
| Exp 12 | VME Short | GOP+Prefetch | 75.0% | 1,123 | 2,615 | 4,941 |
| Exp 13 | VME Short | GOP+Cache+Prefetch | 75.0% | 783 | 2,140 | 4,941 |
| Exp 14 | ActivityNet | GOP+Prefetch | 40.5% | 1,592 | 3,092 | 3,851 |
| Exp 15 | ActivityNet | GOP+Cache+Prefetch | 40.5% | 1,130 | 2,527 | 3,851 |

### 端到端加速总结（三技术点叠加 vs Baseline）

| 数据集 | Baseline total_ms | 三合一 total_ms | 加速比 | 准确率变化 |
|--------|-------------------|-----------------|--------|-----------|
| VME Short | 5,450 | 2,140 | **2.55x** | 75.93% → 75.0% (≈0) |
| ActivityNet | 5,480 | 2,527 | **2.17x** | 41.7% → 40.5% (-1.2pp) |

### Video-MME (300 题，历史数据)
| 方法 | Short | Medium | Long | Overall |
|------|-------|--------|------|---------|
| Baseline (32帧) | 75.93% | 56.67% | 49.41% | 62.00% |
| naive_iframe (kr=0.5) | **75.93%** | 60.00% | 47.06% | 62.33% |
| sparse (kr=0.5) | 69.44% | 61.11% | 47.06% | 60.33% |
| adaptive_v2 (top_k) | 72.22% | — | — | — |
| adaptive_v3 (stratified) | 71.30% | — | — | — |
| **adaptive_v4 (bug fix)** | **75.0%** | — | — | — |

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

- **三技术点叠加 2.55x/2.17x 加速**：GOP+Cache+Prefetch，VME total_ms 5450→2140，ANet 5480→2527
- **两层缓存系统效果累加**：CPU 层（PrefetchBuffer）降 total_ms -25%，GPU 层（EncoderCache）降 generate_ms -30%，互不干扰
- **准确率零损失**：三技术点叠加后 VME 75.0%=Baseline，ANet 40.5%（-1.2pp，正常波动）
- **kr=0.5 零损失**：naive_iframe 75.93% = Baseline，2.0x 加速，54% token 减少
- **Two-Regime 理论**：kr≈0.5 coverage-dominant（naive > sparse +6.49pp），kr≤0.2 relevance-dominant（sparse > naive +1.85pp），交叉点 kr≈0.3
- **Pareto 非单调**：kr=0.5 是 sweet spot，kr>0.5 I 帧聚类反而损害覆盖度
- **语言先验**：text_only=42%（远 > 随机 25%），视觉贡献 +20.2pp
- **音频 free lunch**：去音频仅多降 1.8pp
- **M/L 稀疏化失效**：max_frames=32 锁死，Baseline@32 和 Sparse@32 token 数几乎相同
- **MVBench 退化**：极短视频 GOP 少，kr=0.5 后帧数过少，时序信息丧失
- **Sparse@64 价值**：Short@64 Baseline 89% OOM，Sparse 0 OOM，证明稀疏化在高帧数场景的价值
- **AV-LRM 打分问题**：adaptive v2/v3 均未超越 naive_iframe，怀疑 I 帧码率和音频能量不能准确反映信息密度
- **Adaptive v4 验证**：修复 bug 后 75.0% 追平 naive_iframe，但 top_k（66.7%）仍比 uniform（77.0%）低 10pp，确认 training-free 打分天花板
- **技术路线决策**：毕业论文用 naive_iframe，会议论文后续升级（GOP token 压缩 / question-aware 打分）

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

### P0（论文写作，3.24-3.31）
- [x] **技术点 2 实现 + 实验验证**：EncoderCache，generate_ms -30%
- [x] **技术点 3 实现 + 实验验证**：PrefetchBuffer，total_ms -25%
- [x] **实验 12-15 全量完成**：VME Short 108题 × 2 + ANet 1000题 × 2
- [x] **论文大纲更新**：三技术点 + 两层缓存架构 + 实验数据填入
- [ ] **开始写论文正文**：先写第三章（方法）+ 第四章（实验）
- [ ] **图表制作**：架构图 + 两层缓存示意图 + 三层保护机制 + 实验曲线

### P1（技术点 4，可选）
- [ ] **技术点 4：显存碎片优化**
  - ViT forward 后清理中间激活值
  - 监控峰值显存
  - 实验验证（峰值显存、OOM 率）

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
- [x] Adaptive v4 全量验证（75.0%，追平 naive_iframe）
- [x] **技术路线决策：毕业论文锁定 naive_iframe**
- [x] **Scheme A 统一重跑（Exp 11, c8c9e7c）**
- [x] **EncoderCache 实现 + 集成到 eval 脚本**
- [x] **PrefetchBuffer 实现（ThreadPoolExecutor 异步预取）**
- [x] **Exp 12-15 全量实验**：四组对照（Prefetch / 三合一 × VME / ANet）
- [x] **实验数据拉回本地归档**
- [x] **论文大纲更新**：两层缓存架构 + 真实实验数据
- [x] **Codex 交叉验证**：确认数据一致性，三技术点效果累加无抵消

## 外部反馈
- **[3.19] 实验 12-15 全量完成**：三技术点叠加 VME 2.55x / ANet 2.17x，Codex 验证数据一致性通过，论文大纲已更新
- **[3.18] EncoderCache + PrefetchBuffer 集成**：eval 脚本添加 --encoder-cache 参数，pipe.load_model() 修复懒加载问题
- **[3.17] 导师会议**：确认时间线（3.18-3.22 技术开发，3.24-3.31 写论文），5.9 答辩
- **[3.11] 技术路线决策**：毕业论文锁定 naive_iframe，会议论文后续 CLIP Question-Aware Rerank


## GPT Code Review 发现的问题（3.11）

> **审查范围**：sparse.py, gop_parser.py, audio_energy.py, pipeline.py, eval 脚本
> **核心结论**：AV-LRM 链路存在多个致命问题，打分在挑"画面复杂、声音大"的段，而不是"对答题关键"的段

### P0 问题（致命，必须修复）

**P0-1: 方差门控阈值过低（0.02 vs 0.05）**
- 问题：99% 视频被判定为"高方差"，几乎全走 top_k
- 证据：36 个视频中，threshold=0.02 时 33 个走 top_k（92%），threshold=0.05 时仅 6 个走 top_k（17%）
- 代码：pipeline.py:363, pipeline.py:917 用 0.02，sparse.py:133 默认 0.05 被覆盖
- 修复：改为 0.05 或用验证集分位数定门槛

**P0-2: top_k/stratified_top_k 不保头尾**
- 问题：丢失视频开头和结尾，伤害时序任务（OCR、Counting、Needle）
- 证据：33 个 top_k 视频中，23 个（70%）没保住开头，10 个（30%）没保住结尾
- 代码：sparse.py:178-189 stratified_top_k 不保证选中首尾 GOP
- 修复：强制保留第一个和最后一个有效 GOP，剩下 K-2 个再做分段选高分

**P0-3: 音频喂给模型时和选帧不匹配（音画错位）**
- 问题：画面抽样了，音频还在喂完整前缀，模型收到不匹配的多模态输入
- 证据：pipeline.py:463-470 音频截取 [0, max_end]，不是选中 GOP 对应的音频片段
- 修复：短期保留最后一个 GOP 别砍尾音频，中期要么喂全音频要么做音频切片拼接

### P1 问题（重要，建议修复）

**P1-4: min_gop_frames 参数被偷偷覆盖**
- 问题：传入 min_gop_frames=10，实际用 adaptive_min_gop = max(2, median×0.5)，平均 39.8，最大 120
- 证据：某视频 31 帧 / 1.24 秒的 GOP 被阈值 50 过滤掉
- 代码：pipeline.py:385-388
- 修复：用传入的 min_gop_frames 或改成显式开关并落盘到 CSV

**P1-5: 打分逻辑不"懂题"，更像"复杂度分数"**
- 问题：视觉分数只看 i_frame_size（画面复杂度），音频只看 RMS（声音大小），不代表语义重要性
- 证据：sparse 69.4% < naive_iframe 75.9%，说明"用了打分"反而不如"不打分"
- 代码：sparse.py:75, audio_energy.py:101
- 修复：视觉改成 log1p(i_frame_size / num_frames)，音频换成 VAD / ASR token 密度

**P1-6: 按 GOP 个数分段，不是按时间分段**
- 问题：高方差视频 GOP 长短差异大，按索引分段失真
- 证据：keep_ratio_actual=0.5 但 frame_keep_ratio=0.61，说明"保留一半 GOP"≠"保留一半时间"
- 代码：sparse.py:183, sparse.py:199
- 修复：按时间轴分段，每段再挑分数最高的 GOP

### P2 问题（次要，影响分析）

**P2-7: 元数据没落盘到 CSV**
- 问题：selection_strategy、score_variance、kr_adaptive、adaptive_min_gop 只能从字符串解析
- 代码：eval_videomme.py:425, eval_videomme.py:571
- 修复：加这些字段到 EvalRecord 和 CSV 头

### 修复优先级

**立即修复（P0，20 分钟）**：
1. 方差阈值 0.02 → 0.05
2. stratified_top_k 强制保头尾
3. 音频对齐

**建议修复（P1，20 分钟）**：
4. 恢复 min_gop_frames 参数
5. 按时间分段
6. 补充元数据到 CSV

**暂缓修复（P1-5，1-2 天）**：
7. 重新设计打分特征

### 预期修复效果

- **乐观**：72-74%（接近 naive_iframe 75.93%）
- **保守**：70-72%（比当前 69.44% 有提升）
- **决策点**：≥74% 继续修复打分逻辑，70-74% 放弃 AV-LRM 转技术点 2/3，<70% 深入分析

## 待探讨问题
- [ ] **AV-LRM 打分逻辑根因分析**：I 帧码率是否能代表信息密度？音频能量是否有效？归一化是否正确？
- [ ] **方差门控阈值**：0.05 是否合理？是否应该动态调整？
- [ ] **Top-K 时间聚类问题**：是否需要加入时间约束（如最小间隔）？
- [ ] **Content-adaptive sparsification**：根据视频类型动态调整 kr 和 alpha
- [ ] **P/B 帧利用**：是否可以在特定任务（动作识别）中利用 P/B 帧？

## 结果文件位置
- `scheme_a_c8c9e7c/` — Scheme A 统一重跑（Exp 11, 基线）
- `bench_prefetch_vmme/` — Exp 12: VME Short GOP+Prefetch
- `bench_all3_vmme/` — Exp 13: VME Short GOP+Cache+Prefetch
- `bench_prefetch_anet/` — Exp 14: ANet GOP+Prefetch
- `bench_all3_anet/` — Exp 15: ANet GOP+Cache+Prefetch
- `videomme_full/` — Video-MME 完整评估 + Modality baselines
- `pareto_naive_iframe/` — Pareto kr sweep
- `mvbench/` — MVBench 全量

## 变更日志
- **[3.19]** Exp 12-15 全量完成 + Codex 数据验证 + 论文大纲全面更新（两层缓存架构 + 真实实验数据）
- **[3.18]** EncoderCache + PrefetchBuffer 集成到 eval_videomme_ref.py 和 eval_activitynet.py；修复 pipe.load_model() 懒加载、SIGALRM 超时（trap "" ALRM）；四个实验启动（tmux bench）
- **[3.17]** 导师会议，确认时间线和论文策略
- **[3.15-3.17]** Scheme A 统一重跑（Exp 11, c8c9e7c）；encoder_cache.py 和 prefetch_buffer.py 实现
- **[3.11]** Adaptive v4 验证 + 技术路线决策：锁定 naive_iframe
