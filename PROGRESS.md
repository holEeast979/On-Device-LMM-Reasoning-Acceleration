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
**技术点 1（GOP 级 token 稀疏化）已完成，准备技术点 2/3 开发**

核心实验已全部完成。经过 adaptive v2/v3/v4/v5 探索，确认 **naive_iframe（均匀选 I 帧）** 为最优方案。

**研究定位**：聚焦 **10s-180s 短视频**（Video-MME Short 11s-2min + ActivityNet-QA 30-180s），对齐端侧多模态推理加速的典型场景。

**论文策略**：
- **毕业论文**：用 naive_iframe 作为核心方法（training-free + codec-aware + 零损失加速），材料充足
  - **当前任务：V5 实验已完成，确认 naive_iframe 为最优方案。准备开发技术点 2（RingBuffer）+ 技术点 3（显存优化）
- **会议论文**：后续升级，CLIP question-aware rerank 为首选方向

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
| **adaptive_v4 (bug fix)** | **75.0%** | — | — | — |
| **adaptive_v5 (gate invert)** | **73.15%** | — | — | — |

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

### P0（论文写作准备，3.11-3.25）
- [x] **Adaptive v5 门控反转实验**：验证完成，确认 naive_iframe 为最优方案
- [x] **代码全面检查**：GPT Review 发现 7 个问题，修复 P0+P1-4+P2-7，adaptive v4 达 75.0%
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
- [x] GPT Code Review + P0/P1/P2 修复
- [x] Adaptive v4 全量验证（75.0%，追平 naive_iframe）
- [x] **技术路线决策：毕业论文锁定 naive_iframe，会议论文后续升级**

## 外部反馈
- **[3.11] 技术路线转折**：adaptive v4 修复后 75.0% 追平 naive_iframe，但 top_k 仍低 10pp；确认 training-free 打分天花板太低，毕业论文锁定 naive_iframe，会议论文后续升级硬核方案
- **[3.11] 论文策略确认**：毕业论文用 naive_iframe（材料充足），会议论文需补充技术点 2/3 或 content-adaptive
- **[3.11] 技术点 2/3 可行性评估**：RingBuffer 3-5 天，显存优化 1-2 天，时间可控
- **[3.11] 与开题报告对齐**：RingBuffer = 推流中间件，逻辑自洽
- **[3.10] Adaptive v3 结果**：stratified_top_k 未超越 naive_iframe，AV-LRM 打分逻辑可能有问题
- **[3.9] 研究 Scope 确认**：10s-180s 短视频（Video-MME Short + ActivityNet-QA），对齐同方向论文
- **[3.2] M/L 放弃**：稀疏化在 M/L 基本失效，论文不再 claim 15min+ 有效性


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

## 会议论文备选方向

### CLIP Question-Aware Rerank（最有价值）
- **核心思路**：GOP I 帧解码 → CLIP 计算每帧和问题的相似度 → 选最相关的帧
- **优势**：
  - 仍然 training-free（冻结 CLIP 不训练）
  - 解决 question-agnostic 的根本矛盾（看了问题再选帧）
  - 架构轻量（CLIP ViT-B/32 ~600MB，推理 ~5ms/帧）
  - Novelty 足够：codec-aware GOP + question-aware CLIP 双层 training-free
- **实现成本**：中等（3-5 天）
- **预期效果**：有望超越 naive_iframe 75.93%

### 其他备选
- GOP 感知 token 压缩（压缩已选帧的 token，而非选帧）
- Codec-aware temporal pooling（利用 GOP 结构做时间维度 token 合并）
- 音频特征升级：RMS → VAD + ASR token density

## 待探讨问题
- [ ] **方差门控反转验证**：V5 实验中，高方差 → uniform_boosted 效果如何
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
- `videomme/adaptive_v4/` — Adaptive v4 (P0+P1 bug fix, 75.0%)

- **[3.11]** Adaptive v5 门控反转实验失败：73.15%（vs v4 75.0%），uniform_boosted 57.1% vs stratified_top_k 66.7%；确认最优策略是全部用 uniform（naive_iframe），完成技术点 1 所有探索
## 变更日志
- **[3.11]** Adaptive v4 修复验证：75.0%（追平 naive_iframe 75.93%），top_k 66.7% vs uniform 77.0%；确认 training-free 打分天花板，毕业论文锁定 naive_iframe，会议论文后续升级
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

**v4（Bug 修复：方差阈值 + 保头尾 + 音频对齐）**：
- Video-MME Short: 75.0%（追平 naive_iframe）
- 策略分布：uniform 77.0% (87题), stratified_top_k 66.7% (21题)
- 结论：修复 bug 后追平 baseline，但 top_k 仍比 uniform 低 10pp

**v5（门控反转：高方差→uniform_boosted）**：
- Video-MME Short: 73.15%（比 v4 低 1.85pp）
- 策略分布：uniform 77.0% (87题), uniform_boosted 57.1% (21题)
- 结论：门控反转假设错误，uniform_boosted 比 stratified_top_k 更差 9.6pp

**最终结论**：
- training-free + question-agnostic 打分的天花板就是 naive_iframe
- 毕业论文锁定 naive_iframe，放弃 adaptive 所有变体
- 会议论文后续升级：CLIP question-aware rerank


**v4（Bug 修复：方差阈值 + 保头尾 + 音频对齐）**：
- Video-MME Short: 75.0%（追平 naive_iframe）
- 策略分布：uniform 77.0% (87题), stratified_top_k 66.7% (21题)
- 结论：修复 bug 后追平 baseline，但 top_k 仍比 uniform 低 10pp

**v5（门控反转：高方差→uniform_boosted）**：
- Video-MME Short: 73.15%（比 v4 低 1.85pp）
- 策略分布：uniform 77.0% (87题), uniform_boosted 57.1% (21题)
- 同一批 7 个高方差视频：uniform_boosted 57.1% vs stratified_top_k 66.7%（-9.6pp）
- 结论：门控反转假设错误，uniform_boosted 比 stratified_top_k 更差

**最终结论**：
- training-free + question-agnostic 打分的天花板就是 naive_iframe
- 毕业论文锁定 naive_iframe，放弃 adaptive 所有变体
- 会议论文后续升级：CLIP question-aware rerank


### 多轮缓存（Encoder Cache, 3.15）

**Smoke Test (10 视频)**:
- Uncached: 80.0% acc, 3395ms avg
- Cached: 80.0% acc, 2998ms avg  
- Speedup: 1.13x (13% improvement)
- Prediction match: PASS (0 mismatches)

**GOP + Cache 验证**:
- Without cache: 6240ms
- With cache Q1: 4905ms (1.27x)
- With cache Q2: 3503ms (1.78x)
- Cache hits: video=1, audio=1
- Visual tokens: 4322 (GOP 稀疏化生效)


## 变更日志（续）
- **[3.15]** 多轮缓存 Codex Review + 修复：修复缓存键设计、线程安全、AB/BA 设计、逐题对比；Smoke test 通过（10 视频，1.13x 加速）
- **[3.15]** GOP + Cache 联合验证：确认两个技术点可以叠加工作，第 2 轮加速 1.78x
- **[3.15]** 修复 pipeline.py bug：`_select_naive` 方法中 `min_gop_frames` 变量未定义，改为 `min_frames`
- **[3.15]** Git 分支整理：创建 feature/av-lrm-sparse 分支保存 AV-LRM 打分研究快照；主分支标记 sparse 为 deprecated（保留代码以复现实验，推荐用 naive_iframe）；提交 encoder_cache 相关代码到主分支
