# FasterOmni 研究故事线

> **本文件用途**：记录研究过程中的关键问题、解决方案、证据与决策转折点  
> **目标读者**：写论文时快速回顾"为什么这么做"、审稿回复时引用决策依据  
> **维护原则**：只记录主线决策，不堆原始数据（数据在 PROGRESS_ARCHIVE.md）

---

## 整体架构与进度

**论文定位**：面向端侧设备的 training-free 多模态推理加速系统

**三大技术支柱**（独立可解耦）：
1. ✅ **GOP 级 token 稀疏化**（AV-LRM 打分 + 帧选择）← **当前已完成**
2. ⬜ **显存碎片优化**（ViT 后清理激活值 + 分块 Encoding）← 待开发
3. ⬜ **Ring Buffer 流水线**（CPU/GPU 异步预取，批量场景隐藏预处理延迟）← 待开发

**当前进度**：技术点 1 已完成初步验证与边界探索，正在收尾实验（MVBench 少 GOP 修复 + latency/token 对比）

---

## 技术点 1：GOP 级 token 稀疏化（Phase 1-4）

### 故事线概览

**起点假设**：视觉方差 + 音频能量打分（AV-LRM）能在 GOP 级帧选择中优于简单的均匀采样  
**核心转折**：kr=0.5 时 AV-LRM 反而不如 naive I 帧均匀选取，引发 Two-Regime 理论  
**阶段性结论**：
- 核心方法：GOP-aware I 帧选取（naive_iframe）为主力
- 理论贡献：Two-Regime 理论（覆盖度 vs 精准度的交叉点在 kr≈0.3）
- 适用范围：Short 视频（10-120s），kr=0.5 零损失 2.1x 加速
- 边界限制：M/L 视频因 max_frames=32 限制稀疏化无效，需要技术点 2（显存优化）解锁

---

## Phase 1: 初始验证与意外发现（2.15 - 2.18）

### 节点 1.1 — 串行 Pipeline 跑通
**时间**：2.15  
**问题**：需要验证 GOP 级稀疏化的基本可行性  
**动作**：实现 GOP 解析 → AV-LRM 打分 → I 帧解码 → 模型推理的串行 pipeline  
**关键证据**：
- 4 视频 TTFT 对比：平均 3.77x 加速
- Visual tokens 减少 92%，音频 tokens 一致（408=408）
**决策**：基础链路可行，推进到完整 benchmark 评估

### 节点 1.2 — Video-MME 完整评估
**时间**：2.16 - 2.18  
**问题**：ActivityNet-QA 采样 bug（50 题仅 16 独立视频），需要更可靠的 benchmark  
**动作**：切换到 Video-MME（300 题选择题，零歧义），实现增量 CSV + 超时保护 + 断点恢复  
**关键证据**：
- Baseline: 62.0% (300/300, 0 errors)
- Sparse (kr=0.5): 59.0% (-3.0pp)
- Short 视频：2x 加速，-5.6pp
- M/L 视频：sparse 无效（max_frames=32 截断导致 token 数与 baseline 相同）
**决策**：Short 视频是最佳优化目标，M/L 需要后续解决

### 节点 1.3 — kr 消融与音频兜底验证
**时间**：2.18  
**问题**：准确率对 kr 不敏感（68.5%~70.4%），怀疑是音频在"兜底"  
**动作**：
- kr 消融：0.2/0.3/0.5/0.7/0.9 五组对比
- 去音频消融：sparse vs sparse_no_audio
**关键证据**：
- kr=0.2 即可达到 3.7x 加速，仅 -5.5pp
- 去音频后仅多降 1.8pp（67.6% vs 69.4%）
**决策**：音频兜底效应很小，视觉稀疏化本身鲁棒；音频作为 free lunch 保留

---

## Phase 2: 负结果深挖与理论构建（2.19 - 2.22）

### 节点 2.1 — GPT 5.2 Review 触发补充实验
**时间**：2.19  
**问题**：GPT Review 指出 3 个 Critical 问题：
1. 新颖性风险：kr 不敏感可能意味着"选哪几帧不重要"，AV-LRM 贡献点站不住
2. 公平性混淆：sparse 截断音频，speedup 混入了音频 token 减少效应
3. 适用范围：M/L 无效是结构性问题
**动作**：补充 5 个对照实验（P0 优先级）
**决策**：Phase 2 重点转向"证明 AV-LRM 的价值边界"

### 节点 2.2 — Naive Baselines 对比：核心转折
**时间**：2.19  
**问题**：AV-LRM 是否真的优于简单方法？  
**动作**：实现 naive_uniform / naive_random / naive_iframe 三种策略，kr=0.5 同帧数对比  
**关键证据**（Short 108 题）：
- **naive_iframe: 75.93% = Baseline 75.93%**（零损失！）
- naive_uniform: 74.07% (-1.9pp)
- naive_random: 73.15% (-2.8pp)
- **sparse (AV-LRM): 69.44% (-6.5pp，最差）**
- 所有方法 VisTok 完全一致（4939），确认帧数匹配
**决策**：**重大发现**——kr=0.5 时 AV-LRM 不如 naive，覆盖度 > 精准打分

### 节点 2.3 — 极低预算验证：排名反转
**时间**：2.19  
**问题**：AV-LRM 在所有场景都不如 naive 吗？  
**动作**：补充 kr=0.2 极端稀疏验证（4 种策略对比）  
**关键证据**（Short 108 题）：
- **sparse (AV-LRM): 70.37%（最优）**
- naive_iframe: 68.52% (-1.85pp)
- naive_uniform: 67.59% (-2.78pp)
- naive_random: 63.89% (-6.48pp，暴跌）
**决策**：**排名反转**——极低预算时 AV-LRM 占优，naive_random 在极端稀疏下崩溃

### 节点 2.4 — Two-Regime 理论形成
**时间**：2.19 - 2.22  
**问题**：如何解释 kr=0.5 和 kr=0.2 的排名反转？  
**理论构建**（GPT 5.2 提出）：
- **Coverage-dominant（kr≈0.5）**：中等预算下，时间覆盖度比精准打分重要 → naive_iframe 天然接近场景切换，所以强
- **Relevance-dominant（kr≤0.2）**：极低预算下，必须精准投放 → AV-LRM 占优
- **交叉点在 kr≈0.3**：两种方法持平（69.44%）
**关键证据**：
- kr=0.5: naive 75.93% vs sparse 69.44%（+6.49pp）
- kr=0.2: sparse 70.37% vs naive 68.52%（+1.85pp）
**决策**：Two-Regime 是独立学术贡献，为部署策略提供量化依据

### 节点 2.5 — Modality Baselines：语言先验验证
**时间**：2.20 - 2.21  
**问题**：kr 不敏感是否因为 benchmark bias（语言先验强）？  
**动作**：实现 text_only / audio_only / video_only 三种 modality baseline（6 模式 × 300 题）  
**关键证据**：
- text_only: 42.0%（远 > 随机 25%）
- audio_only: 51.3% (+9.3pp)
- video_only: 62.2% (+20.2pp)
- baseline: 61.9%（video_only ≥ baseline，音频可能干扰 Medium）
- Long 视频：baseline 仅比 text_only 高 4.3pp（91% 靠语言先验）
**决策**：语言先验显著但不能完全解释 kr 不敏感；Short 视频视觉贡献 +32.4pp，是最佳优化目标

### 节点 2.6 — Sparse@64 闭环：扩展能力验证
**时间**：2.21  
**问题**：稀疏化能否扩展帧预算边界？  
**动作**：Baseline@64 vs Sparse@64 对比（Short 108 题）  
**关键证据**：
- Baseline@64: 12 valid, **96 OOM (89%)**，准确率 83.3%*（存活偏差）
- Sparse@64: 108 valid, **0 OOM**，准确率 70.37%
**决策**：直接证明稀疏化扩展帧预算边界——相同硬件下可 max_frames 从 32→64 而不 OOM

### 节点 2.7 — Bootstrap CI 与 Non-inferiority
**时间**：2.21  
**问题**：统计严谨性验证（GPT Review Major #1）  
**动作**：实现 bootstrap_ci.py（10,000 次 bootstrap，配对检验）  
**关键证据**：
- sparse vs baseline: 95% CI [-5.7, +1.4]（跨零，无显著差异）
- naive_iframe vs baseline: 95% CI [-2.8, +4.3]（跨零，无显著差异）
- Short: naive_iframe vs baseline CI [-4.6, +5.6]（完全无差异）
**决策**：论文可用结论——"naive_iframe 砍掉 54% visual tokens，准确率统计无差异，2x 加速零损失"

---

## Phase 3: 扩展验证与边界探索（2.21 - 2.24）

### 节点 3.1 — MVBench 全量：短视频不兼容
**时间**：2.21 - 2.22  
**问题**：方法在其他 benchmark 上表现如何？  
**动作**：接入 MVBench（3600 题 × 20 任务类别），修复 1-GOP 视频崩溃 bug  
**关键证据**：
- Baseline: 66.94% (3318/3600, 282 OOM)
- naive_iframe(0.5): 53.59% (-13.35pp, 3579/3600, 21 errors)
- 7/18 任务严重退化（Δ > -20pp）：counterfactual_inference -36.0pp, object_existence -33.0pp
- 根因：MVBench 视频极短（kr=0.5 后仅 ~3 帧 vs baseline ~30 帧），时序信息丧失
**决策**：MVBench 短视频不兼容，但支持 Adaptive kr 的必要性——不同视频需要不同帧预算

### 节点 3.2 — Pareto 曲线：非单调现象
**时间**：2.22  
**问题**：为什么 kr=0.7/0.9 准确率反而下降？  
**动作**：naive_iframe kr sweep（0.2/0.3/0.5/0.7/0.9）  
**关键证据**：
- kr=0.5: 75.93%（峰值）
- kr=0.7: 71.30% (-4.63pp)
- kr=0.9: 70.37% (-5.56pp)
- 根因：max_frames=32 截断 + I 帧时间聚类。kr=0.7 时 17% 视频触及上限，32 个 I 帧的时间覆盖度不如 32 个均匀帧（同视频同帧数：I 帧 77.8% vs 均匀 83.3%，差 5.5pp）
**决策**：存在最优稀疏度，超过后 I 帧聚类反而损害覆盖度；kr=0.5 是 sweet spot

### 节点 3.3 — Adaptive kr 实现与 M/L 失效
**时间**：2.24  
**问题**：如何解决 M/L 视频稀疏化无效问题？  
**动作**：实现 Adaptive kr（`kr_adaptive = min(kr_base, max_frames/n_valid)`），全集实验（292 题 × 2 模式）  
**关键证据**：
- naive_iframe: Short 77.0% | Medium 60.0% | Long 47.1%
- sparse: Short 70.4% | Medium 61.1% | Long 47.1%
- **M/L 视频 vistok ≈ 10737（与 baseline 完全相同）**
- 根因诊断：M/L 视频 n_valid=100~1000，`kr_adaptive = min(0.5, 32/n_valid)` → K=32，选出 32 个 I 帧 = baseline 的 32 帧
**决策**：Adaptive kr 的"防截断"设计在 max_frames=32 下无法让 M/L 视频受益；需要提高 max_frames 上限（显存优化）

### 节点 3.4 — 技术点 1 定位调整
**时间**：2.22 - 2.24  
**问题**：稀疏化模块的贡献如何定位？  
**讨论结论**（GPT 5.2 + 用户）：
- **从**："AV-LRM 准确率领先"
- **到**："资源约束下可运行性 + Pareto 前沿 + 鲁棒性贡献"
- **核心方法**：GOP-aware I 帧选取（naive_iframe）为主力，AV-LRM 为低预算场景补充
- **理论贡献**：Two-Regime 理论（覆盖度 vs 精准度的交叉点在 kr≈0.3），为动态 kr 管理提供理论基础
**关键支撑**：
- kr=0.5 零损失（75.93% = Baseline），2.1x 加速
- Two-Regime 理论量化验证
- Sparse@64 零 OOM vs Baseline@64 89% OOM
**决策**：Short 视频为主战场，M/L 转为 scope/limitation 分析；动态 kr 管理作为后续优化方向

---

## Phase 4: 技术点 1 收尾（2.24 - 现在）

### 当前卡点
**问题**：Layer 2 Adaptive kr 在 M/L 视频上"数学上不截断，但效果上等价 baseline@32"  
**待讨论方向**：
1. 提高 max_frames（64/128）+ 显存优化 → 稀疏化在更大帧预算下发挥作用
2. 降低 K + 提升选帧质量 → 用更少但更精准的 I 帧
3. 混合策略 → Short 用 Layer 1，M/L 用不同帧预算策略

### P0 待办（先做）
1. **MVBench 少 GOP 退化修复** — 增加低 `n_valid` 边界策略（避免 kr=0.5 过度稀疏到 ~3 帧）
2. **补齐 sparse vs baseline 的 latency/token 对比** — 作为论文"加速贡献"主证据

### P1 待办（随后）
3. **Video-MME alpha 稳健性复核** — 强调区间稳定，不按单一数据集单点最优定参（默认建议 `alpha=0.3`）
4. **Short 场景误差分析** — 定位 70.4% vs 77.0% 的主要掉点题型
5. **跨模型验证（Qwen3.5）** — 在同 pipeline 上补 `Qwen3.5-4B`（必做）+ `Qwen3.5-9B`（可选），验证 training-free 方法的可迁移性

---

## 技术点 2：显存碎片优化（Phase 5，待开发）

**目标**：提高 OOM 边界，解锁 M/L 视频的稀疏化能力

**核心问题**：当前 max_frames=32 限制导致 M/L 视频稀疏化无效（Adaptive kr 在 M/L 上 K=32，等价 baseline）

**技术方案**（三层递进）：
1. **ViT 后清理激活值** — hook 清理中间激活，降低峰值显存
2. **输入规整化** — 统一输入尺寸，减少碎片
3. **分块编码** — GOP 边界分块编码，进一步降峰值

**预期效果**：max_frames 从 32 → 64+，M/L 视频 Adaptive kr 不再被截断，稀疏化策略发挥作用

**与技术点 1 的联动**：
- 显存优化提高 max_frames 上限
- Adaptive kr 自动适配新上限
- M/L 视频保留更多高分帧，覆盖度与精准度兼顾

**待探索问题**：
- [ ] 显存优化的实际收益（能提升多少 max_frames？）
- [ ] 与通用 VLM 优化的差异化（我们的 GOP-aware 分块 vs 通用方法）
- [ ] 对 Short 视频的影响（是否有额外开销？）

---

## 技术点 3：Ring Buffer 流水线（Phase 6，待开发）

**目标**：端到端延迟优化，批量场景隐藏预处理延迟

**核心问题**：当前串行处理（GOP 解析 → I 帧解码 → ViT 编码 → LLM），CPU/GPU 串行等待

**技术方案**：
- **帧级 CPU/GPU 解耦** — CPU 解码 batch N+1 的同时 GPU 编码 batch N
- **Ring Buffer 异步预取** — 预取下一批帧，降低 GPU 空闲时间

**预期效果**：批量推理场景（如视频问答服务）吞吐量提升

**与技术点 1/2 的联动**：
- 技术点 1 减少帧数 → 减少 CPU 解码压力
- 技术点 2 降低峰值显存 → 允许更大 batch size
- 技术点 3 流水线并行 → 隐藏预处理延迟

**待探索问题**：
- [ ] 单视频推理场景是否受益？（可能无收益，因为没有"下一个视频"可预取）
- [ ] 实现复杂度 vs 收益权衡
- [ ] 作为论文主贡献 vs Future Work 的定位

---

## 整体系统故事线（待技术点 2/3 完成后整合）

**论文核心叙事**（四段式）：

1. **问题**：多模态大模型处理视频时，visual tokens 占计算量大头。现有加速方法要么需要额外训练，要么需要额外前向传播计算 attention score。端侧设备算力有限，这些额外开销本身就是负担。

2. **洞察**：视频编码格式（H.264/H.265）的 GOP 结构天然标记了帧间冗余——I 帧是信息密度最高的关键帧，这个信息解码时就能拿到，零额外计算。

3. **方法**：围绕 GOP 结构设计一套完整的加速流水线：
   - **GOP 级帧选择（稀疏化）** — 利用 I 帧做 token 压缩，kr=0.5 零损失
   - **AV-LRM 智能选帧** — 低预算时补充精准度，和 naive 形成互补（Two-Regime）
   - **GOP 边界分块编码** — 显存优化，解锁中长视频
   - **CPU/GPU 异步流水线** — 端到端延迟优化

4. **贡献**：
   - 首个 training-free 的 codec-aware 视频 LMM 加速系统
   - Two-Regime 理论（coverage-dominant vs relevance-dominant），为动态 kr 管理提供理论指导
   - 三大技术支柱的系统性整合，端到端加速
   - 双 benchmark（Video-MME + MVBench）验证，Pareto 曲线展示 accuracy-efficiency tradeoff

**核心卖点**：codec-aware、training-free、系统性整合、端侧多模态

**与 CoPE-VideoLM 的区分**：CoPE 训练 encoder 走学术路线（-93% tokens 但需要预训练+微调），我们不改模型走部署路线（即插即用，适合端侧）

---

## 当前状态总结（技术点 1 完成度：~90%）

### 研究方法论
1. **负结果也是贡献**：AV-LRM 不如 naive 的发现引出 Two-Regime 理论，比单纯"方法 A 优于方法 B"更有学术价值
2. **补充对照实验的重要性**：GPT Review 触发的 5 个补充实验（naive baselines / modality baselines / Sparse@64 / Bootstrap CI / kr=0.2）彻底改变了论文定位
3. **统计严谨性**：Bootstrap CI 和 non-inferiority framing 让"零损失"的说法站得住脚
4. **边界探索**：MVBench 和 Adaptive kr 实验暴露了方法的适用边界，转为 limitation 分析反而增强可信度

### 工程教训
1. **[2.20] 代码污染事件**：不要批量修改 generate 调用参数，逐条改逐条验证
2. **增量 CSV + 断点恢复**：300 题实验崩溃不丢数据，paper-level 工程质量
3. **超时保护**：C 扩展永久阻塞 → SIGALRM→SIG_DFL 内核级 watchdog
4. **架构重构**：帧选择与推理引擎解耦，generate 参数只有一处，避免重复代码

### 论文写作启示
1. **故事线调整**：从"证明 A 优于 B"到"探索 A 和 B 的适用边界"
2. **坦诚评价**：AV-LRM 坦诚评价（PROGRESS.md 已写入）反而增强可信度
3. **理论贡献**：Two-Regime 理论为部署策略提供量化依据，是独立贡献
4. **差异化定位**：与 CoPE-VideoLM 的区分——training-free vs 需训练，部署路线 vs 学术路线

---

## 附录：关键数据速查

### Video-MME Short (108 题)
- Baseline: 75.93%, 2189ms, 10737 tokens
- naive_iframe(0.5): 75.93% (0.0pp), 1096ms (2.0x), 4939 tokens (-54%)
- sparse(0.5): 69.44% (-6.5pp), 1085ms (2.0x), 4939 tokens (-54%)
- sparse(0.2): 70.37% (-5.6pp), 586ms (3.7x), 2192 tokens (-80%)

### MVBench (3600 题)
- Baseline: 66.94%, 1054ms, 5570 tokens, 29.8 frames
- naive_iframe(0.5): 53.59% (-13.35pp), 196ms (5.4x), 628 tokens (-89%), 3.0 frames

### Two-Regime 交叉点
- kr=0.3: naive 69.44% ≈ sparse 69.44%（持平）
- kr=0.5: naive 75.93% >> sparse 69.44%（+6.49pp，coverage-dominant）
- kr=0.2: sparse 70.37% > naive 68.52%（+1.85pp，relevance-dominant）

