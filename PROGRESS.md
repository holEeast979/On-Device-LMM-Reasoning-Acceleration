# FasterOmni 开发进度

> **本文件是跨对话、跨平台的唯一进度真实来源。**
> Windsurf 每次对话开始时读取此文件获取上下文，结束时更新此文件。
> Obsidian Agent 关注 `待探讨问题` 和 `外部反馈` 区域。

---

## 当前阶段

**Phase 1 完成，等待 GPT Review 后进入 Phase 2**。核心发现：①精度对 kr 不敏感（kr=0.2~0.9 准确率 68.5%~70.4%）②音频兜底效应很小（仅 1.8pp），视觉稀疏化本身鲁棒。GPT Review Prompt 已就绪（`gpt_review_prompt.md`），Review 结果将决定 Phase 2 优先级。

---

## 项目背景

研究 **Qwen2.5-Omni-7B** 多模态大模型的推理加速，构建**端侧多模态推理加速架构**。

三大技术支柱（独立可解耦）：
1. **GOP 级 token 稀疏化**（AV-LRM 打分 + Top-K/Uniform 选择）← Phase 1 已完成初步验证
2. **显存碎片优化**（ViT 后清理激活值 + 分块 Encoding）← Phase 2/3
3. **Ring Buffer 流水线**（CPU/GPU 异步预取，批量场景隐藏预处理延迟）← Phase 3

**硬件**：32GB 显存，OOM 边界 ~25-30k tokens（约 34s 视频全量帧）。

**论文故事线**（去音频消融后确认）：
- **主线**："GOP 感知推理加速" — 视觉稀疏化本身鲁棒，kr 不敏感
- **加分项**：音频保留作为 free lunch（零成本恢复 1.8pp）

---

## 模块状态

| 模块 | 文件 | 状态 | 最后验证 | 阻塞 | 下一步 |
|------|------|------|---------|------|--------|
| GOP 解析 | `fasteromni/modules/gop_parser.py` | ✅ 完成 | 16 个视频验证通过，GOP 中位数 5 | — | — |
| 音频能量 | `fasteromni/modules/audio_energy.py` | ✅ 完成 | RMS 能量按 GOP 时间窗口切分正常 | — | — |
| AV-LRM 打分 | `fasteromni/modules/sparse.py` | ✅ 完成 | 归一化顺序+Uniform 偏置已修 | — | alpha 消融显示 GOP 粒度下无区分度 |
| I 帧解码 | `fasteromni/modules/frame_decoder.py` | ✅ 完成 | 全解码后过滤 keyframe（已知限制） | — | 后续考虑 seek-based 选择性解码 |
| Pipeline | `fasteromni/pipeline.py` | ✅ 完成 | baseline+sparse+sparse_no_audio 三模式 | — | — |
| 评估器(EM) | `fasteromni/evaluator.py` | ✅ 完成 | NLTK 版 11/11 self-test 通过 | — | Video-MME 不需要 EM |
| ActivityNet 评估 | `fasteromni/eval_accuracy.py` | ✅ 完成 | 50 样本消融跑通 | 仅 16 独立视频 | 不作为论文主实验 |
| Video-MME 评估 | `fasteromni/eval_videomme.py` | ✅ 完成 | Baseline+Sparse 300/300, kr消融 6×108 | 死锁已修 | 去音频消融 |
| 消融脚本 | `fasteromni/run_ablation.py` | ✅ 完成 | ActivityNet 消融跑通 | — | 在 Video-MME 上重新消融 |

---

## 实验数据

### 修复后真 TTFT（4 视频，ActivityNet-QA，max_new_tokens=1）

| 视频 | Baseline TTFT | Sparse TTFT | 加速比 | Vis Token 减少 | Audio Token |
|------|--------------|-------------|--------|---------------|-------------|
| v_1YU4MSK80cQ (16.2s) | 1646ms | 299ms | **5.51x** | 94% | 408=408 ✅ |
| v_2uUNiV8xmEo (25.2s) | 1163ms | 625ms | **1.86x** | 92% | 634=634 ✅ |
| v_RLBfyIVpocE (13.5s) | 1019ms | 265ms | **3.85x** | 92% | 341=341 ✅ |
| v_G_rVqf_hwXw (17.7s) | 2318ms | 599ms | **3.87x** | 88% | 447=447 ✅ |
| **平均** | | | **3.77x** | **92%** | **一致** |

### ActivityNet-QA 消融（50 样本，EM 评估，16 独立视频）

**keep_ratio 消融**：

| keep_ratio | Accuracy | Acc Drop | Avg Gen(ms) | Speedup |
|-----------|----------|----------|-------------|---------|
| baseline | 56.0% | — | 1382 | 1.00x |
| 0.20 | 54.0% | -2.0% | 183 | **7.56x** |
| 0.30 | 50.0% | -6.0% | 201 | 6.86x |
| 0.50 | 52.0% | -4.0% | 221 | 6.24x |
| 0.70 | 52.0% | -4.0% | 247 | 5.59x |
| 0.90 | 54.0% | -2.0% | 280 | 4.94x |

⚠️ 准确率几乎平（统计噪声范围），因为：1) 仅 16 独立视频 2) 音频"兜底"效应

**alpha 消融**：完全无影响。原因：GOP 中位数仅 5，5 选 3 时不同 alpha 选出相同集合。

### Video-MME 完整评估（100 视频 300 题，max_frames=32）

**kr 消融（Short 视频 36 视频 108 题）**：

| kr | Accuracy | Acc Drop | Avg Gen(ms) | Speedup | Vis Token |
|----|----------|----------|-------------|---------|----------|
| baseline | 75.9% | — | 2,189 | 1.0x | 10,737 |
| 0.9 | 70.4% | -5.5pp | 1,624 | 1.3x | 7,794 |
| 0.7 | 68.5% | -7.4pp | 1,410 | 1.6x | 6,658 |
| 0.5 | 69.4% | -6.5pp | 1,103 | **2.0x** | 4,939 |
| 0.3 | 69.4% | -6.5pp | 783 | **2.8x** | 3,190 |
| 0.2 | 70.4% | -5.5pp | 589 | **3.7x** | 2,192 |

⚠️ **关键发现：准确率对 kr 不敏感**（68.5%~70.4% 范围内波动），延迟与 visual_tokens 近似线性。

**去音频消融（Short 视频 36 视频 108 题，kr=0.5）**：

| 模式 | Accuracy | Acc Drop | Avg Gen(ms) | Speedup | Vis Token | Audio Token |
|------|----------|----------|-------------|---------|----------|-------------|
| baseline | 75.9% | — | 2,189 | 1.0x | 10,737 | ~2,900 |
| sparse (kr=0.5) | 69.4% | -6.5pp | 1,103 | 2.0x | 4,939 | ~2,400 |
| sparse_no_audio | 67.6% | -8.3pp | 863 | **2.5x** | 4,939 | **0** |

⚠️ **音频兜底效应很小**：音频仅恢复 1.8pp（占总损失 22%）。视觉稀疏化本身鲁棒，精度对 kr 不敏感的主因是 GOP 选择保留了关键视觉信息。

**Baseline vs Sparse 完整对比**：

| Duration | N | B Acc | S Acc | Drop | B Gen(ms) | S Gen(ms) | Speedup | Vis Token 减少 |
|----------|---|-------|-------|------|-----------|-----------|---------|---------------|
| Short | 108 | 75.0% | 69.4% | **-5.6pp** | 2,177 | 1,092 | **1.99x** | **54.0%** |
| Medium | 90 | 62.2% | 57.8% | -4.4pp | 3,256 | 3,189 | 1.02x | 0.3% |
| Long | 102 | 48.0% | 49.0% | +1.0pp | 3,178 | 3,138 | 1.01x | 0.0% |
| **Overall** | **300** | **62.0%** | **59.0%** | **-3.0pp** | **2,841** | **2,417** | **1.18x** | **19.3%** |

两者均 0 errors（OOM 修复生效）。

⚠️ **关键发现：M/L 视频 sparse 无效**。原因：M/L 视频 GOP 数量多（100+），kr=0.5 选完后 I 帧数仍超 max_frames=32 → 被截断到 32 帧 → 与 baseline 帧数、token 数完全一致。Sparse 只在 Short 视频（GOP 数 5-10）上有效。

**按 Task Type 对比**：

| Task Type | Baseline | Sparse | Diff | N |
|-----------|----------|--------|------|---|
| Spatial Perception | 71.4% | 85.7% | **+14.3pp** | 7 |
| Object Recognition | 65.1% | 72.1% | **+7.0pp** | 43 |
| Information Synopsis | 82.4% | 82.4% | 0.0pp | 34 |
| Action Recognition | 57.6% | 57.6% | 0.0pp | 33 |
| OCR Problems | 52.6% | 52.6% | 0.0pp | 19 |
| Temporal Reasoning | 34.6% | 30.8% | -3.8pp | 26 |
| Action Reasoning | 51.7% | 44.8% | -6.9pp | 29 |
| Object Reasoning | 66.7% | 58.3% | -8.3pp | 48 |
| Attribute Perception | 81.5% | 70.4% | **-11.1pp** | 27 |
| Counting Problem | 41.7% | 29.2% | **-12.5pp** | 24 |

规律：计数/属性类损失大；总结/空间类不受影响甚至提升。

---

## 关键设计决策

⚠️ = 不可变更的硬约束

1. ⚠️ **TTFT 只能用 `max_new_tokens=1` 测量**，否则测的是生成时间不是 prefill 时间
2. ⚠️ **音频必须通过 processor 传入**（`proc(audio=, use_audio_in_video=True)`），不能手动塞 input_features
3. ⚠️ **Sparse 必须走 video tensor 路径**（不是 image），确保 tokens/frame 一致（~150）
4. **variance_threshold = 0.02**（0.05 时 90% 走 Uniform，过于保守）
5. **Video-MME baseline 用 max_frames=32**（64 帧 OOM，32 帧 12890 tokens 可运行）
6. **评估主实验用 Video-MME 选择题**（零歧义），ActivityNet-QA + GPT-judge 作为补充

---

## 待办事项（按优先级排序）

### Phase 1 ✅ 已完成

| 任务 | 结果 |
|------|------|
| Baseline 完整评估 | ✅ 300/300, 62.0%, 0 errors |
| Sparse 完整评估 | ✅ 300/300, 59.0%, 0 errors |
| Baseline vs Sparse 深度对比 | ✅ Short 2x加速 -5.6pp; M/L 因 max_frames=32 无效 |
| Short kr 消融 (6组×108) | ✅ 精度对 kr 不敏感 (68.5%~70.4%)，kr=0.2 即 3.7x 加速 |
| Short 去音频消融 | ✅ 108/108, 67.6%，音频仅恢复 1.8pp |
| 死锁修复 + 自动恢复 | ✅ SIG_DFL 内核级 watchdog + 增量 CSV |
| GPT Review Prompt | ✅ `gpt_review_prompt.md` 已就绪 |

### Phase 2（GPT Review 驱动 — 补充实验 + 策略改进）

> ⚠️ 优先级按 GPT Review 严重程度排列。P0 是"不做论文站不住"，P1 是"做了论文更强"。

| # | 优先级 | 任务 | GPT Review 级别 | 说明 | 预估 |
|---|--------|------|-----------------|------|------|
| 1 | **P0** | **Naive baselines 对比** | Critical #1 | 等间隔/随机/只取I帧不打分，同帧数对比 → 证明 AV-LRM 优于 naive | ~2h |
| 2 | **P0** | **Modality baselines** | Major #2 | text-only / audio-only / video-only 下界 → 确认模型吃了多少视觉信息 | ~2h |
| 3 | **P0** | **Sparse@64 vs Baseline@64** | Critical #3 | 验证稀疏化扩展能力边界，打穿"只对短视频有效" | ~30min |
| 4 | **P0** | **音频公平性修复** | Critical #2 | baseline 也做音频截断，或明确报告音频 token 差异（实测仅 7.9%） | ~1h |
| 5 | **P1** | **Per-video 统计** | Major #1 | 按视频为单位报告均值/方差/置信区间 + 配对检验 | ~1h |
| 6 | **P1** | **M/L sparse 策略重设计** | — | kr 直接控制帧数 / GOP 内选帧 / max_tokens 替代 max_frames | ~3h |
| 7 | **P1** | **Content-adaptive** | — | 动态 kr（解决逐视频波动大的 tail case） | ~2h |
| 8 | **P2** | AV-LRM 在高 GOP 场景验证 | Major #4 | 在 M/L（GOP 100+）证明打分公式优于 naive | 依赖 #6 |
| 9 | **P2** | 论文表格 + Pareto 曲线图 | — | 全部补充实验完成后生成 | ~1h |

### Phase 3（架构扩展 — 其他两大技术支柱）

| # | 任务 | 说明 |
|---|------|------|
| 8 | **显存碎片优化** | ViT 后 hook 清理激活值 → 降低峰值 → 支持更长视频 |
| 9 | **Ring Buffer 流水线** | CPU/GPU 异步预取，批量场景隐藏预处理延迟 |
| 10 | 多 benchmark 交叉验证 | Video-MME + MVBench / LongVideoBench |
| 11 | 跨模型泛化验证 | 在其他 Omni 模型上验证框架通用性 |
| 12 | Patch 级稀疏化 | 帧内部哪些区域重要（可选探索方向） |

---

## 已知问题

- [x] **音频"兜底"效应已验证**：去音频后仅多降 1.8pp（67.6% vs 69.4%），兜底效应很小，视觉稀疏本身鲁棒
- [ ] **GOP 粒度太粗**：短视频中位数仅 5 个 GOP，alpha 参数和打分公式无法体现价值 → Phase 2 帧级选择
- [ ] **I 帧解码是全解码**：`container.decode()` 遍历全帧再过滤 keyframe，CPU 侧无加速（但不是瓶颈）
- [x] **ActivityNet-QA 采样 bug**：按 QA 对采样而非按视频，50 题仅 16 独立视频→已切换到 Video-MME 规避，不再使用 ActivityNet-QA 作主实验
- [x] **Video-MME "short" 实际 52-111s**：已确认是官方定义，baseline 用 max_frames=32 解决
- [x] **Sparse OOM 修复**：max_frames=32 上限 + 音频截断到选中 GOP 时间范围，300/300 全部跑通
- [x] **eval 结果覆盖问题**：已修复（每个 mode 保存到独立子目录）
- [ ] **M/L 视频 sparse 无效**：max_frames=32 限制使 sparse 在 M/L 上帧数、token 数与 baseline 完全一致。需要改进帧选择策略或提高 max_frames
- [ ] **Short 视频逐视频波动大**：部分视频准确率暴跌 66pp（关键帧丢失），部分提升 33pp（去噪效果）
- [x] **音频+视频提取死锁**：某些视频导致 `torchvision.io.read_video`/`av.open`/`librosa.load` C 扩展永久阻塞，SIGALRM 无法打断。修复：monkey-patch `fetch_video` + `process_audio_info` + `librosa.load`，全部改为 ffmpeg/ffprobe subprocess + timeout。详见变更日志 [2.18 PM]

---

## GPT Code Review 修复状态

| # | 问题 | 状态 | 说明 |
|---|------|------|------|
| 1 | 音频没进入模型 | ✅ | `proc(audio=, use_audio_in_video=True)` |
| 2 | TTFT 口径错误 | ✅ | `generate_ms` + `max_new_tokens=1` |
| 3 | I 帧全解码 | ⚠️ 已知限制 | 加速来自模型侧 token 减少 |
| 4 | 归一化含短 GOP | ✅ | 先过滤再归一化 |
| 5 | Uniform 前部偏置 | ✅ | `np.linspace` |
| 6 | 评估函数 | ✅ | 切换到 Video-MME 选择题 |
| 7 | 样本量+顺序偏置 | ⏳ | Video-MME 300 题可解决 |
| 8 | 阈值写死 | ⏳ | 低优先级 |

---

## 外部反馈

> 从 Obsidian / GPT / 导师收集的结论和方向变化

- **[2.16] GPT Code Review**：8 个问题，已修 6 个。核心发现：音频链路断裂、TTFT 口径错误。修复后加速比从 2.39x 升至 3.77x（因为修前 max_new_tokens=32 掩盖了 prefill 差异）。
- **[2.15] Claude 架构建议**：先串行跑通再拆并行，Phase 1-4 路线图（已大部分落地）。
- **[2.15] GPT 执行清单**：当天完成 GOP 解析 + 串行 Baseline + 稀疏化初测。
- **[2.18] Jarvis 讨论 - content-adaptive sparsification**：结论是 Phase 2 优化项，不是核心贡献。理由：①kr 消融已证明精度对 kr 不敏感，动态调 kr 收益空间小 ②alpha 在短视频无区分度 ③需要额外分类器判断视频类型，增加复杂度 ④论文核心应简洁，固定 kr 就能讲清框架价值。定位：稀疏化 pipeline 的一个可选优化环节，Phase 2 用于优化逐视频波动大的 tail case。
- **[2.18] Jarvis 讨论 - GOP 粒度上限**：短视频 GOP 中位数仅 5，alpha 和打分公式无区分度，这是 H.264 编码特性决定的已知限制。但不影响主线贡献：加速收益来自"砍掉多少 token"而非"选哪几个 GOP"，精度对 kr 不敏感也印证了这一点。GOP 级筛选对长视频（GOP 100+）有足够选择空间，打分公式和 alpha 预期有效。短视频暴露的粒度不足问题引出 Phase 2 方向：帧级选择（GOP 内选 P/B 帧），在更细粒度上优化短视频筛选。
- **[2.18] Jarvis 讨论 - M/L 视频 sparse 失效分析**：由于硬件显存约束（32GB），M/L 视频必须加 max_frames=32 限制，导致 sparse 选完 GOP 后 I 帧数仍超上限被截断，与 baseline 完全一致。但反过来看，这恰好是 sparse 的价值所在：结合稀疏化可以提高 max_frames 上限（如 64 甚至更高）而不 OOM，从而让端侧设备支持更长视频。这是"稀疏化扩展能力边界"的贡献点，待 P1 #3（Sparse@64 + Baseline@64 OOM 验证）实验确认。
- **[2.18] Jarvis 讨论 - Prefill 53% "Other" 开销已定性**：回溯 1 月初 10 视频 TTFT 分解实验（1.6 日后进展汇总），确认 Others 是**固定开销（Fixed Overhead）**，$R^2=0.04$，与 token 数量无相关性。具体来源：①PyTorch/HF 框架调度开销（generate 调用链、模型初始化）②CUDA context + kernel launch ③内存分配器预热 + KV cache 初始化 ④CPU-GPU 同步等待（Wall-clock 计时包含）⑤Python GIL 开销。vLLM 对比实验已验证可压缩此部分。注意：换了 Qwen2.5-Omni + FasterOmni pipeline 后 53% 这个具体数字可能变化，但"固定开销"的定性结论不变。此问题可从待探讨列表划掉。
- **[2.19] GPT-5.2 Phase 1 Review**：质量很高，发现多个致命问题。分级如下：
  - **Critical**：①**新颖性/归因风险** — kr 不敏感可能意味着"选哪几帧不重要"，必须补 naive baselines（等间隔/随机/只取 I 帧不打分等）否则 AV-LRM 贡献点站不住 ②**公平性混淆** — sparse 截断音频到选中 GOP 时间跨度，baseline 用全音频，speedup 混入了"音频 token 减少"效应（实测仅 7.9%，但需明确说明）③**适用范围** — M/L 无效是结构性问题，必须做 Sparse@64 vs Baseline@64 打穿
  - **Major**：①**统计方法** — 108 题来自 36 视频（每视频 3 题），题级不独立，需按视频为单位报告+配对检验 ②**kr 不敏感解释** — 可能是 benchmark bias（语言先验强），需补 text-only/audio-only/video-only 下界确认模型吃了多少视觉信息 ③**Counting 暴跌** — TopK + uniform coverage（固定首尾 + K-2 按分数选）④**AV-LRM 有效性** — 写清楚是可选插件，或在高 GOP 场景证明优于 naive ⑤**CPU 解码口径** — 论文别说"只解码 I 帧快"，最多说"可优化"
  - **Minor**：①gop_parser.py 实际用 PyAV 不是 ffprobe，表述不一致 ②gop_parser/audio_energy 仍有 av.open 无超时 ③decode_i_frames_seek 可能取错 keyframe
  - **Positive**：①增量 CSV+resume+内核级超时是 paper-level 工程 ②关键口径已修正结论可信 ③负面点提前暴露 Phase 2 有抓手
  - **GPT 建议最小代价方案**：补 3 个对照（text-only / audio-only / naive-uniform 同帧数）+ 1 个关键实验（Baseline@64 vs Sparse@64）

---

## 编码 Agent 规范

> 以下规则适用于所有编码 Agent（Windsurf / Cursor / GPT 等）在本项目中工作时遵守。

1. **不要直接跑完整实验**。只跑 smoke test（`--max-videos 3`）验证正确性，用户自己控制完整实验运行。
2. **进度同步**：每次任务完成后更新 `PROGRESS.md`（当前阶段 + 已知问题 + 变更日志）。
3. **不改第三方库**：对 `qwen_omni_utils` 等依赖的修复一律用 monkey-patch，不直接改源码。
4. **`pipeline.py` 保持简洁**：所有 hack/workaround 放在 `eval_videomme.py` 开头的 monkey-patch 区域，pipeline.py 只做直接调用。
5. **超时保护**：所有外部 I/O（视频/音频读取）必须用 `subprocess.run(timeout=N)` 包裹，不依赖 SIGALRM（C 扩展无法打断）。
6. **实验数据安全**：每个 mode/kr 值独立输出目录，增量 CSV 实时写入，崩溃不丢已完成数据。
7. **GPT Review 规范**：每完成一个 Phase，更新 `gpt_review_prompt.md` 并发给 GPT 审查。Review 结果记录在外部反馈区域，作为下一阶段优先级调整的依据。
8. **关注同方向工作**：边做边关注视频 token 稀疏化领域的相关工作（导师要求），记录在下方“相关工作”区域。

---

## 待探讨问题（供离线 Agent 讨论）

- [x] **音频兜底假说**：已验证。去音频后仅多降 1.8pp（67.6% vs 69.4%），音频兜底效应很小。视觉稀疏化本身鲁棒是主因。
- [x] **GOP 粒度上限**：H.264 短视频 GOP 中位数仅 5，这是编码特性决定的。帧级选择（每 GOP 内选帧）能否让 alpha 参数发挥价值？→ 已讨论，见外部反馈 [2.18]
- [x] **论文故事线**：去音频消融显示音频兜底效应小（1.8pp），主线应为"GOP 感知推理加速"，音频作为 free lunch 加分项。
- [x] **content-adaptive sparsification**：根据视频类型（风景/运动/对话）动态调整 kr 和 alpha，这是 Phase 2 还是论文核心贡献？→ 已讨论，见外部反馈 [2.18]
- [x] **Video-MME vs ActivityNet-QA**：长视频（medium/long）对稀疏化的压力是否会暴露短视频掩盖的问题？→ 已讨论，见外部反馈 [2.18]
- [x] **Prefill 中 53% "Other" 开销**：之前 token-scaling 实验发现 ViT 17% + Audio 13% + LLM 17% + Other 53%。这个 Other 能否进一步分解？→ 已定性为固定开销，见外部反馈 [2.18]

---

## 相关工作（视频 token 稀疏化方向）

> 边做边关注的同方向工作，用于论文 Related Work 和差异化分析。

| 方法 | 核心思路 | 与我们的差异 | 状态 |
|------|----------|------------|------|
| FastV | Attention-based token pruning in ViT | 帧内 patch 级 vs 我们帧级 GOP 级 | 待调研 |
| LLaVA-PruMerge | 融合+剪枝 visual tokens | 模型内部 vs 我们预处理阶段 | 待调研 |
| TokenPacker | 压缩 visual token sequence | token 压缩 vs 帧选择 | 待调研 |
| VideoLLM-online | 流式视频理解 | 在线处理 vs 我们离线 | 待调研 |

---

## 变更日志

- **[2.19]** GPT-5.2 Phase 1 Review 完成，发现 3 Critical + 5 Major 问题。Phase 2 优先级据此重排。
- **[2.19]** Phase 1 收尾：更新三大技术支柱架构、Phase 2 方向、GPT Review 规范、相关工作跟踪区域。
- **[2.19]** 去音频消融完成：108/108，67.6%。音频仅恢复 1.8pp，视觉稀疏本身鲁棒。修复 sparse_no_audio 模式 bug（use_audio_in_video=False）+ resume 逻辑加固（忽略无效记录）。
- **[2.18 PM]** **死锁修复（三阶段）**：C 扩展永久阻塞 → ①monkey-patch 改 ffmpeg subprocess ②SIGALRM→SIG_DFL 内核级 watchdog（GIL 被 C 扩展独占时 Python 线程全部阻塞，threading.Timer 无效）③增量 CSV 自动恢复（重启跳过已完成样本）。
- **[2.18 PM]** kr 消融完成：6 模式 × 108 题全部跑通。核心发现：精度对 kr 不敏感（68.5%~70.4%），kr=0.2 即 3.7x 加速 -5.5pp。
- **[2.18 PM]** Baseline 完整评估完成：300/300, 62.0%, 0 errors。Short 2x 加速 -5.6pp，M/L 因 max_frames=32 sparse 无效。
- **[2.18 AM]** Sparse 完整评估完成：300/300, 59.0%, 0 errors。
- **[2.17]** PROGRESS.md 创建。eval_videomme.py 优化（实时进度 + 超时 + 增量 CSV）。pipeline.py 修复（OOM: max_frames + 音频截断）。
- **[2.16]** Video-MME 评估 Pipeline 完成。GPT Code Review 6/8 修复。ActivityNet-QA 消融完成。
- **[2.15]** 串行 Pipeline 跑通。首次 TTFT 对比完成。

