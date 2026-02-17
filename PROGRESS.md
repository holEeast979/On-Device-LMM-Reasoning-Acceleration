# FasterOmni 开发进度

> **本文件是跨对话、跨平台的唯一进度真实来源。**
> Windsurf 每次对话开始时读取此文件获取上下文，结束时更新此文件。
> Obsidian Agent 关注 `待探讨问题` 和 `外部反馈` 区域。

---

## 当前阶段

**Phase 1 收尾**：Video-MME 完整评估 + 消融实验。评估 Pipeline 已写好并 smoke test 通过，但 100 视频完整评估尚未跑完。

---

## 项目背景

研究 **Qwen2.5-Omni-7B** 多模态大模型的推理加速，核心方法是 **GOP 级视频 token 稀疏化（AV-LRM）**。

三个独立可解耦的技术点：
1. **GOP 级 token 稀疏化**（AV-LRM 打分 + Top-K/Uniform 选择）← 当前主线
2. **显存管理优化**（ViT 后清理激活值 + 分块 Encoding）
3. **Ring Buffer**（CPU/GPU 异步预取，批量场景隐藏预处理延迟）

**硬件**：32GB 显存，OOM 边界 ~25-30k tokens（约 34s 视频全量帧）。

**论文故事线候选**：
- "首个面向端侧的音视频联合稀疏化框架"（差异化最强）
- "基于 GOP 感知的多模态推理加速"

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
| Video-MME 评估 | `fasteromni/eval_videomme.py` | 🟡 90% | smoke test 3 视频 9 题通过 | 100 视频完整评估未跑完 | **重跑完整评估（带实时进度）** |
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

### Video-MME smoke test（3 视频 9 题）

| 模式 | Accuracy | Avg Gen(ms) | Avg VisTok |
|------|----------|-------------|-----------|
| baseline (32帧) | 66.7% | 2406 | 11520 |
| sparse (kr=0.5) | 55.6% | 1079 | 4560 |

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

## 已知问题

- [ ] **音频"兜底"效应未验证**：kr=0.2 不掉精度可能是因为完整音频 token 在补偿，需去音频消融分离贡献
- [ ] **GOP 粒度太粗**：短视频中位数仅 5 个 GOP，alpha 参数和打分公式无法体现价值
- [ ] **I 帧解码是全解码**：`container.decode()` 遍历全帧再过滤 keyframe，CPU 侧无加速（但不是瓶颈）
- [ ] **ActivityNet-QA 采样 bug**：按 QA 对采样而非按视频，50 题仅 16 独立视频（已切换到 Video-MME 规避）
- [ ] **Video-MME "short" 实际 52-111s**：远超预期，baseline 需限帧

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

## 待办事项

### 紧急（Phase 1 收尾）

| 优先级 | 任务 | 预估时间 | 命令参考 |
|--------|------|---------|---------|
| **P0** | Video-MME 完整评估 (100 视频 300 题, baseline+sparse) | ~30-60min | `python fasteromni/eval_videomme.py --max-videos 0 --modes baseline sparse --max-frames 32` |
| **P0** | Video-MME keep_ratio 消融 (0.2/0.3/0.5/0.7/0.9) | ~2h | `python fasteromni/eval_videomme.py --sweep keep_ratio --max-frames 32` |
| **P0** | 去音频消融 (sparse vs sparse_no_audio) | ~30min | `python fasteromni/eval_videomme.py --modes sparse sparse_no_audio --max-frames 32` |
| **P1** | 输出论文级表格 + Pareto 曲线图 | ~1h | 消融完成后生成 |

### 中期（Phase 2）

| 任务 | 说明 | 依赖 |
|------|------|------|
| P/B 帧选择策略 | 在 GOP 内选关键帧，解决"只取 I 帧太粗"的问题 | Phase 1 数据确认稀疏化有效 |
| 显存管理优化 | ViT 后 hook 清理激活值 → 降低峰值 → 支持更长视频 | 无 |
| alpha 在帧级验证 | 帧级选择下 alpha 是否有区分度 | P/B 帧策略实现后 |

### 远期（Phase 3）

| 任务 | 说明 |
|------|------|
| 多 benchmark 交叉验证 | Video-MME + ActivityNet-QA (GPT-judge) |
| 与 naive 方法对比 | uniform frame sampling / random sampling vs AV-LRM |
| Ring Buffer CPU/GPU 异步预取 | 批量场景隐藏预处理延迟 |
| Patch 级稀疏化 | 帧内部哪些区域重要 |
| 长视频能力验证 | sparse 模式处理 baseline OOM 的长视频 |

---

## 外部反馈

> 从 Obsidian / GPT / 导师收集的结论和方向变化

- **[2.16] GPT Code Review**：8 个问题，已修 6 个。核心发现：音频链路断裂、TTFT 口径错误。修复后加速比从 2.39x 升至 3.77x（因为修前 max_new_tokens=32 掩盖了 prefill 差异）。
- **[2.15] Claude 架构建议**：先串行跑通再拆并行，Phase 1-4 路线图（已大部分落地）。
- **[2.15] GPT 执行清单**：当天完成 GOP 解析 + 串行 Baseline + 稀疏化初测。

---

## 待探讨问题（供离线 Agent 讨论）

- [ ] **音频兜底假说**：如果去掉音频后 kr=0.2 准确率暴跌，说明 GOP 稀疏化本身价值不大？还是说"利用完整音频弥补视觉损失"本身就是一个有价值的设计？
- [ ] **GOP 粒度上限**：H.264 短视频 GOP 中位数仅 5，这是编码特性决定的。帧级选择（每 GOP 内选帧）能否让 alpha 参数发挥价值？
- [ ] **论文故事线**：选"音视频联合稀疏化框架"还是"GOP 感知推理加速"？前者差异化更强但需要证明音频的参与确实有用。
- [ ] **content-adaptive sparsification**：根据视频类型（风景/运动/对话）动态调整 kr 和 alpha，这是 Phase 2 还是论文核心贡献？
- [ ] **Video-MME vs ActivityNet-QA**：长视频（medium/long）对稀疏化的压力是否会暴露短视频掩盖的问题？
- [ ] **Prefill 中 53% "Other" 开销**：之前 token-scaling 实验发现 ViT 17% + Audio 13% + LLM 17% + Other 53%。这个 Other 能否进一步分解？

---

## 变更日志

- **[2.16]** Video-MME 评估 Pipeline 完成 (`eval_videomme.py`)，smoke test 3 视频 9 题通过。Baseline OOM 问题修复 (max_frames=32)。Pipeline 增加 `skip_audio` 参数支持去音频消融。
- **[2.16]** GPT Code Review 8 个问题修了 6 个。NLTK 评估器升级完成。ActivityNet-QA 消融完成（发现 alpha 无影响、音频兜底效应）。
- **[2.15]** 串行 Pipeline 跑通：GOP 解析 → AV-LRM 打分 → I 帧解码 → 模型推理。首次 TTFT 对比完成。
