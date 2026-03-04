# FasterOmni 开发进度
> **本文件是跨对话,跨平台的唯一进度真实来源.**
> Windsurf 每次对话开始时读取此文件获取上下文,结束时更新此文件.
> Obsidian Agent 关注 `待探讨问题` 和 `外部反馈` 区域.

## 文档体系说明

本项目维护三份核心文档，各司其职：

1. **PROGRESS.md**（本文件）— 实时看板
   - 当前状态快照：正在做什么、下一步是什么
   - 待办事项清单（P0/P1/P2 优先级）
   - 已知问题与阻塞点
   - 适用对象：Agent 每次对话开始时读取，快速了解当前进度

2. **PROGRESS_ARCHIVE.md** — 完整历史
   - 所有实验数据的完整记录（数据表格、变更日志、外部反馈）
   - 深度上下文：决策来龙去脉、踩坑详情、编码规范
   - 适用对象：需要查找历史细节时查阅（如"为什么选 Video-MME"、"[2.20] 代码污染事件详情"）

3. **STORY.md** — 研究叙事线
   - 问题→动作→证据→决策的主线记录
   - 关键转折点与理论发现（如 Two-Regime 理论的形成过程）
   - 三大技术支柱的整体架构与联动关系
   - 适用对象：写论文时串联故事、答辩时回顾决策依据、审稿回复时引用证据
   - **维护原则**：只记录主线决策，不堆原始数据；每个节点固定格式（时间/问题/动作/证据/决策）

**三份文档的关系**：
- PROGRESS.md 管"现在在干嘛"（实时状态）
- PROGRESS_ARCHIVE.md 管"发生过什么"（完整日志）
- STORY.md 管"为什么这么做"（决策逻辑）

---

## 当前阶段
**技术点 1（GOP 级 token 稀疏化）收尾阶段**：核心方法已验证（kr=0.5 零损失 2.1x 加速），边界探索已完成（Short 有效，M/L 受限于 max_frames=32），正在收尾实验（MVBench 少 GOP 修复 + latency/token 对比）。

**研究定位**：聚焦 Short 视频（10-120s），M/L 视频不在端侧研究 scope（文献支撑：端侧模型普遍评估 10-120s）。

## 项目背景
研究 **Qwen2.5-Omni-7B** 多模态大模型的推理加速，构建**端侧多模态推理加速系统**。

**三大技术支柱**（独立可解耦）：
1. ✅ **GOP 级 token 稀疏化**（AV-LRM 打分 + 帧选择）← **当前已完成 ~90%**
2. ⬜ **显存碎片优化**（ViT 后清理激活值 + 分块 Encoding）← 待开发
3. ⬜ **Ring Buffer 流水线**（CPU/GPU 异步预取，批量场景隐藏预处理延迟）← 待开发

**硬件**：RTX 5090 32GB，OOM 边界 ~25-30k tokens（约 34s 视频全量帧）。

**论文定位**：边缘服务器/轻量部署（单 GPU 32GB），training-free codec-aware 加速系统。

## 模块状态
| GOP 解析 | `fasteromni/modules/gop_parser.py` | 完成 | 16 个视频验证通过,GOP 中位数 5 | — | — |
| 音频能量 | `fasteromni/modules/audio_energy.py` | 完成 | RMS 能量按 GOP 时间窗口切分正常 | — | — |
| AV-LRM 打分 | `fasteromni/modules/sparse.py` | 完成 | 归一化顺序+Uniform 偏置已修 | — | alpha 消融显示 GOP 粒度下无区分度 |
| I 帧解码 | `fasteromni/modules/frame_decoder.py` | 完成 | 全解码后过滤 keyframe(已知限制) | — | 后续考虑 seek-based 选择性解码 |
| Pipeline | `fasteromni/pipeline.py` | 完成 | baseline+sparse+sparse_no_audio 三模式 | — | — |
| 评估器(EM) | `fasteromni/evaluator.py` | 完成 | NLTK 版 11/11 self-test 通过 | — | Video-MME 不需要 EM |
| ActivityNet 评估 | `fasteromni/eval_accuracy.py` | 完成 | 50 样本消融跑通 | 仅 16 独立视频 | 不作为论文主实验 |
| Video-MME 评估 | `fasteromni/eval_videomme.py` | 🟡 90% | smoke test 3 视频 9 题通过 | 100 视频完整评估未跑完 | **重跑完整评估(带实时进度)** |
| 消融脚本 | `fasteromni/run_ablation.py` | 完成 | ActivityNet 消融跑通 | — | 在 Video-MME 上重新消融 |

## 实验数据

### 修复后真 TTFT(4 视频,ActivityNet-QA,max_new_tokens=1)
| v_1YU4MSK80cQ (16.2s) | 1646ms | 299ms | **5.51x** | 94% | 408=408 |
| v_2uUNiV8xmEo (25.2s) | 1163ms | 625ms | **1.86x** | 92% | 634=634 |
| v_RLBfyIVpocE (13.5s) | 1019ms | 265ms | **3.85x** | 92% | 341=341 |
| v_G_rVqf_hwXw (17.7s) | 2318ms | 599ms | **3.87x** | 88% | 447=447 |
| **平均** | | | **3.77x** | **92%** | **一致** |

### ActivityNet-QA 消融(50 样本,EM 评估,16 独立视频)
**keep_ratio 消融**:

| baseline | 56.0% | — | 1382 | 1.00x |
| 0.20 | 54.0% | -2.0% | 183 | **7.56x** |
| 0.30 | 50.0% | -6.0% | 201 | 6.86x |
| 0.50 | 52.0% | -4.0% | 221 | 6.24x |
| 0.70 | 52.0% | -4.0% | 247 | 5.59x |
| 0.90 | 54.0% | -2.0% | 280 | 4.94x |

️ 准确率几乎平(统计噪声范围),因为:1) 仅 16 独立视频 2) 音频"兜底"效应

**alpha 消融**:完全无影响.原因:GOP 中位数仅 5,5 选 3 时不同 alpha 选出相同集合.

### Video-MME smoke test(3 视频 9 题)
baseline (32帧), Accuracy=66.7%, Avg Gen(ms)=2406, Avg VisTok=11520
sparse (kr=0.5), Accuracy=55.6%, Avg Gen(ms)=1079, Avg VisTok=4560

## 关键发现（技术点 1）

### Short 视频（10-120s）
- **kr=0.5 零损失**：naive_iframe 75.93% = Baseline 75.93%，2.1x 加速，54% token 减少
- **Two-Regime 理论**：kr=0.5 coverage-dominant（naive 优于 sparse +6.49pp），kr=0.2 relevance-dominant（sparse 优于 naive +1.85pp），交叉点 kr≈0.3
- **Pareto 曲线非单调**：kr=0.5 是峰值，kr=0.7/0.9 反而下降（I 帧聚类损害覆盖度）
- **语言先验显著**：text_only=42%（远 > 随机 25%），但 video_only=62.2% 说明视觉贡献 +20.2pp
- **音频作为 free lunch**：去音频仅多降 1.8pp，保留全量音频零额外优化

### M/L 视频（4-60min）— 方法边界量化
- **Adaptive kr 在 M/L 上"数学正确，效果失效"**：
  - Medium: naive_iframe 和 sparse 的 vistok **完全相同**（11052±1810）
  - Long: naive_iframe 和 sparse 的 vistok **完全相同**（10737±2185）
  - 根因：M/L 视频 n_valid=100~1000，`kr_adaptive = min(0.5, 32/n_valid)` → K=32，稀疏化被"数学上限"锁死
- **max_frames=32 是硬瓶颈**：Baseline Medium vistok=11082，Long vistok=10558，与 Adaptive kr 几乎相同
- **意外发现**：Medium 上 Adaptive kr 反超 baseline +3.3pp（60.0% vs 56.7%），说明 I 帧选择在中等视频上不劣于均匀采样
- **论文价值**：
  - 清晰定义了方法边界（Short 有效，M/L 受限于硬件）
  - 量化分析了失效根因（max_frames 上限，不是方法本身的问题）
  - 指向了技术点 2（显存优化）的必要性

### MVBench（短视频时序密集型任务）
- **7/18 任务严重退化**（Δ > -20pp）：counterfactual_inference -36.0pp, object_existence -33.0pp
- **根因**：视频极短（kr=0.5 后仅 ~3 帧 vs baseline ~30 帧），时序信息丧失
- **论文价值**：暴露了方法边界，指向 Layer 3（Motion-Aware P/B 帧补偿）的必要性

## 关键设计决策
️ = 不可变更的硬约束

1. ️ **TTFT 只能用 `max_new_tokens=1` 测量**,否则测的是生成时间不是 prefill 时间
2. ️ **音频必须通过 processor 传入**(`proc(audio=, use_audio_in_video=True)`),不能手动塞 input_features
3. ️ **Sparse 必须走 video tensor 路径**(不是 image),确保 tokens/frame 一致(~150)
4. **variance_threshold = 0.02**(0.05 时 90% 走 Uniform,过于保守)
5. **Video-MME baseline 用 max_frames=32**(64 帧 OOM,32 帧 12890 tokens 可运行)
6. **评估主实验用 Video-MME 选择题**(零歧义),ActivityNet-QA + GPT-judge 作为补充

## 已知问题
- [ ] **音频"兜底"效应未验证**:kr=0.2 不掉精度可能是因为完整音频 token 在补偿,需去音频消融分离贡献
- [ ] **GOP 粒度太粗**:短视频中位数仅 5 个 GOP,alpha 参数和打分公式无法体现价值
- [ ] **I 帧解码是全解码**:`container.decode()` 遍历全帧再过滤 keyframe,CPU 侧无加速(但不是瓶颈)
- [ ] **ActivityNet-QA 采样 bug**:按 QA 对采样而非按视频,50 题仅 16 独立视频(已切换到 Video-MME 规避)
- [ ] **Video-MME "short" 实际 52-111s**:远超预期,baseline 需限帧

## GPT Code Review 修复状态
1, 问题=音频没进入模型, 状态=, 说明=`proc(audio=, use_audio_in_video=True)`
2, 问题=TTFT 口径错误, 状态=, 说明=`generate_ms` + `max_new_tokens=1`
3, 问题=I 帧全解码, 状态=️ 已知限制, 说明=加速来自模型侧 token 减少
4, 问题=归一化含短 GOP, 状态=, 说明=先过滤再归一化
5, 问题=Uniform 前部偏置, 状态=, 说明=`np.linspace`
6, 问题=评估函数, 状态=, 说明=切换到 Video-MME 选择题
7, 问题=样本量+顺序偏置, 状态=⏳, 说明=Video-MME 300 题可解决
8, 问题=阈值写死, 状态=⏳, 说明=低优先级

## 待办事项

### 紧急(Phase 1 收尾)
**P0**, 任务=Video-MME 完整评估 (100 视频 300 题, baseline+sparse), 预估时间=~30-60min, 命令参考=`python fasteromni/eval_videomme.py --max-videos 0 --modes baseline sparse --max-frames 32`
**P0**, 任务=Video-MME keep_ratio 消融 (0.2/0.3/0.5/0.7/0.9), 预估时间=~2h, 命令参考=`python fasteromni/eval_videomme.py --sweep keep_ratio --max-frames 32`
**P0**, 任务=去音频消融 (sparse vs sparse_no_audio), 预估时间=~30min, 命令参考=`python fasteromni/eval_videomme.py --modes sparse sparse_no_audio --max-frames 32`
**P1**, 任务=输出论文级表格 + Pareto 曲线图, 预估时间=~1h, 命令参考=消融完成后生成

### 中期(Phase 2)
P/B 帧选择策略, 说明=在 GOP 内选关键帧,解决"只取 I 帧太粗"的问题, 依赖=Phase 1 数据确认稀疏化有效
显存管理优化, 说明=ViT 后 hook 清理激活值 → 降低峰值 → 支持更长视频, 依赖=无
alpha 在帧级验证, 说明=帧级选择下 alpha 是否有区分度, 依赖=P/B 帧策略实现后

### 远期(Phase 3)
- 多 benchmark 交叉验证: Video-MME + ActivityNet-QA (GPT-judge)
- 与 naive 方法对比: uniform frame sampling / random sampling vs AV-LRM
- Ring Buffer CPU/GPU 异步预取: 批量场景隐藏预处理延迟
- Patch 级稀疏化: 帧内部哪些区域重要
- 长视频能力验证: sparse 模式处理 baseline OOM 的长视频

## 外部反馈
> 从 Obsidian / GPT / 导师收集的结论和方向变化

- **[3.2] 讨论结论(中后段同步)**:确认 `kr_adaptive = min(0.5, 32/n_valid)` 在 M/L 场景会导致 `n_valid>=64` 时 K 恒等于 32,与 baseline 等价,稀疏化在 M/L 基本失效.论文层面不再 claim 15min+ 端侧有效性,Video-MME Medium/Long 结果继续上报但定位为边界分析(scope/limitation),核心贡献收敛到 Short 场景的效率-精度权衡.
- **[3.2] 参数策略结论**:alpha 不采用单数据集单点最优,采用稳健区间策略(避免数据集偏置).当前消融显示 alpha 对结果波动小(约 2%),默认推荐 `alpha=0.3`（保留双模态叙事）,并在论文中强调区间稳定性而非“最高点”.
- **[3.2] 方法边界与后续**:I-only 继续作为正文主线;PTBG(P/B 帧门控保留)作为 Future Work/可选附录小实验(3 配置: I-only / I+P/B always / I+P/B gated),不在当前主线强行并入,避免稀释主贡献.
- **[3.2] 执行优先级**:P0 先做两件事——(1) 修复 MVBench 少 GOP 场景退化(低 n_valid 边界策略), (2) 补齐 sparse vs baseline 的 latency/token 对比,作为论文“加速贡献”主证据;M/L 不再硬攻提分.
- **[2.16] GPT Code Review**:8 个问题,已修 6 个.核心发现:音频链路断裂,TTFT 口径错误.修复后加速比从 2.39x 升至 3.77x(因为修前 max_new_tokens=32 掩盖了 prefill 差异).
- **[2.15] Claude 架构建议**:先串行跑通再拆并行,Phase 1-4 路线图(已大部分落地).
- **[2.15] GPT 执行清单**:当天完成 GOP 解析 + 串行 Baseline + 稀疏化初测.

## 待探讨问题(供离线 Agent 讨论)
- [ ] **音频兜底假说**:如果去掉音频后 kr=0.2 准确率暴跌,说明 GOP 稀疏化本身价值不大?还是说"利用完整音频弥补视觉损失"本身就是一个有价值的设计?
- [ ] **GOP 粒度上限**:H.264 短视频 GOP 中位数仅 5,这是编码特性决定的.帧级选择(每 GOP 内选帧)能否让 alpha 参数发挥价值?
- [ ] **论文故事线**:选"音视频联合稀疏化框架"还是"GOP 感知推理加速"?前者差异化更强但需要证明音频的参与确实有用.
- [ ] **content-adaptive sparsification**:根据视频类型(风景/运动/对话)动态调整 kr 和 alpha,这是 Phase 2 还是论文核心贡献?
- [ ] **Video-MME vs ActivityNet-QA**:长视频(medium/long)对稀疏化的压力是否会暴露短视频掩盖的问题?
- [ ] **Prefill 中 53% "Other" 开销**:之前 token-scaling 实验发现 ViT 17% + Audio 13% + LLM 17% + Other 53%.这个 Other 能否进一步分解?

## 变更日志
- **[2.16]** Video-MME 评估 Pipeline 完成 (`eval_videomme.py`),smoke test 3 视频 9 题通过.Baseline OOM 问题修复 (max_frames=32).Pipeline 增加 `skip_audio` 参数支持去音频消融.
- **[2.16]** GPT Code Review 8 个问题修了 6 个.NLTK 评估器升级完成.ActivityNet-QA 消融完成(发现 alpha 无影响,音频兜底效应).
- **[2.15]** 串行 Pipeline 跑通:GOP 解析 → AV-LRM 打分 → I 帧解码 → 模型推理.首次 TTFT 对比完成.