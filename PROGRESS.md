# FasterOmni 开发进度
> **跨对话、跨平台的唯一进度真实来源。**
> 新 session 开始时读取本文件获取上下文，结束时更新本文件。
> 最后更新：2026-03-22

## ⚡ 新 Agent 快速上手

**你在做什么**：FasterOmni — Qwen2.5-Omni-7B 多模态大模型的端侧推理加速系统。四个技术点全部完成，当前进入论文写作阶段。

**连接方式**：
- SSH: `ssh -o StrictHostKeyChecking=no -p 37667 root@connect.bjb2.seetacloud.com`
- ⚠️ 端口经常变化（AutoDL 开关机会变），如果连不上问用户要新端口
- ⚠️ 跑实验前 `nvidia-smi` 确认有 GPU（AutoDL 有无卡模式）

**关键文件位置**：
- 远程代码：`/root/scripts/fasteromni/` (pipeline.py, encoder_cache.py, prefetch_buffer.py, memory_optimizer.py)
- 远程评估脚本：`/root/scripts/eval_videomme_ref.py`, `/root/scripts/eval_activitynet.py`
- 远程实验结果：`/root/autodl-tmp/results/`
- 远程 Profiling 脚本：`/root/scripts/exp_memory_profiling.py`
- 本地实验归档：`~/Desktop/AutoDL-ClaudeCode/Results/`（按实验编号组织）
- 论文大纲：`~/Desktop/毕业论文➕专业综合拓展/论文大纲.md`

**文档体系**：
- **PROGRESS.md**（本文件）— 实时看板：当前状态 + 待办 + 关键数据
- **STORY.md** — 研究叙事线：Phase 6.6-8.4，每个阶段的问题→动作→证据→决策
- **PROGRESS_ARCHIVE.md** — 完整历史归档：Phase 1-5 早期实验、GPT Review 细节、工程备忘
- **NEXT_ACTION.md** — 下一次 session 的具体任务清单

**进度文档以 `/root/scripts/` 为准**，不要写到 `/root/autodl-tmp/`。

---

## 当前阶段

**四个技术点全部完成，进入论文写作阶段。**

| 技术点 | 状态 | 核心效果 |
|--------|------|---------|
| 1. GOP 级 token 稀疏化 | ✅ | token -54%, 2x 加速, 零损失 (naive_iframe, kr=0.5) |
| 2. 编码器缓存 EncoderCache | ✅ | generate_ms -30%（同视频多问题复用 ViT/Whisper） |
| 3. 预取流水线 PrefetchBuffer | ✅ | total_ms -25%（CPU/GPU 异步并行） |
| 4. 显存碎片优化 | ✅ | 能力边界 58→61 帧, Medium@64 100%无OOM |

**时间线**：3.24-3.31 写毕业论文（Word模板），4月初初稿，5月9日答辩。

---

## 核心实验数据

### 三技术点叠加加速（Exp 11-15, commit dc997bf）

| 数据集 | Baseline total_ms | 三合一 total_ms | 加速比 | 准确率 |
|--------|-------------------|-----------------|--------|--------|
| VME Short (108题) | 5,450 | 2,140 | **2.55x** | 75.0% (=75.93%) |
| ActivityNet (1000题) | 5,480 | 2,527 | **2.17x** | 40.5% (≈41.7%) |

### 显存优化边界验证（Exp 17, 2026-03-21）

| 实验 | 配置 | 无优化 | 有优化 | 说明 |
|------|------|--------|--------|------|
| 17b | Short@61 baseline | 24/36 (33.3% OOM) | **36/36 (0%)** | 显存优化核心价值 |
| 17d | Medium@64 kr=0.2 | 23/30 (23.3% OOM) | 26/30 (13.3%) | memopt 救回 3 个 |
| 17e | Medium@64 kr=0.1 | **30/30 (0%)** | **30/30 (0%)** | 稀疏化足够，无需 memopt |
| 17f | Long@64 kr=0.1 | 14/34 (58.8% OOM) | 15/34 (55.9%) | 物理硬限，memopt 仅救 1 个 |

### 其他关键数据

- **kr=0.5 零损失**：naive_iframe 75.93% = Baseline，token -54%，2x 加速
- **Bootstrap CI**：95% CI [-2.8, +4.3]，跨零→无显著差异→零损失统计可靠
- **MVBench**：53.59%（-13.35pp），极短视频 GOP 不足导致失效
- **Sparse@64**：Baseline 89% OOM → Sparse 0% OOM
- **显存优化延迟代价**：generate_ms +44%（expandable_segments 开销），边界场景可接受

---

## 待办事项

### P0（论文写作准备 + 补跑实验）
- [x] **补跑 M/L 无优化对照**：17d/e/f no_opt 已完成（3.22），论文表 4-10 已更新
- [ ] **开始写论文正文**：先第四章（实验，数据齐全直接填表），再第三章（方法）
- [ ] **图表制作**：架构图（手绘）+ Profiling 对比图 + 实验表格

### P1（可选扩展）
- [ ] ActivityNet-QA 8000+ 样本扩展验证
- [ ] 跨模型验证（Qwen3.5-4B）

---

## 模块状态

| 模块 | 文件 | 状态 |
|------|------|------|
| Pipeline（三种推理模式） | pipeline.py | ✅ |
| 编码器缓存 | encoder_cache.py | ✅ |
| 预取流水线 | prefetch_buffer.py | ✅ |
| 显存优化 | memory_optimizer.py | ✅ |
| Video-MME 评估 | eval_videomme_ref.py | ✅ |
| ActivityNet 评估 | eval_activitynet.py | ✅ |
| 显存 Profiling | exp_memory_profiling.py | ✅ |

## 远程结果目录（/root/autodl-tmp/results/）

| 目录 | 内容 |
|------|------|
| fasteromni/scheme_a_c8c9e7c/ | Exp 11: Baseline + GOP + GOP+Cache |
| fasteromni/bench_prefetch_vmme/ | Exp 12: VME GOP+Prefetch |
| fasteromni/bench_all3_vmme/ | Exp 13: VME GOP+Cache+Prefetch |
| fasteromni/bench_prefetch_anet/ | Exp 14: ANet GOP+Prefetch |
| fasteromni/bench_all3_anet/ | Exp 15: ANet GOP+Cache+Prefetch |
| exp17/ | Exp 17: 显存优化全量验证（17a-17f） |

## 本地结果目录（~/Desktop/AutoDL-ClaudeCode/Results/）

| 目录 | 对应论文 |
|------|---------|
| 17_显存优化实验/表4-9_Short6*/ | 表 4-9: 显存优化 OOM 消除 |
| 17_显存优化实验/图4-1_Profiling61_*/ | 图 4-1: Profiling 对比图 |
| 17_显存优化实验/表4-10_Medium_*/ | 表 4-10: M/L 能力扩展 |
| 17_显存优化实验/表4-10_Long_*/ | 表 4-10: Long 视频物理极限 |
| 11_SchemeA_统一重跑_c8c9e7c/ | 表 4-1, 4-11: 主实验 + 系统性能 |
| 12-15_Prefetch/三合一/ | 表 4-8, 4-11: 缓存加速 + 系统性能 |

## 变更日志
- **[3.22]** 补跑 17d/e/f 无优化对照：Medium kr=0.2 OOM 23.3%→13.3%, kr=0.1 两者都 0%, Long 58.8%→55.9%；论文大纲表4-10更新为有/无优化对照
- **[3.21]** Exp 17 全量验证完成 + 论文大纲更新表4-9/4-10/4-11 + 进度文档精简
- **[3.20]** 技术点 4 显存优化实现 + Exp 16 smoke test
- **[3.19]** Exp 12-15 全量完成，三技术点叠加 2.55x/2.17x
- **[3.18]** EncoderCache + PrefetchBuffer 集成到 eval 脚本
- **[3.17]** 导师会议：5.9 答辩，3.24 开始写论文
- **[3.11]** 技术路线决策：锁定 naive_iframe，放弃 AV-LRM

## 代码清理待办（2026-03-21 记录）

以下是 pipeline.py 中遗留的无用逻辑，来自早期 adaptive 模式（已弃用）：

1. ** 参数及 adaptive 分支**：所有实验用  模式（），adaptive 分支（）从未在最终实验中使用。可删除 adaptive 相关代码。
2. ** 变量**：fixed 模式下直接赋值 ，没有做任何自适应计算。可直接用  替代。
3. ** 的 AV-LRM 打分逻辑**（, , ）：实验中 naive_iframe 策略完全不走打分，这些函数只有  /  调用，最终论文方案（naive_iframe）不需要。
4. ****：只被 AV-LRM 打分引用，naive_iframe 不需要音频能量特征。
5. ** / **：方差门控逻辑属于 AV-LRM 打分策略，naive_iframe 不涉及。

以上属于历史遗留，不影响运行结果，但增加了代码理解难度。后续有空可清理。


## 代码清理待办（2026-03-21 记录）

以下是 pipeline.py 中遗留的无用逻辑，来自早期 adaptive 模式（已弃用）：

1. gop_filter_mode 参数及 adaptive 分支：所有实验用 fixed 模式（min_gop_frames=10），adaptive 分支（adaptive_min_gop = max(2, median*0.5)）从未在最终实验中使用。可删除 adaptive 相关代码。
2. kr_adaptive 变量：fixed 模式下直接赋值 kr_adaptive = keep_ratio，没有做任何自适应计算。可直接用 keep_ratio 替代。
3. sparse.py 的 AV-LRM 打分逻辑（score_gops, select_gops, ScoredGOP）：实验中 naive_iframe 策略完全不走打分，这些函数只有 run_sparse / run_adaptive 调用，最终论文方案（naive_iframe）不需要。
4. audio_energy.py：只被 AV-LRM 打分引用，naive_iframe 不需要音频能量特征。
5. variance_threshold / score_variance：方差门控逻辑属于 AV-LRM 打分策略，naive_iframe 不涉及。

以上属于历史遗留，不影响运行结果，但增加了代码理解难度。后续有空可清理。

---

## 技术概念备忘

### Codec（编解码器）= Coder + Decoder

Codec 是视频文件的压缩格式标准（如 H.264、H.265），跟 AI 模型无关。它把原始视频（几十 GB）压缩成 .mp4（几百 MB），压缩时把帧组织成 GOP 结构：

```
GOP 1                    GOP 2
[I] [P] [B] [P] [B]      [I] [P] [B] [P] [B] ...
 ↑                        ↑
 关键帧(完整画面)          关键帧(完整画面)
     ↑
     预测帧(只存差异)
```

- **I 帧**（Intra-coded）：完整编码一整帧，可独立解码
- **P 帧**（Predictive）：只存和前一帧的差异（运动向量 + 残差）
- **B 帧**（Bi-predictive）：存前后两帧差异，压缩率最高

这些信息存在视频文件里，用 `ffprobe` 即可查看每帧类型。

### 我们的方法与 Codec 的关系

"Codec-aware" = 方法知道并利用了视频压缩格式的结构信息（I/P/B 帧类型、GOP 边界）。

具体流程：
1. `av.open()` 打开视频 → codec 解析出 GOP 结构
2. 识别 I 帧（`frame.key_frame == True`）
3. 只保留 I 帧（信息最完整），跳过 P/B 帧
4. 在 I 帧上均匀采样 → 送给 ViT 编码

| | 传统方法 | 我们的方法（codec-aware） |
|---|---|---|
| 看待视频 | 一串等间隔 RGB 帧 | 有结构的 GOP 序列（I/P/B）|
| 帧选择 | 均匀采第 0, 10, 20... 帧 | 只选 I 帧（信息完整的关键帧）|
| 是否利用压缩信息 | 否，全部解码成 RGB 后采样 | 是，利用帧类型做筛选 |

---

## 论文写作状态（3.22 更新）

**论文初稿已完成最终审查，可进入手动查重阶段。**

已修复的问题：
1. 表4-3 kr=0.5 准确率统一为 75.00%（与表4-1 Exp11 数据一致）
2. L457 延迟 2,144ms → 2,161ms（与表4-11 一致）
3. 参考文献[13] EMA-VFI → 正确的 EMA 论文
4. 新增 [20] ReMoRa、[21] CoPE-VideoLM 引用
5. 表4-7 补充 Baseline 对照行
6. AI 用语旨在填补这一空白→ 自然表述
7. 因果逻辑修正：近零损失是实验结果，不是研究目标

参考文献：19 → 21 条，全覆盖。图表编号连续无跳号。
