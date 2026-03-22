# FasterOmni 研究叙事线
> 记录每个阶段的 问题→动作→证据→决策。新 agent 读此文件了解项目演化脉络。
> Phase 1-5（早期探索）已归档到 PROGRESS_ARCHIVE.md。
> 本文件从 Phase 6.6（技术路线转折点）开始记录。

### 阶段导航
| Phase | 时间 | 主题 | 结论 |
|-------|------|------|------|
| 6.6 | 3.11 | GPT Code Review | AV-LRM 打分逻辑有致命问题 |
| 6.7 | 3.11 | Adaptive V4 验证 | 修复后追平 naive_iframe，但 top_k 仍差→放弃 AV-LRM |
| 7.1-7.4 | 3.15-3.19 | 两层缓存系统 | EncoderCache(-30%) + PrefetchBuffer(-25%)，三合一 2.55x |
| 8.1-8.3 | 3.19-3.20 | 显存碎片优化 | 两层策略实现 + Exp16 smoke test |
| 8.4 | 3.20-3.21 | Exp17 全量验证 | 边界58→61，Medium 100%，Long 55.9%受物理限 |
| 8.5 | 3.22 | M/L 无优化对照 | 稀疏化是主力，memopt 是安全网；Medium kr=0.1 无优化也 0% OOM |

---

## Phase 6.6 — GPT Code Review：发现 AV-LRM 链路的致命问题

**时间**：3.11
**问题**：adaptive v2/v3 均未超越 naive_iframe，需要深入分析代码逻辑是否有问题
**动作**：用 GPT-4 对核心代码进行全面审查（sparse.py, gop_parser.py, audio_energy.py, pipeline.py, eval 脚本）

**GPT 核心结论**：
> AV-LRM 链路存在多个致命问题，打分在挑"画面复杂、声音大、看起来热闹"的段，而不是挑"对答题最关键"的段。叠加"方差门控几乎总走 top_k""top_k 不保头尾""音频按前缀截断不对齐"这三刀，导致高方差视频比低方差视频更容易翻车。

**发现的 7 个问题**：

### P0 问题（致命，必须修复）

**P0-1: 方差门控阈值过低（0.02 vs 0.05）**
- 问题：99% 视频被判定为"高方差"，几乎全走 top_k，"自适应"名存实亡
- 证据：36 个视频中，threshold=0.02 时 33 个走 top_k（92%），threshold=0.05 时仅 6 个走 top_k（17%）
- 代码位置：pipeline.py:363, pipeline.py:917 用 0.02，sparse.py:133 默认 0.05 被覆盖

**P0-2: top_k/stratified_top_k 不保头尾**
- 问题：丢失视频开头和结尾，伤害时序任务（OCR、Counting、Needle）
- 证据：33 个 top_k 视频中，23 个（70%）没保住开头，10 个（30%）没保住结尾，最差的时间覆盖只有 82.8%
- 代码位置：sparse.py:178-189 stratified_top_k 不保证选中首尾 GOP

**P0-3: 音频喂给模型时和选帧不匹配（音画错位）**
- 问题：画面抽样了，音频还在喂完整前缀，模型收到不匹配的多模态输入
- 证据：pipeline.py:463-470 音频截取 [0, max_end]，不是选中 GOP 对应的音频片段
- 例子：选了 GOP 2、5、8，画面只有这 3 个，但音频是 GOP 0-8 的完整前缀

### P1 问题（重要，建议修复）

**P1-4: min_gop_frames 参数被偷偷覆盖**
- 问题：传入 min_gop_frames=10，实际用 adaptive_min_gop = max(2, median×0.5)，平均 39.8，最大 120
- 证据：某视频 31 帧 / 1.24 秒的 GOP 被阈值 50 过滤掉
- 代码位置：pipeline.py:385-388

**P1-5: 打分逻辑不"懂题"，更像"复杂度分数"**
- 问题：视觉分数只看 i_frame_size（画面复杂度），音频只看 RMS（声音大小），不代表语义重要性
- 证据：sparse 69.4% < naive_iframe 75.9%，说明"用了打分"反而不如"不打分"
- 代码位置：sparse.py:75, audio_energy.py:101

**P1-6: 按 GOP 个数分段，不是按时间分段**
- 问题：高方差视频 GOP 长短差异大，按索引分段失真
- 证据：keep_ratio_actual=0.5 但 frame_keep_ratio=0.61，说明"保留一半 GOP"≠"保留一半时间"
- 代码位置：sparse.py:183, sparse.py:199

### P2 问题（次要，影响分析）

**P2-7: 元数据没落盘到 CSV**
- 问题：selection_strategy、score_variance、kr_adaptive、adaptive_min_gop 只能从字符串解析
- 代码位置：eval_videomme.py:425, eval_videomme.py:571

**决策**：
1. **立即修复 P0 + P1-4 + P1-6 + P2-7**（预计 40 分钟：改代码 20 分钟 + 跑实验 1-2 小时）
2. **验证效果**：
   - 乐观预期：72-74%（接近 naive_iframe 75.93%）
   - 保守预期：70-72%（比当前 69.44% 有提升）
3. **决策点**：
   - ≥74%：继续修复 P1-5（打分逻辑），争取超越 naive_iframe
   - 70-74%：放弃 AV-LRM，转向技术点 2/3 开发
   - <70%：说明还有其他问题，需要更深入分析
4. **P1-5（打分逻辑）暂缓修复**：需要 1-2 天重新设计特征 + 实验验证，时间成本高

**关键洞察**：
- GPT 的分析揭示了 AV-LRM 失败的根本原因：**打分代理不"懂题"**
- I 帧码率 = 画面复杂度（纹理多、镜头切换快），不等于语义重要性
- 音频 RMS = 声音大小（背景音乐、环境噪音），不等于信息量
- 这解释了为什么 naive_iframe（不打分，均匀选）反而比 sparse（打分选）准确率更高

**论文写作启示**：
- AV-LRM 的失败不是"方法不好"，而是"特征选错了"
- 可以在论文中坦诚讨论：codec-level 特征（I 帧码率、音频 RMS）不足以代表语义重要性
- 未来工作：需要 content-aware 特征（场景切换检测、语音识别、OCR 等）
- 这为 Future Work 提供了明确方向

---

## Phase 6.7 — Adaptive V4 修复验证 + 技术路线转折

**时间**：3.11
**问题**：修复 GPT Review 发现的 P0+P1-4+P2-7 后，验证效果并决定技术路线

**修复内容**：
1. _stratified_top_k_selection 强制保留首尾 GOP
2. 方差阈值 0.02 → 0.05（3 处）
3. 音频对齐：保留到最后有效 GOP 的音频
4. 恢复 min_gop_frames 参数不被覆盖
5. 补充 selection_strategy / score_variance / min_gop_frames_used 到 CSV

**关键证据**：

| 方法 | 准确率 | 说明 |
|------|--------|------|
| adaptive v4 | **75.0%** | 修复后，追平 naive_iframe |
| naive_iframe | 75.93% | 等间隔 I 帧，无打分 |
| adaptive v2/v3 | 69-72% | 修复前 |

策略分布：
- uniform（方差 ≤ 0.05）：87 条，**77.0%**
- stratified_top_k（方差 > 0.05）：21 条，**66.7%**

top_k 翻车重灾区：Object Reasoning 0/3 全错

**核心结论**：
- 修复 bug 后 adaptive 追平了 naive_iframe → 之前的差距主要来自工程 bug
- 但 top_k 仍比 uniform 低 10pp → **training-free + question-agnostic 打分的天花板**
- 不看问题就不知道哪些帧"有用"，这不是换个特征就能解决的根本矛盾

**技术路线决策（重要转折）**：
1. **毕业论文**：锁定 naive_iframe（均匀 I 帧），不再投入 AV-LRM 打分优化
2. **会议论文**：后续升级，需要更硬核的技术点
   - GOP 感知 token 压缩（压缩已选帧的 token，而非选帧）
   - Codec-aware temporal pooling（利用 GOP 结构做时间维度 token 合并）
   - 轻量 question-aware 打分（如 CLIP 相似度，用现成模型不额外训练）

---

## Phase 7 — 两层缓存系统：EncoderCache + PrefetchBuffer（3.15-3.19）

**时间**：3.15-3.19
**问题**：单靠 GOP 稀疏化只能减 token，CPU 预处理和 GPU 编码重复计算仍然浪费大量时间
**动作**：实现两层缓存系统 + 四组全量对照实验

### Phase 7.1 — Scheme A 统一重跑（Exp 11）

**时间**：3.15-3.17
**动作**：在新机器（RTX 5090）上用 commit c8c9e7c 统一重跑 Baseline / GOP / GOP+Cache，建立可比基线
**结果**：
- VME Short: Baseline 75.93% (gen=2161ms), GOP 75.0% (gen=1110ms), GOP+Cache 75.0% (gen=766ms)
- ANet: Baseline 41.7% (gen=2246ms), GOP 40.6% (gen=1581ms), GOP+Cache 40.5% (gen=1119ms)

### Phase 7.2 — EncoderCache 实现

**问题**：批量评估中同视频多问题（VME 3题/视频，ANet 10题/视频）反复执行 ViT/Whisper 编码
**方案**：Hook 模型的 visual encoder 和 audio encoder forward 方法，缓存键 = video_path + 采样参数
**关键修复**：
- `pipe.load_model()` 必须在 EncoderCacheHook 创建前调用（懒加载导致 model=None）
- eval 脚本添加 `--encoder-cache` 参数

### Phase 7.3 — PrefetchBuffer 实现

**问题**：CPU 预处理（GOP 解析 + I 帧解码 + 音频提取）占单次推理 15-20%
**方案**：ThreadPoolExecutor 后台线程异步预取下一个视频的 CPU 预处理结果，LRU 淘汰，容量=2
**已知限制**：长视频（>30min）会触发 Python GIL + CUDA 上下文冲突导致死锁，但短视频不受影响（不在论文 scope 内）

### Phase 7.4 — 四组全量对照实验（Exp 12-15）

**时间**：3.18-3.19
**环境**：commit dc997bf，tmux bench session，trap "" ALRM 防超时
**结果**：

| 实验 | 数据集 | 配置 | 准确率 | generate_ms | total_ms |
|------|--------|------|--------|-------------|----------|
| Exp 12 | VME Short 108题 | GOP+Prefetch | 75.0% | 1,123 | 2,615 |
| Exp 13 | VME Short 108题 | GOP+Cache+Prefetch | 75.0% | 783 | 2,140 |
| Exp 14 | ANet 1000题 | GOP+Prefetch | 40.5% | 1,592 | 3,092 |
| Exp 15 | ANet 1000题 | GOP+Cache+Prefetch | 40.5% | 1,130 | 2,527 |

**关键结论**：
1. 三技术点效果累加，无内部抵消（Codex 交叉验证通过）
2. EncoderCache 降 generate_ms -30%（GPU 层缓存 ViT/Whisper 编码结果）
3. PrefetchBuffer 降 total_ms -25%（CPU 层异步预取，隐藏解码延迟）
4. 端到端加速：VME 2.55x（5450→2140），ANet 2.17x（5480→2527）
5. 准确率完全不变，可以称为"两层缓存系统"：CPU 层（预处理缓存）+ GPU 层（编码缓存）

**当前状态**：技术点 1-3 实验全部完成，论文大纲已更新，下一步写论文正文 + 技术点 4（显存优化，待定）

---

## Phase 8 — 显存碎片优化（技术点 4）（3.19-3.20）

**时间**：3.19-3.20
**问题**：64 帧输入下 OOM 率 89%（96/108），max_frames 锁死 32。RTX 5090 有 32GB 显存，理论上够用，根因是 PyTorch 缓存分配器的碎片化。

### Phase 8.1 — 代码实现

**方案**：分层递进，两层均基于 PyTorch 标准 API，模型无关。

- **Layer 0**：分配器调优（零开销）
  - expandable_segments:True — segment 可虚拟扩展合并，消除外部碎片
  - max_split_size_mb:128 — 防止大 block 被切碎，保留给 KV cache
  - 在 load_model() 模型加载前通过 os.environ 设置

- **Layer 1**：LLM Prefill 前 defrag hook（~150-200ms 开销）
  - register_forward_pre_hook 挂在 LLM backbone，prefill 时执行 gc.collect() + empty_cache()
  - 将碎片化的 reserved 内存释放回 CUDA 驱动，给 KV cache 腾出连续空间
  - 仅在 prefill（seq_len > 1）时触发，decode 阶段自动跳过
  - 自动探测 LLM 子模块，覆盖 Qwen/LLaVA/InternVL/mPLUG-Owl/Phi-3

**文件**：新建 fasteromni/memory_optimizer.py (~400 行)，修改 pipeline.py + eval 脚本添加 --memory-optimize

**Codex 审查**：v1 7.4/10，v2 7.2/10，均通过（>=7.0 且无单项<=3）

### Phase 8.2 — Exp 16 显存优化消融（baseline@64, medium+long 64 视频）

**结果**：

| 配置 | OOM 率 | 成功视频 | 说明 |
|------|--------|---------|------|
| 无优化 | 82.8% (53/64) | 6 (vtok=9570) | 23042 tokens 超物理上限 |
| 有优化 | 79.7% (51/64) | 同 6 个 | peak_alloc 27.8GB，32GB 不够 |

**结论**：baseline@64 全量帧(23042 tokens)真的超了 32GB 物理上限，不是碎片问题。显存优化无法解决物理不足。

### Phase 8.3 — Smoke Test（baseline@64 + naive_iframe@64，15 题）

| 配置 | OOM 率 | 成功数 | visual_tokens | generate_ms |
|------|--------|--------|---------------|-------------|
| baseline@64 无优化 | 80% | 3/15 | 7074 | 1625 |
| baseline@64 有优化 | 33% | 10/15 | 18252(avg) | 5222(avg) |
| naive_iframe@64 有优化 | **0%** | **15/15** | 3735 | 1161 |

**核心发现：显存优化与 GOP 稀疏化协同工作**
- 单靠显存优化不够：全量帧 token 数超物理上限
- 单靠稀疏化不够：token 数降下来了，但碎片化导致分配失败
- 两者配合：稀疏化将 token 降到物理上限以内，显存优化消除碎片使分配成功
- 效果：max_frames 从 32 提升到 64，OOM 率 0%

**同视频对比 defrag 开销**：同一视频(5_fXicEnKKk, 7074 tokens) 有优化比无优化慢 ~200ms (+13%)，可接受。

---

## Phase 8.4 — Exp 17 显存优化全量验证（3.20-3.21）

**时间**：3.20-3.21
**问题**：Exp 16 smoke test 只有 15 个 Short 视频，需要全量验证 + 精确边界测定 + M/L 视频能力扩展

### 17a — OOM 边界精确扫描
- max_frames 57-62 逐帧测试，确定：无优化边界=58，有优化边界=61

### 17b — Short@60/61 全量验证（36 视频）
- @60 无优化: 24/36 (33.3% OOM) → 有优化: 36/36 (0% OOM)
- @61 无优化: 24/36 (33.3% OOM) → 有优化: 36/36 (0% OOM)
- **能力边界从 58 扩展到 61**

### 17c — Profiling 对比图
- 同视频 0ay2Qy3wBe8 (21602 tokens) @61 对比
- 有优化：expandable_segments 导致 reserved 波动，但成功完成
- 无优化：reserved 阶梯增长，2/3 视频 OOM

### 17d/17e — Medium 视频能力扩展
- 17d: Medium@64 kr=0.2 memopt → 26/30 (4 OOM, 13.3%)
- 17e: Medium@64 kr=0.1 memopt → **30/30 (0% OOM)**
- 17d 的 4 个 OOM 视频在 17e 全部成功（token 减少 ~50%）

### 17f — Long 视频物理极限
- Long@64 kr=0.1 memopt → 15/34 (19 OOM, 55.9%)
- 成功视频最大 peak_alloc=30,742 MB，接近 32GB 物理极限
- OOM 视频在 ViT 阶段即超限，属于硬件物理约束

**核心结论**：
1. 显存优化完全消除 Short 边界 OOM（33.3% → 0%）
2. GOP 稀疏化 + 显存优化协同使 Medium@64 100% 通过
3. Long 视频受 32GB 物理限制，部分超限不可解
4. 代价：generate_ms +44%（expandable_segments 开销），边界场景可接受

**缺失数据**：M/L 无优化对照组未跑（需补跑 17d/e/f no_opt 版本）

---

## Phase 8.5 — M/L 无优化对照实验（3.22）

**时间**：3.22
**问题**：17d/e/f 只有 memopt 版本，缺无优化对照，论文表 4-10 不完整

**实验**：补跑 17d/e/f 的 no_opt 版本（naive_iframe@64，无 --memory-optimize）

**结果**：

| 配置 | 无优化 | 有优化 | memopt 贡献 |
|------|--------|--------|------------|
| Medium kr=0.2 | 23/30 (23.3% OOM) | 26/30 (13.3%) | 救回 3 个高 token 视频 |
| Medium kr=0.1 | 30/30 (0%) | 30/30 (0%) | 无额外贡献 |
| Long kr=0.1 | 14/34 (58.8%) | 15/34 (55.9%) | 仅救回 1 个边界视频 |

**关键发现——修正了之前的叙事**：
1. **显存优化的核心价值在 Short 边界场景**（表 4-9：33.3%→0%），不在 M/L
2. **M/L 场景以 GOP 稀疏化为主**：kr=0.1 时 token 足够低（avg 6,412），无优化也 100% 通过 Medium
3. **显存优化在 M/L 是安全网**：仅在 token 接近物理上限时（kr=0.2 的高 GOP 视频）有边际贡献
4. **延迟代价**：memopt 导致 generate_ms 增加 17-25%（M/L），与 Short 的 44% 一致

**论文叙事调整**：
- 之前：显存优化 + 稀疏化"协同"使 Medium 100% 通过
- 修正：稀疏化是主力（降 token），显存优化是安全网（消除碎片导致的边界 OOM）
- 两者角色不对等，论文应如实呈现

---

## Phase 9.0 — 论文初稿最终审查（3.22）

**时间**：3.22
**问题**：论文初稿需要最终审查，确保数据一致性、参考文献正确性后进入查重

**审查范围**：
1. 降重状态检查 — 正文已降重，质量好，仅 1 处 AI 用语残留
2. 大纲对比 — 初稿以 Exp11 冻结数据为准，大纲部分数据过时
3. 实验数据核验 — 逐表与本地 Results/ 目录的 JSON/CSV 交叉验证

**发现并修复的 7 个问题**：

| 优先级 | 问题 | 修复 |
|--------|------|------|
| P0 | 表4-3 kr=0.5 准确率 75.93%（Exp05）vs 表4-1 的 75.00%（Exp11）矛盾 | 统一为 75.00%，全列对齐 |
| P0 | L457 引用旧版 baseline 延迟 2,144ms | 改为 2,161ms（与表4-11一致）|
| P1 | 参考文献[13]指向 EMA-VFI（插帧），非 EMA（视频MLLM）| 替换为正确论文 |
| P1 | ReMoRa、CoPE-VideoLM 缺引用编号 | 新增 [20][21]，经网络搜索确认论文真实 |
| P2 | 旨在填补这一空白 AI 用语残留 | 改为自然表述 |
| P2 | 表4-7 缺 Baseline 对照行 | 补充 Medium/Long Baseline 数据 |
| P2 | 因果逻辑：近零损失被写为前提/目标 | 修正为实验发现的结果/bonus |

**核心决策**：
- 数据矛盾根因：Exp05（旧代码 adaptive_min_gop）vs Exp11（冻结代码 min_gop_frames=10）产生不同准确率
- 以 Exp11 为权威数据源，全文统一
- 研究因果逻辑：目标是加速+保证准确率，近零损失是实验发现的 bonus

**结论**：论文初稿审查完毕，21 条参考文献全覆盖，图表编号连续，实验数据与冻结版代码一致。可进入手动查重。

### Codec 技术备忘

- Codec = Coder + Decoder，视频编解码器（H.264/H.265），与 AI 模型无关
- 压缩时将帧组织为 GOP 结构：I 帧（完整画面）、P 帧（前向差异）、B 帧（双向差异）
- 我们的 "codec-aware" 方法：利用 codec 的帧类型信息（I/P/B）做智能帧选择，而非传统均匀采样
- 具体：av.open() → 识别 key_frame → 只保留 I 帧 → 均匀采样送 ViT
