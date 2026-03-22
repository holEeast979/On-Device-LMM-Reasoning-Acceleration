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
