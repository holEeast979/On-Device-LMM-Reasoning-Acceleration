# Next Action — 显存优化开发（技术点 4）

> 更新：2026-03-19
> 前置：技术点 1-3 全部完成，实验 11-15 已归档

## 当前状态

### 已完成（技术点 1-3）

| 技术点 | 状态 | 效果 |
|--------|------|------|
| 技术点 1：GOP 稀疏化 | ✅ | visual tokens -54%, generate_ms -48% |
| 技术点 2：EncoderCache | ✅ | generate_ms 再降 30%（同视频多问题复用 ViT/Whisper） |
| 技术点 3：PrefetchBuffer | ✅ | total_ms 再降 25%（异步 CPU 预处理） |
| 技术点 4：显存优化 | ⬜ | 待开发（本次任务） |

### 三技术点叠加实验数据（已冻结）

**Video-MME Short (108题, 36视频)**
| 配置 | generate_ms | total_ms | acc | visual_tokens |
|------|------------|----------|-----|---------------|
| Baseline | 2,161 | 5,450 | 75.93% | 10,739 |
| GOP 稀疏化 | 1,110 | 3,466 | 75.00% | 4,941 |
| GOP+Cache | 766 | 3,309 | 75.00% | 4,941 |
| GOP+Prefetch | 1,123 | 2,615 | 75.00% | 4,941 |
| GOP+Cache+Prefetch | 783 | 2,140 | 75.00% | 4,941 |

**ActivityNet-QA (1000题, 100视频)**
| 配置 | generate_ms | total_ms | acc | visual_tokens |
|------|------------|----------|-----|---------------|
| Baseline | 2,246 | 5,480 | 41.70% | 7,745 |
| GOP 稀疏化 | 1,581 | 4,128 | 40.60% | 3,851 |
| GOP+Cache | 1,119 | 3,731 | 40.50% | 3,851 |
| GOP+Prefetch | 1,592 | 3,092 | 40.50% | 3,851 |
| GOP+Cache+Prefetch | 1,130 | 2,527 | 40.50% | 3,851 |

**端到端加速比**: VME 2.55x, ANet 2.17x

---

## 进入新 session 后立即做

### 1. 显存优化开发（技术点 4）

**目标**：降低峰值显存，解锁 max_frames > 32（当前 64帧 OOM 率 89%）

**思路**：
- ViT forward 后立即释放中间激活值 (`del + torch.cuda.empty_cache()`)
- 监控 `torch.cuda.max_memory_allocated()` 对比优化前后
- 可能方向：gradient checkpointing for inference、分块 ViT 编码

**实现步骤**：
1. 先在 AutoDL 上跑 baseline 64帧，记录峰值显存（nvidia-smi + torch API）
2. 定位显存热点（ViT encoder 中间激活、KV cache、audio encoder）
3. 实现显存优化（del 激活 / empty_cache / 分块编码）
4. 对比优化前后：峰值显存、OOM 率、推理速度影响
5. 跑 Sparse@64 验证优化后 OOM 率降为 0%

**实验设计**（表 4-9）：
| 方法 | 峰值显存 (GB) | 64帧 OOM 率 | max_frames 上限 |
|------|--------------|-------------|----------------|
| 优化前 | 待测 | 89% (@64帧) | 32 |
| 优化后 | 待测 | 0% (目标) | 64+ |

**注意事项**：
- 跑实验前 `nvidia-smi` 确认有 GPU（AutoDL 有无卡模式）
- SSH Python 脚本需 `flush=True`
- 显存优化不应影响准确率（必须验证）

### 2. 完成后更新论文大纲

- 填写表 4-9 真实数据
- 更新表 4-10 加入四技术点叠加行
- 如果 max_frames 能提到 64，需要跑 64帧实验验证准确率

### 3. 开始写论文

- 时间线：3.24-3.31 写毕业论文（Word 模板）
- 先写第三章（方法）+ 第四章（实验）
- 答辩 5月9日

---

## 关键文件位置

### AutoDL 服务器
- SSH: `ssh -o StrictHostKeyChecking=no -p 28758 root@connect.bjb2.seetacloud.com`
- 代码: `/root/scripts/fasteromni/` (pipeline.py, encoder_cache.py, prefetch_buffer.py)
- Eval 脚本: `/root/scripts/eval_videomme_ref.py`, `/root/scripts/eval_activitynet.py`
- 代码版本: commit `dc997bf`（技术点 1-3 集成版）
- 结果目录: `/root/autodl-tmp/results/fasteromni/`
- 进度文档: `/root/autodl-tmp/PROGRESS.md`, `/root/autodl-tmp/STORY.md`

### 本地
- 实验数据: `Results/` 目录（11-15 已归档）
- 论文大纲: `~/Desktop/毕业论文➕专业综合拓展/论文大纲.md`（表格已标注数据来源）
- Result 索引: `Results/Result_locate.md`
- 进度: `PROGRESS_UPDATE.md`, `STORY_GPT_REVIEW.md`

### 数据目录映射
| 目录 | 内容 |
|------|------|
| `Results/11_SchemeA_统一重跑_c8c9e7c/` | Baseline + GOP + GOP+Cache |
| `Results/12_Prefetch_VideoMME_dc997bf/` | VME: GOP+Prefetch |
| `Results/13_三合一_VideoMME_dc997bf/` | VME: GOP+Cache+Prefetch |
| `Results/14_Prefetch_ActivityNet_dc997bf/` | ANet: GOP+Prefetch |
| `Results/15_三合一_ActivityNet_dc997bf/` | ANet: GOP+Cache+Prefetch |
