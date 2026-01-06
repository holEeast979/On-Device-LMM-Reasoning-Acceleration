# Experiments

本项目的实验脚本，用于验证 Qwen2.5-Omni 模型的性能瓶颈和优化方向。

## 实验列表

| 实验 | 脚本 | 目的 |
|------|------|------|
| Exp1 | `exp1_modality_bottleneck.py` | 模态瓶颈分析：识别视频/音频/文本各模态的处理瓶颈 |
| Exp2 | `exp2_projection_compare.py` | 投影层对比：比较不同投影策略的效率 |
| Exp3 | `exp3_frame_ablation.py` | 帧数消融：分析不同帧数对推理时间的影响 |
| Exp4 | `exp4_serial_vs_parallel.py` | 串行 vs 并行：比较串行和并行处理多模态输入的差异 |
| Exp5 | `exp5_module_profiler.py` | 模块级 Profiling：细粒度分析各模块耗时 |
| Exp7 | `exp7_video_audio_encode.py` | 视频/音频编码器分析 |
| Exp8 | `exp8_dual_gpu_parallel.py` | 双 GPU 并行：跨设备并行推理实验 |
| Exp9 | `exp9_audio_length_scaling.py` | 音频长度缩放：分析音频时长对处理时间的影响 |
| Exp10 | `exp10_defect_verification.py` | 缺陷验证：音频 padding 浪费 + 多轮无复用验证 |

## 运行方式

```bash
# 单独运行某个实验
python exp/exp1_modality_bottleneck.py

# 使用统一框架运行 (推荐)
python benchmark/run.py audio-padding --manifest /path/to/manifest.csv
python benchmark/run.py multiturn --manifest /path/to/manifest.csv
```

## 输出

所有实验结果默认输出到 `/root/autodl-tmp/results/`，包括：

- `*_results.csv`: 原始测量数据
- `*_summary.csv`: 统计摘要
- `*.png`: 可视化图表
- `*.json`: 结构化数据

## 关键发现

### 音频 Padding 浪费 (Exp10)

- 使用 `padding=max_length` 时，短音频被 padding 到 30s，造成大量计算浪费
- 使用 `do_not_pad` 可显著降低 audio encoder 耗时

### 多轮无复用 (Exp10)

- 同视频多轮对话时，视频/音频编码器每轮都重新计算
- 无 KV Cache 复用，导致 prefill 成本重复
