# Unified Benchmark Framework

统一测量框架，用于对 Qwen2.5-Omni 模型进行标准化性能测试。

## 快速开始

```bash
# 运行 audio-padding 测量
python benchmark/run.py audio-padding --manifest /path/to/manifest.csv

# 运行 multiturn 测量
python benchmark/run.py multiturn --manifest /path/to/manifest.csv

# 运行 token-prefill 测量
python benchmark/run.py token-prefill --manifest /path/to/manifest.csv

# 运行 ttft-breakdown 测量
python benchmark/run.py ttft-breakdown --manifest /path/to/manifest.csv
```

## 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-dir` | `/root/autodl-tmp/Qwen2.5-Omni-7B` | 模型路径 |
| `--dtype` | `bf16` | 数据类型 |
| `--manifest` | - | 数据集 manifest.csv 路径 |
| `--out-dir` | `/root/autodl-tmp/results/motivation` | 输出目录 |
| `--n-samples` | 3 | 采样数量 |
| `--repeats` | 1 | 重复次数 |
| `--warmup` | 0 | 预热次数 |
| `--profile-mem` | false | 是否记录 GPU 峰值内存 |

## Spec 说明

### audio-padding

验证音频 padding 浪费：对同一视频截断不同秒数，分别测量 `padding=max_length` vs `do_not_pad` 的性能差异。

**输出**: `audio_padding_results.csv`, `audio_padding_summary.csv`

### multiturn

验证多轮无复用问题：同视频同问题连续两轮 generate，记录各阶段耗时，统计 forward hook 调用次数。

**输出**: `multiturn_results.csv`, `multiturn_summary.csv`, `multiturn_ratio.json`

### token-prefill

测量 Token prefill 延迟，分析不同输入长度对 prefill 时间的影响。

**输出**: `token_prefill_results.csv`

### ttft-breakdown

TTFT (Time To First Token) 分解：分离视频编码、音频编码、LLM prefill、decode 各阶段耗时。

**输出**: `ttft_breakdown_results.csv`, `ttft_breakdown_summary.csv`

## 架构

```
benchmark/
├── run.py              # CLI 入口
├── runner.py           # BenchmarkRunner：模型加载、音视频提取、通用工具
├── unified_runner.py   # 统一执行器
└── specs/              # 可插拔的测量规格
    ├── __init__.py
    ├── audio_padding.py
    ├── multiturn.py
    ├── token_prefill.py
    └── ttft_breakdown.py
```

每个 spec 实现 `register_subcommand()` 函数注册到 CLI，实现 `run()` 函数执行测量逻辑。
