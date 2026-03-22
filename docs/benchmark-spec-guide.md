# LMM TTFT Benchmark 框架规范

## 项目背景

本项目研究 **Qwen2.5-Omni-7B** 多模态大模型的推理性能，重点分析 **TTFT (Time To First Token)** 的各阶段耗时分布和 scaling 关系。

### 研究目标
1. 分析各阶段（Visual Encoder、Audio Encoder、Prefill）与 Token 数量的 scaling 关系
2. 验证已知的性能缺陷（如音频 padding 浪费、多轮对话 KV Cache 不复用）
3. 为后续优化提供量化依据

### 硬件环境
- GPU: 32GB 显存（实测 OOM 边界约 25-30k tokens，约 34s 视频）
- Flash Attention: 已启用（PyTorch 2.x SDPA 自动调用）

---

## 1. 核心架构：Benchmark Spec 模式

所有实验逻辑必须封装在 `/root/scripts/benchmark/specs/` 目录下的独立模块中。

### 文件结构
```
/root/scripts/benchmark/
├── run.py              # 统一入口点
├── runner.py           # BenchmarkRunner 核心类
├── unified_runner.py   # 实验运行框架（重复、warmup、保存）
└── specs/
    ├── __init__.py     # 注册所有 specs
    ├── token_scaling.py
    ├── ttft_10videos.py
    ├── vllm_comparison.py
    └── ...
```

### Spec 实现模板
```python
SPEC_NAME = "my-experiment"

def register_subcommand(subparsers, common_parser) -> None:
    p = subparsers.add_parser(SPEC_NAME, parents=[common_parser], help="...")
    p.add_argument("--my-param", type=int, default=10)
    p.set_defaults(_spec_run=run)

def run(args: argparse.Namespace, runner: BenchmarkRunner) -> str:
    # 实验逻辑
    return out_dir
```

### 运行方式
```bash
python benchmark/run.py <spec-name> \
    --model-dir /root/autodl-tmp/Qwen2.5-Omni-7B \
    --manifest /path/to/manifest.csv \
    --out-dir /root/autodl-tmp/results \
    --repeats 5 --warmup 1
```

---

## 2. 计时标准（关键！）

### 阶段划分（与学术界对齐）

```
┌─────────────────────────────────────────────────────────────────┐
│                    Preprocess (CPU)                              │
│  extract_ms + pack_ms + audio_feature_ms                         │
│  (视频解码、图像预处理、音频特征提取 - 在 generate 前完成)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 TTFT = model.generate() 耗时                     │
│  ┌──────────────┬──────────────┬────────────────┬─────────────┐ │
│  │Visual Encoder│Audio Encoder │     Prefill    │   Others    │ │
│  │ (CUDA Hook)  │ (CUDA Hook)  │ Embed+LLM Fwd  │  调度开销   │ │
│  └──────────────┴──────────────┴────────────────┴─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 计时方法

| 指标 | 测量方式 | 说明 |
|------|----------|------|
| `extract_ms` | `time.perf_counter()` | 视频解码、帧提取 |
| `pack_ms` | `time.perf_counter()` | 图像预处理、tokenize |
| `audio_feature_ms` | `time.perf_counter()` | 音频 mel 特征提取 |
| `preprocess_ms` | 上述三者之和 | **Preprocess 总时间** |
| `ttft_ms` | `run_generate_with_breakdown()` | **TTFT (wall-clock)** |
| `visual_encoder_ms` | `ModuleCudaEventTimer` | Visual Encoder GPU 时间 |
| `audio_encoder_ms` | `ModuleCudaEventTimer` | Audio Encoder GPU 时间 |
| `prefill_ms` | `ThinkerPrefillCapture - visual - audio` | **Embedding Merge + LLM Forward** |
| `llm_prefill_ms` | `LLMPrefillCudaEventCapture` | 仅 LLM Forward（细粒度分析） |
| `others_ms` | `ttft - visual - audio - prefill` | **调度开销**（内存分配、数据传输等） |

### 计算公式

```python
# Thinker.forward 包含 visual + audio + embedding_merge + llm
thinker_forward_ms = ThinkerPrefillCapture.prefill_forward_ms

# Prefill = Embedding Merge + LLM Forward
prefill_ms = thinker_forward_ms - visual_encoder_ms - audio_encoder_ms

# Others = 调度开销（内存分配、数据传输、同步等）
others_ms = ttft_ms - visual_encoder_ms - audio_encoder_ms - prefill_ms
```

### 重要注意事项

⚠️ **Prefill 现在精确测量 Embedding Merge + LLM Forward**：
- Embedding 融合 (masked_scatter + get_rope_index)
- LLM forward (真正的 attention 计算)

⚠️ **Others 单独分离调度开销**：
- GPU 内存分配
- CPU-GPU 数据传输
- 同步开销

因此 **Prefill 与 Token 数量的关系可能不是纯线性**（实测 R² ≈ 0.50），这是正常的。

---

## 3. 必须复用的核心组件

### BenchmarkRunner (`/root/scripts/benchmark/runner.py`)
- `extract_av_from_video()` - 返回 `ExtractedAV`，含 `extract_ms`
- `prepare_base_inputs()` - 返回 `PackedInputs`，含 `pack_ms`
- `build_audio_features()` - 返回 `af, mel_frames, audio_feature_ms`
- `attach_audio_features()` - 合并音频特征
- `run_generate_with_breakdown()` - **统一计时逻辑**，返回详细分解
- `get_token_stats()` - 获取 token 统计

### Profiling Hooks (`/root/scripts/profiling_utils.py`)
```python
import profiling_utils as P

# 设置 Hooks
visual_timer = P.ModuleCudaEventTimer()
audio_timer = P.ModuleCudaEventTimer()
thinker_capture = P.ThinkerPrefillCapture()
llm_prefill_capture = P.LLMPrefillCudaEventCapture()

visual_timer.register(model.thinker.visual)
audio_timer.register(model.thinker.audio_tower)
thinker_capture.register(model.thinker)
llm_prefill_capture.register(model.thinker.model)

# 使用统一计时逻辑
breakdown = runner.run_generate_with_breakdown(
    model, inputs,
    visual_timer=visual_timer,
    audio_timer=audio_timer,
    thinker_capture=thinker_capture,
    llm_prefill_capture=llm_prefill_capture,
)
# breakdown 包含: ttft_ms, visual_encoder_ms, audio_encoder_ms, prefill_ms, llm_prefill_ms, others_ms
```

### UnifiedRunner (`/root/scripts/benchmark/unified_runner.py`)
```python
ur = UnifiedRunner(base=runner, spec_name=SPEC_NAME, out_dir=out_dir, args=args)
df = ur.run(cases=cases, repeats=5, warmup=1, run_once=run_once, clear_cache=True)
```

---

## 4. 实验输出标准

### 必须记录的字段

```python
return {
    # 预处理时间（分离！）
    "extract_ms": float(av.extract_ms),
    "pack_ms": float(base.pack_ms),
    "audio_feature_ms": float(audio_feature_ms),
    "preprocess_ms": float(av.extract_ms + base.pack_ms + audio_feature_ms),
    
    # TTFT 分解（使用统一计时逻辑）
    "visual_encoder_ms": breakdown["visual_encoder_ms"],
    "audio_encoder_ms": breakdown["audio_encoder_ms"],
    "prefill_ms": breakdown["prefill_ms"],      # = Embedding Merge + LLM Forward
    "llm_prefill_ms": breakdown["llm_prefill_ms"],  # = 仅 LLM Forward（可选）
    "others_ms": breakdown["others_ms"],        # = 调度开销
    "ttft_ms": breakdown["ttft_ms"],
    
    # Token 统计
    "visual_tokens": ...,
    "audio_tokens": ...,
    "text_tokens": ...,
    "total_tokens": ...,
    
    # 元数据
    "duration": float(duration),
    "mel_frames": int(mel_frames),
}
```

### 输出目录结构
```
/root/autodl-tmp/results/<spec-name>/
├── <spec-name>_results.csv      # 原始数据
├── <spec-name>_summary.csv      # 汇总统计
├── <spec-name>_plot.png         # 可视化
└── <spec-name>_analysis.json    # 分析结果
```

---

## 5. 已知问题与发现

### Token-Scaling 实验发现的问题

1. **Prefill R² = 0.50**：相同 token 数量的视频 prefill 时间差 2.2 倍
   - 原因：`prefill_ms` 是"剩余时间"，包含 Embedding 融合、内存分配等开销
   - 这些开销的变异性大，导致拟合度低

2. **Other 占 TTFT 的 53%**：
   - Visual Encoder: 17%
   - Audio Encoder: 13%
   - LLM Prefill: 17%
   - **Other (未分解): 53%**

3. **Visual Encoder 是 O(n) 线性**：R² = 0.986，斜率 ≈ 0.088 ms/token

4. **Audio Encoder 是 O(1) 常数**：固定 ~22ms，与 token 数量无关

---

## 6. 交互规范

每次任务完成或回复结束前，必须调用检查点工具：

```bash
# Cwd 必须是扩展目录，不是工作区目录！
node /root/.windsurf-server/extensions/ask-continue.ask-continue-2.8.9/ac.js \
    "任务完成情况" "/root/scripts" <端口号>
```

端口号从 `/root/scripts/.ask_continue_port` 读取。
