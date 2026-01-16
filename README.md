# FasterOmni

**Accelerating On-Device Omni-modal LMM Inference**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active_Research-yellow)](https://github.com/holEeast979)

## Introduction

This project develops **FasterOmni**, a lightweight middleware for **On-Device Omni-modal LMMs** (e.g., Qwen2.5-Omni). Unlike server-side engines (vLLM, SGLang) that optimize throughput, FasterOmni targets **TTFT latency** and **memory efficiency** in single-request, resource-constrained environments.

**Target Model:** Qwen2.5-Omni-7B

**Benchmarks:** Video-MME, ActivityNet-QA

---

## Research Focus

### Core Problem

Existing inference engines (vLLM, SGLang) focus on **Decoder optimization** (PagedAttention, CUDA Graph, continuous batching), but our profiling reveals that **Encoder + Projection stages dominate TTFT** in multimodal tasks, leaving limited room for decoder-only optimization.

### Three Optimization Directions

| Direction | Problem | Solution Approach |
|-----------|---------|-------------------|
| **Multi-turn KV Cache** | Encoder re-runs for same video across turns | Cache visual/audio embeddings across conversation turns |
| **Video-Audio Sparsification** | Redundant tokens from dense sampling | Joint video-audio token pruning with semantic awareness |
| **Pipeline Parallelism** | Sequential processing of modalities | Overlap video/audio encoding with LLM prefill |

---

## Project Structure

```
├── benchmark/                # Unified benchmark framework
│   ├── run.py                # CLI entry point
│   ├── runner.py             # BenchmarkRunner core
│   ├── unified_runner.py     # Experiment execution framework
│   └── specs/                # Experiment specifications
│       ├── token_scaling.py  # Token-TTFT scaling analysis
│       ├── ttft_10videos.py  # TTFT breakdown (10 videos)
│       ├── vllm_comparison.py # HF vs vLLM comparison
│       └── gpu_memory_trace.py # GPU memory profiling
├── utils/
│   └── profiling_utils.py    # Timing & memory profiling tools
├── common.py                 # Model loaders & shared utilities
└── tools/                    # Data preparation scripts
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/holEeast979/On-Device-LMM-Reasoning-Acceleration.git
cd On-Device-LMM-Reasoning-Acceleration
pip install -r requirements.txt
```

### Run Experiments

```bash
# Token-TTFT Scaling Analysis
python benchmark/run.py token-scaling \
    --model-dir /path/to/Qwen2.5-Omni-7B \
    --manifest /path/to/manifest.csv \
    --out-dir ./results

# TTFT Breakdown (10 videos)
python benchmark/run.py ttft-10videos \
    --model-dir /path/to/Qwen2.5-Omni-7B \
    --manifest /path/to/manifest.csv

# HF vs vLLM Comparison
python benchmark/run.py vllm-comparison \
    --model-dir /path/to/Qwen2.5-Omni-7B \
    --manifest /path/to/manifest.csv \
    --backend hf  # or vllm

# GPU Memory Trace
python benchmark/run.py gpu-memory-trace \
    --model-dir /path/to/Qwen2.5-Omni-7B \
    --video-path /path/to/video.mp4
```

---

## Roadmap

- [x] **Profiling Framework:** Unified benchmark runner with timing & memory analysis
- [x] **TTFT Breakdown:** Decompose latency into encoder/prefill/other stages
- [x] **vLLM Comparison:** Quantify HF vs vLLM performance gap (~4-5x speedup)
- [ ] **Multi-turn Cache:** Implement embedding cache for cross-turn reuse
- [ ] **Sparsification:** Video-audio joint token pruning
- [ ] **Pipeline Optimization:** Overlap modality encoding stages

---

## Author

**Haodong Zhang (HolEast)**

- Incoming MSc Student @ CUHK
- Focus: Edge AI, LLMOps, Applied LLMs
- [GitHub](https://github.com/holEeast979)

---

## License

MIT License - see [LICENSE](LICENSE) for details.
