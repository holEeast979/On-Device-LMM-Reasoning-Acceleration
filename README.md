# Edge-LMM-Accelerator

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Research-yellow)](https://github.com/holEeast979)

> **Accelerating Large Multimodal Models (LMMs) inference on resource-constrained edge devices (Jetson Orin, Raspberry Pi).**

## Introduction

Running modern Large Multimodal Models (like Qwen2-VL, LLaVA) on edge devices is challenging due to limited memory and compute power. This project aims to bridge the gap between heavy LMMs and accessible hardware.

We focus on **fine-grained profiling**, **architecture analysis**, and **inference optimization** (Quantization, Token Pruning) to achieve real-time multimodal interaction on the edge.

**Hardware Focus:** NVIDIA Jetson Orin / Nano, Consumer GPUs.

---

## Key Findings (So Far)

Based on our benchmarks on **Qwen2-VL**, **Phi-3.5-Vision**, and **BLIP-2**, we have debunked several common assumptions about LMM latency:

### 1. The Real Bottleneck is ViT, Not Video Decoding
Contrary to popular belief, video decoding accounts for only **~18%** of total latency. The **Vision Transformer (ViT) Encoder** is the absolute bottleneck, consuming **92-94%** of inference time in high-resolution settings.

| Component | Latency Share (Approx.) | Status |
| :--- | :--- | :--- |
| **ViT Encoder** | **92% - 94%** | **Critical Bottleneck** |
| Video Decoding | 18% | Acceptable |
| Projection / Merger | < 1% | Negligible |
| LLM Prefill | ~5% | Fast for short prompts |

### 2. Architecture Matters: Q-Former vs. MLP
For video inference tasks, **Q-Former (BLIP-2 style)** architectures are significantly faster than **MLP (LLaVA style)** architectures due to efficient token compression.

- **Speedup:** **9x faster** (923ms → 105ms).
- **Trade-off:** Q-Former sacrifices some spatial details but gains massive speed, making it ideal for edge video understanding.

### 3. Diminishing Returns of Frame Rate
Doubling input video frames (e.g., 4 → 8 frames) increases latency by **77%** but yields **negligible accuracy gains** on standard QA benchmarks. This suggests that **Dynamic Key-Frame Extraction** is a superior strategy to uniform sampling.

---

## Methodology & Features

### Phase 1: Fine-Grained Profiling (Completed)
We developed a non-intrusive profiling tool using **PyTorch Hooks** to dissect the inference timeline without modifying the model source code.

- **Hook-based Timing:** Measures exact execution time of `PatchEmbed`, `Attention`, `MLP`, and `Projector` layers.
- **Memory Tracking:** Monitors peak VRAM usage per module.
- **Supported Models:**
    - [x] Qwen2-VL-7B
    - [x] Phi-3.5-Vision
    - [x] LLaVA-v1.5
    - [x] BLIP-2

### Phase 2: Optimization (In Progress)
We are currently implementing the following optimizations to address the ViT bottleneck:

- [ ] **FlashAttention Integration:** Porting FlashAttention-2 to edge-compatible kernels to speed up ViT self-attention.
- [ ] **Token Pruning:** Implementing algorithms (like ToMe) to remove redundant visual tokens early in the forward pass.
- [ ] **Quantization:** 4-bit/8-bit quantization (AWQ/GPTQ) for running 7B models on 8GB VRAM devices.

---

## Quick Start

### Installation

```bash
git clone https://github.com/holEeast979/Edge-LMM-Accelerator.git
cd Edge-LMM-Accelerator
pip install -r requirements.txt
```

### Download Models & Data

```bash
# Generate data manifests
python tools/generate_manifests.py

# (Optional) prepare/verify video+audio assets
python tools/prepare_video_audio_data.py
```

### Run Experiments

```bash
# Run individual experiments
python exp/exp1_modality_bottleneck.py
python exp/exp2_projection_compare.py
python exp/exp3_frame_ablation.py
python exp/exp4_serial_vs_parallel.py
python exp/exp5_module_profiler.py
python exp/exp7_video_audio_encode.py
python exp/exp8_dual_gpu_parallel.py
python exp/exp9_audio_length_scaling.py
python exp/exp10_defect_verification.py
```

### Usage: Profiling an LMM

Use our profiler to analyze your own model's bottleneck:

```python
from edge_lmm.profiler import ModelProfiler
from transformers import Qwen2VLForConditionalGeneration

# Load Model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", 
    device_map="cuda"
)

# Attach Profiler
profiler = ModelProfiler(model)
profiler.start()

# Run Inference (Standard HuggingFace code)
output = model.generate(**inputs)

# Print Report
profiler.stop()
profiler.print_summary()
```

**Sample Output:**

```
[Profiler Report]
Total Latency: 1250ms
------------------------------------------------
| Layer Type    | Time (ms) | % Total |
|---------------|-----------|---------|
| ViT Encoder   | 1150.0    | 92.0%   |  <-- BOTTLENECK DETECTED
| Projector     | 5.0       | 0.4%    |
| LLM Decoder   | 95.0      | 7.6%    |
------------------------------------------------
```

---

## Project Structure

```
├── common.py                 # Shared utilities and model loaders
├── profiling_utils.py        # Shared profiling utilities (timers/monitors/hooks)
├── exp/                      # Experiment scripts
│   ├── exp1_modality_bottleneck.py   # Modality encoding bottleneck analysis
│   ├── exp2_projection_compare.py    # Q-Former vs MLP comparison
│   ├── exp3_frame_ablation.py        # Video frame count ablation
│   ├── exp4_serial_vs_parallel.py    # Serial vs parallel inference
│   ├── exp5_module_profiler.py       # Fine-grained module profiling
│   ├── exp7_video_audio_encode.py    # Video+Audio encode latency breakdown
│   ├── exp8_dual_gpu_parallel.py     # Dual-GPU parallel encode
│   ├── exp9_audio_length_scaling.py  # Audio length scaling
│   └── exp10_defect_verification.py  # Defect verification (padding waste + multiturn redundancy)
├── tools/                    # Data preparation and helper scripts
│   ├── generate_manifests.py
│   └── prepare_video_audio_data.py
└── docs/                     # Notes and documentation
     └── 12.16 日计划.md
```

---

## Roadmap

- [x] **Benchmark Platform:** Setup AutoDL/Colab environments for reproducible testing.
- [x] **Bottleneck Analysis:** Complete profiling for Qwen2-VL and Phi-3.5.
- [ ] **Optimization MVP:** Implement FlashAttention on Jetson Orin.
- [ ] **Library Release:** Package the profiler as a standalone pip tool.

---

## Author

**Haodong Zhang (HolEast)**

- Incoming MSc Student @ CUHK
- Focus: Edge AI, LLMOps, Applied LLMs
- [Portfolio & Resume](https://github.com/holEeast979)

---

## Contributing

This is an active research project. Feedback and discussions are welcome! If you are interested in Edge AI collaboration, feel free to reach out via email.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
