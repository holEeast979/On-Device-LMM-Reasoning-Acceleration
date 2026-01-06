# On-Device LMM Reasoning Acceleration

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Research-yellow)](https://github.com/holEeast979)

> **Accelerating Large Multimodal Models (LMMs) inference for video+audio understanding tasks.**

## Introduction

This project investigates inference optimization for **Omni-modal LMMs** (video + audio + text), focusing on identifying and addressing performance bottlenecks that existing SOTA methods have not covered.

**Current Model:** Qwen2.5-Omni-7B (will extend to smaller models to validate middleware generalization).

**Datasets:** ActivityNet-QA, Video-MME, AudioCaps.

---

## Key Findings (Motivation Experiments)

Our profiling on **Qwen2.5-Omni-7B** revealed several inefficiencies **not addressed by existing SOTA optimization methods**:

### 1. Audio Padding Waste
Whisper-based audio encoders pad all inputs to **30 seconds** (`padding=max_length`), causing significant compute waste for short audio clips.

| Audio Duration | Padding Overhead | Encoder Time |
| :--- | :--- | :--- |
| 5s | 6x | ~500ms |
| 10s | 3x | ~500ms |
| 30s | 1x (no waste) | ~500ms |

**Opportunity:** Dynamic padding (`do_not_pad`) reduces audio encoder latency proportionally to actual audio length.

### 2. Multi-turn KV Cache Not Reused
In multi-turn conversations on the **same video**, the visual/audio encoders and LLM prefill are **re-executed every turn**, with no KV cache reuse.

| Turn | Visual Encoder | Audio Encoder | LLM Prefill |
| :--- | :--- | :--- | :--- |
| Turn 1 | ✅ Run | ✅ Run | ✅ Run |
| Turn 2 | ❌ Re-run | ❌ Re-run | ❌ Re-run |

**Opportunity:** Cache visual/audio embeddings and KV states across turns.

### 3. Serial Encoding Bottleneck
Video and audio encoders run **serially**, missing parallelization opportunities on multi-GPU setups.

**Opportunity:** Parallel encoding on separate devices can reduce TTFT significantly.

---

## Methodology & Progress

### Phase 1: Motivation Experiments ✅ (Completed)
Developed profiling tools and benchmark framework to identify performance gaps in Omni-modal LMMs:

- **Unified Benchmark Framework:** Spec-based runner for reproducible experiments (`benchmark/run.py`)
- **TTFT Breakdown:** Decomposed Time-To-First-Token into visual/audio encoding, LLM prefill, and decode stages
- **Defect Verification:** Quantified audio padding waste and multi-turn redundancy

### Phase 2: Middleware Design (In Progress)
Designing a lightweight middleware layer to address identified inefficiencies:

- [ ] **Dynamic Audio Padding:** Adaptive padding based on actual audio duration
- [ ] **Cross-turn Caching:** Reuse visual/audio embeddings and KV cache across conversation turns
- [ ] **Parallel Encoding:** Offload video/audio encoders to separate devices

### Phase 3: Generalization Validation (Planned)
Validate middleware on smaller models (e.g., Qwen2.5-Omni-3B) to ensure generalization.

---

## Quick Start

### Installation

```bash
git clone https://github.com/holEeast979/On-Device-LMM-Reasoning-Acceleration.git
cd On-Device-LMM-Reasoning-Acceleration
pip install -r requirements.txt
```

### Download Models & Data

```bash
# Prepare datasets using dedicated scripts
# Video-MME (requires yt-dlp for YouTube downloads)
python tools/prepare_video_mme.py --out-root /root/autodl-tmp/data --max-samples 100 --validate

# ActivityNet-QA
python tools/generate_activitynet_manifest.py

# AudioCaps (requires audiocaps-download package)
python tools/prepare_audiocaps.py --out-root /root/autodl-tmp/data --max-samples 100 --validate
```

**Dataset Statistics** (current setup):
| Dataset | Videos/Audios | QA Pairs | Size |
|---------|---------------|----------|------|
| Video-MME | 100 videos | 2700 QA | ~12GB |
| ActivityNet-QA | 103 videos | - | ~1.7GB |
| AudioCaps | 100 audios | - | ~120MB |

### Run Experiments

```bash
# Unified benchmark runner (spec-based)
python benchmark/run.py audio-padding --manifest /root/autodl-tmp/data/video_mme/manifest.csv
python benchmark/run.py multiturn --manifest /root/autodl-tmp/data/video_mme/manifest.csv
python benchmark/run.py token-prefill --manifest /root/autodl-tmp/data/video_mme/manifest.csv

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

Use our profiler to analyze Qwen2.5-Omni's bottleneck:

```python
from profiling_utils import TorchCudaMemPeakMonitor, Timer
import common as C

# Load Model
model, processor = C.load_qwen25_omni("/path/to/Qwen2.5-Omni-7B", dtype="bf16")

# Attach memory monitor
mem_monitor = TorchCudaMemPeakMonitor()
mem_monitor.start()

# Run inference with timing
with Timer("generate"):
    output = model.generate(**inputs)

# Get memory stats
mem_monitor.stop()
print(f"Peak GPU memory: {mem_monitor.peak_allocated_bytes / 1e9:.2f} GB")
```

**Sample TTFT Breakdown Output:**

```
[TTFT Breakdown]
Total TTFT: 1850ms
------------------------------------------------
| Stage              | Time (ms) | % Total |
|--------------------|-----------|---------|
| Video Encoder      | 850.0     | 45.9%   |
| Audio Encoder      | 520.0     | 28.1%   |  <-- PADDING WASTE
| LLM Prefill        | 450.0     | 24.3%   |
| First Token Decode | 30.0      | 1.6%    |
------------------------------------------------
```

---

## Project Structure

```
├── benchmark/                # Unified benchmark framework (runner + specs)
│   ├── run.py                # CLI entry
│   ├── runner.py             # Shared runner utilities
│   └── specs/                # Spec-based experiments (audio-padding/multiturn/token-prefill)
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
│   ├── prepare_video_mme.py
│   ├── prepare_audiocaps.py
│   ├── prepare_msvd_qa.py
│   └── datasets/             # Manifest utilities
└── docs/                     # Documentation
    ├── benchmark-framework.md
    ├── dataset-tools.md
    └── experiments.md
```

---

## Roadmap

- [x] **Motivation Experiments:** Identify inefficiencies not covered by existing SOTA methods
- [x] **Benchmark Framework:** Unified spec-based runner for reproducible experiments
- [x] **Dataset Preparation:** ActivityNet-QA, Video-MME, AudioCaps pipelines
- [ ] **Middleware Implementation:** Dynamic padding, cross-turn caching, parallel encoding
- [ ] **Generalization Validation:** Test on smaller models (3B, 1B)

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
