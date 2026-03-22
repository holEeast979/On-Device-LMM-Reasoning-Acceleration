# FasterOmni Experiments

Profiling experiments for identifying performance bottlenecks in Qwen2.5-Omni.

## Experiment List

| Exp | Script | Purpose |
|-----|--------|---------|
| 1 | `exp1_serial_vs_parallel.py` | Serial vs parallel encoding comparison |
| 2 | `exp2_ttft_breakdown.py` | **TTFT breakdown analysis (core)** - decompose latency |
| 3 | `exp3_dual_gpu_parallel.py` | Dual-GPU parallel encoding |
| 4 | `exp4_defect_verification.py` | **Defect verification (core)** - audio padding + multi-turn |

## Running Experiments

```bash
# Run individual experiments
python exp/exp2_ttft_breakdown.py          # Core: TTFT breakdown
python exp/exp4_defect_verification.py     # Core: Defect verification

# Use unified benchmark framework (recommended)
python benchmark/run.py audio-padding --manifest /path/to/manifest.csv
python benchmark/run.py multiturn --manifest /path/to/manifest.csv
python benchmark/run.py ttft-breakdown --manifest /path/to/manifest.csv
```

## Output

Results are saved to `/root/autodl-tmp/results/`:

- `*_results.csv`: Raw measurement data
- `*_summary.csv`: Statistical summary
- `*.png`: Visualization charts
- `*.json`: Structured data

## Key Findings

### Audio Padding "Resource Lock" (Exp4)

- `padding=max_length` pads all audio to 30s, causing ~6x compute waste for short clips
- `do_not_pad` reduces audio encoder latency proportionally to actual audio length
- Short audio clips consume ~30-40MB extra VRAM purely due to padding

### Multi-turn "Groundhog Day" (Exp4)

- Same video multi-turn: visual/audio encoders re-run every turn
- No KV cache reuse, causing redundant prefill computation
- Turn 2 latency ≈ Turn 1 latency (no speedup)
