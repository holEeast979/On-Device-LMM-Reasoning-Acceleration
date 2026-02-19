#!/bin/bash
# Phase 2 P0 实验串行脚本
# 按优先级顺序自动执行，跑完一个接下一个
# 用法: bash run_phase2_experiments.sh

set -e
cd /root/scripts

echo "========================================"
echo "Phase 2 实验开始: $(date)"
echo "========================================"

# ---- 实验 1: Naive Baselines 全量（P0 #1，Short 108 题）----
echo ""
echo "[1/3] Naive Baselines 全量对比 (Short, kr=0.5)"
echo "预计时间: ~2h"
python fasteromni/eval_videomme.py \
    --duration short \
    --keep-ratio 0.5 \
    --modes baseline sparse naive_uniform naive_random naive_iframe \
    --out-dir /root/autodl-tmp/results/fasteromni/naive_comparison
echo "[1/3] 完成: $(date)"

# ---- 实验 2: Sparse@64 vs Baseline@64（P0 #3，Short 108 题）----
echo ""
echo "[2/3] Sparse@64 vs Baseline@64 (Short, kr=0.5, max-frames=64)"
echo "预计时间: ~1h"
python fasteromni/eval_videomme.py \
    --duration short \
    --keep-ratio 0.5 \
    --modes baseline sparse \
    --max-frames 64 \
    --out-dir /root/autodl-tmp/results/fasteromni/sparse64
echo "[2/3] 完成: $(date)"

# ---- 实验 3: Naive Baselines kr=0.2（验证极端稀疏）----
echo ""
echo "[3/3] Naive Baselines 极端稀疏对比 (Short, kr=0.2)"
echo "预计时间: ~2h"
python fasteromni/eval_videomme.py \
    --duration short \
    --keep-ratio 0.2 \
    --modes sparse naive_uniform naive_random naive_iframe \
    --out-dir /root/autodl-tmp/results/fasteromni/naive_comparison_kr02
echo "[3/3] 完成: $(date)"

echo ""
echo "========================================"
echo "所有实验完成: $(date)"
echo "结果目录:"
echo "  /root/autodl-tmp/results/fasteromni/naive_comparison"
echo "  /root/autodl-tmp/results/fasteromni/sparse64"
echo "  /root/autodl-tmp/results/fasteromni/naive_comparison_kr02"
echo "========================================"
