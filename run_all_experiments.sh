#!/bin/bash
# ============================================================
# FasterOmni 全量实验脚本
# 
# 包含：
#   Part 1: MVBench 全量（baseline + sparse + naive_iframe × 3600 题）
#   Part 2: Pareto 补数据（naive_iframe kr sweep × Video-MME Short 108 题）
#
# 运行方式：
#   tmux new -s experiments
#   bash run_all_experiments.sh 2>&1 | tee /root/autodl-tmp/results/fasteromni/experiment_log_$(date +%Y%m%d_%H%M).log
#
# 数据安全：所有输出到新目录，不覆盖已有数据
# 断点恢复：增量 CSV 机制，中断后重跑会自动跳过已完成的题
# ============================================================

set -e
cd /root/scripts

RESULTS_BASE="/root/autodl-tmp/results/fasteromni"
TIMESTAMP=$(date +%Y%m%d_%H%M)

echo "============================================================"
echo " FasterOmni 全量实验"
echo " 开始时间: $(date)"
echo " 结果目录: ${RESULTS_BASE}"
echo "============================================================"

# ──────────────────────────────────────────────────────────────
# Part 1: MVBench 全量
# 3 modes × 3600 题 ≈ 10,800 次推理
# 预计 ~4 小时
# ──────────────────────────────────────────────────────────────

MVBENCH_DIR="${RESULTS_BASE}/mvbench"
echo ""
echo "============================================================"
echo " Part 1: MVBench 全量评估"
echo " modes: baseline, sparse(kr=0.5), naive_iframe(kr=0.5)"
echo " 输出: ${MVBENCH_DIR}"
echo " 预计: ~4 小时 (10,800 次推理)"
echo "============================================================"

python fasteromni/eval_mvbench.py \
    --modes baseline sparse naive_iframe \
    --keep-ratio 0.5 \
    --max-frames 0 \
    --out-dir "${MVBENCH_DIR}"

echo ""
echo "[Part 1 完成] $(date)"
echo ""

# ──────────────────────────────────────────────────────────────
# Part 2: Pareto 补数据 - naive_iframe kr sweep
# Video-MME Short only, 108 题 × 5 个 kr 值
# 预计 ~1.5 小时
# ──────────────────────────────────────────────────────────────

PARETO_DIR="${RESULTS_BASE}/pareto_naive_iframe"
mkdir -p "${PARETO_DIR}"

echo "============================================================"
echo " Part 2: Pareto 补数据 (naive_iframe kr sweep)"
echo " kr values: 0.2, 0.3, 0.5, 0.7, 0.9"
echo " Video-MME Short only (108 题 × 5 = 540 次推理)"
echo " 输出: ${PARETO_DIR}"
echo " 预计: ~1.5 小时"
echo "============================================================"

for kr in 0.2 0.3 0.5 0.7 0.9; do
    echo ""
    echo "--- naive_iframe kr=${kr} ---"
    
    KR_DIR="${PARETO_DIR}/naive_iframe_kr${kr}"
    mkdir -p "${KR_DIR}"
    
    python fasteromni/eval_videomme.py \
        --modes naive_iframe \
        --keep-ratio ${kr} \
        --duration short \
        --max-frames 32 \
        --out-dir "${KR_DIR}"
    
    echo "[naive_iframe kr=${kr} 完成] $(date)"
done

echo ""
echo "============================================================"
echo " 全部实验完成!"
echo " 结束时间: $(date)"
echo " MVBench 结果: ${MVBENCH_DIR}"
echo " Pareto 结果:  ${PARETO_DIR}"
echo "============================================================"

# 汇总
echo ""
echo "=== 磁盘使用 ==="
du -sh ${MVBENCH_DIR} ${PARETO_DIR}
df -h /root/autodl-tmp | tail -1
