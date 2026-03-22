#\!/bin/bash
source ~/.bashrc
cd /root/scripts

echo "=== Starting re-run at $(date) ==="

echo ""
echo "[1/3] ActivityNet baseline (est ~100 min)"
python3 fasteromni/eval_activitynet.py \
  --modes baseline \
  --keep-ratio 0.5 \
  --max-frames 32 \
  --out-dir /root/autodl-tmp/results/fasteromni/activitynet \
  2>&1 | tee /root/autodl-tmp/results/fasteromni/activitynet/rerun_baseline.log

echo ""
echo "[2/3] Video-MME baseline (est ~11 min)"
python3 fasteromni/eval_videomme.py \
  --modes baseline \
  --keep-ratio 0.5 \
  --max-frames 32 \
  --duration short \
  --out-dir /root/autodl-tmp/results/fasteromni/videomme \
  2>&1 | tee /root/autodl-tmp/results/fasteromni/videomme/rerun_baseline.log

echo ""
echo "[3/3] Video-MME adaptive (est ~7 min)"
python3 fasteromni/eval_videomme.py \
  --modes adaptive \
  --keep-ratio 0.5 \
  --max-frames 32 \
  --duration short \
  --out-dir /root/autodl-tmp/results/fasteromni/videomme \
  2>&1 | tee /root/autodl-tmp/results/fasteromni/videomme/rerun_adaptive.log

echo ""
echo "=== All done at $(date) ==="
