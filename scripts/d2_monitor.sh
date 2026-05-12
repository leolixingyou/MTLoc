#!/bin/bash
# Monitor D.2 training, run eval when done, output results
cd /workspace/MTLoc

# Wait for all training to finish
while ps aux | grep "train_kitti.*d2_kitti" | grep -v grep > /dev/null; do
    sleep 300  # check every 5 min
done

echo "=== D.2 TRAINING COMPLETE ==="
date

# Find best checkpoints
for v in B C D; do
    BEST=$(ls runs/d2_kitti_${v}_depth10/checkpoints/*.ckpt 2>/dev/null | grep -v last | sort -t'-' -k3 | head -1)
    echo "D.2 $v best: $BEST"
    
    if [ -n "$BEST" ]; then
        GPU=$(($(echo $v | tr 'BCD' '012')))
        CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_kitti.py \
            --adapter_ckpt "$BEST" \
            --max_num_val 3773 \
            --save_per_sample "runs/d1_eval_results/d2_${v}_per_sample.npz" \
            --coarse_to_fine \
            --device cuda:0 \
            > "runs/d1_eval_results/eval_d2_${v}.log" 2>&1 &
    fi
done

# Wait for evals
wait

echo "=== D.2 EVAL COMPLETE ==="
for v in B C D; do
    echo "=== D.2 $v ==="
    tail -c 2000 "runs/d1_eval_results/eval_d2_${v}.log" | tr '\r' '\n' | grep -E "(lateral|longitudinal|yaw|xy |per-sample)"
done
