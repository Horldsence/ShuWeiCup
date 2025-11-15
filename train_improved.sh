#!/bin/bash

# Agricultural Disease Recognition - Improved Training Script
# ============================================================
# "Talk is cheap. Show me the code." - Linus Torvalds
#
# This script runs training with FIXED hyperparameters:
# - Learning rate: 5e-4 (up from 1e-4)
# - Batch size: 32 (down from 64)
# - Warmup: 5 epochs (new)
# - Cosine annealing after warmup (fixed)

echo "============================================================"
echo "Training Task 1: 61-class Disease Classification"
echo "============================================================"
echo ""
echo "Key improvements:"
echo "  ✅ Learning rate: 1e-4 → 5e-4 (5x increase)"
echo "  ✅ Batch size: 64 → 32 (better gradient signal)"
echo "  ✅ Warmup: 0 → 5 epochs (prevent feature destruction)"
echo "  ✅ Scheduler: Fixed cosine to start AFTER warmup"
echo ""
echo "Expected results:"
echo "  - Baseline (old): ~27.6% accuracy"
echo "  - Improved (new): >70% accuracy (target: 80%+)"
echo ""
echo "============================================================"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run training with improved hyperparameters
python train.py \
    --model-type baseline \
    --backbone resnet50 \
    --pretrained \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --loss-type weighted_ce \
    --use-class-weights \
    --label-smoothing 0.1 \
    --dropout 0.3 \
    --image-size 224 \
    --use-amp \
    --compile \
    --num-workers 4 \
    --save-dir checkpoints/task1_improved \
    --save-freq 5 \
    --log-interval 10 \
    --seed 42

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Check results: ls -lh checkpoints/task1_improved/"
echo "  2. View TensorBoard: tensorboard --logdir checkpoints/task1_improved/logs"
echo "  3. Compare with baseline: grep 'Best' checkpoints/*/best.pth"
echo ""
