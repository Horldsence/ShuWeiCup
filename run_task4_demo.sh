#!/bin/bash
# Task 4: Inference Demo - Run predictions on 10 sample images
# ============================================================

set -e  # Exit on error

# Configuration
CHECKPOINT="checkpoints/task4_multitask/multitask/best.pth"
VAL_META="data/cleaned/metadata/val_metadata.csv"
VAL_DIR="data/cleaned/val"
OUT_DIR="outputs/task4_inference_demo"
NUM_SAMPLES=10
SEED=42

echo "========================================"
echo "Task 4: Inference Demo"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Checkpoint:   $CHECKPOINT"
echo "  Val metadata: $VAL_META"
echo "  Val dir:      $VAL_DIR"
echo "  Output dir:   $OUT_DIR"
echo "  Num samples:  $NUM_SAMPLES"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please ensure training has completed successfully."
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Run inference demo
echo "Running inference on $NUM_SAMPLES random samples..."
echo ""

python task4_inference_demo.py \
    --checkpoint "$CHECKPOINT" \
    --val-meta "$VAL_META" \
    --val-dir "$VAL_DIR" \
    --out-dir "$OUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --use-cam

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Inference demo completed!"
    echo "========================================"
    echo ""
    echo "Output images saved to: $OUT_DIR"
    echo ""
    echo "Generated images:"
    ls -lh "$OUT_DIR"/*.jpg 2>/dev/null | head -15
    echo ""
    echo "Each image shows:"
    echo "  - Left: Original image"
    echo "  - Right: Grad-CAM heatmap overlay"
    echo "  - Bottom: Predictions for all tasks with confidence scores"
    echo ""
else
    echo ""
    echo "========================================"
    echo "❌ Inference demo failed with exit code: $EXIT_CODE"
    echo "========================================"
    exit $EXIT_CODE
fi
