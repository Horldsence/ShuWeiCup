#!/bin/bash

# Agricultural Disease Recognition - FAST Training Script
# ============================================================
# "Talk is cheap. Show me the code." - Linus Torvalds
#
# 快速训练版本:
# - 使用balanced dataset (10,837 vs 31,541 samples)
# - 训练速度提升 ~3x per epoch
# - 使用优化后的超参数和class weights
#
# 预期时间:
#   原始: ~2小时/epoch × 50 epochs = 100小时
#   快速: ~40分钟/epoch × 50 epochs = 33小时
#   节省: ~67小时 (67% faster)

echo "============================================================"
echo "Fast Training: 61-class Disease Classification"
echo "============================================================"
echo ""
echo "数据集:"
echo "  原始训练集: 31,541 samples"
echo "  精简训练集: 10,837 samples (每类最多200张)"
echo "  验证集: 4,533 samples (不变)"
echo ""
echo "训练策略:"
echo "  ✅ Balanced dataset (3x faster per epoch)"
echo "  ✅ Sqrt-smoothed class weights (49:1)"
echo "  ✅ Strong data augmentation"
echo "  ✅ Higher resolution (320x320)"
echo "  ✅ Optimized hyperparameters"
echo ""
echo "预期结果:"
echo "  训练时间: ~33小时 (vs 100小时原始)"
echo "  准确率: 60-70% (可能略低于完整数据集)"
echo "  适用场景: 快速验证、超参数搜索、原型开发"
echo ""
echo "============================================================"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if balanced dataset exists
if [ ! -f "data/cleaned/metadata/train_metadata_balanced.csv" ]; then
    echo "⚠️  Balanced dataset not found. Creating..."
    python create_balanced_dataset.py
    echo ""
fi

# Run fast training
echo "============================================================"
echo "Starting Fast Training"
echo "============================================================"
echo ""

python train.py \
    --model-type baseline \
    --backbone resnet50 \
    --pretrained \
    \
    --train-meta data/cleaned/metadata/train_metadata_balanced.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --class-weights data/cleaned/metadata/class_weights_sqrt.csv \
    \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    \
    --optimizer adamw \
    --scheduler cosine \
    --loss-type weighted_ce \
    --label-smoothing 0.1 \
    \
    --dropout 0.4 \
    --image-size 320 \
    \
    --use-amp \
    --compile \
    --num-workers 4 \
    \
    --save-dir checkpoints/task1_fast \
    --save-freq 5 \
    --log-interval 10 \
    --seed 42

echo ""
echo "============================================================"
echo "Fast Training completed!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Checkpoints: checkpoints/task1_fast/"
echo "  Training curves: checkpoints/task1_fast/training_curves.png"
echo ""
echo "Visualize:"
echo "  python visualize_training.py --checkpoint-dir checkpoints/task1_fast/"
echo ""
echo "Compare with full dataset:"
echo "  python visualize_training.py --compare \\"
echo "    checkpoints/task1_fast/best.pth \\"
echo "    checkpoints/task1_stage2/best.pth"
echo ""
echo "Training stats:"
echo "  Samples per epoch: 10,837 (vs 31,541)"
echo "  Speedup: ~3x per epoch"
echo "  Total time saved: ~67 hours"
echo ""
