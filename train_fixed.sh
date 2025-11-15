#!/bin/bash

# Agricultural Disease Recognition - FIXED Training Script
# ============================================================
# "Talk is cheap. Show me the code." - Linus Torvalds
#
# 真正的问题诊断:
# ❌ 不是学习率问题 (6 epoch就稳定了)
# ✅ 是数据问题: 极端类别不平衡 (2445:1)
#
# 修复方案:
# 1. 更强的数据增强 (已更新dataset.py)
# 2. 更温和的class weights (sqrt smoothing: 49x vs 2445x)
# 3. 更高的图像分辨率 (224 → 320)
# 4. 两阶段训练: 先训练头,再fine-tune全部

echo "============================================================"
echo "Training Task 1: 61-class Disease Classification (FIXED)"
echo "============================================================"
echo ""
echo "问题诊断:"
echo "  ❌ 6 epoch就稳定 = 快速过拟合major classes"
echo "  ❌ Class 44/45只有1个训练样本"
echo "  ❌ 原class weights: 2445:1 (灾难性)"
echo ""
echo "修复方案:"
echo "  ✅ 数据增强: 显著增强 (旋转45°, 更多color jitter, cutout等)"
echo "  ✅ Class weights: sqrt smoothing (49:1, 更温和)"
echo "  ✅ 图像分辨率: 224 → 320 (捕捉更多细节)"
echo "  ✅ 训练策略: 两阶段训练"
echo ""
echo "预期改进:"
echo "  Before: ~30% accuracy (过早收敛)"
echo "  After:  60-75% accuracy (proper learning)"
echo ""
echo "============================================================"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Stage 1: Train classifier head only (10 epochs)
echo ""
echo "============================================================"
echo "STAGE 1: Train Classifier Head (Backbone Frozen)"
echo "============================================================"
echo "Strategy: Quick adaptation of classification head"
echo "Epochs: 10"
echo "LR: 1e-3 (higher for head)"
echo ""

python train.py \
    --model-type baseline \
    --backbone resnet50 \
    --pretrained \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --loss-type weighted_ce \
    --class-weights data/cleaned/metadata/class_weights_sqrt.csv \
    --label-smoothing 0.1 \
    --dropout 0.5 \
    --image-size 320 \
    --use-amp \
    --num-workers 4 \
    --save-dir checkpoints/task1_stage1 \
    --save-freq 5 \
    --log-interval 10 \
    --seed 42

echo ""
echo "============================================================"
echo "STAGE 2: Fine-tune Entire Model"
echo "============================================================"
echo "Strategy: Unfreeze backbone, fine-tune end-to-end"
echo "Epochs: 40"
echo "LR: 3e-4 (lower for fine-tuning)"
echo "Resume from: Stage 1 best checkpoint"
echo ""

python train.py \
    --model-type baseline \
    --backbone resnet50 \
    --pretrained \
    --epochs 40 \
    --batch-size 24 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --loss-type weighted_ce \
    --class-weights data/cleaned/metadata/class_weights_sqrt.csv \
    --label-smoothing 0.1 \
    --dropout 0.4 \
    --image-size 320 \
    --use-amp \
    --num-workers 4 \
    --save-dir checkpoints/task1_stage2 \
    --save-freq 5 \
    --log-interval 10 \
    --resume checkpoints/task1_stage1/best.pth \
    --seed 42

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Stage 1: checkpoints/task1_stage1/"
echo "  Stage 2: checkpoints/task1_stage2/"
echo ""
echo "Visualize:"
echo "  python visualize_training.py --checkpoint-dir checkpoints/task1_stage2/"
echo ""
echo "Compare stages:"
echo "  python visualize_training.py --compare \\"
echo "    checkpoints/task1_stage1/best.pth \\"
echo "    checkpoints/task1_stage2/best.pth"
echo ""
