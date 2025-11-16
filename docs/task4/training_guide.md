# Task 4 训练脚本使用指南

## 快速开始

### 基础训练命令

```bash
python task4train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --epochs 25 \
    --batch-size 64 \
    --lr 3e-4
```

---

## 命令行参数详解

### 数据路径参数

```bash
--train-meta TRAIN_META
    训练集元数据CSV文件路径
    默认: data/cleaned/metadata/train_metadata.csv
    
--val-meta VAL_META
    验证集元数据CSV文件路径
    默认: data/cleaned/metadata/val_metadata.csv
    
--train-dir TRAIN_DIR
    训练图像根目录
    默认: data/cleaned/train
    
--val-dir VAL_DIR
    验证图像根目录
    默认: data/cleaned/val
```

**示例**：
```bash
--train-meta data/custom/train.csv \
--val-meta data/custom/val.csv \
--train-dir data/custom/images/train \
--val-dir data/custom/images/val
```

---

### 模型架构参数

```bash
--backbone BACKBONE
    骨干网络名称（timm支持的任意模型）
    可选值: resnet50, resnet101, efficientnet_b0~b7, convnext_tiny/small/base
    默认: resnet50
    
--pretrained
    使用ImageNet预训练权重
    默认: True
    
--dropout DROPOUT
    Dropout比率
    范围: [0.1, 0.5]
    默认: 0.3
```

**示例**：
```bash
# ResNet101 + 高Dropout
--backbone resnet101 --dropout 0.4

# EfficientNet-B3 + 预训练
--backbone efficientnet_b3 --pretrained

# ConvNeXt Tiny（参数量小）
--backbone convnext_tiny --dropout 0.25
```

---

### 训练超参数

```bash
--epochs EPOCHS
    训练总轮数
    推荐: 20-50
    默认: 25
    
--batch-size BATCH_SIZE
    批次大小
    根据GPU内存调整
    默认: 64
    
--lr LR
    学习率
    范围: [1e-5, 1e-3]
    默认: 3e-4
    
--weight-decay WEIGHT_DECAY
    权重衰减（L2正则化）
    范围: [1e-5, 1e-3]
    默认: 1e-4
    
--optimizer OPTIMIZER
    优化器类型
    可选值: adam, adamw, sgd
    默认: adamw
    
--scheduler SCHEDULER
    学习率调度器
    可选值: cosine, step, onecycle, reduce_on_plateau
    默认: cosine
```

**示例**：
```bash
# 快速原型（少轮次）
--epochs 15 --batch-size 128 --lr 5e-4

# 精细训练（多轮次 + 小学习率）
--epochs 50 --batch-size 32 --lr 1e-4 --scheduler onecycle

# SGD + Momentum
--optimizer sgd --lr 1e-2 --weight-decay 5e-4
```

---

### 任务权重参数

#### 静态权重模式

```bash
--task-weights WEIGHTS
    手动指定四个任务的权重（逗号分隔）
    顺序: label_61, crop, disease, severity
    格式: "w1,w2,w3,w4"
    默认: "1.0,0.3,0.3,0.4"
```

**示例**：
```bash
# 标准配置（主任务主导）
--task-weights "1.0,0.3,0.3,0.4"

# 平衡配置（所有任务等权）
--task-weights "1.0,1.0,1.0,1.0"

# 严重度优先（强调诊断）
--task-weights "1.0,0.2,0.2,0.8"

# 只训练主任务（等价单任务）
--task-weights "1.0,0.0,0.0,0.0"
```

#### 动态权重模式

```bash
--dynamic-task-weights
    启用基于验证损失的动态权重调整
    与 --task-weights 互斥（动态模式优先）
    
--weight-update-interval INTERVAL
    动态权重更新间隔（轮数）
    默认: 1
    
--weight-smoothing ALPHA
    权重EMA平滑系数
    范围: [0.5, 0.95]
    默认: 0.8
```

**示例**：
```bash
# 启用动态权重（推荐）
--dynamic-task-weights

# 自定义更新策略
--dynamic-task-weights \
--weight-update-interval 2 \
--weight-smoothing 0.9
```

---

### 数据增强参数

```bash
--image-size IMAGE_SIZE
    输入图像尺寸（正方形）
    常用值: 224, 256, 320, 384
    默认: 224
    
--augment-strength STRENGTH
    数据增强强度
    可选值: light, medium, strong
    默认: medium
    
--mixup-alpha ALPHA
    Mixup增强的alpha参数
    设为0禁用Mixup
    范围: [0.0, 1.0]
    默认: 0.2
    
--cutmix-prob PROB
    CutMix应用概率
    范围: [0.0, 1.0]
    默认: 0.0（未启用）
```

**示例**：
```bash
# 高分辨率 + 强增强
--image-size 384 --augment-strength strong

# 禁用Mixup
--mixup-alpha 0.0

# 启用CutMix
--mixup-alpha 0.2 --cutmix-prob 0.5
```

---

### 诊断报告参数

```bash
--report-samples NUM
    生成诊断报告的样本数
    默认: 50
    
--report-topk K
    报告中包含的Top-K预测数
    默认: 5
    
--include-cam
    在报告中包含Grad-CAM可视化
    默认: False
    
--cam-samples NUM
    单独生成的Grad-CAM可视化样本数
    默认: 12
```

**示例**：
```bash
# 生成完整诊断报告（包含CAM）
--report-samples 100 --report-topk 10 --include-cam

# 快速诊断（无可视化）
--report-samples 20 --report-topk 3

# 大量CAM样本（调试用）
--cam-samples 50
```

---

### 协同效应对比参数

```bash
--compare-synergy
    启用多任务 vs 单任务协同效应对比实验
    
--compare-epochs EPOCHS
    单任务对比模型训练轮数
    默认: 8（快速对比）
    
--compare-metric METRIC
    对比的主要指标
    可选值: accuracy, macro_f1, per_class_recall
    默认: macro_f1
```

**示例**：
```bash
# 完整对比实验
--compare-synergy --compare-epochs 20

# 快速验证（少轮次）
--compare-synergy --compare-epochs 5 --compare-metric accuracy
```

---

### 输出与日志参数

```bash
--out-dir OUT_DIR
    输出目录（模型、日志、报告）
    默认: checkpoints/task4_multitask
    
--save-interval INTERVAL
    模型保存间隔（轮数）
    默认: 5
    
--log-interval INTERVAL
    日志打印间隔（训练步数）
    默认: 10
    
--tensorboard
    启用TensorBoard日志
    默认: False
    
--seed SEED
    随机种子（可复现性）
    默认: 42
```

**示例**：
```bash
# 自定义输出路径
--out-dir experiments/exp001_resnet101

# 频繁保存检查点
--save-interval 2

# 启用TensorBoard
--tensorboard --out-dir runs/task4_tb
```

---

## 完整使用示例

### 示例1：标准多任务训练

```bash
python task4train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone resnet50 \
    --epochs 25 \
    --batch-size 64 \
    --lr 3e-4 \
    --task-weights "1.0,0.3,0.3,0.4" \
    --out-dir checkpoints/task4_standard
```

### 示例2：动态权重 + 高分辨率

```bash
python task4train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone efficientnet_b3 \
    --epochs 30 \
    --batch-size 48 \
    --lr 2e-4 \
    --image-size 320 \
    --dynamic-task-weights \
    --augment-strength strong \
    --out-dir checkpoints/task4_dynamic
```

### 示例3：完整诊断报告生成

```bash
python task4train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --epochs 25 \
    --batch-size 64 \
    --lr 3e-4 \
    --report-samples 100 \
    --report-topk 10 \
    --include-cam \
    --cam-samples 30 \
    --out-dir checkpoints/task4_diagnosis
```

### 示例4：协同效应验证

```bash
python task4train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --epochs 25 \
    --batch-size 64 \
    --lr 3e-4 \
    --compare-synergy \
    --compare-epochs 15 \
    --compare-metric macro_f1 \
    --out-dir checkpoints/task4_synergy
```

### 示例5：快速原型验证

```bash
python task4train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone resnet18 \
    --epochs 10 \
    --batch-size 128 \
    --lr 5e-4 \
    --image-size 224 \
    --report-samples 10 \
    --out-dir checkpoints/task4_fast_test
```

### 示例6：大模型 + TensorBoard

```bash
python task4train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone resnet101 \
    --epochs 40 \
    --batch-size 32 \
    --lr 1e-4 \
    --image-size 384 \
    --weight-decay 5e-5 \
    --dynamic-task-weights \
    --tensorboard \
    --out-dir checkpoints/task4_large_model
```

---

## 输出文件结构

训练完成后，`--out-dir` 目录包含以下文件：

```
checkpoints/task4_multitask/
├── models/
│   ├── best_model.pth              # 最佳验证性能模型
│   ├── last_model.pth              # 最后一轮模型
│   └── checkpoint_epoch_N.pth      # 定期检查点
│
├── logs/
│   ├── training.log                # 详细训练日志
│   ├── metrics_history.csv         # 每轮指标CSV
│   └── config.json                 # 训练配置备份
│
├── reports/
│   ├── diagnostic_reports.json     # 诊断报告JSON
│   ├── diagnostic_reports.csv      # 诊断报告CSV
│   └── report_summary.txt          # 报告摘要
│
├── visualizations/
│   ├── training_curves.png         # 训练曲线图
│   ├── task_losses.png             # 各任务损失曲线
│   ├── task_weights.png            # 动态权重变化（如启用）
│   ├── confusion_matrix_61.png     # 61类混淆矩阵
│   ├── confusion_matrix_severity.png # 严重度混淆矩阵
│   └── per_class_metrics.png       # 每类性能柱状图
│
├── gradcam/
│   ├── sample_0001_class12_sev1_correct.png
│   ├── sample_0002_class34_sev2_wrong.png
│   └── ...
│
└── synergy_comparison/             # 如启用 --compare-synergy
    ├── comparison_report.txt
    ├── comparison_metrics.csv
    ├── mtl_vs_stl.png
    └── single_task_severity_model.pth
```

---

## 诊断报告格式

### JSON格式

```json
{
  "metadata": {
    "model": "resnet50",
    "num_samples": 50,
    "timestamp": "2024-01-15T10:30:00"
  },
  "samples": [
    {
      "image_name": "image_001.jpg",
      "image_path": "data/cleaned/val/class_12/image_001.jpg",
      
      "label_61": {
        "predicted": 12,
        "true": 12,
        "class_name": "Tomato___Late_blight",
        "confidence": 0.923,
        "correct": true,
        "top5": [
          {"rank": 1, "class": 12, "name": "Tomato___Late_blight", "prob": 0.923},
          {"rank": 2, "class": 13, "name": "Tomato___Leaf_Mold", "prob": 0.045},
          {"rank": 3, "class": 15, "name": "Tomato___Septoria_leaf_spot", "prob": 0.018},
          {"rank": 4, "class": 11, "name": "Tomato___Early_blight", "prob": 0.009},
          {"rank": 5, "class": 14, "name": "Tomato___Bacterial_spot", "prob": 0.003}
        ]
      },
      
      "crop": {
        "predicted": 9,
        "true": 9,
        "crop_name": "Tomato",
        "confidence": 0.987,
        "correct": true
      },
      
      "disease": {
        "predicted": 2,
        "true": 2,
        "disease_name": "Late_blight",
        "confidence": 0.894,
        "correct": true
      },
      
      "severity": {
        "predicted": 2,
        "true": 2,
        "severity_name": "Serious",
        "confidence": 0.856,
        "correct": true
      },
      
      "diagnosis_summary": "诊断结果：检测到番茄的晚疫病，严重程度为严重。置信度：92.3%。请立即采取防治措施！",
      "cam_path": "gradcam/sample_0001_class12_sev2_correct.png"
    }
  ],
  
  "overall_statistics": {
    "accuracy_61": 0.874,
    "accuracy_crop": 0.956,
    "accuracy_disease": 0.891,
    "accuracy_severity": 0.823,
    "macro_f1_61": 0.861,
    "macro_f1_severity": 0.810
  }
}
```

### CSV格式

```csv
image_name,pred_61,true_61,conf_61,correct_61,pred_crop,true_crop,pred_disease,true_disease,pred_severity,true_severity,conf_severity,diagnosis
image_001.jpg,12,12,0.923,True,9,9,2,2,2,2,0.856,"番茄晚疫病，严重"
image_002.jpg,34,34,0.867,True,5,5,1,1,1,1,0.745,"玉米锈病，一般"
...
```

---

## TensorBoard可视化

如果启用了 `--tensorboard`，可以查看实时训练曲线：

```bash
tensorboard --logdir checkpoints/task4_multitask/runs
```

访问 `http://localhost:6006` 查看：

- **Scalars**：
  - `Loss/train_total`：总训练损失
  - `Loss/train_61`：61类任务训练损失
  - `Loss/train_severity`：严重度任务训练损失
  - `Accuracy/val_61`：61类验证准确率
  - `MacroF1/val_severity`：严重度验证Macro-F1
  - `Weights/task_*`：动态权重变化（如启用）

- **Histograms**：
  - 各层参数分布
  - 梯度分布

- **Images**（如启用）：
  - 训练样本可视化
  - Grad-CAM热力图

---

## 性能调优建议

### 针对不同GPU内存

#### 高端GPU（>= 24GB）
```bash
--backbone resnet101 \
--batch-size 128 \
--image-size 384 \
--num-workers 8
```

#### 中端GPU（12-16GB）
```bash
--backbone resnet50 \
--batch-size 64 \
--image-size 256 \
--num-workers 4
```

#### 入门GPU（6-8GB）
```bash
--backbone resnet34 \
--batch-size 32 \
--image-size 224 \
--num-workers 2
```

#### CPU训练（不推荐）
```bash
--backbone resnet18 \
--batch-size 8 \
--image-size 224 \
--num-workers 0
```

---

### 针对不同训练阶段

#### 快速原型（探索超参数）
```bash
--epochs 10 \
--batch-size 128 \
--backbone resnet18 \
--report-samples 10
```

#### 基准训练（验证方法）
```bash
--epochs 25 \
--batch-size 64 \
--backbone resnet50 \
--report-samples 50
```

#### 精细优化（追求最优）
```bash
--epochs 50 \
--batch-size 32 \
--backbone resnet101 \
--image-size 384 \
--dynamic-task-weights \
--augment-strength strong \
--report-samples 100
```

---

### 针对不同目标

#### 优先准确率
```bash
--task-weights "1.0,0.2,0.2,0.2" \
--dynamic-task-weights false \
--dropout 0.2
```

#### 优先严重度分类
```bash
--task-weights "0.8,0.2,0.2,1.0" \
--compare-synergy \
--report-samples 100
```

#### 平衡所有任务
```bash
--dynamic-task-weights \
--weight-smoothing 0.9 \
--compare-synergy
```

---

## 常见问题排查

### 问题1：某个任务损失不下降

**症状**：
```
Epoch 10: Loss_61=0.8, Loss_crop=0.05, Loss_disease=2.5, Loss_severity=0.6
```
disease任务损失异常高。

**解决方案**：
```bash
# 方案1：增大该任务权重
--task-weights "1.0,0.3,0.8,0.4"

# 方案2：启用动态权重
--dynamic-task-weights

# 方案3：检查数据标签是否正确
```

---

### 问题2：动态权重震荡

**症状**：
权重在各轮次剧烈波动。

**解决方案**：
```bash
# 增大平滑系数
--weight-smoothing 0.95

# 增大更新间隔
--weight-update-interval 2
```

---

### 问题3：多任务性能不如单任务

**症状**：
对比实验显示 MTL < STL。

**可能原因**：
- 任务冲突（负迁移）
- 权重失衡
- 容量不足

**解决方案**：
```bash
# 检查梯度冲突（需修改代码）
# 使用更大模型
--backbone resnet101

# 尝试不同权重
--task-weights "1.0,0.1,0.1,0.1"  # 降低辅助任务影响
```

---

### 问题4：诊断报告置信度过低

**症状**：
所有预测置信度 < 0.6。

**解决方案**：
```bash
# 温度校准（需在代码中启用）
# 或训练更多轮次
--epochs 40

# 或使用更大模型
--backbone efficientnet_b4
```

---

## 高级用法

### 自定义任务权重策略

编辑代码实现自定义策略：

```python
# 在 task4train.py 中添加
def custom_weight_strategy(epoch, val_losses):
    """
    示例：指数衰减权重
    """
    w_61 = 1.0
    w_crop = 0.5 * (0.95 ** epoch)
    w_disease = 0.5 * (0.95 ** epoch)
    w_severity = 0.4
    return [w_61, w_crop, w_disease, w_severity]
```

---

### 从检查点恢复训练

```bash
python task4train.py \
    --resume checkpoints/task4_multitask/models/checkpoint_epoch_10.pth \
    --epochs 30 \
    --out-dir checkpoints/task4_resumed
```

---

### 只运行推理和报告生成

```bash
python task4train.py \
    --eval-only \
    --load-model checkpoints/task4_multitask/models/best_model.pth \
    --val-meta data/test/test_metadata.csv \
    --val-dir data/test/images \
    --report-samples 200 \
    --include-cam \
    --out-dir results/task4_inference
```

---

### 批量实验（超参数搜索）

使用shell脚本：

```bash
#!/bin/bash

for lr in 1e-4 3e-4 1e-3; do
  for batch_size in 32 64 128; do
    python task4train.py \
      --lr $lr \
      --batch-size $batch_size \
      --out-dir experiments/lr${lr}_bs${batch_size}
  done
done
```

---

## 性能基准

### 预期性能指标（ResNet50，25 epochs）

| 任务 | 准确率 | Macro-F1 | 训练时间（V100） |
|------|--------|----------|------------------|
| 61类病害 | 87-90% | 85-88% | ~2小时 |
| 作物类型 | 95-97% | 94-96% | - |
| 病害类型 | 89-92% | 87-90% | - |
| 严重度 | 82-85% | 80-83% | - |

### 协同效应提升（相比单任务严重度分类）

```
单任务 Macro-F1: 78-80%
多任务 Macro-F1: 80-83%
相对提升: +2-4%
```

---

## 引用

如果使用本代码，请引用：

```bibtex
@misc{shuweicamp2024multitask,
  title={Multi-Task Learning for Agricultural Disease Diagnosis},
  author={ShuWeiCamp Team},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

---

**文档版本**：v1.0  
**最后更新**：2024年  
**维护者**：ShuWeiCamp Team