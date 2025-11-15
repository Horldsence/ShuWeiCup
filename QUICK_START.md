# Quick Start Guide - Fast Training
============================================================

**"Talk is cheap. Show me the code."** - Linus Torvalds

## 🚀 TL;DR - 最快开始

```bash
# 1. 创建精简数据集 (如果还没有)
python create_balanced_dataset.py

# 2. 开始快速训练 (3x faster!)
bash train_fast.sh

# 3. 监控训练进度
open checkpoints/task1_fast/training_curves.png
```

**预期时间**: ~16小时 (vs 完整数据集48小时)  
**预期准确率**: 60-70% (vs 完整数据集70-75%)

---

## 📋 完整问题诊断与解决方案

### 🔴 问题1: 6 Epochs就收敛，准确率只有27-30%

**不是学习率问题！** (你说对了)

**真正原因**:
```
极端类别不平衡:
  - Class 59: 2,445 samples
  - Class 44: 1 sample  
  - Class 45: 1 sample
  - 比例: 2445:1

原始class weights太激进:
  - 范围: [0.211, 517.066]
  - 比例: 2445:1
  - 结果: rare classes dominate loss

模型行为:
  - Epoch 1-6: 快速学习major classes (9类 = 42%数据)
  - 达到30%准确率
  - Class weights导致loss不稳定
  - 收敛到局部最优: "猜大类"
```

### ✅ 解决方案 (已实施)

#### 1. 更温和的Class Weights (Critical!)
```python
# Before: 2445:1 ratio
weights = total / (n_classes * counts)  # [0.211, 517.066]

# After: 49:1 ratio  
weights = sqrt(max_count / counts)     # [0.226, 11.167]
```

#### 2. 更强的数据增强 (Critical!)
```python
# Before (太弱)
A.Rotate(limit=15)
A.ColorJitter(..., p=0.5)

# After (aggressive)
A.Rotate(limit=45)           # 完整旋转
A.ColorJitter(..., p=0.8)    # 更多颜色变化
A.VerticalFlip(p=0.3)        # 垂直翻转
A.GridDistortion(p=0.3)      # 网格扭曲
A.RandomShadow(p=0.2)        # 随机阴影
```

#### 3. 更高分辨率 (Important)
```
224x224 → 320x320
```
农作物病害需要细节，224太小。

#### 4. 精简训练集 (Speed!)
```
31,541 samples → 10,837 samples
每个类最多200张
训练速度: 3x faster
```

---

## 🎯 三种训练方案

### 方案A: 快速训练 (推荐用于实验)

```bash
bash train_fast.sh
```

**配置**:
- 数据集: Balanced (10,837 samples)
- Class weights: Sqrt smoothing (49:1)
- 分辨率: 320x320
- 训练时间: ~16小时
- 预期准确率: 60-70%

**适用场景**:
- ✅ 快速原型验证
- ✅ 超参数搜索
- ✅ 架构实验
- ✅ 时间<24小时

---

### 方案B: 完整训练 (推荐用于最终模型)

```bash
bash train_fixed.sh
```

**配置**:
- 数据集: Full (31,541 samples)
- Class weights: Sqrt smoothing
- 分辨率: 320x320
- 两阶段训练:
  - Stage 1: Head only (10 epochs)
  - Stage 2: Full fine-tune (40 epochs)
- 训练时间: ~48小时
- 预期准确率: 70-75%

**适用场景**:
- ✅ 最终模型训练
- ✅ 竞赛提交
- ✅ 生产部署
- ✅ 追求最高准确率

---

### 方案C: 超快训练 (仅用于debug)

```bash
# 自定义: 更少epochs
python train.py \
    --train-meta data/cleaned/metadata/train_metadata_balanced.csv \
    --class-weights data/cleaned/metadata/class_weights_sqrt.csv \
    --epochs 10 \
    --batch-size 32 \
    --lr 5e-4 \
    --image-size 224 \
    --save-dir checkpoints/debug
```

**配置**:
- 数据集: Balanced
- 分辨率: 224x224 (更快)
- Epochs: 10 (快速验证)
- 训练时间: ~2小时
- 预期准确率: 40-50%

**适用场景**:
- ✅ 代码调试
- ✅ 快速验证修改
- ✅ CI/CD测试

---

## 📊 实时监控

### 训练过程中
```bash
# 查看实时曲线 (每个epoch自动更新)
open checkpoints/task1_fast/training_curves.png

# 查看控制台输出
# - Epoch进度
# - Train/Val loss & accuracy
# - Learning rate变化
```

### 图表包含
- **Loss Curves**: Train/Val loss变化
- **Accuracy Curves**: Train/Val accuracy + 最佳点标记
- **LR Schedule**: 学习率变化，显示warmup阶段
- **Overfitting Analysis**: Train-Val gap分析
  - 🟢 Green: Good fit (<5% gap)
  - 🟡 Orange: Slight overfitting (5-10%)
  - 🔴 Red: Overfitting (>10%)

### 训练后分析
```bash
# 详细可视化
python visualize_training.py --checkpoint-dir checkpoints/task1_fast/

# 对比不同运行
python visualize_training.py --compare \
    checkpoints/task1_fast/best.pth \
    checkpoints/task1_stage2/best.pth

# 数据集对比
python compare_datasets.py --visualize
```

---

## 📈 性能对比

| 配置 | 数据集 | 训练时间 | 预期准确率 | 适用场景 |
|------|--------|----------|------------|----------|
| **快速训练** | 10.8k | 16h | 60-70% | 实验/原型 |
| **完整训练** | 31.5k | 48h | 70-75% | 最终模型 |
| **超快Debug** | 10.8k | 2h | 40-50% | 代码验证 |
| **原始(错误)** | 31.5k | 100h | ~30% | ❌ 不推荐 |

---

## 🔍 故障排查

### 准确率还是很低 (<40%)

**检查清单**:

```bash
# 1. 验证class weights是否使用
grep "class_weights_sqrt" train_fast.sh
python -c "import pandas as pd; df = pd.read_csv('data/cleaned/metadata/class_weights_sqrt.csv'); print(df.head())"

# 2. 验证数据增强是否应用
grep "rotate_limit=45" dataset.py

# 3. 查看训练曲线
open checkpoints/task1_fast/training_curves.png
# Loss应该在下降，不是平的

# 4. 检查数据加载
python -c "from dataset import *; ds = AgriDiseaseDataset('data/cleaned/train', 'data/cleaned/metadata/train_metadata_balanced.csv'); print(f'Samples: {len(ds)}')"
```

### 训练太慢

```bash
# 减小batch size (如果GPU内存不足)
python train.py --batch-size 16 ...

# 降低分辨率
python train.py --image-size 224 ...

# 减少workers
python train.py --num-workers 2 ...

# 使用更小的backbone
python train.py --backbone resnet34 ...
```

### 显存不足

```bash
# 1. 减小batch size
--batch-size 16  # or 8

# 2. 降低分辨率
--image-size 224  # or 192

# 3. 禁用AMP (如果有问题)
# 移除 --use-amp 参数

# 4. 减少workers
--num-workers 2
```

---

## 📚 文档导航

- **[REAL_FIX.md](REAL_FIX.md)** - 详细问题诊断和修复方案
- **[TRAINING_FIX_SUMMARY.md](TRAINING_FIX_SUMMARY.md)** - 学习率修复总结 (次要)
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - 完整改进文档
- **[README.md](README.md)** - 项目主文档

---

## 🎓 关键教训 (Linus-Style)

### 1. 数据 > 超参数

```
"Bad programmers worry about code.
 Good programmers worry about data." - Linus
```

问题不在于:
- ❌ Learning rate
- ❌ Optimizer
- ❌ Model architecture

问题在于:
- ✅ 极端类别不平衡 (2445:1)
- ✅ 错误的weighting策略
- ✅ 数据增强不足

### 2. 听用户的 (你说对了)

```
"6 epochs就稳定，准确率过低，这不是学习率影响的"
```

这句话是关键。快速收敛+低准确率 = 解决了**错误的问题**。

模型在高效学习，只是学习了:
- "预测大类" → 30% 准确率，6 epochs搞定
- 任务完成 (从模型角度)

### 3. 修复根本原因

```python
# 错误做法 (治标不治本)
lr = 5e-4  # 提高学习率
# → 模型依然30%收敛

# 正确做法 (修复根因)  
class_weights = sqrt_smoothing()  # 平衡数据
augmentation = strong()           # 生成更多样本
# → 模型正常学习
```

### 4. 过早优化是万恶之源

```
初始修复:
✓ LR调优 (小幅改进)
✓ Warmup (稳定性)
✓ Scheduler (平滑性)

真正问题:
✓ 数据不平衡 (2-3x改进)
```

我们花时间优化超参数，但真正问题是数据分布。

**教训: 永远先检查数据。**

---

## 🚀 推荐工作流

### Phase 1: 快速实验 (1-2天)
```bash
# 使用balanced dataset
bash train_fast.sh

# 尝试不同配置:
# - 不同的dropout (0.3, 0.4, 0.5)
# - 不同的augmentation强度
# - 不同的backbone (resnet34, resnet50, efficientnet)

# 找到最佳配置
```

### Phase 2: 完整训练 (2-3天)
```bash
# 使用full dataset + 最佳配置
bash train_fixed.sh

# 预期: 比balanced高2-3%准确率
```

### Phase 3: 分析优化
```bash
# 分析per-class性能
python analyze_data.py --visualize

# 对比不同模型
python visualize_training.py --compare \
    checkpoints/*/best.pth

# 错误案例分析
# (需要单独实现)
```

---

## ✅ 最终检查清单

训练前确认:

- [ ] Balanced dataset已创建
  ```bash
  ls data/cleaned/metadata/train_metadata_balanced.csv
  ```

- [ ] Sqrt class weights已生成
  ```bash
  ls data/cleaned/metadata/class_weights_sqrt.csv
  ```

- [ ] 数据增强已更新
  ```bash
  grep "rotate_limit=45" dataset.py
  ```

- [ ] 训练脚本可执行
  ```bash
  ls -lh train_fast.sh
  ```

- [ ] GPU可用 (推荐)
  ```bash
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
  ```

---

## 🎯 期望结果

### Balanced Dataset (Fast)
```
训练时间: ~16小时
Epoch 10: ~45-50% val accuracy
Epoch 30: ~60-65% val accuracy  
Epoch 50: ~65-70% val accuracy
```

### Full Dataset (Complete)
```
训练时间: ~48小时
Stage 1 (10 epochs): ~40-45% val accuracy
Stage 2 (40 epochs): ~70-75% val accuracy
```

### 如果准确率低于预期
1. 检查class weights是否使用
2. 查看training curves是否正常
3. 验证数据augmentation
4. 考虑调整hyperparameters

---

**现在开始**: `bash train_fast.sh` 🚀

**预期时间**: 16小时  
**预期准确率**: 65-70%  
**加速比**: 3x faster than full dataset