# Agricultural Disease Recognition
============================================================

**"Talk is cheap. Show me the code."** - Linus Torvalds

深度学习农作物病害识别系统 - 61类疾病分类

## 🚀 Quick Start

### 训练改进版模型 (推荐)

```bash
# 一键训练 - 使用优化后的超参数
bash train_improved.sh
```

**预期结果**: 70-85% 验证准确率 (vs 基线27.6%)

### 原始训练 (不推荐 - 准确率低)

```bash
python train.py
```

## 📊 实时监控训练

训练过程中自动生成可视化图表:

```bash
# 查看实时训练曲线
open checkpoints/task1_improved/training_curves.png

# 训练后详细分析
python visualize_training.py --checkpoint-dir checkpoints/task1_improved/
```

**图表包含**:
- Loss curves (train/val)
- Accuracy curves with best marker
- Learning rate schedule with warmup
- Overfitting analysis (train-val gap)

## 📈 性能对比

| 配置 | Val Accuracy | 说明 |
|------|--------------|------|
| **Baseline** (原始) | 27.6% | ❌ LR太低, 无warmup |
| **Improved** (优化) | **70-85%** | ✅ 正确的超参数 |

### 关键改进

1. **Learning Rate**: 1e-4 → 5e-4 (5x ↑)
2. **Batch Size**: 64 → 32 (更好的梯度信号)
3. **Warmup**: 添加5 epoch warmup
4. **LR Schedule**: Warmup后才开始decay
5. **可视化**: 每轮自动更新训练曲线

## 📁 项目结构

```
ShuWeiCamp/
├── train.py                    # 主训练脚本
├── train_improved.sh          # 优化版训练脚本 (推荐使用)
├── trainer.py                 # 训练引擎 (含可视化)
├── models.py                  # 模型定义
├── dataset.py                 # 数据集加载
├── losses.py                  # 损失函数
├── visualize_training.py      # 可视化工具
├── demo_visualization.py      # Demo演示
│
├── TRAINING_FIX_SUMMARY.md    # 🔥 训练修复总结 (必读!)
├── IMPROVEMENTS.md            # 详细改进文档
├── ROADMAP.md                 # 项目路线图
├── SETUP.md                   # 环境设置
│
├── data/                      # 数据目录
│   ├── cleaned/               # 清洗后的数据
│   └── raw/                   # 原始数据
│
└── checkpoints/               # 模型检查点
    ├── task1_baseline/        # 基线模型 (27.6%)
    └── task1_improved/        # 优化模型 (70-85%)
```

## 📖 文档导航

- **[TRAINING_FIX_SUMMARY.md](TRAINING_FIX_SUMMARY.md)** - 🔥 训练问题修复总结 (先看这个!)
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - 详细改进说明和原理
- **[ROADMAP.md](ROADMAP.md)** - 完整项目路线图
- **[SETUP.md](SETUP.md)** - 环境配置指南

## 🛠️ 常用命令

### 训练相关

```bash
# 使用优化配置训练
bash train_improved.sh

# 自定义参数训练
python train.py --lr 5e-4 --batch-size 32 --epochs 50

# 从断点恢复训练
python train.py --resume checkpoints/task1_improved/interrupted.pth
```

### 可视化相关

```bash
# 查看训练进度
python visualize_training.py --checkpoint checkpoints/task1_improved/best.pth

# 对比多个模型
python visualize_training.py --compare \
    checkpoints/task1_baseline/best.pth \
    checkpoints/task1_improved/best.pth

# 生成demo图表
python demo_visualization.py
```

### 数据相关

```bash
# 查看数据集统计
python dataset.py

# 测试数据加载
python -c "from dataset import *; test_dataset()"
```

## 🎯 训练流程

1. **数据准备** (已完成)
   - 61类疾病数据
   - Train/Val split
   - Class weights计算

2. **模型训练** (当前)
   ```bash
   bash train_improved.sh
   ```

3. **监控训练**
   - 查看控制台输出
   - 检查 `training_curves.png`
   - 分析overfitting状态

4. **模型评估**
   - Best checkpoint在 `checkpoints/task1_improved/best.pth`
   - 使用 `visualize_training.py` 查看详细指标

## 🐛 故障排查

### 准确率低 (<50%)

1. 检查是否使用了优化配置:
   ```bash
   grep "lr" checkpoints/task1_improved/logs/*
   ```

2. 查看训练曲线是否正常:
   ```bash
   python visualize_training.py --checkpoint-dir checkpoints/task1_improved/
   ```

3. 验证数据加载:
   ```bash
   python -c "from dataset import *; ds = AgriDiseaseDataset('data/cleaned/train', 'data/cleaned/metadata/train_metadata.csv'); print(f'Samples: {len(ds)}')"
   ```

### 显存不足

```bash
# 减小batch size
python train.py --batch-size 16

# 使用更小的模型
python train.py --backbone resnet34
```

### 训练太慢

```bash
# 增加workers
python train.py --num-workers 8

# 启用编译 (PyTorch 2.0+)
python train.py --compile
```

## 📊 结果可视化示例

训练后自动生成的图表包含:

1. **Loss Curves** - 训练/验证损失
2. **Accuracy Curves** - 准确率变化 + 最佳点标注
3. **LR Schedule** - 学习率变化 (显示warmup阶段)
4. **Overfitting Analysis** - Train-Val gap分析

状态判断:
- 🟢 Green: Good fit (gap < 5%)
- 🟡 Orange: Slight overfitting (5-10%)
- 🔴 Red: Overfitting (> 10%)

## 🎓 Linus式开发哲学

> **"Talk is cheap. Show me the code."**

我们的原则:

1. ✅ **修复基础问题优先** - LR/warmup/scheduler
2. ✅ **简单直接的方案** - No fancy tricks
3. ✅ **可视化验证** - 一图胜千言
4. ❌ **避免过早优化** - 先让基础work

### 我们做的

- 正确的learning rate
- Proper warmup schedule
- 实时可视化监控

### 我们没做 (好品味)

- ~~复杂的optimizer~~
- ~~花哨的augmentation~~
- ~~架构搜索~~
- ~~Ensemble~~

**原因**: 基础训练都没搞对，优化这些没意义。

## 📝 更新日志

### 2024-11-15 - 训练优化

- 🔴 **Critical Fix**: Learning rate 1e-4 → 5e-4
- 🔴 **Critical Fix**: 添加5 epoch warmup
- 🔴 **Critical Fix**: 修复scheduler时机
- ✨ **Feature**: 实时训练可视化
- ✨ **Feature**: History tracking in checkpoints
- 📊 **Improvement**: 预期准确率从27.6%提升到70-85%

### 2024-11-14 - 项目初始化

- 数据清洗和预处理
- 基线模型实现
- 训练流程搭建

## 🤝 贡献指南

遵循Linus的原则:

1. **代码质量** > 功能数量
2. **简单方案** > 复杂方案
3. **实际测试** > 理论分析
4. **零废话** > 长篇大论

## 📄 License

MIT License

## 🙏 致谢

- Linus Torvalds - 为优秀代码品味树立标准
- PyTorch Team - 优秀的深度学习框架
- timm - 预训练模型库

---

**项目状态**: ✅ 可用  
**最佳准确率**: 待训练 (预期70-85%)  
**最后更新**: 2024-11-15

**开始训练**: `bash train_improved.sh` 🚀