# Task 1 训练脚本说明文档

## 概述

`task1train.py` 是农业病害识别系统的主训练脚本，用于完成 **61类作物病害分类任务**。该脚本实现了一个完整的深度学习训练流程，集成了多种先进的训练技巧和优化策略。

### 核心目标
- **多类别分类**：准确识别61种不同的作物病害类别
- **长尾分布处理**：应对数据集中类别样本数量严重不均衡的问题
- **高精度优化**：通过多种训练策略提升模型性能

---

## 文件结构

### 主要组件

```
task1train.py
├── 辅助类定义
│   ├── CosineClassifier       # 余弦分类器
│   ├── CenterLoss            # 中心损失
│   ├── ModelEMA              # 模型指数移动平均
│   └── build_weighted_sampler # 加权采样器构建
├── 数据增强函数
│   └── apply_mixup_cutmix    # Mixup/CutMix数据增强
├── 训练核心
│   └── custom_train_loop     # 自定义训练循环
├── 工具函数
│   ├── set_seed              # 随机种子设置
│   └── parse_args            # 命令行参数解析
└── 主函数
    └── main                  # 训练主流程
```

---

## 核心组件详解

### 1. CosineClassifier（余弦分类器）

**类定义位置**：L38-48

**功能说明**：
实现基于余弦相似度的分类层，相比传统线性分类器，能更好地处理特征空间的角度关系。

**关键参数**：
- `in_features` (int)：输入特征维度
- `num_classes` (int)：类别数量（61）
- `scale` (float, 默认30.0)：缩放因子，用于控制logits的数值范围

**工作原理**：
1. 对输入特征和权重向量进行L2归一化
2. 计算归一化后的余弦相似度
3. 通过scale参数放大相似度得分

**数学表达式**：
```
output = scale × (normalize(features) · normalize(weight)^T)
```

**优势**：
- 减少特征幅值对分类的影响
- 提升模型对特征方向的敏感度
- 对长尾分布数据更友好

---

### 2. CenterLoss（中心损失）

**类定义位置**：L51-59

**功能说明**：
实现中心损失函数，通过最小化同类样本特征到类中心的距离，增强类内紧凑性。

**关键参数**：
- `num_classes` (int)：类别数量
- `feat_dim` (int)：特征维度

**工作原理**：
1. 为每个类别维护一个可学习的中心向量
2. 计算每个样本特征与其类别中心的欧氏距离
3. 最小化这些距离以聚类同类样本

**损失计算**：
```
loss = mean(||features - centers[targets]||²)
```

**应用场景**：
- 配合交叉熵损失使用
- 提升特征的判别能力
- 改善类间分离度

---

### 3. build_weighted_sampler（加权采样器）

**函数位置**：L62-76

**功能说明**：
构建加权随机采样器，用于处理类别不平衡问题，确保稀有类别有足够的训练机会。

**关键参数**：
- `metadata_df` (DataFrame)：包含样本标签的元数据
- `label_col` (str, 默认"label_61")：标签列名
- `power` (float, 默认0.5)：平衡强度指数

**权重计算公式**：
```
weight_i = 1 / (count_i)^power
```

**平衡策略**：
- `power=1.0`：完全平衡，所有类别期望采样次数相同
- `power=0.5`：平方根平衡，适度缓解不平衡
- `power=0.0`：无平衡，等同于随机采样

**实际效果**：
- 稀有类别获得更高采样权重
- 避免模型过度偏向多数类
- 提升整体macro-F1分数

---

### 4. ModelEMA（模型指数移动平均）

**类定义位置**：L79-102

**功能说明**：
维护模型参数的指数移动平均副本，用于验证和推理，提供更稳定的预测性能。

**关键参数**：
- `model` (nn.Module)：待跟踪的模型
- `decay` (float, 默认0.999)：衰减系数

**更新公式**：
```
shadow_t = decay × shadow_{t-1} + (1 - decay) × param_t
```

**核心方法**：

#### `__init__(model, decay)`
初始化EMA影子参数，复制当前模型所有可训练参数。

#### `update(model)`
每次训练步骤后更新影子参数：
- 对每个参数应用指数移动平均
- 只更新requires_grad=True的参数

#### `apply_to(model)`
将影子参数应用到模型：
- 用于验证阶段
- 通常比原始参数更稳定

**使用场景**：
- 训练时使用原始参数
- 验证和推理时使用EMA参数
- 减少参数更新的噪声影响

---

### 5. apply_mixup_cutmix（数据增强）

**函数位置**：L105-145

**功能说明**：
实现Mixup和CutMix两种数据增强策略，通过混合不同样本来增强模型泛化能力。

**关键参数**：
- `images` (Tensor)：输入图像批次 [B, C, H, W]
- `targets` (Tensor)：标签 [B]
- `mixup_alpha` (float)：Mixup Beta分布参数
- `cutmix_alpha` (float)：CutMix Beta分布参数
- `mixup_prob` (float)：应用Mixup的概率
- `cutmix_prob` (float)：应用CutMix的概率

**返回值**：
```python
(mixed_images, targets_a, targets_b, lambda_value)
```

#### Mixup策略
**原理**：线性插值混合两个样本

```python
lambda ~ Beta(alpha, alpha)
mixed_image = lambda × image_i + (1 - lambda) × image_j
mixed_label = lambda × label_i + (1 - lambda) × label_j
```

**特点**：
- 全局混合，整张图像都被影响
- 生成平滑的决策边界
- 减少过拟合

#### CutMix策略
**原理**：裁剪并粘贴图像区域

```python
1. 生成随机矩形区域
2. 用另一样本的对应区域替换
3. 标签权重 = 1 - (裁剪区域面积 / 总面积)
```

**裁剪框计算**：
```python
cut_ratio = sqrt(1 - lambda)
cut_h = H × cut_ratio
cut_w = W × cut_ratio
center = (random_y, random_x)
```

**特点**：
- 保留局部特征信息
- 鼓励模型关注多个区域
- 提升定位能力

**执行逻辑**：
1. 以`mixup_prob`概率尝试Mixup
2. 若不执行Mixup，以`cutmix_prob`概率尝试CutMix
3. 两者都不执行则返回原始数据

---

### 6. custom_train_loop（自定义训练循环）

**函数位置**：L148-422

**功能说明**：
实现灵活可控的训练循环，集成多种训练策略和动态调整机制。

**核心参数**：
- `model` (nn.Module)：待训练模型
- `optimizer`：优化器
- `scheduler`：学习率调度器
- `train_loader`：训练数据加载器
- `val_loader`：验证数据加载器
- `device`：计算设备 (cuda/mps/cpu)
- `config` (dict)：训练配置字典

**配置参数详解**：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `epochs` | int | 50 | 总训练轮数 |
| `use_amp` | bool | True | 是否启用混合精度训练 |
| `clip_grad` | float | 1.0 | 梯度裁剪阈值 |
| `use_ema` | bool | True | 是否使用模型EMA |
| `ema_decay` | float | 0.999 | EMA衰减系数 |
| `use_mixup` | bool | True | 是否使用Mixup |
| `mixup_alpha` | float | 0.2 | Mixup混合强度 |
| `use_cutmix` | bool | True | 是否使用CutMix |
| `cutmix_alpha` | float | 1.0 | CutMix混合强度 |
| `center_loss_weight` | float | 0.01 | 中心损失权重 |
| `use_weighted_sampler` | bool | True | 是否使用加权采样 |

#### 子函数：adaptive_loss（自适应损失）

**位置**：L211-229

**功能**：根据训练阶段动态调整损失权重

**策略**：
1. **早期阶段**（epoch < focal_start_epoch）：
   - 使用标准交叉熵损失
   - 配合标签平滑（label smoothing）
   
2. **后期阶段**（epoch >= focal_start_epoch）：
   - 切换到Focal Loss
   - 专注于困难样本
   - gamma参数控制聚焦程度

3. **标签平滑调整**（epoch >= smooth_reduce_epoch）：
   - 减小平滑系数
   - 使决策边界更加锐利

#### 子函数：run_val（验证运行）

**位置**：L231-283

**功能**：执行完整的验证流程

**执行步骤**：
1. 切换模型到评估模式
2. 应用EMA参数（如果启用）
3. 禁用梯度计算（节省内存）
4. 遍历验证集计算损失和准确率
5. 计算混淆矩阵和分类报告
6. 恢复原始模型参数

**返回指标**：
- `val_loss`：平均验证损失
- `val_acc`：验证准确率
- `confusion_matrix`：混淆矩阵（可选）

---

### 7. 训练流程关键特性

#### 动态学习率调整
使用OneCycleLR调度器：
- 学习率先线性增大到最大值
- 然后余弦衰减到最小值
- 配合动量相反方向调整

#### 渐进式图像尺寸调整
**progressive_resize模式**：
- 初期：使用较小图像（如224×224）
- 中期：逐步增大（如256×256）
- 后期：使用最大尺寸（如320×320）

**优势**：
- 加快早期训练速度
- 后期捕获更多细节
- 提升最终精度

#### 采样器动态切换
**策略**：
- 前N轮：使用加权采样器（平衡类别）
- 后续轮次：切换到随机采样（还原真实分布）

**时机**：通过`sampler_switch_epoch`参数控制

#### 早停机制
**触发条件**：
- 验证准确率在patience轮内无提升
- 或验证损失开始上升

**行为**：
- 保存当前最佳模型
- 提前终止训练
- 避免过拟合

---

## 命令行参数说明

### 数据相关参数

```bash
--train-dir          # 训练数据目录，默认: data/cleaned/train
--val-dir            # 验证数据目录，默认: data/cleaned/val
--train-meta         # 训练元数据CSV路径
--val-meta           # 验证元数据CSV路径
--class-weights      # 类别权重CSV路径
```

### 模型架构参数

```bash
--model-type         # 模型类型: baseline/multitask/fewshot
--backbone           # 骨干网络: resnet50/efficientnet_b3等
--pretrained         # 是否使用预训练权重（默认True）
--dropout            # Dropout比率，默认0.3
```

### 训练超参数

```bash
--epochs             # 训练轮数，默认50
--batch-size         # 批次大小，默认64
--lr                 # 学习率，默认3e-4
--weight-decay       # 权重衰减，默认1e-4
--optimizer          # 优化器: adamw/sgd
--image-size         # 图像尺寸，默认224
```

### 高级训练策略

```bash
# 渐进式调整
--progressive-resize        # 启用渐进式尺寸调整
--progressive-sizes         # 尺寸序列，如"224,256,320"
--resize-epochs            # 切换轮次，如"10,20"

# 损失函数
--focal-start-epoch        # 切换到Focal Loss的轮次，默认20
--focal-gamma              # Focal Loss gamma值，默认1.5
--label-smoothing          # 标签平滑系数，默认0.1
--smooth-reduce-epoch      # 减小标签平滑的轮次，默认12

# 数据增强
--mixup-alpha              # Mixup alpha值，默认0.2
--mixup-prob               # Mixup应用概率，默认0.5
--cutmix-alpha             # CutMix alpha值，默认1.0
--cutmix-prob              # CutMix应用概率，默认0.5

# 采样策略
--use-weighted-sampler     # 启用加权采样
--sampler-power            # 采样权重指数，默认0.5
--sampler-switch-epoch     # 切换采样器的轮次，默认15

# 其他
--use-ema                  # 启用EMA
--ema-decay                # EMA衰减系数，默认0.999
--center-loss-weight       # 中心损失权重，默认0.01
--clip-grad                # 梯度裁剪阈值，默认1.0
--use-amp                  # 启用混合精度训练
```

### 输出与日志

```bash
--save-dir           # 模型保存目录
--log-interval       # 日志打印间隔步数
--save-interval      # 模型保存间隔轮数
--tensorboard        # 启用TensorBoard日志
--seed               # 随机种子，默认42
```

---

## 使用示例

### 基础训练

```bash
python task1train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone resnet50 \
    --epochs 50 \
    --batch-size 64 \
    --lr 3e-4
```

### 高级训练（全部特性）

```bash
python task1train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone efficientnet_b3 \
    --epochs 80 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 5e-5 \
    --optimizer adamw \
    --progressive-resize \
    --progressive-sizes "224,256,320" \
    --resize-epochs "20,40" \
    --use-weighted-sampler \
    --sampler-power 0.5 \
    --sampler-switch-epoch 20 \
    --mixup-alpha 0.3 \
    --mixup-prob 0.6 \
    --cutmix-alpha 1.0 \
    --cutmix-prob 0.4 \
    --focal-start-epoch 25 \
    --focal-gamma 2.0 \
    --label-smoothing 0.1 \
    --smooth-reduce-epoch 15 \
    --use-ema \
    --ema-decay 0.9995 \
    --center-loss-weight 0.02 \
    --use-amp \
    --save-dir checkpoints/task1_advanced \
    --tensorboard
```

### 少样本场景训练

```bash
python task1train.py \
    --model-type fewshot \
    --train-meta data/cleaned/metadata/train_metadata_fewshot_10.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --backbone resnet50 \
    --epochs 100 \
    --batch-size 16 \
    --lr 5e-4 \
    --use-weighted-sampler \
    --sampler-power 0.3 \
    --mixup-alpha 0.5 \
    --label-smoothing 0.15
```

---

## 输出文件说明

训练完成后，在`--save-dir`目录下会生成以下文件：

### 模型文件
- `best_model.pth`：最佳验证准确率模型
- `last_model.pth`：最后一轮训练模型
- `checkpoint_epoch_N.pth`：定期检查点（如果启用）

### 日志文件
- `training.log`：详细训练日志
- `config.yaml`：训练配置备份
- `metrics.csv`：每轮指标记录

### 可视化文件（如果启用TensorBoard）
- `runs/`目录：TensorBoard事件文件
  - 损失曲线
  - 准确率曲线
  - 学习率变化
  - 参数分布直方图

---

## 关键技术点总结

### 1. 长尾分布处理
- **加权采样**：提升稀有类别采样频率
- **类别权重**：损失函数中给稀有类别更高权重
- **Focal Loss**：专注于困难样本
- **中心损失**：增强类内聚合度

### 2. 泛化能力提升
- **数据增强**：Mixup + CutMix
- **标签平滑**：防止过度自信
- **Dropout**：随机失活防止过拟合
- **权重衰减**：L2正则化

### 3. 训练稳定性
- **梯度裁剪**：防止梯度爆炸
- **混合精度**：加速训练同时保持数值稳定
- **EMA**：参数平滑降低验证波动
- **学习率调度**：OneCycle策略

### 4. 训练效率
- **渐进式尺寸**：早期小图像快速训练
- **自动混合精度**：减少显存占用和训练时间
- **动态采样切换**：平衡准确率和效率

---

## 性能优化建议

### 针对不同数据集规模

#### 小数据集（< 10k样本）
```bash
--epochs 100
--mixup-alpha 0.5
--label-smoothing 0.2
--use-weighted-sampler
--dropout 0.5
```

#### 中数据集（10k-100k样本）
```bash
--epochs 50
--mixup-alpha 0.3
--label-smoothing 0.1
--use-weighted-sampler
--sampler-switch-epoch 20
```

#### 大数据集（> 100k样本）
```bash
--epochs 30
--mixup-alpha 0.2
--label-smoothing 0.05
--batch-size 128
```

### 针对不同计算资源

#### GPU内存充足（≥ 24GB）
```bash
--batch-size 128
--image-size 320
--use-amp
```

#### GPU内存受限（8-16GB）
```bash
--batch-size 32
--image-size 224
--use-amp
--gradient-accumulation-steps 4
```

#### CPU训练
```bash
--batch-size 8
--image-size 224
--use-amp false
--num-workers 0
```

---

## 常见问题排查

### 1. 训练损失不下降
**可能原因**：
- 学习率过大或过小
- 批次大小不合适
- 数据预处理错误

**解决方案**：
```bash
# 调整学习率
--lr 1e-4  # 减小学习率

# 调整批次大小
--batch-size 32  # 增大批次

# 检查数据
python visualize_samples.py  # 可视化数据样本
```

### 2. 过拟合
**症状**：训练准确率高，验证准确率低

**解决方案**：
```bash
--dropout 0.5              # 增大Dropout
--label-smoothing 0.2      # 增大标签平滑
--mixup-alpha 0.5          # 增强数据增强
--weight-decay 1e-3        # 增大权重衰减
```

### 3. 欠拟合
**症状**：训练和验证准确率都低

**解决方案**：
```bash
--backbone efficientnet_b4  # 使用更大模型
--epochs 100                # 增加训练轮数
--lr 3e-3                   # 增大学习率
--batch-size 64             # 增大批次
```

### 4. 内存溢出
**解决方案**：
```bash
--batch-size 16            # 减小批次
--image-size 224           # 减小图像尺寸
--backbone resnet34        # 使用更小模型
--use-amp                  # 启用混合精度
```

---

## 扩展与定制

### 添加新的骨干网络

在`models.py`中注册新模型：

```python
def create_model(backbone="resnet50", num_classes=61, **kwargs):
    if backbone == "your_new_model":
        model = timm.create_model("your_new_model", pretrained=True)
        # ... 自定义修改
    return model
```

### 添加新的损失函数

在`losses.py`中定义：

```python
class YourCustomLoss(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... 初始化
    
    def forward(self, logits, targets):
        # ... 计算损失
        return loss
```

### 添加新的数据增强

修改`dataset.py`中的transform：

```python
def get_train_transform(image_size=224):
    return A.Compose([
        # ... 现有增强
        A.YourCustomAugmentation(...),  # 新增
        A.Normalize(...),
        ToTensorV2(),
    ])
```

---

## 参考文献

1. **Mixup**: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
2. **CutMix**: Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers" (ICCV 2019)
3. **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
4. **Center Loss**: Wen et al. "A Discriminative Feature Learning Approach for Deep Face Recognition" (ECCV 2016)
5. **Label Smoothing**: Szegedy et al. "Rethinking the Inception Architecture" (CVPR 2016)
6. **OneCycle**: Smith et al. "Super-Convergence: Very Fast Training of Neural Networks" (2018)

---

## 更新日志

- **v1.0** (初始版本)：基础训练流程
- **v1.1**：添加EMA和中心损失
- **v1.2**：集成Mixup/CutMix
- **v1.3**：添加渐进式尺寸调整
- **v1.4**：优化长尾分布处理
- **v1.5**（当前版本）：完善动态训练策略

---

**文档编写日期**：2024年
**脚本版本**：v1.5
**维护者**：ShuWeiCamp Team