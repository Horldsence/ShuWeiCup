# Task 2 训练脚本说明文档

## 概述

`task2train.py` 是少样本学习（Few-Shot Learning）训练脚本，专门用于处理 **每类仅有少量标注样本** 的场景。该脚本基于 **ArcFace + 原型网络（Prototype Network）** 架构，通过度量学习和原型对齐策略，在极少样本下实现高效的61类作物病害分类。

### 核心目标
- **少样本学习**：每类仅有5-20个标注样本
- **度量学习**：学习可判别的特征嵌入空间
- **原型对齐**：通过类原型约束增强泛化能力
- **迁移学习**：利用预训练模型和渐进式解冻策略

---

## 文件结构

### 主要组件

```
task2train.py
├── 工具函数
│   ├── set_seed                    # 随机种子设置
│   ├── get_fewshot_train_transform # 训练数据增强
│   ├── get_fewshot_val_transform   # 验证数据预处理
│   └── fewshot_collate             # 数据批次整理
├── 核心模块
│   ├── ArcFaceHead                 # ArcFace分类头
│   ├── PrototypeLoss               # 原型损失
│   ├── compute_prototypes          # 原型计算
│   └── FewShotArcFaceModel         # 完整模型封装
├── 训练评估
│   ├── macro_f1                    # 宏平均F1计算
│   ├── apply_mixup                 # Mixup数据增强
│   ├── train_one_epoch             # 单轮训练
│   └── validate                    # 验证函数
└── 主流程
    ├── parse_args                  # 参数解析
    └── main                        # 主函数
```

---

## 核心组件详解

### 1. ArcFaceHead（ArcFace分类头）

**类定义位置**：L137-162

**功能说明**：
实现ArcFace（Additive Angular Margin）损失的分类头，通过在角度空间添加边界来增强类间分离度。

**关键参数**：
- `in_features` (int)：输入特征维度
- `num_classes` (int)：类别数量（61）
- `scale` (float, 默认30.0)：特征缩放因子
- `margin` (float, 默认0.30)：角度边界（弧度）

**工作原理**：

1. **特征归一化**：
```python
x_norm = F.normalize(features, dim=1)  # 特征向量归一化
w_norm = F.normalize(weight, dim=1)    # 权重向量归一化
```

2. **计算余弦相似度**：
```python
logits = x_norm @ w_norm.T  # 余弦相似度 = cos(θ)
```

3. **添加角度边界**（训练时）：
```python
theta = arccos(logits)                    # 角度
target_theta = theta[target_idx] + margin # 目标类角度加边界
logits[target_idx] = cos(target_theta)    # 重新计算logits
```

4. **缩放输出**：
```python
output = scale * logits  # 放大到合适范围
```

**数学表达式**：
```
L_ArcFace = -log( exp(s·cos(θ_yi + m)) / 
                  (exp(s·cos(θ_yi + m)) + Σ_j≠yi exp(s·cos(θ_j))) )
```

其中：
- `s`：scale参数
- `m`：margin参数
- `θ_yi`：样本与目标类权重的夹角
- `θ_j`：样本与第j类权重的夹角

**优势**：
- **强判别性**：角度边界使类间距离更大
- **可解释性**：在超球面上操作，几何意义清晰
- **少样本友好**：归一化减少样本数量对幅值统计的依赖

---

### 2. PrototypeLoss（原型损失）

**类定义位置**：L168-173

**功能说明**：
计算样本特征与其类原型之间的距离，促使同类样本在特征空间中聚集。

**工作原理**：
1. 根据标签索引提取对应类别的原型向量
2. 计算样本特征与原型的欧氏距离平方
3. 对批次内所有样本取平均

**损失计算公式**：
```python
loss = mean( ||features - prototypes[labels]||² )
```

**数学表达式**：
```
L_proto = (1/N) Σ_i ||f_i - p_yi||²
```

其中：
- `f_i`：第i个样本的特征向量
- `p_yi`：第i个样本所属类别的原型
- `N`：批次大小

**与ArcFace的协同作用**：
- **ArcFace**：优化类间分离（最大化类间距离）
- **PrototypeLoss**：优化类内聚合（最小化类内距离）
- **联合效果**：形成紧凑且分离的特征分布

---

### 3. compute_prototypes（原型计算）

**函数位置**：L177-202

**功能说明**：
遍历数据集计算每个类别的原型向量（特征均值），用于原型损失和推理。

**输入参数**：
- `model` (nn.Module)：特征提取模型
- `dataloader` (DataLoader)：数据加载器
- `device` (torch.device)：计算设备

**返回值**：
- `prototypes` (Tensor)：形状为 [num_classes, feat_dim] 的原型矩阵

**计算流程**：

```python
1. 初始化累加器
   sums = {}      # 每类特征和
   counts = {}    # 每类样本数

2. 遍历数据集（无梯度）
   for images, labels in dataloader:
       features = model.extract_features(images)
       for feat, label in zip(features, labels):
           sums[label] += feat
           counts[label] += 1

3. 计算均值原型
   prototypes[k] = sums[k] / counts[k]
```

**使用场景**：
- **训练前**：计算初始原型
- **训练中**：定期更新原型（通过EMA平滑）
- **推理时**：用于最近邻分类

---

### 4. FewShotArcFaceModel（少样本ArcFace模型）

**类定义位置**：L208-252

**功能说明**：
封装完整的少样本学习模型，包含骨干网络、池化层、Dropout和ArcFace分类头。

**架构设计**：

```
Input Image [B, 3, H, W]
    ↓
Backbone (ResNet50/EfficientNet等)
    ↓
Feature Map [B, C, H', W']
    ↓
Global Average Pooling
    ↓
Feature Vector [B, feat_dim]
    ↓
Dropout (dropout=0.3)
    ↓
ArcFace Head
    ↓
Logits [B, num_classes]
```

**关键参数**：
- `backbone_name` (str)：骨干网络名称（如"resnet50"）
- `num_classes` (int)：类别数量
- `pretrained` (bool)：是否使用预训练权重
- `dropout` (float)：Dropout比率

**初始化策略**：
1. **加载骨干网络**：
```python
self.backbone = timm.create_model(
    backbone_name, 
    pretrained=True,    # 使用ImageNet预训练
    num_classes=0,       # 移除原分类头
    global_pool=""       # 保留特征图
)
```

2. **冻结骨干网络**：
```python
for p in self.backbone.parameters():
    p.requires_grad = False  # 初始完全冻结
```

3. **添加自定义头部**：
```python
self.global_pool = nn.AdaptiveAvgPool2d(1)
self.dropout = nn.Dropout(dropout)
self.arcface_head = ArcFaceHead(feat_dim, num_classes)
```

**核心方法**：

#### `extract_features(x)`
提取特征向量：
```python
feature_map = self.backbone(x)       # [B, C, H, W]
pooled = self.global_pool(feature_map)  # [B, C, 1, 1]
features = pooled.flatten(1)         # [B, C]
features = self.dropout(features)    # Dropout
return features
```

#### `forward(x, labels=None)`
完整前向传播：
```python
features = self.extract_features(x)
logits = self.arcface_head(features, labels)
return logits, features
```

**参数量统计**：
脚本会自动打印：
```
Total params: XX.XXM | Trainable params: YY.YYM
```

---

### 5. 训练策略详解

#### Strategy A: 头部学习率放大（head-lr-scale）

**原理**：
对分类头使用比骨干网络更大的学习率，促进快速适应新任务。

**实现**：
```python
param_groups = [
    {'params': backbone.parameters(), 'lr': lr},
    {'params': head.parameters(), 'lr': lr * head_lr_scale}
]
optimizer = AdamW(param_groups)
```

**默认值**：`head_lr_scale = 3.0`

**效果**：
- 头部快速学习新类别特征
- 骨干网络缓慢微调，保留预训练知识

---

#### Strategy B: 高Dropout防过拟合

**原理**：
在少样本场景下，过拟合风险极高，使用较大Dropout率增强正则化。

**默认值**：`dropout = 0.3`

**位置**：特征提取后、分类头前

**效果**：
- 强制模型学习冗余特征
- 降低对特定样本的依赖

---

#### Strategy C: 早期禁用Mixup

**原理**：
Mixup会生成虚拟样本，在极少样本下可能引入过多噪声，后期关闭以强化真实样本学习。

**实现**：
```python
if epoch < mixup_disable_epoch:
    images, labels_a, labels_b, lam = apply_mixup(images, labels, mixup_alpha)
else:
    # 使用原始样本
```

**默认值**：`mixup_disable_epoch = 8`

**效果**：
- 前期：Mixup增强泛化
- 后期：聚焦真实决策边界

---

#### Strategy D: 标签平滑

**原理**：
将one-hot标签软化为概率分布，防止模型过度自信。

**数学表达式**：
```
y_smooth = (1 - ε) × y_hard + ε / K
```

其中：
- `ε`：平滑系数（默认0.05）
- `K`：类别数
- `y_hard`：one-hot标签

**实现**：
```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
```

**效果**：
- 减少过拟合
- 提升模型校准度（calibration）

---

#### Strategy E: 原型损失权重

**原理**：
在ArcFace损失基础上增加原型损失，双重约束特征空间。

**总损失**：
```python
total_loss = loss_ce + proto_weight * loss_proto
```

**默认值**：`proto_weight = 0.4`

**平衡考量**：
- 过小：原型约束不足，类内分散
- 过大：过度依赖原型，损失ArcFace优势

---

#### Strategy F: 延迟解冻（Delayed Unfreeze）

**原理**：
分阶段解冻骨干网络，先让头部收敛，再微调骨干网络深层。

**解冻策略**：
```python
Epoch 0-2:  全部冻结（只训练分类头）
Epoch 3+:   解冻layer4（最后一个残差块）
            layer1-3保持冻结
```

**实现代码**：
```python
if epoch == unfreeze_epoch:
    for name, param in model.backbone.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True
    print("[Unfreeze] Enabled layer4")
```

**默认值**：`unfreeze_epoch = 3`

**效果**：
- 避免早期梯度破坏预训练特征
- 后期微调捕获任务特定细节

---

### 6. 原型EMA更新机制

**功能**：维护原型的指数移动平均，平滑原型更新过程。

**更新公式**：
```python
proto_new = proto_ema * proto_old + (1 - proto_ema) * proto_current
```

**参数**：
- `proto_ema` (float, 默认0.7)：EMA衰减系数
- `proto_refresh_interval` (int, 默认1)：刷新间隔（轮数）

**更新时机**：
```python
if (epoch + 1) % proto_refresh_interval == 0:
    proto_current = compute_prototypes(model, train_loader, device)
    if epoch == 0:
        prototypes = proto_current
    else:
        prototypes = proto_ema * prototypes + (1 - proto_ema) * proto_current
```

**优势**：
- 减少原型波动
- 平滑训练过程
- 提升稳定性

---

### 7. 数据增强策略

#### 训练增强（get_fewshot_train_transform）

**位置**：L96-106

**Pipeline**：
```python
Resize(256, 256)                     # 调整尺寸
↓
HorizontalFlip(p=0.5)                # 水平翻转
↓
ColorJitter(                         # 颜色抖动
    brightness=0.15,
    contrast=0.15,
    saturation=0.15,
    hue=0.05,
    p=0.6
)
↓
Normalize(ImageNet统计量)            # 标准化
↓
ToTensorV2()                         # 转张量
```

**设计原则**：
- **温和增强**：避免破坏关键特征
- **稳定性优先**：使用固定的Resize而非RandomCrop
- **颜色保留**：轻微ColorJitter，保持病害颜色特征

---

#### 验证增强（get_fewshot_val_transform）

**位置**：L109-116

**Pipeline**：
```python
Resize(256, 256)                     # 固定尺寸
↓
Normalize(ImageNet统计量)            # 标准化
↓
ToTensorV2()                         # 转张量
```

**特点**：
- 无随机性
- 确保可复现
- 与训练保持一致的归一化

---

### 8. 训练与验证循环

#### train_one_epoch（单轮训练）

**位置**：L289-366

**流程图**：
```
开始
 ↓
设置模型为训练模式
 ↓
遍历训练批次
 ├─ 数据移至设备
 ├─ 应用Mixup（如果启用）
 ├─ 前向传播（获取logits和features）
 ├─ 计算损失
 │   ├─ ArcFace损失（或CE+Mixup损失）
 │   └─ 原型损失（如果启用）
 ├─ 反向传播
 ├─ 梯度裁剪（可选）
 ├─ 优化器更新
 ├─ 学习率调度
 └─ 记录指标
 ↓
返回平均损失和准确率
```

**关键代码片段**：
```python
# Mixup处理
if use_mixup and epoch < mixup_disable_epoch:
    mixed_images, labels_a, labels_b, lam = apply_mixup(...)
    loss_ce = lam * ce_loss(logits, labels_a) + (1-lam) * ce_loss(logits, labels_b)
else:
    loss_ce = ce_loss(logits, labels)

# 原型损失
if proto_weight > 0:
    loss_proto = proto_loss_fn(features, labels, prototypes)
    total_loss = loss_ce + proto_weight * loss_proto
```

---

#### validate（验证函数）

**位置**：L370-413

**评估指标**：
- **Accuracy**：整体准确率
- **Macro-F1**：不考虑类别样本数的F1平均
- **混淆矩阵**：类别间混淆情况

**流程**：
```python
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        logits, _ = model(images)
        predictions = logits.argmax(dim=1)
        # 累积预测和真实标签

# 计算指标
acc = correct / total
macro_f1 = compute_macro_f1(all_preds, all_labels, num_classes)
```

**Macro-F1计算**（L258-269）：
```python
for class_id in range(num_classes):
    TP = ((preds == c) & (targets == c)).sum()
    FP = ((preds == c) & (targets != c)).sum()
    FN = ((preds != c) & (targets == c)).sum()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall)
    
macro_f1 = mean(f1_scores)
```

---

## 命令行参数说明

### 数据路径参数

```bash
--train-meta         # 训练元数据CSV路径
                     # 示例: data/cleaned/metadata/train_metadata_fewshot_10.csv
                     
--val-meta           # 验证元数据CSV路径
                     # 示例: data/cleaned/metadata/val_metadata.csv
                     
--train-dir          # 训练图像根目录
                     # 示例: data/cleaned/train
                     
--val-dir            # 验证图像根目录
                     # 示例: data/cleaned/val
```

### 模型架构参数

```bash
--backbone           # 骨干网络
                     # 可选: resnet50, resnet101, efficientnet_b3, etc.
                     # 默认: resnet50
                     
--pretrained         # 使用预训练权重（默认True）

--dropout            # Dropout率
                     # 默认: 0.3
                     # 范围: [0.1, 0.5]
```

### 训练超参数

```bash
--epochs             # 训练轮数
                     # 默认: 30
                     # 建议: 少样本场景使用50-100
                     
--batch-size         # 批次大小
                     # 默认: 8
                     # 说明: 少样本场景通常使用较小批次
                     
--lr                 # 基础学习率
                     # 默认: 3e-4
                     # 范围: [1e-5, 1e-3]
                     
--head-lr-scale      # 头部学习率倍数（Strategy A）
                     # 默认: 3.0
                     # 范围: [1.0, 5.0]
                     
--weight-decay       # 权重衰减
                     # 默认: 1e-4
                     
--optimizer          # 优化器类型
                     # 可选: adamw, sgd
                     # 默认: adamw
```

### ArcFace参数

```bash
--arcface-scale      # ArcFace缩放因子
                     # 默认: 30.0
                     # 范围: [10.0, 64.0]
                     
--arcface-margin     # ArcFace角度边界
                     # 默认: 0.30
                     # 范围: [0.2, 0.5]
```

### 原型学习参数

```bash
--proto-weight       # 原型损失权重（Strategy E）
                     # 默认: 0.4
                     # 范围: [0.0, 1.0]
                     
--proto-ema          # 原型EMA衰减系数
                     # 默认: 0.7
                     # 范围: [0.5, 0.9]
                     
--proto-refresh-interval  # 原型刷新间隔（轮数）
                          # 默认: 1
```

### 数据增强参数

```bash
--image-size         # 图像尺寸
                     # 默认: 256
                     
--mixup-alpha        # Mixup混合强度
                     # 默认: 0.2
                     # 设为0禁用Mixup
                     
--mixup-disable-epoch  # 禁用Mixup的轮次（Strategy C）
                       # 默认: 8
                       
--label-smoothing    # 标签平滑系数（Strategy D）
                     # 默认: 0.05
```

### 解冻策略参数

```bash
--unfreeze-epoch     # 解冻layer4的轮次（Strategy F）
                     # 默认: 3
                     
--freeze-stage12     # 是否保持stage1-2冻结
                     # 默认: True
```

### 输出与日志

```bash
--save-dir           # 模型保存目录
                     # 默认: checkpoints/task2_fewshot
                     
--log-interval       # 日志打印间隔（步数）
                     # 默认: 10
                     
--save-best          # 保存最佳模型（默认True）

--tensorboard        # 启用TensorBoard日志

--seed               # 随机种子
                     # 默认: 42
```

---

## 使用示例

### 1. 标准10-shot训练

```bash
python task2train.py \
    --train-meta data/cleaned/metadata/train_metadata_fewshot_10.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone resnet50 \
    --epochs 50 \
    --batch-size 8 \
    --lr 3e-4 \
    --head-lr-scale 3.0 \
    --proto-weight 0.4 \
    --save-dir checkpoints/task2_10shot
```

### 2. 极少样本5-shot训练

```bash
python task2train.py \
    --train-meta data/cleaned/metadata/train_metadata_fewshot_5.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone resnet50 \
    --epochs 100 \
    --batch-size 4 \
    --lr 2e-4 \
    --head-lr-scale 5.0 \
    --proto-weight 0.5 \
    --mixup-alpha 0.3 \
    --mixup-disable-epoch 15 \
    --dropout 0.4 \
    --label-smoothing 0.1 \
    --unfreeze-epoch 5 \
    --save-dir checkpoints/task2_5shot
```

### 3. 较多样本20-shot训练

```bash
python task2train.py \
    --train-meta data/cleaned/metadata/train_metadata_fewshot_20.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone efficientnet_b3 \
    --epochs 30 \
    --batch-size 16 \
    --lr 5e-4 \
    --head-lr-scale 2.0 \
    --proto-weight 0.3 \
    --mixup-alpha 0.2 \
    --mixup-disable-epoch 5 \
    --unfreeze-epoch 2 \
    --save-dir checkpoints/task2_20shot
```

### 4. 无Mixup纯ArcFace训练

```bash
python task2train.py \
    --train-meta data/cleaned/metadata/train_metadata_fewshot_10.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone resnet50 \
    --epochs 50 \
    --batch-size 8 \
    --lr 3e-4 \
    --mixup-alpha 0.0 \
    --proto-weight 0.0 \
    --arcface-margin 0.40 \
    --save-dir checkpoints/task2_pure_arcface
```

---

## 输出文件说明

训练完成后，`--save-dir`目录包含：

### 模型文件
```
best_model.pth          # 最佳验证Macro-F1模型
last_model.pth          # 最后一轮模型
checkpoint_epoch_N.pth  # 定期检查点（可选）
```

### 日志文件
```
training.log            # 详细训练日志
config.json             # 训练配置JSON
metrics_history.csv     # 每轮指标记录
```

### TensorBoard文件（如果启用）
```
runs/
├── scalars/
│   ├── train_loss
│   ├── train_acc
│   ├── val_loss
│   ├── val_acc
│   └── val_macro_f1
└── hparams/           # 超参数记录
```

### 可视化文件
```
training_curves.png     # 训练曲线图
confusion_matrix.png    # 最终混淆矩阵
confusion_matrix.csv    # 混淆矩阵CSV
```

---

## 核心技术原理

### 1. ArcFace vs Softmax

**传统Softmax**：
```
L = -log(exp(W_yi^T x) / Σ_j exp(W_j^T x))
```
- 优化内积 W^T x
- 受特征幅值影响大
- 类间分离度有限

**ArcFace**：
```
L = -log(exp(s·cos(θ_yi + m)) / 
         (exp(s·cos(θ_yi + m)) + Σ_j≠yi exp(s·cos(θ_j))))
```
- 优化角度距离
- 归一化后不受幅值影响
- 强制类间角度分离

**几何解释**：
```
            类别1
              ↑
              |  ← margin
样本特征 ─────→ (角度θ)
              |  ← margin
              ↓
            类别2
```

---

### 2. 原型网络（Prototypical Network）

**基本思想**：
每个类别由一个原型向量表示，分类时找最近的原型。

**原型定义**：
```
p_k = (1/|S_k|) Σ_{x∈S_k} f(x)
```
- `p_k`：第k类原型
- `S_k`：第k类支持集
- `f(x)`：特征提取函数

**分类规则**（推理时）：
```
y = argmin_k d(f(x), p_k)
```
其中 d 是距离度量（通常为欧氏距离）。

**与ArcFace融合**：
- **训练时**：ArcFace提供判别性监督，原型损失增强聚类
- **推理时**：可选择用ArcFace logits或最近原型

---

### 3. 少样本学习的挑战与解决

| 挑战 | 解决方案 |
|------|----------|
| **过拟合严重** | 高Dropout + 标签平滑 + 轻微数据增强 |
| **类内方差大** | 原型损失 + ArcFace边界 |
| **特征不稳定** | 原型EMA + 预训练骨干网络 |
| **梯度破坏** | 延迟解冻 + 头部学习率放大 |
| **样本稀缺** | Mixup + 元学习策略 |

---

### 4. 迁移学习策略

**三阶段训练**：

**阶段1：头部预热（Epoch 0-2）**
```
骨干网络: 完全冻结
分类头:   高学习率训练
目标:     快速适应新任务，避免破坏预训练特征
```

**阶段2：深层解冻（Epoch 3-10）**
```
骨干网络: 解冻layer4（最后残差块）
分类头:   继续训练
目标:     微调高层语义特征
```

**阶段3：全局优化（Epoch 10+）**
```
骨干网络: layer4持续微调
分类头:   收敛到最优
目标:     整体优化，提升泛化
```

**为什么不解冻全部？**
- layer1-2学习低级特征（边缘、纹理），通用性强
- layer4学习高级语义，任务相关性高
- 少样本下全部微调易过拟合

---

## 性能优化与调参指南

### 针对不同样本量

#### 5-shot（极少样本）
```bash
--epochs 100
--batch-size 4
--lr 2e-4
--head-lr-scale 5.0
--dropout 0.4
--mixup-alpha 0.3
--mixup-disable-epoch 20
--proto-weight 0.6
--label-smoothing 0.1
--unfreeze-epoch 10
```

**说明**：
- 更多轮次充分学习
- 更大Dropout和标签平滑
- 更强原型约束
- 延迟解冻避免梯度破坏

---

#### 10-shot（标准少样本）
```bash
--epochs 50
--batch-size 8
--lr 3e-4
--head-lr-scale 3.0
--dropout 0.3
--mixup-alpha 0.2
--mixup-disable-epoch 8
--proto-weight 0.4
--label-smoothing 0.05
--unfreeze-epoch 3
```

**说明**：
- 平衡的配置
- 适度的正则化
- 较早解冻微调

---

#### 20-shot（充足样本）
```bash
--epochs 30
--batch-size 16
--lr 5e-4
--head-lr-scale 2.0
--dropout 0.2
--mixup-alpha 0.15
--mixup-disable-epoch 5
--proto-weight 0.3
--label-smoothing 0.03
--unfreeze-epoch 2
```

**说明**：
- 较少轮次避免过拟合
- 降低正则化强度
- 更早解冻利用数据

---

### 针对不同骨干网络

#### ResNet50（推荐）
- **参数量**：~25M
- **速度**：快
- **精度**：高
- **配置**：默认设置即可

#### EfficientNet-B3
- **参数量**：~12M
- **速度**：中等
- **精度**：更高
- **调整**：
```bash
--backbone efficientnet_b3
--lr 2e-4           # 降低学习率
--batch-size 12     # 适当增大批次
```

#### ResNet101
- **参数量**：~45M
- **速度**：慢
- **精度**：最高
- **调整**：
```bash
--backbone resnet101
--batch-size 4      # 减小批次（内存限制）
--gradient-accumulation-steps 2  # 累积梯度
```

---

### 硬件资源配置

#### GPU >= 16GB
```bash
--batch-size 16
--image-size 320
--num-workers 4
```

#### GPU 8-12GB
```bash
--batch-size 8
--image-size 256
--num-workers 2
```

#### GPU 4-6GB
```bash
--batch-size 4
--image-size 224
--num-workers 0
--backbone resnet34  # 使用更小模型
```

---

## 常见问题排查

### 1. 验证准确率不提升

**症状**：训练准确率上升，验证准确率停滞或下降

**可能原因**：
- 过拟合
- 原型未更新
- 学习率过大

**解决方案**：
```bash
# 增强正则化
--dropout 0.4
--label-smoothing 0.1
--weight-decay 5e-4

# 检查原型更新
--proto-refresh-interval 1

# 降低学习率
--lr 1e-4
--head-lr-scale 2.0
```

---

### 2. 训练损失震荡

**症状**：损失曲线剧烈波动

**可能原因**：
- 学习率过大
- 批次过小
- 原型不稳定

**解决方案**：
```bash
# 调整学习率
--lr 1e-4

# 增大批次
--batch-size 16

# 平滑原型
--proto-ema 0.9
--proto-refresh-interval 2
```

---

### 3. 某些类别完全错误

**症状**：混淆矩阵显示某些类别召回率为0

**可能原因**：
- 原型未正确初始化
- 类别样本过少
- 特征不判别

**解决方案**：
```bash
# 增强原型约束
--proto-weight 0.6

# 增大ArcFace边界
--arcface-margin 0.40

# 使用Mixup
--mixup-alpha 0.3
```

---

### 4. 内存溢出

**解决方案**：
```bash
# 方案1：减小批次
--batch-size 4

# 方案2：减小图像
--image-size 224

# 方案3：使用小模型
--backbone resnet34

# 方案4：梯度累积
--gradient-accumulation-steps 4
```

---

## 高级技巧

### 1. 推理时融合策略

训练完成后，推理时可以融合多种预测：

```python
# 加载模型和原型
model.load_state_dict(torch.load('best_model.pth'))
prototypes = torch.load('prototypes.pth')

# ArcFace预测
logits, features = model(images)
pred_arcface = logits.argmax(dim=1)

# 原型预测（最近邻）
distances = torch.cdist(features, prototypes)
pred_proto = distances.argmin(dim=1)

# 融合
pred_final = (pred_arcface + pred_proto) / 2  # 简单平均
```

### 2. 测试时增强（TTA）

```python
from torchvision import transforms as T

tta_transforms = [
    T.Compose([]),                    # 原图
    T.Compose([T.HorizontalFlip()]),  # 水平翻转
    T.Compose([T.Rotate(10)]),        # 旋转10度
]

preds = []
for transform in tta_transforms:
    aug_images = transform(images)
    logits, _ = model(aug_images)
    preds.append(logits)

# 平均logits
final_logits = torch.stack(preds).mean(dim=0)
```

### 3. 伪标签半监督

利用未标注数据：

```python
# 1. 在少量标注数据上训练
model = train(labeled_data)

# 2. 对未标注数据预测
pseudo_labels = model.predict(unlabeled_data)

# 3. 选择高置信度样本
high_conf_mask = pseudo_labels.max(dim=1).values > 0.9
pseudo_labeled_data = unlabeled_data[high_conf_mask]

# 4. 联合训练
train(labeled_data + pseudo_labeled_data)
```

---

## 与Task1的对比

| 维度 | Task1（全样本） | Task2（少样本） |
|------|----------------|----------------|
| **样本量** | 每类100+ | 每类5-20 |
| **分类器** | 线性层/余弦分类器 | ArcFace头 |
| **损失函数** | CE + Focal + Center | ArcFace + Prototype |
| **数据增强** | 强增强（Mixup+CutMix） | 轻增强（Mixup+轻微色彩） |
| **正则化** | Dropout 0.3 | Dropout 0.3-0.5 |
| **训练策略** | 全参数训练 | 冻结+延迟解冻 |
| **训练轮数** | 30-50 | 50-100 |
| **批次大小** | 64-128 | 4-16 |
| **优化目标** | 整体准确率 | Macro-F1（平衡） |

---

## 理论基础与参考文献

### 核心论文

1. **ArcFace**  
   Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"  
   CVPR 2019  
   https://arxiv.org/abs/1801.07698

2. **Prototypical Networks**  
   Snell et al. "Prototypical Networks for Few-shot Learning"  
   NeurIPS 2017  
   https://arxiv.org/abs/1703.05175

3. **Few-Shot Learning Survey**  
   Wang et al. "Generalizing from a Few Examples: A Survey on Few-Shot Learning"  
   ACM Computing Surveys 2020

4. **Label Smoothing**  
   Szegedy et al. "Rethinking the Inception Architecture for Computer Vision"  
   CVPR 2016

### 相关技术

- **度量学习**：学习可判别的嵌入空间
- **元学习**：学习如何快速学习
- **迁移学习**：利用预训练知识
- **数据增强**：扩充稀缺样本

---

## 实验建议

### 超参数搜索

建议搜索的关键参数：

| 参数 | 搜索范围 | 优先级 |
|------|----------|--------|
| `lr` | [1e-5, 1e-3] | 高 |
| `proto_weight` | [0.0, 1.0] | 高 |
| `arcface_margin` | [0.2, 0.5] | 中 |
| `dropout` | [0.2, 0.5] | 中 |
| `head_lr_scale` | [2.0, 5.0] | 中 |
| `mixup_alpha` | [0.0, 0.5] | 低 |

### 消融实验

验证各策略的有效性：

```bash
# 基线（只有ArcFace）
python task2train.py --proto-weight 0.0 --mixup-alpha 0.0

# +原型损失
python task2train.py --proto-weight 0.4 --mixup-alpha 0.0

# +Mixup
python task2train.py --proto-weight 0.4 --mixup-alpha 0.2

# +延迟解冻
python task2train.py --proto-weight 0.4 --mixup-alpha 0.2 --unfreeze-epoch 3
```

---

## 总结

Task2脚本专为少样本场景设计，核心特点：

1. **ArcFace头**：强判别性角度边界
2. **原型约束**：类内聚合，提升稳定性
3. **延迟解冻**：保护预训练特征
4. **轻量增强**：避免破坏关键信息
5. **EMA平滑**：减少原型波动

通过合理配置六大策略（A-F），即使在每类仅5-10样本的极端场景下，也能达到可接受的性能。

---

**文档版本**：v1.0  
**编写日期**：2024年  
**维护者**：ShuWeiCamp Team