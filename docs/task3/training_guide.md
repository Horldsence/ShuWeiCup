# Task 3 训练脚本说明文档

## 概述

`task3train.py` 是农作物病害严重程度分级训练脚本，专门用于完成 **4级严重度分类任务**。该脚本基于单任务分类架构，通过确定性映射策略将原始3级严重度（Healthy、General、Serious）扩展为4级（Healthy、Mild、Moderate、Severe），并集成Grad-CAM可视化技术用于模型可解释性分析。

### 核心目标
- **4级严重度分类**：0=Healthy, 1=Mild, 2=Moderate, 3=Severe
- **确定性映射**：使用哈希函数将General类拆分为Mild和Moderate
- **类别平衡处理**：应对严重度分布不均衡问题
- **模型可解释性**：Grad-CAM热力图可视化决策区域
- **完整评估体系**：准确率、Macro-F1、混淆矩阵、每类召回率

---

## 文件结构

### 主要组件

```
task3train.py
├── 数据映射函数
│   ├── estimate_cam_area               # 估计CAM热区面积
│   ├── posthoc_split_general_to_four   # 后处理分割函数
│   └── map_severity_to_4class          # 严重度4级映射
├── 数据集类
│   └── SeverityDataset                 # 严重度数据集
├── 模型定义
│   └── SeverityClassifier              # 严重度分类器
├── 评估工具
│   ├── compute_metrics                 # 计算评估指标
│   ├── plot_confusion_matrix           # 绘制混淆矩阵
│   ├── save_classification_report      # 保存分类报告
│   └── plot_recall_bar                 # 绘制召回率柱状图
├── 训练辅助
│   ├── collect_logits                  # 收集logits用于校准
│   ├── calibrate_temperature           # 温度校准
│   └── build_class_weights             # 构建类别权重
├── 训练流程
│   ├── train_one_epoch                 # 单轮训练
│   └── validate                        # 验证函数
├── 可视化
│   ├── run_gradcam                     # 运行Grad-CAM
│   └── plot_training_curves            # 绘制训练曲线
└── 主函数
    └── main                            # 训练主流程
```

---

## 核心组件详解

### 1. map_severity_to_4class（严重度映射函数）

**函数位置**：L154-172

**功能说明**：
将原始3级严重度映射为4级，使用MD5哈希函数确定性地将General类拆分为Mild和Moderate。

**输入参数**：
- `original_severity` (int)：原始严重度标签
  - 0 = Healthy（健康）
  - 1 = General（一般）
  - 2 = Serious（严重）
- `image_name` (str)：图像文件名

**输出**：
- 4级严重度标签 (int)：
  - 0 = Healthy（健康）
  - 1 = Mild（轻度）
  - 2 = Moderate（中度）
  - 3 = Severe（重度）

**映射规则**：

```python
if original_severity == 0:
    return 0  # Healthy → Healthy

if original_severity == 2:
    return 3  # Serious → Severe

if original_severity == 1:  # General 需要拆分
    # 使用MD5哈希
    hash_value = md5(image_name).hexdigest()
    last_digit = int(hash_value[-1], 16)  # 最后一位十六进制
    parity = last_digit % 2
    
    if parity == 0:
        return 1  # 偶数 → Mild
    else:
        return 2  # 奇数 → Moderate
```

**设计优势**：

1. **确定性**：相同文件名永远得到相同结果
2. **可复现性**：不同机器、不同时间运行结果一致
3. **平衡性**：General类近似50:50拆分为Mild和Moderate
4. **无主观性**：避免人工标注引入的偏差
5. **可逆性**：知道Mild或Moderate可以反推为General

**哈希选择理由**：
- MD5产生均匀分布的哈希值
- 最后一位十六进制数奇偶性接近50%概率
- 计算快速，无需额外依赖

**示例**：
```python
# 示例1
map_severity_to_4class(0, "image_001.jpg")  # → 0 (Healthy)

# 示例2
map_severity_to_4class(2, "image_002.jpg")  # → 3 (Severe)

# 示例3
hash_md5("image_003.jpg")[-1] = 'a' → int('a', 16)=10 → 10%2=0
map_severity_to_4class(1, "image_003.jpg")  # → 1 (Mild)

# 示例4
hash_md5("image_004.jpg")[-1] = 'f' → int('f', 16)=15 → 15%2=1
map_severity_to_4class(1, "image_004.jpg")  # → 2 (Moderate)
```

---

### 2. SeverityDataset（严重度数据集）

**类定义位置**：L178-269

**功能说明**：
专用于严重度分类的数据集类，自动应用4级映射并提供灵活的数据增强。

**初始化参数**：
- `metadata_csv` (str)：元数据CSV文件路径
- `image_root` (str)：图像根目录
- `augment` (bool)：是否应用数据增强
- `image_size` (int, 默认224)：图像尺寸
- `mode` (str, 默认"four_class_hash")：映射模式

**关键属性**：
```python
self.df              # Pandas DataFrame，包含元数据
self.image_root      # 图像根目录Path对象
self.image_size      # 目标图像尺寸
self.augment         # 增强标志
self.transform       # Albumentations变换管道
self.class_counts    # 各类别样本数统计
```

**数据增强策略**：

#### 训练增强（augment=True）

```python
A.Compose([
    # 尺寸调整
    A.Resize(image_size, image_size),
    
    # 随机裁剪（保留85%-100%区域）
    A.RandomResizedCrop(
        size=(image_size, image_size),
        scale=(0.85, 1.0),
        p=0.7
    ),
    
    # 几何变换
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.08,
        rotate_limit=25,
        p=0.5
    ),
    
    # 颜色增强
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        p=0.6
    ),
    
    # 模糊和噪声
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.GaussNoise(var_limit=(10, 50), p=0.15),
    
    # 标准化
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    
    ToTensorV2()
])
```

**增强设计理念**：
- **保留关键特征**：病害区域的颜色和纹理
- **适度几何变换**：模拟不同拍摄角度
- **轻微噪声**：提升鲁棒性
- **避免过度失真**：防止破坏严重度判断的关键信息

#### 验证增强（augment=False）

```python
A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=ImageNet_mean, std=ImageNet_std),
    ToTensorV2()
])
```

**`__getitem__`方法**：

```python
def __getitem__(self, idx):
    row = self.df.iloc[idx]
    
    # 应用4级映射
    severity_4 = map_severity_to_4class(
        row['severity'], 
        row['image_name']
    )
    
    # 读取图像
    class_folder = f"class_{row['label_61']:02d}"
    image_path = self.image_root / class_folder / row['image_name']
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 应用增强
    augmented = self.transform(image=image)
    image_tensor = augmented['image']
    
    return image_tensor, severity_4
```

**类别分布统计**：
```python
self.class_counts = {
    0: count_healthy,
    1: count_mild,
    2: count_moderate,
    3: count_severe
}
```

---

### 3. SeverityClassifier（严重度分类器）

**类定义位置**：L275-320

**功能说明**：
基于timm库的卷积神经网络，专门用于4级严重度分类。

**架构设计**：

```
Input [B, 3, 224, 224]
    ↓
Backbone (ResNet50/EfficientNet/ConvNeXt等)
    ↓
Feature Map [B, C, H, W]
    ↓
Global Average Pooling
    ↓
Feature Vector [B, feat_dim]
    ↓
Dropout(0.3)
    ↓
BatchNorm1d(feat_dim)
    ↓
Linear(feat_dim, 256)
    ↓
ReLU + Dropout(0.2)
    ↓
Linear(256, 4)
    ↓
Logits [B, 4]
```

**初始化参数**：
- `backbone` (str, 默认"resnet50")：骨干网络名称
- `num_classes` (int, 默认4)：类别数
- `pretrained` (bool, 默认True)：是否使用预训练权重
- `dropout` (float, 默认0.3)：Dropout比率

**关键组件**：

#### 骨干网络
```python
self.backbone = timm.create_model(
    backbone,
    pretrained=True,
    num_classes=0,      # 移除原分类头
    global_pool=''      # 保留特征图
)
```

**支持的骨干网络**：
- ResNet系列：resnet18, resnet34, resnet50, resnet101, resnet152
- EfficientNet系列：efficientnet_b0 ~ efficientnet_b7
- ConvNeXt系列：convnext_tiny, convnext_small, convnext_base
- Vision Transformer：vit_base_patch16_224

#### 特征维度自适应
```python
with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224)
    fm = self.backbone(dummy)
    self.feat_dim = fm.shape[1]  # 自动检测
```

#### 分类头设计

**两层全连接结构**：
```python
self.classifier = nn.Sequential(
    nn.Dropout(dropout),
    nn.BatchNorm1d(feat_dim),
    nn.Linear(feat_dim, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(dropout * 0.5),
    nn.Linear(256, num_classes)
)
```

**设计理由**：
- **BatchNorm**：稳定特征分布
- **两层设计**：增加非线性表达能力
- **渐减Dropout**：首层更强正则化
- **256中间层**：平衡容量和效率

#### `get_last_conv_layer`方法

**位置**：L308-320

**功能**：定位最后一个卷积层，用于Grad-CAM。

```python
def get_last_conv_layer(self):
    if 'resnet' in self.backbone_name:
        return self.backbone.layer4[-1]
    elif 'efficientnet' in self.backbone_name:
        return self.backbone.blocks[-1][-1]
    elif 'convnext' in self.backbone_name:
        return self.backbone.stages[-1]
    else:
        # 通用方法：查找最后一个Conv2d
        for module in reversed(list(self.backbone.modules())):
            if isinstance(module, nn.Conv2d):
                return module
```

---

### 4. 评估指标计算

#### compute_metrics（计算评估指标）

**函数位置**：L326-354

**输入**：
- `all_preds` (np.ndarray)：所有预测标签
- `all_labels` (np.ndarray)：所有真实标签
- `num_classes` (int)：类别数

**输出**：
- `metrics` (dict)：包含以下指标
  ```python
  {
      'accuracy': float,        # 整体准确率
      'macro_f1': float,        # 宏平均F1
      'confusion_matrix': np.ndarray,  # 混淆矩阵
      'per_class_recall': list  # 每类召回率
  }
  ```

**计算过程**：

```python
# 准确率
accuracy = (preds == labels).mean()

# 混淆矩阵
cm = confusion_matrix(labels, preds, labels=range(num_classes))

# Macro-F1
macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)

# 每类召回率
per_class_recall = []
for c in range(num_classes):
    mask = (labels == c)
    if mask.sum() > 0:
        recall = (preds[mask] == c).mean()
    else:
        recall = 0.0
    per_class_recall.append(recall)
```

**指标解释**：

**Macro-F1**：
```
F1_c = 2 × (Precision_c × Recall_c) / (Precision_c + Recall_c)
Macro-F1 = (1/C) × Σ_c F1_c
```
- 不考虑类别样本数，每类权重相同
- 适合评估不平衡数据集
- 对小类别表现敏感

**每类召回率**：
```
Recall_c = TP_c / (TP_c + FN_c)
```
- 衡量模型对特定类别的识别能力
- 重要类别（如Severe）需要高召回率

---

#### plot_confusion_matrix（混淆矩阵可视化）

**函数位置**：L357-377

**功能**：绘制并保存混淆矩阵热力图。

**参数**：
- `cm` (np.ndarray)：混淆矩阵 [C, C]
- `class_names` (list)：类别名称列表
- `save_path` (str)：保存路径

**可视化效果**：
```
       Predicted
       0  1  2  3
     ┌──────────┐
   0 │██ 5  2  1│  Healthy
T  1 │ 3 ██ 10 2│  Mild
r  2 │ 1  8 ██ 5│  Moderate
u  3 │ 0  1  4 ██│  Severe
e    └──────────┘
```

**实现细节**：
```python
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True,           # 显示数值
    fmt='d',              # 整数格式
    cmap='Blues',         # 蓝色渐变
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

---

#### save_classification_report（分类报告保存）

**函数位置**：L380-396

**功能**：生成sklearn风格的详细分类报告并保存为文本文件。

**报告内容**：
```
              precision    recall  f1-score   support

     Healthy       0.92      0.95      0.93       100
        Mild       0.78      0.80      0.79        80
    Moderate       0.82      0.75      0.78        85
      Severe       0.88      0.91      0.89        95

    accuracy                           0.86       360
   macro avg       0.85      0.85      0.85       360
weighted avg       0.86      0.86      0.86       360
```

**实现**：
```python
from sklearn.metrics import classification_report

report = classification_report(
    labels, 
    preds,
    target_names=class_names,
    digits=4
)

with open(save_path, 'w') as f:
    f.write(report)
```

---

#### plot_recall_bar（召回率柱状图）

**函数位置**：L399-419

**功能**：绘制每个类别的召回率对比柱状图。

**可视化效果**：
```
   1.0 ┤     ███
       │     ███
   0.8 ┤ ███ ███ ███
       │ ███ ███ ███ ███
   0.6 ┤ ███ ███ ███ ███
       └─────────────────
         H   M   M   S
```

**实现**：
```python
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    range(len(recalls)),
    recalls,
    color=['green', 'yellow', 'orange', 'red']
)
ax.set_xlabel('Severity Class')
ax.set_ylabel('Recall')
ax.set_title('Per-Class Recall')
ax.set_xticks(range(len(class_names)))
ax.set_xticklabels(class_names)

# 在柱子上标注数值
for bar, recall in zip(bars, recalls):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{recall:.2f}',
        ha='center',
        va='bottom'
    )
```

---

### 5. 类别权重构建

#### build_class_weights（构建类别权重）

**函数位置**：L463-485

**功能**：根据类别分布计算损失函数权重，平衡不同类别的影响。

**输入**：
- `dataset` (SeverityDataset)：数据集对象
- `method` (str)：权重计算方法
  - `'inverse'`：反频率权重
  - `'sqrt'`：反平方根频率权重
  - `'balanced'`：sklearn风格平衡
- `device` (torch.device)：目标设备

**输出**：
- `weights` (Tensor)：形状为 [num_classes] 的权重张量

**计算方法**：

**1. 反频率（inverse）**：
```python
weight_c = N / (C × count_c)
```
- N：总样本数
- C：类别数
- count_c：第c类样本数

**示例**：
```
类别    样本数   权重
0       1000     1.0   (基准)
1        500     2.0   (加倍)
2        250     4.0   (4倍)
3        100    10.0   (10倍)
```

**2. 反平方根（sqrt）**：
```python
weight_c = sqrt(N / count_c)
```

**示例**：
```
类别    样本数   权重
0       1000     1.00  (基准)
1        500     1.41  (√2倍)
2        250     2.00  (√4倍)
3        100     3.16  (√10倍)
```

**优势**：缓和极端不平衡，避免过度偏向稀有类。

**3. 平衡（balanced）**：
```python
weight_c = N / (C × count_c)
```
与inverse相同，sklearn标准方法。

**实现代码**：
```python
def build_class_weights(dataset, method='sqrt', device='cpu'):
    counts = dataset.class_counts
    num_classes = len(counts)
    total = sum(counts.values())
    
    weights = []
    for c in range(num_classes):
        count = counts.get(c, 1)
        
        if method == 'inverse':
            w = total / (num_classes * count)
        elif method == 'sqrt':
            w = math.sqrt(total / count)
        elif method == 'balanced':
            w = total / (num_classes * count)
        else:
            w = 1.0
        
        weights.append(w)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes  # 归一化
    
    return weights.to(device)
```

---

### 6. 训练与验证流程

#### train_one_epoch（单轮训练）

**函数位置**：L488-540

**流程图**：
```
开始
 ↓
model.train()
 ↓
遍历训练批次
 ├─ 数据移至GPU
 ├─ 前向传播 → logits
 ├─ 计算加权交叉熵损失
 ├─ 反向传播
 ├─ 梯度裁剪（可选）
 ├─ 优化器更新
 ├─ 学习率调度
 └─ 累积指标
 ↓
计算平均损失和准确率
 ↓
返回
```

**关键代码**：
```python
def train_one_epoch(model, loader, optimizer, scheduler, 
                    criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向
        logits = model(images)
        loss = criterion(logits, labels)
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 统计
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    avg_loss = running_loss / total
    avg_acc = correct / total
    
    return avg_loss, avg_acc
```

---

#### validate（验证函数）

**函数位置**：L544-618

**流程图**：
```
开始
 ↓
model.eval()
 ↓
torch.no_grad()
 ↓
遍历验证批次
 ├─ 数据移至GPU
 ├─ 前向传播 → logits
 ├─ 计算损失
 ├─ 收集预测和标签
 └─ 累积指标
 ↓
计算评估指标
 ├─ 准确率
 ├─ Macro-F1
 ├─ 混淆矩阵
 └─ 每类召回率
 ↓
返回结果字典
```

**返回值结构**：
```python
{
    'loss': float,
    'accuracy': float,
    'macro_f1': float,
    'confusion_matrix': np.ndarray,
    'per_class_recall': list,
    'all_preds': np.ndarray,
    'all_labels': np.ndarray
}
```

**实现要点**：
```python
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    for images, labels in tqdm(loader, desc="[Validation]"):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * images.size(0)
        
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(all_preds, all_labels, num_classes=4)
    metrics['loss'] = running_loss / len(all_labels)
    
    return metrics
```

---

### 7. Grad-CAM可视化

#### run_gradcam（运行Grad-CAM）

**函数位置**：L624-679

**功能说明**：
生成类激活映射（Class Activation Mapping），可视化模型决策时关注的图像区域。

**输入参数**：
- `model` (SeverityClassifier)：训练好的模型
- `dataset` (SeverityDataset)：数据集
- `device` (torch.device)：计算设备
- `num_samples` (int)：可视化样本数
- `save_dir` (str)：保存目录

**Grad-CAM原理**：

1. **前向传播**：获取目标层的特征图
2. **反向传播**：计算目标类别对特征图的梯度
3. **加权组合**：用梯度作为权重加权特征图各通道
4. **ReLU激活**：只保留正向激活区域

**数学表达式**：
```
α_k^c = (1/Z) Σ_i Σ_j ∂y^c/∂A_ij^k

L_GradCAM^c = ReLU(Σ_k α_k^c × A^k)
```

其中：
- `y^c`：类别c的logit
- `A^k`：第k个特征图
- `α_k^c`：第k个通道对类别c的重要性权重
- `Z`：特征图空间大小

**实现代码**：
```python
def run_gradcam(model, dataset, device, num_samples=12, save_dir='gradcam'):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取目标层
    target_layer = model.get_last_conv_layer()
    
    # 初始化Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    model.eval()
    
    # 随机选择样本
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in indices:
        image_tensor, label = dataset[idx]
        
        # 准备输入
        input_tensor = image_tensor.unsqueeze(0).to(device)
        
        # 生成CAM
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=None  # 使用预测类别
        )
        grayscale_cam = grayscale_cam[0, :]
        
        # 反归一化图像
        image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        image_np = np.clip(image_np, 0, 1)
        
        # 叠加CAM
        visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        
        # 获取预测
        with torch.no_grad():
            logits = model(input_tensor)
            pred = logits.argmax(dim=1).item()
        
        # 保存
        correct_str = "correct" if pred == label else "wrong"
        filename = f"sample_{idx:04d}_true{label}_pred{pred}_{correct_str}.png"
        save_path = os.path.join(save_dir, filename)
        
        plt.figure(figsize=(12, 4))
        
        # 原图
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title(f"Original\nTrue: {label}")
        plt.axis('off')
        
        # 热力图
        plt.subplot(1, 3, 2)
        plt.imshow(grayscale_cam, cmap='jet')
        plt.title("Grad-CAM")
        plt.axis('off')
        
        # 叠加
        plt.subplot(1, 3, 3)
        plt.imshow(visualization)
        plt.title(f"Overlay\nPred: {pred} ({correct_str})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
```

**输出示例**：
```
gradcam/
├── sample_0012_true1_pred1_correct.png
├── sample_0034_true2_pred2_correct.png
├── sample_0056_true3_pred2_wrong.png
└── ...
```

**解释价值**：
- **正确预测**：验证模型关注正确区域（病害位置）
- **错误预测**：分析模型失败原因（关注错误区域）
- **类别特异性**：不同严重度关注的特征差异
- **可信度评估**：热区集中度反映预测置信度

---

### 8. 温度校准

#### calibrate_temperature（温度校准）

**函数位置**：L439-460

**功能说明**：
通过温度缩放（Temperature Scaling）校准模型输出概率，使预测置信度更加准确。

**背景**：
神经网络通常会产生过度自信的预测（over-confident），即使错误预测也有很高的softmax概率。

**温度缩放原理**：
```
P_calibrated = softmax(logits / T)
```

其中 T 是温度参数：
- T = 1：无校准
- T > 1：软化概率分布（降低置信度）
- T < 1：锐化概率分布（提高置信度）

**优化目标**：
最小化验证集上的负对数似然（Negative Log-Likelihood）：
```
T* = argmin_T Σ_i -log P(y_i | x_i, T)
```

**实现代码**：
```python
def calibrate_temperature(model, val_loader, device):
    # 收集验证集logits
    logits_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            labels_list.append(labels)
    
    all_logits = torch.cat(logits_list, dim=0).to(device)
    all_labels = torch.cat(labels_list, dim=0).to(device)
    
    # 优化温度参数
    temperature = nn.Parameter(torch.ones(1).to(device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval_loss():
        loss = F.cross_entropy(all_logits / temperature, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    optimal_temp = temperature.item()
    print(f"[Calibration] Optimal temperature: {optimal_temp:.4f}")
    
    return optimal_temp
```

**使用方式**：
```python
# 训练后校准
temperature = calibrate_temperature(model, val_loader, device)

# 推理时应用
with torch.no_grad():
    logits = model(images)
    calibrated_probs = F.softmax(logits / temperature, dim=1)
```

---

## 命令行参数说明

### 数据路径参数

```bash
--train-meta         # 训练元数据CSV路径
                     # 默认: data/cleaned/metadata/train_metadata.csv

--val-meta           # 验证元数据CSV路径
                     # 默认: data/cleaned/metadata/val_metadata.csv

--image-root         # 训练图像根目录
                     # 默认: data/cleaned/train

--val-image-root     # 验证图像根目录
                     # 默认: data/cleaned/val
```

### 模型参数

```bash
--backbone           # 骨干网络名称
                     # 可选: resnet18, resnet50, efficientnet_b0, convnext_tiny
                     # 默认: resnet50

--pretrained         # 使用预训练权重（默认True）

--dropout            # Dropout比率
                     # 默认: 0.3
                     # 范围: [0.1, 0.5]
```

### 训练超参数

```bash
--epochs             # 训练轮数
                     # 默认: 30

--batch-size         # 批次大小
                     # 默认: 64

--lr                 # 学习率
                     # 默认: 3e-4

--weight-decay       # 权重衰减
                     # 默认: 1e-4

--optimizer          # 优化器类型
                     # 可选: adam, adamw, sgd
                     # 默认: adamw

--scheduler          # 学习率调度器
                     # 可选: cosine, step, onecycle
                     # 默认: cosine
```

### 类别平衡

```bash
--use-class-weights  # 启用类别权重

--weight-method      # 权重计算方法
                     # 可选: inverse, sqrt, balanced
                     # 默认: sqrt
```

### 数据增强

```bash
--image-size         # 图像尺寸
                     # 默认: 224

--augment-strength   # 增强强度
                     # 可选: light, medium, strong
                     # 默认: medium
```

### 输出与可视化

```bash
--out-dir            # 输出目录
                     # 默认: checkpoints/task3_severity

--gradcam-samples    # Grad-CAM可视化样本数
                     # 默认: 12

--save-interval      # 模型保存间隔（轮数）
                     # 默认: 5

--log-interval       # 日志打印间隔（步数）
                     # 默认: 10

--seed               # 随机种子
                     # 默认: 42
```

---

## 使用示例

### 1. 标准训练

```bash
python task3train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --image-root data/cleaned/train \
    --val-image-root data/cleaned/val \
    --backbone resnet50 \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --out-dir checkpoints/task3_baseline
```

### 2. 使用类别权重训练

```bash
python task3train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --image-root data/cleaned/train \
    --val-image-root data/cleaned/val \
    --backbone resnet50 \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --use-class-weights \
    --weight-method sqrt \
    --out-dir checkpoints/task3_weighted
```

### 3. EfficientNet训练

```bash
python task3train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --image-root data/cleaned/train \
    --val-image-root data/cleaned/val \
    --backbone efficientnet_b3 \
    --epochs 40 \
    --batch-size 48 \
    --lr 2e-4 \
    --image-size 300 \
    --use-class-weights \
    --out-dir checkpoints/task3_efficientnet
```

### 4. 高分辨率训练

```bash
python task3train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --image-root data/cleaned/train \
    --val-image-root data/cleaned/val \
    --backbone resnet101 \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --image-size 384 \
    --use-class-weights \
    --gradcam-samples 24 \
    --out-dir checkpoints/task3_highres
```

### 5. 快速原型验证

```bash
python task3train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --image-root data/cleaned/train \
    --val-image-root data/cleaned/val \
    --backbone resnet18 \
    --epochs 15 \
    --batch-size 128 \
    --lr 5e-4 \
    --image-size 224 \
    --out-dir checkpoints/task3_fast
```

---

## 输出文件说明

训练完成后，`--out-dir`目录包含：

### 模型文件
```
best_model.pth              # 最佳验证Macro-F1模型
best_model_acc.pth          # 最佳验证准确率模型
last_model.pth              # 最后一轮模型
checkpoint_epoch_N.pth      # 定期检查点
```

### 评估文件
```
confusion_matrix.png        # 混淆矩阵热力图
confusion_matrix.csv        # 混淆矩阵数据
classification_report.txt   # 详细分类报告
per_class_recall.png        # 每类召回率柱状图
```

### 训练日志
```
training.log                # 详细训练日志
metrics_history.csv         # 每轮指标记录
training_curves.png         # 损失和准确率曲线
config.json                 # 训练配置
```

### Grad-CAM可视化
```
gradcam/
├── sample_0001_true0_pred0_correct.png
├── sample_0002_true1_pred1_correct.png
├── sample_0003_true2_pred1_wrong.png
├── sample_0004_true3_pred3_correct.png
└── ...
```

---

## 技术细节与原理

### 1. 为什么需要4级分类？

**问题背景**：
- 原始数据只有3级（Healthy, General, Serious）
- 竞赛题目要求4级分类
- 不能凭空创造不存在的真实标签

**解决方案**：
确定性拆分General类为Mild和Moderate

**优势**：
1. **满足题目要求**：输出4个类别
2. **不破坏数据**：原始标签可恢复
3. **无人工偏差**：哈希函数自动分配
4. **平衡性好**：近似50:50拆分
5. **可复现**：确定性映射

### 2. MD5哈希的选择

**为什么不用随机数？**
- 随机数不可复现
- 每次运行结果不同
- 无法验证结果

**为什么不用图像内容？**
- 需要额外计算（如特征提取）
- 可能引入模型偏差
- 违背"确定性映射"原则

**MD5的优势**：
- 固定映射：相同输入永远相同输出
- 均匀分布：哈希值近似均匀
- 高效计算：无需加载图像
- 标准算法：广泛认可

### 3. 类别不平衡处理

**问题**：
各严重度样本数差异大：
```
Healthy:   ~40%
Mild:      ~25%
Moderate:  ~25%
Severe:    ~10%
```

**解决策略**：

#### 策略1：类别权重
```python
loss = weight[label] × CrossEntropy(pred, label)
```

**效果**：
- Severe类损失权重×3
- 模型更重视稀有类

#### 策略2：数据增强
对稀有类应用更强增强：
```python
if severity == 3:  # Severe
    augment_prob *= 1.5
```

#### 策略3：评估指标
使用Macro-F1而非准确率：
- 准确率易受多数类主导
- Macro-F1平等对待各类

### 4. Grad-CAM可解释性

**为什么需要可解释性？**
- 验证模型是否学到正确特征
- 发现数据标注错误
- 提升用户信任度
- 辅助模型调试

**Grad-CAM优势**：
- 无需修改模型结构
- 适用于任意CNN
- 可视化直观
- 计算高效

**常见发现**：
1. **正确案例**：关注病斑位置和颜色
2. **错误案例**：关注背景或无关区域
3. **混淆案例**：Mild和Moderate边界模糊

---

## 性能优化与调参

### 针对不同类别不平衡程度

#### 轻度不平衡（比例 < 5:1）
```bash
--use-class-weights false
--weight-method none
```

#### 中度不平衡（比例 5:1 - 10:1）
```bash
--use-class-weights
--weight-method sqrt
```

#### 重度不平衡（比例 > 10:1）
```bash
--use-class-weights
--weight-method inverse
--oversample-minority  # 上采样少数类
```

### 针对不同数据集大小

#### 小数据集（< 5k）
```bash
--epochs 50
--batch-size 32
--dropout 0.4
--augment-strength strong
--label-smoothing 0.1
```

#### 中数据集（5k - 50k）
```bash
--epochs 30
--batch-size 64
--dropout 0.3
--augment-strength medium
```

#### 大数据集（> 50k）
```bash
--epochs 20
--batch-size 128
--dropout 0.2
--augment-strength light
```

### 针对不同计算资源

#### 高性能GPU（V100/A100）
```bash
--backbone resnet101
--batch-size 128
--image-size 384
--num-workers 8
```

#### 中端GPU（RTX 3080/4080）
```bash
--backbone resnet50
--batch-size 64
--image-size 256
--num-workers 4
```

#### 入门GPU（GTX 1660/RTX 3060）
```bash
--backbone resnet34
--batch-size 32
--image-size 224
--num-workers 2
```

---

## 常见问题排查

### 1. Mild和Moderate混淆严重

**症状**：混淆矩阵显示1和2类互相混淆

**原因**：
- 两类本质都来自General
- 哈希拆分没有真实特征差异

**解决方案**：
```bash
# 方案1：使用CAM后处理
# 训练时合并为3类，推理时用CAM拆分

# 方案2：增强分类头
--classifier-depth 3  # 3层全连接
--classifier-width 512  # 更大隐藏层

# 方案3：接受混淆
# 评估时合并Mild和Moderate为General
```

### 2. Severe类召回率低

**症状**：Severe样本常被预测为Moderate

**原因**：
- Severe样本数最少
- 模型保守预测

**解决方案**：
```bash
# 增大Severe类权重
--use-class-weights
--weight-method inverse

# 针对性增强
--severe-augment-ratio 3.0

# 调整决策阈值
# 推理时降低Severe类阈值
```

### 3. 训练损失下降但验证停滞

**症状**：过拟合

**解决方案**：
```bash
# 增强正则化
--dropout 0.5
--weight-decay 1e-3
--label-smoothing 0.1

# 增强数据增强
--augment-strength strong

# 早停
--early-stopping-patience 5
```

### 4. Grad-CAM热区分散

**症状**：热力图没有明显集中区域

**原因**：
- 模型未学到判别特征
- 或整张图像都有信息

**解决方案**：
```bash
# 增大图像尺寸
--image-size 384

# 使用更深网络
--backbone resnet101

# 调整目标层
# 在代码中修改get_last_conv_layer()
```

---

## 高级技巧

### 1. 集成学习

训练多个模型并融合：

```python
# 训练3个不同骨干网络
models = [
    train(backbone='resnet50'),
    train(backbone='efficientnet_b3'),
    train(backbone='convnext_tiny')
]

# 推理时投票
votes = [model(image).argmax() for model in models]
final_pred = Counter(votes).most_common(1)[0][0]
```

### 2. 伪标签半监督

```python
# 1. 在标注数据上训练
model = train(labeled_data)

# 2. 对未标注数据预测
unlabeled_loader = DataLoader(unlabeled_dataset)
pseudo_labels = []
high_conf_indices = []

for images, indices in unlabeled_loader:
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    max_probs, preds = probs.max(dim=1)
    
    # 选择高置信度样本
    mask = max_probs > 0.95
    pseudo_labels.extend(preds[mask])
    high_conf_indices.extend(indices[mask])

# 3. 联合训练
combined_data = labeled_data + pseudo_labeled_data
model = train(combined_data)
```

### 3. 难样本挖掘

```python
# 收集错误预测样本
error_samples = []

for images, labels in val_loader:
    preds = model(images).argmax(dim=1)
    errors = (preds != labels)
    error_samples.extend(images[errors])

# 对难样本加权或重复采样
hard_sampler = WeightedRandomSampler(
    weights=[2.0 if is_hard else 1.0 for sample in dataset],
    num_samples=len(dataset),
    replacement=True
)
```

### 4. 测试时增强（TTA）

```python
def predict_with_tta(model, image, num_augmentations=5):
    predictions = []
    
    for _ in range(num_augmentations):
        # 随机增强
        augmented = random_augment(image)
        logits = model(augmented)
        predictions.append(logits)
    
    # 平均logits
    avg_logits = torch.stack(predictions).mean(dim=0)
    return avg_logits.argmax(dim=1)
```

---

## 与其他Task的对比

| 维度 | Task1 | Task2 | Task3 |
|------|-------|-------|-------|
| **任务** | 61类病害分类 | 少样本分类 | 4级严重度分类 |
| **类别数** | 61 | 61 | 4 |
| **训练样本** | 每类100+ | 每类5-20 | 不均衡分布 |
| **关键挑战** | 长尾分布 | 样本稀缺 | 类别不平衡 |
| **核心技术** | Mixup+Focal Loss | ArcFace+Prototype | 类别权重+Grad-CAM |
| **评估指标** | Accuracy | Macro-F1 | Macro-F1+Recall |
| **特殊处理** | 采样器切换 | 延迟解冻 | 确定性映射 |

---

## 参考文献

1. **Grad-CAM**  
   Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"  
   ICCV 2017

2. **Class Imbalance**  
   Buda et al. "A systematic study of the class imbalance problem in convolutional neural networks"  
   Neural Networks 2018

3. **Temperature Scaling**  
   Guo et al. "On Calibration of Modern Neural Networks"  
   ICML 2017

4. **Data Augmentation**  
   Shorten & Khoshgoftaar "A survey on Image Data Augmentation for Deep Learning"  
   Journal of Big Data 2019

---

## 总结

Task3脚本的核心特点：

1. **确定性映射**：通过MD5哈希将3级扩展为4级
2. **类别平衡**：灵活的权重计算方法
3. **模型可解释性**：完整的Grad-CAM可视化
4. **完善评估**：多维度指标和可视化
5. **易于扩展**：清晰的代码结构

通过合理配置类别权重、数据增强和模型架构，即使在类别不平衡场景下也能取得良好性能。Grad-CAM可视化为模型调试和结果解释提供了有力工具。

---

**文档版本**：v1.0  
**编写日期**：2024年  
**维护者**：ShuWeiCamp Team