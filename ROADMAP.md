# 农作物病害识别系统技术路线图

> "Bad programmers worry about the code. Good programmers worry about data structures and their relationships."  
> —— 献给真正关心数据结构的工程师

---

## 【核心判断】

✅ **值得做：这是个真实的农业问题，不是学术界臆想出来的玩具问题**

### 关键洞察

1. **数据结构**：61个类别不是扁平的，而是天然的三层层次结构（作物-疾病-严重程度）
2. **复杂度消除**：一个统一架构可以解决所有4个任务，不要为每个任务单独设计
3. **最大风险**：数据清理不彻底会浪费大量调试时间

---

## 【问题本质分析】

### 三个关键问题

#### 1. "这是个真问题还是臆想出来的？"
- ✅ **真问题**：农业病害识别有实际应用价值
- ✅ **真数据**：30,000+高质量标注图像
- ✅ **真需求**：早期检测可避免大规模传播，减少农药使用

#### 2. "有更简单的方法吗？"
- ❌ 不要上来就用最新的Transformer、Diffusion模型
- ✅ 从成熟的ResNet/ConvNeXt开始，先把最简单的跑通
- ✅ 利用数据的层次结构，而不是暴力61分类

#### 3. "会破坏什么吗？"
- 新系统，无向后兼容问题
- 但要注意：接口设计要稳定，输出格式要统一
- 部署后不能随意改变模型输出结构

---

## 【数据结构设计】

### 核心数据表示（消除特殊情况）

```python
# 好品味的数据结构
class DiseaseLabel:
    """统一的三元组表示，消除Healthy作为特殊情况"""
    crop_type: str      # 10类：Apple, Cherry, Corn, ...
    disease: str | None # 28类（None表示健康）
    severity: int       # 0=健康, 1=一般, 2=严重
    label_61: int       # 原始61类标签（用于Task 1）

# 示例
Apple Healthy        -> (Apple, None, 0)
Apple Scab General   -> (Apple, Scab, 1)
Apple Scab Serious   -> (Apple, Scab, 2)
Apple Frogeye Spot   -> (Apple, Frogeye, 1)  # 只有一级
```

**为什么这样设计？**
- 消除了"Healthy"作为特殊情况的处理
- 统一了所有标签的表示方式
- 自然支持多任务学习
- 易于扩展新疾病/新严重程度

---

## 【架构设计】

### 统一多任务架构

```
Input Image (3×224×224)
    ↓
Feature Extractor (ResNet50/ConvNeXt)
    ↓
Feature Vector (2048-d)
    ↓
    ├─→ Head 1: 61-class classifier     → Task 1: 标准分类
    ├─→ Head 2: 10-class crop           → 辅助任务（提升泛化）
    ├─→ Head 3: 28-class disease        → 辅助任务（层次化）
    └─→ Head 4: 4-class severity        → Task 3: 严重程度分级

Task 2 (少样本): 冻结backbone，只训练分类head
Task 4 (多任务): 同时训练所有head，共享特征
```

**参数预算**：
- Backbone (ResNet50): ~23M
- 4个分类head: ~0.5M
- **总计**: ~24M << 50M限制 ✅

---

## 【实施路线】

### Phase 0: 环境准备 (Day 0, 4小时)

#### 技术栈选择（实用主义）

```python
# 不要用最新最炫的，用经过验证的
torch==2.1.0           # 稳定版本
timm==0.9.12           # 预训练模型库
albumentations==1.3.1  # 数据增强首选
opencv-python==4.8.1   # 图像处理
pandas==2.1.3          # 数据管理
matplotlib==3.8.2      # 可视化
grad-cam==1.4.8        # 可解释性
```

#### 项目结构

```
project/
├── data/
│   ├── raw/                    # 原始数据
│   ├── cleaned/                # 清理后的数据
│   └── splits/                 # 训练/验证划分
├── src/
│   ├── data/
│   │   ├── dataset.py         # 数据集定义
│   │   ├── transforms.py      # 数据增强
│   │   └── cleaner.py         # 数据清理
│   ├── models/
│   │   ├── baseline.py        # 单任务baseline
│   │   └── multitask.py       # 多任务模型
│   ├── training/
│   │   ├── trainer.py         # 训练循环
│   │   └── metrics.py         # 评估指标
│   └── utils/
│       ├── visualization.py   # Grad-CAM等
│       └── logger.py          # 日志记录
├── configs/
│   ├── task1_baseline.yaml
│   ├── task2_fewshot.yaml
│   └── task4_multitask.yaml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_error_analysis.ipynb
└── results/
    └── experiments/
```

---

### Phase 1: 数据理解与清理 (Day 1-2, 12小时)

**优先级最高！数据结构错了，后面全是补丁。**

#### Step 1.1: 数据探索 (4小时)

```python
# notebooks/01_data_exploration.ipynb
# 核心问题：
# 1. 每个类别有多少样本？（分布不平衡？）
# 2. 图像尺寸、质量如何？
# 3. "duplicate"标记的图像占比？
# 4. 是否有标注错误？

import json
import pandas as pd
from collections import Counter

def analyze_dataset(json_path):
    """分析数据集统计信息"""
    # 读取JSON标签
    # 统计类别分布
    # 检查duplicate标记
    # 可视化样本
    pass

# 关键指标
stats = {
    'total_samples': 32768,
    'num_classes': 61,
    'class_distribution': Counter(),  # 每类样本数
    'duplicate_count': 0,             # 重复样本数
    'image_sizes': [],                # 图像尺寸分布
    'label_hierarchy': {}             # 层次化标签结构
}
```

#### Step 1.2: 数据清理 (4小时)

```python
# src/data/cleaner.py

def clean_dataset(src_dir, dst_dir, json_path):
    """
    清理数据集：
    1. 删除标记为"duplicate"的图像
    2. 检查文件完整性（能否正常读取）
    3. 统一图像格式
    4. 构建层次化标签映射
    """
    # 好品味：一次遍历完成所有检查
    for image_info in json_data:
        if 'duplicate' in image_info.get('tags', []):
            continue  # 直接跳过，不复制
        
        # 验证图像可读
        try:
            img = cv2.imread(image_path)
            assert img is not None
        except:
            print(f"Corrupted: {image_path}")
            continue
        
        # 解析层次化标签
        label = parse_hierarchical_label(image_info['label'])
        
        # 保存清理后的数据
        save_cleaned_sample(image, label, dst_dir)
```

#### Step 1.3: 数据划分策略 (4小时)

```python
# 关键决策：如何划分训练/验证集？

# 方案1: 随机划分（简单但可能不公平）
# 方案2: 分层采样（保持类别分布）✅ 推荐
# 方案3: 按作物类型划分（测试泛化能力）

def stratified_split(data, val_ratio=0.15):
    """
    分层采样，确保每个类别在训练/验证集中比例一致
    注意：原数据已有validation set，这里是为了交叉验证
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    splitter = StratifiedShuffleSplit(n_splits=1, 
                                      test_size=val_ratio,
                                      random_state=42)
    
    for train_idx, val_idx in splitter.split(data, labels):
        return train_idx, val_idx
```

---

### Phase 2: Baseline实现 (Day 3-4, 16小时)

**目标：快速构建一个端到端的完整系统**

#### Step 2.1: 数据加载Pipeline (4小时)

```python
# src/data/dataset.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

class AgriDiseaseDataset:
    """简单但完整的数据集类"""
    
    def __init__(self, data_dir, labels_df, transform=None):
        self.data_dir = data_dir
        self.labels_df = labels_df
        self.transform = transform
        
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.data_dir / self.labels_df.iloc[idx]['image_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取标签（多任务）
        labels = {
            'label_61': self.labels_df.iloc[idx]['label_61'],
            'crop': self.labels_df.iloc[idx]['crop_id'],
            'disease': self.labels_df.iloc[idx]['disease_id'],
            'severity': self.labels_df.iloc[idx]['severity'],
        }
        
        # 数据增强
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, labels

# 数据增强策略（从简单开始）
train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

#### Step 2.2: Baseline模型 (4小时)

```python
# src/models/baseline.py

import timm
import torch.nn as nn

class BaselineModel(nn.Module):
    """简单的单任务分类器（Task 1）"""
    
    def __init__(self, num_classes=61, pretrained=True):
        super().__init__()
        # 使用timm加载预训练模型
        self.model = timm.create_model(
            'resnet50',  # 或 'convnext_small'
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)

# 为什么选ResNet50？
# 1. 参数量合适（23M < 50M）
# 2. 训练稳定，容易收敛
# 3. 预训练权重质量高（ImageNet-1K）
# 4. 社区成熟，遇到问题容易找到解决方案
```

#### Step 2.3: 训练循环 (4小时)

```python
# src/training/trainer.py

class Trainer:
    """简洁的训练器"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 处理类别不平衡
        class_weights = compute_class_weights(train_loader.dataset)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 优化器（不要用SGD，调参太麻烦）
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度（简单有效）
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )
    
    def train_epoch(self):
        """单个epoch训练"""
        self.model.train()
        total_loss = 0
        
        for images, labels in self.train_loader:
            images, labels = images.cuda(), labels.cuda()
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total
```

#### Step 2.4: 初次训练与评估 (4小时)

```bash
# 训练baseline（Task 1）
python train.py --config configs/task1_baseline.yaml \
                --epochs 50 \
                --batch-size 64 \
                --lr 1e-4

# 预期结果：
# - Accuracy: 85-90%（第一次尝试）
# - 训练时间: 2-4小时（单卡V100）
```

---

### Phase 3: 迭代优化 (Day 5-6, 16小时)

**基于实际问题进行针对性优化，不要盲目堆trick**

#### Step 3.1: 错误分析 (4小时)

```python
# notebooks/02_error_analysis.ipynb

def analyze_errors(model, val_loader):
    """
    分析模型错误的类型：
    1. 哪些类别容易混淆？
    2. 错误样本有什么共同特征？
    3. 是数据问题还是模型问题？
    """
    errors = []
    
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        
        # 收集错误样本
        wrong_mask = predictions != labels
        if wrong_mask.any():
            for idx in wrong_mask.nonzero():
                errors.append({
                    'image': images[idx],
                    'true_label': labels[idx],
                    'pred_label': predictions[idx],
                    'confidence': outputs[idx].softmax(dim=0).max()
                })
    
    # 可视化混淆矩阵
    plot_confusion_matrix(errors)
    
    # 分析高频错误
    error_pairs = Counter([
        (e['true_label'], e['pred_label']) for e in errors
    ])
    
    return errors, error_pairs

# 常见错误类型及对策：
# 1. 同一作物不同疾病混淆 → 增强疾病特征（数据增强）
# 2. 严重程度判断错误 → 使用层次化分类器
# 3. 背景干扰 → 更强的数据增强（random crop）
# 4. 某些类别始终错误 → 检查数据质量
```

#### Step 3.2: 针对性改进 (8小时)

```python
# 改进策略（基于错误分析结果）

# 1. 更强的数据增强
advanced_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),  # 叶片可能任意方向
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 2. 尝试更好的backbone
models_to_try = [
    'resnet50',           # baseline
    'resnet101',          # 更深
    'convnext_small',     # 现代CNN
    'efficientnet_b3',    # 高效
]

# 3. Label Smoothing（减少过拟合）
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

# 4. 测试时增强（TTA）
def predict_with_tta(model, image, num_augments=5):
    """测试时数据增强"""
    predictions = []
    
    for _ in range(num_augments):
        aug_image = tta_transform(image=image)['image']
        pred = model(aug_image.unsqueeze(0))
        predictions.append(pred)
    
    # 平均预测结果
    return torch.stack(predictions).mean(dim=0)
```

#### Step 3.3: 超参数调优 (4小时)

```python
# 关键超参数（不要调太多）
hyperparams = {
    'lr': [1e-4, 5e-5, 1e-5],           # 学习率
    'weight_decay': [1e-4, 1e-5],       # 正则化
    'batch_size': [64, 128],            # batch size
    'img_size': [224, 256, 288],        # 输入尺寸
}

# 使用验证集准确率选择最佳配置
# 预期提升：+3-5%
```

---

### Phase 4: 多任务学习 (Day 7-8, 16小时)

#### Step 4.1: 多任务模型 (6小时)

```python
# src/models/multitask.py

class MultiTaskModel(nn.Module):
    """统一的多任务架构（Task 4）"""
    
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        
        # 共享特征提取器
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            global_pool=''  # 保留空间信息（用于Grad-CAM）
        )
        
        # 自适应池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 获取特征维度
        feature_dim = self.backbone.num_features
        
        # 多个任务头
        self.head_61class = nn.Linear(feature_dim, 61)
        self.head_crop = nn.Linear(feature_dim, 10)
        self.head_disease = nn.Linear(feature_dim, 28)
        self.head_severity = nn.Linear(feature_dim, 4)
    
    def forward(self, x, return_features=False):
        # 特征提取
        features = self.backbone(x)
        pooled = self.global_pool(features).flatten(1)
        
        # 多任务输出
        outputs = {
            'label_61': self.head_61class(pooled),
            'crop': self.head_crop(pooled),
            'disease': self.head_disease(pooled),
            'severity': self.head_severity(pooled),
        }
        
        if return_features:
            outputs['features'] = features  # 用于Grad-CAM
        
        return outputs

# 参数量：~24M（在限制内）
```

#### Step 4.2: 多任务训练策略 (6小时)

```python
# 多任务损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights=None):
        super().__init__()
        # 任务权重（可学习或固定）
        if task_weights is None:
            task_weights = {
                'label_61': 1.0,    # 主任务
                'crop': 0.3,        # 辅助任务
                'disease': 0.3,     # 辅助任务
                'severity': 0.8,    # 次主任务
            }
        self.task_weights = task_weights
        
        # 每个任务的损失
        self.criteria = {
            'label_61': nn.CrossEntropyLoss(weight=class_weights_61),
            'crop': nn.CrossEntropyLoss(),
            'disease': nn.CrossEntropyLoss(),
            'severity': nn.CrossEntropyLoss(weight=severity_weights),
        }
    
    def forward(self, outputs, labels):
        total_loss = 0
        losses = {}
        
        for task, criterion in self.criteria.items():
            loss = criterion(outputs[task], labels[task])
            weighted_loss = self.task_weights[task] * loss
            total_loss += weighted_loss
            losses[task] = loss.item()
        
        return total_loss, losses

# 训练循环
def train_multitask(model, train_loader, val_loader, epochs=50):
    criterion = MultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            outputs = model(images)
            loss, task_losses = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证（评估所有任务）
        metrics = evaluate_multitask(model, val_loader)
        print(f"Epoch {epoch}: {metrics}")
```

#### Step 4.3: 可解释性（Grad-CAM）(4小时)

```python
# src/utils/visualization.py

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class ModelInterpreter:
    """模型可解释性工具"""
    
    def __init__(self, model):
        self.model = model
        # 选择最后一个卷积层
        target_layer = [model.backbone.layer4[-1]]
        self.grad_cam = GradCAM(model=model, target_layers=target_layer)
    
    def visualize(self, image, predicted_class):
        """生成Grad-CAM热力图"""
        # 生成类激活图
        cam = self.grad_cam(input_tensor=image, targets=[predicted_class])
        
        # 叠加到原图
        visualization = show_cam_on_image(
            image.cpu().numpy().transpose(1, 2, 0),
            cam[0],
            use_rgb=True
        )
        
        return visualization
    
    def generate_report(self, image, outputs):
        """生成诊断报告（Task 4要求）"""
        # 解析预测结果
        label_61 = outputs['label_61'].argmax().item()
        crop = outputs['crop'].argmax().item()
        disease = outputs['disease'].argmax().item()
        severity = outputs['severity'].argmax().item()
        
        # 置信度
        confidence_61 = outputs['label_61'].softmax(dim=1).max().item()
        
        # 生成报告
        report = {
            'crop_name': CROP_NAMES[crop],
            'disease_name': DISEASE_NAMES[disease] if disease > 0 else 'Healthy',
            'severity_level': SEVERITY_LEVELS[severity],
            'confidence': f"{confidence_61 * 100:.2f}%",
            'diagnosis': self._generate_diagnosis_text(crop, disease, severity),
            'visualization': self.visualize(image, label_61)
        }
        
        return report
    
    def _generate_diagnosis_text(self, crop, disease, severity):
        """生成可读的诊断文本"""
        if disease == 0:  # Healthy
            return f"该{CROP_NAMES[crop]}植株健康，无明显病害。"
        else:
            severity_text = ['', '轻度', '中度', '重度'][severity]
            return (
                f"检测到{CROP_NAMES[crop]}存在{DISEASE_NAMES[disease]}，"
                f"严重程度为{severity_text}。"
                f"建议采取相应的植保措施。"
            )
```

---

### Phase 5: 少样本学习 (Day 9, 8小时)

#### Task 2专项解决方案

```python
# src/models/fewshot.py

class FewShotClassifier(nn.Module):
    """少样本学习方案（每类仅10张图）"""
    
    def __init__(self, pretrained_model_path):
        super().__init__()
        
        # 加载在完整数据集上预训练的模型
        checkpoint = torch.load(pretrained_model_path)
        self.backbone = checkpoint['model'].backbone
        
        # 冻结backbone（关键！）
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 只训练分类头
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 61)
        )
    
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)

# 训练策略
def train_fewshot(model, support_set, query_set):
    """
    support_set: 每类10张图
    query_set: 验证集
    """
    # 强数据增强（至关重要！）
    strong_augment = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.8),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.CoarseDropout(max_holes=12, max_height=40, max_width=40, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # 只训练分类头，学习率可以大一些
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,  # 比全模型训练大10倍
        weight_decay=1e-3
    )
    
    # 训练更多epoch（样本少，容易过拟合）
    for epoch in range(100):
        # 每个epoch对support set做不同的数据增强
        train_one_epoch(model, support_set, strong_augment)
        
        # 早停（避免过拟合）
        val_acc = evaluate(model, query_set)
        if early_stop_criterion(val_acc):
            break

# 预期效果：
# - 完整数据：90% accuracy
# - 10-shot：65-75% accuracy（合理的性能下降）
```

---

### Phase 6: 最终优化与集成 (Day 10, 8小时)

#### Step 6.1: 模型集成

```python
# 简单但有效的集成策略
class EnsembleModel:
    """集成多个模型"""
    
    def __init__(self, model_paths):
        self.models = [load_model(path) for path in model_paths]
    
    def predict(self, x):
        """软投票"""
        predictions = []
        for model in self.models:
            pred = model(x).softmax(dim=1)
            predictions.append(pred)
        
        # 平均预测概率
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred

# 集成哪些模型？
# 1. ResNet50（baseline）
# 2. ConvNeXt-Small（现代架构）
# 3. 多任务模型的label_61 head
# 预期提升：+2-3%
```

#### Step 6.2: 最终评估

```python
# src/evaluation/final_eval.py

def final_evaluation(model, test_loader):
    """生成最终提交结果"""
    
    metrics = {
        # Task 1: 61-class accuracy
        'task1_accuracy': evaluate_accuracy(model, test_loader, task='label_61'),
        
        # Task 2: Few-shot accuracy
        'task2_accuracy': evaluate_fewshot(fewshot_model, test_loader),
        
        # Task 3: Severity classification
        'task3_metrics': {
            'accuracy': evaluate_accuracy(model, test_loader, task='severity'),
            'macro_f1': evaluate_f1(model, test_loader, task='severity'),
            'per_class_recall': evaluate_recall(model, test_loader, task='severity'),
        },
        
        # Task 4: Multi-task performance
        'task4_metrics': evaluate_multitask(multitask_model, test_loader),
    }
    
    # 生成可视化报告
    generate_report(metrics)
    
    return metrics
```

---

## 【性能预期】

### 保守估计（首次尝试）

| 任务 | 指标 | 目标 |
|------|------|------|
| Task 1 | Accuracy | 85-90% |
| Task 2 | Accuracy (10-shot) | 65-75% |
| Task 3 | Macro F1 | 80-85% |
| Task 4 | 综合性能 | 待评估 |

### 优化后（经过迭代）

| 任务 | 指标 | 目标 |
|------|------|------|
| Task 1 | Accuracy | 92-95% |
| Task 2 | Accuracy (10-shot) | 75-80% |
| Task 3 | Macro F1 | 88-92% |
| Task 4 | 综合性能 | 90%+ |

---

## 【潜在的坑与对策】

### 1. 数据不平衡
**问题**：某些疾病类别样本极少  
**对策**：
- ✅ 使用加权loss（不是过采样）
- ✅ Focal Loss（关注难样本）
- ❌ 不要SMOTE（图像数据不适用）

### 2. 过拟合风险
**问题**：模型记住训练集特定背景  
**对策**：
- ✅ 强数据增强（random crop, color jitter）
- ✅ Dropout + Label Smoothing
- ✅ 监控训练/验证loss差距

### 3. 标注噪声
**问题**：除了"duplicate"，可能还有其他错误  
**对策**：
- ✅ 训练中期检查高loss样本
- ✅ 人工审核混淆矩阵中的高频错误对
- ❌ 不要盲目信任标签

### 4. 测试集分布偏移
**问题**：竞赛数据可能与真实场景不同  
**对策**：
- ✅ 不过度优化validation set
- ✅ 使用交叉验证评估稳定性
- ✅ TTA（测试时增强）提高鲁棒性

---

## 【关键决策点】

### 选择1：Backbone架构

```python
# 推荐优先级
candidates = [
    'resnet50',         # 1st choice: 稳定可靠
    'convnext_small',   # 2nd choice: 现代高效
    'efficientnet_b3',  # 3rd choice: 参数效率高
]

# 不推荐
avoid = [
    'vit_*',            # 需要大数据集
    'swin_*',           # 训练不稳定
    'resnet18/34',      # 容量不足
]
```

### 选择2：训练策略

```python
# 分阶段训练（推荐）
stage1 = {
    'epochs': 20,
    'lr': 1e-3,
    'freeze_backbone': True,   # 只训练分类头
}

stage2 = {
    'epochs': 30,
    'lr': 1e-4,
    'freeze_backbone': False,  # 微调整个网络
}

# vs 端到端训练（更简单但可能不如分阶段）
end_to_end = {
    'epochs': 50,
    'lr': 1e-4,
    'freeze_backbone': False,
}
```

### 选择3：损失函数

```python
# Task 1/2: 类别不平衡
loss_options = [
    nn.CrossEntropyLoss(weight=class_weights),  # 首选
    FocalLoss(alpha=0.25, gamma=2.0),           # 备选
]

# Task 3: 严重程度（有序）
severity_loss_options = [
    nn.CrossEntropyLoss(),                       # 简单
    OrdinalRegressionLoss(),                     # 考虑顺序关系
]
```

---

## 【时间分配总结】

| 阶段 | 时间 | 关键产出 |
|------|------|----------|
| Phase 0: 环境准备 | 4h | 项目结构、依赖安装 |
| Phase 1: 数据清理 | 12h | 清洁数据集、标签映射 |
| Phase 2: Baseline | 16h | 端到端系统、初步结果 |
| Phase 3: 优化 | 16h | 错误分析、性能提升 |
| Phase 4: 多任务 | 16h | 多任务模型、可解释性 |
| Phase 5: 少样本 | 8h | Few-shot方案 |
| Phase 6: 集成 | 8h | 模型集成、最终评估 |
| **总计** | **80h** | **完整解决方案** |

---

## 【最后的建议】

### Linus会说什么？

1. **"先把数据结构搞对"**
   - 不要急着写模型代码
   - 花时间理解数据的层次结构
   - 设计统一的标签表示

2. **"消除特殊情况"**
   - 不要为"Healthy"类别写单独的逻辑
   - 用统一的三元组表示所有标签
   - 一个数据pipeline处理所有任务

3. **"从最简单的开始"**
   - 先跑通一个ResNet50 baseline
   - 不要上来就搞Transformer、GAN
   - 复杂性是万恶之源

4. **"解决真实问题"**
   - 关注实际的分类准确率
   - 不要为了论文追求花哨的方法
   - 实用主义第一

5. **"不要破坏用户空间"**
   - 模型输出格式要稳定
   - 接口设计要考虑扩展性
   - 部署后不要随意改变行为

---

## 【参考资源】

```python
# 代码参考
repos = [
    'https://github.com/rwightman/pytorch-image-models',  # timm库
    'https://github.com/jacobgil/pytorch-grad-cam',       # Grad-CAM
    'https://github.com/albumentations-team/albumentations',  # 数据增强
]

# 论文参考（如果需要）
papers = [
    'ResNet: Deep Residual Learning for Image Recognition',
    'ConvNeXt: A ConvNet for the 2020s',
    'Focal Loss for Dense Object Detection',
]

# 但记住：
# "Talk is cheap. Show me the code."
# 不要读太多论文，直接开始写代码！
```

---

**Good luck. And remember: "Good taste" is what separates the great from the merely competent.**

—— 路线图结束 ——