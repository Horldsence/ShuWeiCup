# 项目环境搭建指南

## 目录结构创建

运行以下命令创建完整的项目结构：

```bash
# 从项目根目录执行
mkdir -p data/{raw,cleaned,splits}
mkdir -p src/{data,models,training,utils}
mkdir -p configs
mkdir -p notebooks
mkdir -p results/experiments
mkdir -p checkpoints

# 创建 __init__.py 文件
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/utils/__init__.py
```

## 完整目录结构

```
project/
├── data/
│   ├── raw/                    # 原始数据集
│   │   ├── AgriculturalDisease_trainingset/
│   │   └── AgriculturalDisease_validationset/
│   ├── cleaned/                # 清理后的数据
│   └── splits/                 # 训练/验证划分
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # 数据集定义
│   │   ├── transforms.py      # 数据增强
│   │   └── cleaner.py         # 数据清理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py        # 单任务baseline
│   │   ├── multitask.py       # 多任务模型
│   │   └── fewshot.py         # 少样本学习
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # 训练循环
│   │   ├── metrics.py         # 评估指标
│   │   └── losses.py          # 损失函数
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py   # Grad-CAM等
│       ├── logger.py          # 日志记录
│       └── config.py          # 配置管理
│
├── configs/
│   ├── task1_baseline.yaml
│   ├── task2_fewshot.yaml
│   ├── task3_severity.yaml
│   └── task4_multitask.yaml
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_visualization.ipynb
│
├── results/
│   └── experiments/
│
├── checkpoints/               # 模型检查点
│
├── pyproject.toml
├── ROADMAP.md
├── SETUP.md
└── README.md
```

## 环境安装

### 使用 uv

```bash
# 安装 uv（如果还没安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync
```

## 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "import albumentations; print(f'albumentations: {albumentations.__version__}')"
```

## 下载数据集

将原始数据集放置到 `data/raw/` 目录下：

```
data/raw/
├── AgriculturalDisease_trainingset/
│   ├── images/
│   └── labels.json
└── AgriculturalDisease_validationset/
    ├── images/
    └── labels.json
```

## 快速开始

### 1. 数据探索

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. 数据清理

```bash
uv run python src/data/cleaner.py --src data/raw --dst data/cleaned
```

### 3. 训练 Baseline

```bash
uv run python train.py --config configs/task1_baseline.yaml
```

## 硬件要求

### 最低配置
- GPU: NVIDIA GTX 1660 (6GB VRAM)
- RAM: 16GB
- 存储: 50GB

### 推荐配置
- GPU: NVIDIA RTX 3090 / A100 (24GB+ VRAM)
- RAM: 32GB
- 存储: 100GB SSD

### 训练时间估算
- Baseline (50 epochs): 2-4小时 (单卡 V100)
- Multi-task (50 epochs): 3-5小时 (单卡 V100)
- Few-shot (100 epochs): 1-2小时 (单卡 V100)
