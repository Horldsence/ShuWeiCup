#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 Few-Shot Agricultural Disease Classification
===================================================

"Talk is cheap. Show me the code." - Linus Torvalds

Goal:
  61-class classification with only N=10 images per class (<= 610 samples total).
  Model parameter constraint: trainable params << 20M (freeze bulk of backbone).
  Avoid academic over-engineering; focus on practical margin + prototype stabilization.

Design Philosophy (Linus-style):
  1. Freeze most of the backbone: prevent catastrophic overfit, keep pretrained visual priors.
  2. Partially unfreeze last two residual stages (layer3, layer4) for domain adaptation.
  3. Use ArcFace (cosine margin) head: improves inter-class separation with tiny data.
  4. PrototypeLoss (per-class feature mean) for intra-class compactness, no learnable centers.
  5. Minimal augmentations that preserve lesion texture (no elastic/optical distortions).
  6. Early light Mixup only (optionally), then disable to allow crisp decision boundaries.
  7. OneCycleLR for quick warmup / controlled anneal; head LR boosted vs backbone LR.
  8. Evaluation: Top-1, Top-5, Macro-F1 (fair to balanced few-shot set).
  9. Never rely on "tail" heuristics: dataset artificially balanced (10 shots each).
 10. Simplicity > cleverness. If indentation >3 levels, redesign.

Usage Example:
    python task2train.py \
        --train-meta data/cleaned/metadata/train_metadata_fewshot_10.csv \
        --val-meta data/cleaned/metadata/val_metadata.csv \
        --train-dir data/cleaned/train \
        --val-dir data/cleaned/val \
        --backbone resnet50 \
        --epochs 40 \
        --batch-size 16 \
        --lr 3e-4 \
        --head-lr-scale 5.0 \
        --proto-weight 0.5 \
        --arcface-margin 0.30 \
        --arcface-scale 30.0 \
        --image-size 256 \
        --mixup-alpha 0.2 \
        --mixup-disable-epoch 5 \
        --freeze-stage12 \
        --save-dir checkpoints/task2_fewshot

Few-shot Metadata:
    Generate with create_fewshot_subset.py (already supplied in repo).

"""

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import ColorJitter, Compose, HorizontalFlip, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Reuse dataset utilities
from dataset import AgriDiseaseDataset


# Picklable collate function (avoid lambda for multiprocessing spawn in Python 3.13)
def fewshot_collate(batch):
    """
    Batch: list of tuples (image_tensor, label_dict)
    Returns:
        images: stacked tensor [B, C, H, W]
        labels_dict: {"label_61": LongTensor[B]}
    """
    images = torch.stack([x[0] for x in batch])
    labels = torch.tensor([x[1]["label_61"] for x in batch], dtype=torch.long)
    return images, {"label_61": labels}


# =========================================================
# Utility: Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Set random seed = {seed}")


# =========================================================
# Augmentation (Few-Shot Specific)
# =========================================================
def get_fewshot_train_transform(image_size: int = 256) -> Compose:
    # Preserve lesion texture. Avoid heavy geometric warping.
    # Explicit Resize ensures uniform tensor shapes for DataLoader stacking.
    return Compose(
        [
            Resize(image_size, image_size),
            HorizontalFlip(p=0.5),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.6),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_fewshot_val_transform(image_size: int = 256) -> Compose:
    # Deterministic validation transform with fixed spatial size
    return Compose(
        [
            Resize(image_size, image_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# =========================================================
# ArcFace Head
# =========================================================
class ArcFaceHead(nn.Module):
    def __init__(
        self, in_features: int, num_classes: int, scale: float = 30.0, margin: float = 0.30
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, features: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Normalize features and weights
        x = F.normalize(features, dim=1)
        w = F.normalize(self.weight, dim=1)
        # Cosine similarity
        logits = torch.matmul(x, w.t())  # [B, C]
        if labels is not None:
            # Apply margin only to target class logits
            theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
            target_theta = theta[torch.arange(logits.size(0)), labels] + self.margin
            target_logits = torch.cos(target_theta)
            logits = logits.clone()
            logits[torch.arange(logits.size(0)), labels] = target_logits
        return self.scale * logits


# =========================================================
# Prototype Loss (non-parametric centers)
# =========================================================
class PrototypeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        # features: [B, D]; prototypes: [C, D]; labels: [B]
        target_proto = prototypes[labels]  # [B, D]
        return ((features - target_proto) ** 2).mean()


@torch.no_grad()
def compute_prototypes(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> torch.Tensor:
    model.eval()
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for images, label_dict in dataloader:
        labels = label_dict["label_61"]
        images = images.to(device)
        labels = labels.to(device)
        feats = model.extract_features(images)  # [B, D]
        for f, l in zip(feats, labels):
            l_int = int(l.item())
            if l_int not in sums:
                sums[l_int] = f.clone()
                counts[l_int] = 1
            else:
                sums[l_int] += f
                counts[l_int] += 1
    num_classes = max(counts.keys()) + 1
    feat_dim = next(iter(sums.values())).size(0)
    prototypes = torch.zeros(num_classes, feat_dim, device=device)
    for k in counts.keys():
        prototypes[k] = sums[k] / counts[k]
    return prototypes


# =========================================================
# Few-Shot Model Wrapper
# =========================================================
class FewShotArcFaceModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 61,
        pretrained: bool = True,
        dropout: float = 0.1,
        partial_unfreeze: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Create backbone (no classifier head)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)  # [1, C, H, W]
            feat_dim = feats.shape[1]
        self.feature_dim = feat_dim
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.arcface_head = ArcFaceHead(in_features=self.feature_dim, num_classes=num_classes)

        # Freeze all backbone params first
        for p in self.backbone.parameters():
            p.requires_grad = False

        if partial_unfreeze:
            # Unfreeze last two stages for ResNet-like architectures
            for name, p in self.backbone.named_parameters():
                if "layer3" in name or "layer4" in name:
                    p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Model] Backbone={backbone_name} feat_dim={self.feature_dim}")
        print(
            f"[Model] Total params: {total / 1e6:.2f}M | Trainable params: {trainable / 1e6:.2f}M"
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.backbone(x)  # [B, C, H, W]
        pooled = self.global_pool(fm)  # [B, C, 1, 1]
        vec = pooled.flatten(1)  # [B, C]
        vec = self.dropout(vec)
        return vec

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.extract_features(x)
        logits = self.arcface_head(feats, labels)
        return logits, feats


# =========================================================
# Metrics
# =========================================================
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1)
        correct = 0
        for i in range(targets.size(0)):
            if targets[i] in pred[i]:
                correct += 1
        return 100.0 * correct / targets.size(0)


def macro_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    # One-vs-rest confusion from predictions
    _, preds = torch.max(logits, dim=1)
    f1_vals = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_vals.append(f1)
    return float(sum(f1_vals) / len(f1_vals))


# =========================================================
# Mixup (light, early epochs only)
# =========================================================
def apply_mixup(
    images: torch.Tensor, labels: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam


# =========================================================
# Training / Validation
# =========================================================
def train_one_epoch(
    model: FewShotArcFaceModel,
    loader: DataLoader,
    optimizer,
    scheduler,
    device,
    epoch: int,
    epochs: int,
    ce_loss_fn,
    proto_loss_fn,
    prototypes: torch.Tensor,
    proto_weight: float,
    mixup_alpha: float,
    mixup_disable_epoch: int,
    amp: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    use_mixup = mixup_alpha > 0 and epoch < mixup_disable_epoch

    for images, label_dict in loader:
        labels = label_dict["label_61"]
        images = images.to(device)
        labels = labels.to(device)

        images_aug, y_a, y_b, lam = apply_mixup(images, labels, mixup_alpha if use_mixup else 0.0)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits, feats = model(images_aug, labels=y_a)  # margin applied on y_a
            ce_a = ce_loss_fn(logits, y_a)
            if use_mixup:
                # Recompute logits for second labels (without margin shift)
                logits_b, _ = model(images_aug, labels=y_b)
                ce_b = ce_loss_fn(logits_b, y_b)
                ce_loss = lam * ce_a + (1 - lam) * ce_b
                feats_used = feats  # Use same features; prototypes unaffected by second forward
            else:
                ce_loss = ce_a
                feats_used = feats

            ploss = proto_loss_fn(feats_used, y_a, prototypes)
            loss = ce_loss + proto_weight * ploss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        with torch.no_grad():
            eval_logits, _ = model(images, labels=None)  # eval logits without margin
            preds = torch.argmax(eval_logits, dim=1)
            total_correct += preds.eq(labels).sum().item()

    return {
        "loss": total_loss / total_samples,
        "acc": 100.0 * total_correct / total_samples,
    }


@torch.inference_mode()
def validate(
    model: FewShotArcFaceModel,
    loader: DataLoader,
    device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    ce_fn = nn.CrossEntropyLoss()

    top5_correct = 0

    all_logits = []
    all_labels = []

    for images, label_dict in loader:
        labels = label_dict["label_61"]
        images = images.to(device)
        labels = labels.to(device)

        logits, _ = model(images, labels=None)  # No margin for evaluation
        loss = ce_fn(logits, labels)

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        total_correct += preds.eq(labels).sum().item()

        # Top-5
        _, pred_top5 = logits.topk(5, dim=1)
        for i in range(batch_size):
            if labels[i] in pred_top5[i]:
                top5_correct += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    macro_f1_val = macro_f1(logits_cat, labels_cat, num_classes=num_classes)

    return {
        "loss": total_loss / total_samples,
        "acc": 100.0 * total_correct / total_samples,
        "top5": 100.0 * top5_correct / total_samples,
        "macro_f1": macro_f1_val,
    }


# =========================================================
# Argument Parsing
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Few-Shot Task2 Training (ArcFace + Prototype)")
    # Data
    p.add_argument("--train-dir", type=str, default="data/cleaned/train", help="Train images root")
    p.add_argument("--val-dir", type=str, default="data/cleaned/val", help="Val images root")
    p.add_argument("--train-meta", type=str, required=True, help="Few-shot train metadata CSV")
    p.add_argument("--val-meta", type=str, required=True, help="Validation metadata CSV")
    # Model
    p.add_argument("--backbone", type=str, default="resnet50", help="Backbone architecture (timm)")
    p.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout before head")
    p.add_argument(
        "--freeze-stage12", action="store_true", help="Freeze layers 1&2 only (partial unfreeze)"
    )
    # Training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4, help="Base backbone LR")
    p.add_argument("--head-lr-scale", type=float, default=5.0, help="Scale factor for head LR")
    p.add_argument("--proto-weight", type=float, default=0.5, help="Weight for PrototypeLoss")
    p.add_argument("--arcface-margin", type=float, default=0.30)
    p.add_argument("--arcface-scale", type=float, default=30.0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--mixup-alpha", type=float, default=0.2, help="Mixup alpha (0 disables)")
    p.add_argument("--mixup-disable-epoch", type=int, default=5, help="Epoch to disable Mixup")
    p.add_argument("--no-mixup", action="store_true", help="Force disable Mixup from start")
    p.add_argument(
        "--amp", action="store_true", default=True, help="Use mixed precision (CUDA only)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="checkpoints/task2_fewshot")
    p.add_argument("--save-best-only", action="store_true", default=True)
    p.add_argument(
        "--early-stop-patience", type=int, default=10, help="Early stop patience (val acc plateau)"
    )
    # Optimizer choice
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    return p.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_tf = get_fewshot_train_transform(args.image_size)
    val_tf = get_fewshot_val_transform(args.image_size)

    # Datasets / Loaders
    train_dataset = AgriDiseaseDataset(
        data_dir=args.train_dir,
        metadata_path=args.train_meta,
        transform=train_tf,
        return_multitask=False,
    )
    val_dataset = AgriDiseaseDataset(
        data_dir=args.val_dir,
        metadata_path=args.val_meta,
        transform=val_tf,
        return_multitask=False,
    )

    print(f"[Data] Few-shot train samples: {len(train_dataset)}")
    print(f"[Data] Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=fewshot_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=fewshot_collate,
    )

    # Model
    model = FewShotArcFaceModel(
        backbone_name=args.backbone,
        num_classes=61,
        pretrained=args.pretrained,
        dropout=args.dropout,
        partial_unfreeze=args.freeze_stage12,
    ).to(device)
    # Adjust ArcFace head margin/scale if args differ
    model.arcface_head.margin = args.arcface_margin
    model.arcface_head.scale = args.arcface_scale

    # Parameter groups
    head_params = [p for p in model.arcface_head.parameters() if p.requires_grad] + [
        p for p in model.dropout.parameters() if p.requires_grad
    ]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    param_groups = [
        {"params": head_params, "lr": args.lr * args.head_lr_scale},
        {"params": backbone_params, "lr": args.lr},
    ]

    if args.optimizer == "adamw":
        optimizer = AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr * args.head_lr_scale, args.lr],
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        div_factor=1.0,
        final_div_factor=20.0,
        anneal_strategy="cos",
    )

    ce_loss_fn = nn.CrossEntropyLoss()
    proto_loss_fn = PrototypeLoss()

    # Initial prototypes
    prototypes = compute_prototypes(model, train_loader, device)
    print("[Proto] Initialized prototypes.")

    best_val_acc = 0.0
    best_macro_f1 = 0.0
    epochs_no_improve = 0

    # History for plotting
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top5": [],
        "val_macro_f1": [],
        "head_lr": [],
        "backbone_lr": [],
    }

    # TensorBoard writer
    tb_writer = SummaryWriter(str(save_dir / "logs"))
    print(
        f"[Log] TensorBoard -> {save_dir / 'logs'} (run: tensorboard --logdir {save_dir / 'logs'})"
    )

    mixup_alpha = 0.0 if args.no_mixup else args.mixup_alpha

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Refresh prototypes every epoch (cheap: few-shot)
        prototypes = compute_prototypes(model, train_loader, device)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            ce_loss_fn=ce_loss_fn,
            proto_loss_fn=proto_loss_fn,
            prototypes=prototypes,
            proto_weight=args.proto_weight,
            mixup_alpha=mixup_alpha,
            mixup_disable_epoch=args.mixup_disable_epoch,
            amp=args.amp,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=61,
        )

        print(
            f"[Train] Loss={train_metrics['loss']:.4f} Acc={train_metrics['acc']:.2f}% | "
            f"[Val] Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.2f}% "
            f"Top5={val_metrics['top5']:.2f}% MacroF1={val_metrics['macro_f1']:.3f}"
        )

        # Update history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_top5"].append(val_metrics["top5"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["head_lr"].append(optimizer.param_groups[0]["lr"])
        history["backbone_lr"].append(optimizer.param_groups[1]["lr"])

        # TensorBoard logging
        tb_writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        tb_writer.add_scalar("train/acc", train_metrics["acc"], epoch)
        tb_writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        tb_writer.add_scalar("val/acc", val_metrics["acc"], epoch)
        tb_writer.add_scalar("val/top5", val_metrics["top5"], epoch)
        tb_writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
        tb_writer.add_scalar("lr/head", optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar("lr/backbone", optimizer.param_groups[1]["lr"], epoch)

        improved = False
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_macro_f1 = val_metrics["macro_f1"]
            improved = True
            # Confusion matrix
            with torch.no_grad():
                all_preds = []
                all_trues = []
                for images_cm, labels_cm_dict in val_loader:
                    labels_cm = labels_cm_dict["label_61"].to(device)
                    images_cm = images_cm.to(device)
                    logits_cm, _ = model(images_cm, labels=None)
                    preds_cm = torch.argmax(logits_cm, dim=1)
                    all_preds.append(preds_cm.cpu())
                    all_trues.append(labels_cm.cpu())
                preds_cat = torch.cat(all_preds)
                trues_cat = torch.cat(all_trues)
                num_classes_cm = 61
                cm = torch.zeros(num_classes_cm, num_classes_cm, dtype=torch.int32)
                for t, p in zip(trues_cat.tolist(), preds_cat.tolist()):
                    cm[t, p] += 1
                # Save confusion matrix CSV
                import pandas as pd

                cm_df = pd.DataFrame(cm.numpy())
                cm_path = save_dir / "confusion_matrix_best.csv"
                cm_df.to_csv(cm_path, index=False)
                # Quick plot
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                im = ax_cm.imshow(cm.numpy(), cmap="Blues", aspect="auto")
                ax_cm.set_title("Confusion Matrix (Best)")
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("True")
                plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
                plt.tight_layout()
                fig_cm.savefig(save_dir / "confusion_matrix_best.png", dpi=120)
                plt.close(fig_cm)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_macro_f1": best_macro_f1,
                    "prototypes": prototypes.cpu(),
                },
                save_dir / "best.pth",
            )
            print(
                f"[Checkpoint] New best Acc={best_val_acc:.2f}% MacroF1={best_macro_f1:.3f} saved. CM written."
            )

        if not improved:
            epochs_no_improve += 1
        else:
            epochs_no_improve = 0

        if epochs_no_improve >= args.early_stop_patience:
            print(f"[EarlyStop] No improvement for {epochs_no_improve} epochs. Stopping early.")
            break

        # Disable Mixup when scheduled
        if epoch == args.mixup_disable_epoch and mixup_alpha > 0:
            mixup_alpha = 0.0
            print(f"[Regularization] Mixup disabled at epoch {epoch}")

    # Plot curves
    if history["epoch"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Few-Shot Training Progress", fontsize=16, fontweight="bold")

        # Loss
        axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="tab:blue")
        axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="tab:red")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        # Accuracy & Macro-F1
        axes[1].plot(history["epoch"], history["train_acc"], label="Train Acc", color="tab:blue")
        axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc", color="tab:red")
        axes[1].plot(
            history["epoch"], history["val_macro_f1"], label="Val MacroF1", color="tab:green"
        )
        axes[1].set_title("Accuracy / MacroF1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Percent / Score")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        # Learning Rates
        axes[2].plot(history["epoch"], history["head_lr"], label="Head LR", color="tab:purple")
        axes[2].plot(
            history["epoch"], history["backbone_lr"], label="Backbone LR", color="tab:orange"
        )
        axes[2].set_title("Learning Rates")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].set_yscale("log")
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        plot_path = save_dir / "training_curves.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[Plot] Saved training curves -> {plot_path}")
        # Also dump history to CSV for external analysis
        import pandas as pd

        hist_df = pd.DataFrame(history)
        hist_df.to_csv(save_dir / "training_history.csv", index=False)
        print(f"[History] Saved training history CSV -> {save_dir / 'training_history.csv'}")

    tb_writer.close()

    print("\n" + "=" * 60)
    print(
        f"Training complete. Best Val Acc: {best_val_acc:.2f}% | Best MacroF1: {best_macro_f1:.3f}"
    )
    print(f"Checkpoint directory: {save_dir}")
    print("=" * 60)
    print("Next steps:")
    print("  1. Run evaluation / confusion matrix on validation set.")
    print("  2. Try convnext_tiny backbone for potential extra +3~5% few-shot gain.")
    print("  3. Adjust proto-weight (0.3~0.7) if features collapse or overfit.")
    print("  4. If Acc <35%, allow unfreezing layer2 (broader adaptation).")


if __name__ == "__main__":
    main()
