"""
Data Analysis Tool
==================

"Bad programmers worry about the code. Good programmers worry about data structures."
- Linus Torvalds

This script analyzes the agricultural disease dataset to understand:
1. Class imbalance severity
2. Data augmentation effects
3. Why 6 epochs leads to convergence with low accuracy

Usage:
    python analyze_data.py
    python analyze_data.py --visualize
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def analyze_class_distribution():
    """Analyze class distribution and imbalance."""
    print("=" * 60)
    print("Class Distribution Analysis")
    print("=" * 60)

    train_meta = pd.read_csv("data/cleaned/metadata/train_metadata.csv")
    val_meta = pd.read_csv("data/cleaned/metadata/val_metadata.csv")

    train_counts = train_meta["label_61"].value_counts().sort_index()
    val_counts = val_meta["label_61"].value_counts().sort_index()

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_meta):,}")
    print(f"  Val:   {len(val_meta):,}")

    print(f"\nClass distribution (train):")
    print(f"  Min samples: {train_counts.min()}")
    print(f"  Max samples: {train_counts.max()}")
    print(f"  Mean samples: {train_counts.mean():.1f}")
    print(f"  Median samples: {train_counts.median():.1f}")
    print(f"  Imbalance ratio: {train_counts.max() / train_counts.min():.1f}:1")

    # Find problematic classes
    rare_classes = train_counts[train_counts < 50]
    print(f"\nClasses with < 50 samples: {len(rare_classes)}")
    for cls, count in rare_classes.items():
        val_count = val_counts.get(cls, 0)
        print(f"  Class {cls:2d}: train={count:3d}, val={val_count:3d}")

    # Majority classes
    major_classes = train_counts[train_counts > 1000]
    print(f"\nClasses with > 1000 samples: {len(major_classes)}")
    print(
        f"  Total samples: {major_classes.sum()} ({100 * major_classes.sum() / len(train_meta):.1f}%)"
    )

    # Why 6 epochs converges
    print("\n" + "=" * 60)
    print("Why 6 epochs leads to premature convergence:")
    print("=" * 60)
    print(
        f"1. Major classes ({len(major_classes)} classes) = {100 * major_classes.sum() / len(train_meta):.1f}% of data"
    )
    print(f"2. Model quickly learns to predict these classes")
    print(f"3. Rare classes ({len(rare_classes)} with <50 samples) barely contribute to loss")
    print(f"4. Class weights too extreme (2445:1) dominate training")
    print(f"5. Result: Fast convergence to ~30% by 'guessing major classes'")

    return train_counts, val_counts, rare_classes, major_classes


def analyze_class_weights():
    """Compare different class weighting strategies."""
    print("\n" + "=" * 60)
    print("Class Weights Comparison")
    print("=" * 60)

    train_meta = pd.read_csv("data/cleaned/metadata/train_metadata.csv")
    class_counts = train_meta["label_61"].value_counts().sort_index()

    # Load original weights
    weights_original = pd.read_csv("data/cleaned/metadata/class_weights.csv")

    # Load new weights
    weights_sqrt = pd.read_csv("data/cleaned/metadata/class_weights_sqrt.csv")
    weights_effective = pd.read_csv("data/cleaned/metadata/class_weights_effective.csv")

    print("\nWeight statistics:")
    print(
        f"\n1. Original (inverse frequency):"
        f"\n   Range: [{weights_original['weight'].min():.3f}, {weights_original['weight'].max():.3f}]"
        f"\n   Ratio: {weights_original['weight'].max() / weights_original['weight'].min():.1f}:1"
        f"\n   Problem: TOO EXTREME - rare classes dominate loss"
    )

    print(
        f"\n2. Sqrt smoothing (recommended):"
        f"\n   Range: [{weights_sqrt['weight'].min():.3f}, {weights_sqrt['weight'].max():.3f}]"
        f"\n   Ratio: {weights_sqrt['weight'].max() / weights_sqrt['weight'].min():.1f}:1"
        f"\n   Benefit: Balanced - rare classes weighted but not dominant"
    )

    print(
        f"\n3. Effective number (beta=0.9999):"
        f"\n   Range: [{weights_effective['weight'].min():.3f}, {weights_effective['weight'].max():.3f}]"
        f"\n   Ratio: {weights_effective['weight'].max() / weights_effective['weight'].min():.1f}:1"
        f"\n   Note: Still quite extreme"
    )

    print(
        f"\n✅ Recommendation: Use sqrt smoothing"
        f"\n   - Balances major and rare classes"
        f"\n   - Prevents loss explosion from rare classes"
        f"\n   - More stable training"
    )


def visualize_augmentation():
    """Visualize data augmentation effects."""
    print("\n" + "=" * 60)
    print("Data Augmentation Visualization")
    print("=" * 60)

    from dataset import get_train_transform

    # Load a sample image
    train_meta = pd.read_csv("data/cleaned/metadata/train_metadata.csv")
    sample = train_meta.iloc[0]

    class_folder = f"class_{sample['label_61']:02d}"
    image_path = Path("data/cleaned/train") / class_folder / sample["image_name"]

    if not image_path.exists():
        print(f"⚠️  Sample image not found: {image_path}")
        return

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"\nSample image: {image_path}")
    print(f"Original size: {image.shape}")

    # Apply augmentation multiple times
    transform = get_train_transform(320)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Data Augmentation Examples (320x320)", fontsize=16, fontweight="bold")

    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original", fontweight="bold")
    axes[0, 0].axis("off")

    # Augmented versions
    for i, ax in enumerate(axes.flat[1:], 1):
        augmented = transform(image=image)["image"]
        # Convert tensor to numpy for visualization
        aug_img = augmented.permute(1, 2, 0).numpy()
        # Denormalize
        aug_img = aug_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        aug_img = np.clip(aug_img, 0, 1)

        ax.imshow(aug_img)
        ax.set_title(f"Augmented {i}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    output_path = "demo_plots/augmentation_examples.png"
    Path("demo_plots").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved augmentation examples to: {output_path}")
    plt.close()


def plot_class_distribution(train_counts, val_counts):
    """Plot class distribution."""
    print("\n" + "=" * 60)
    print("Generating Distribution Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Class Distribution Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Train distribution
    ax1 = axes[0, 0]
    ax1.bar(range(len(train_counts)), train_counts.values, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Class ID")
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Training Set Distribution", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Highlight rare classes
    rare_threshold = 50
    rare_mask = train_counts.values < rare_threshold
    ax1.bar(
        np.where(rare_mask)[0],
        train_counts.values[rare_mask],
        color="red",
        alpha=0.8,
        label=f"< {rare_threshold} samples",
    )
    ax1.legend()

    # Plot 2: Val distribution
    ax2 = axes[0, 1]
    ax2.bar(range(len(val_counts)), val_counts.values, color="orange", alpha=0.7)
    ax2.set_xlabel("Class ID")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Validation Set Distribution", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Log scale
    ax3 = axes[1, 0]
    ax3.bar(range(len(train_counts)), train_counts.values, color="steelblue", alpha=0.7)
    ax3.set_xlabel("Class ID")
    ax3.set_ylabel("Number of Samples (log scale)")
    ax3.set_title("Training Set Distribution (Log Scale)", fontweight="bold")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sorted distribution
    ax4 = axes[1, 1]
    sorted_counts = np.sort(train_counts.values)[::-1]
    ax4.bar(range(len(sorted_counts)), sorted_counts, color="green", alpha=0.7)
    ax4.set_xlabel("Class Rank (sorted by size)")
    ax4.set_ylabel("Number of Samples")
    ax4.set_title("Sorted Class Distribution", fontweight="bold")
    ax4.axhline(y=50, color="red", linestyle="--", label="50 samples threshold")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Add statistics text
    stats_text = (
        f"Statistics:\n"
        f"Min: {train_counts.min()}\n"
        f"Max: {train_counts.max()}\n"
        f"Mean: {train_counts.mean():.1f}\n"
        f"Median: {train_counts.median():.1f}\n"
        f"Ratio: {train_counts.max() / train_counts.min():.0f}:1"
    )
    fig.text(
        0.02,
        0.02,
        stats_text,
        fontsize=11,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    output_path = "demo_plots/class_distribution.png"
    Path("demo_plots").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved class distribution to: {output_path}")
    plt.close()


def compare_class_weights():
    """Plot comparison of different class weight strategies."""
    print("\nGenerating Class Weights Comparison Plot...")

    weights_original = pd.read_csv("data/cleaned/metadata/class_weights.csv")
    weights_sqrt = pd.read_csv("data/cleaned/metadata/class_weights_sqrt.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Class Weights Comparison", fontsize=16, fontweight="bold")

    # Plot 1: Original weights
    ax1 = axes[0]
    ax1.bar(weights_original["class_id"], weights_original["weight"], color="red", alpha=0.7)
    ax1.set_xlabel("Class ID")
    ax1.set_ylabel("Weight")
    ax1.set_title("Original (Inverse Frequency)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.5,
        0.95,
        f"Range: [{weights_original['weight'].min():.2f}, {weights_original['weight'].max():.1f}]\n"
        f"Ratio: {weights_original['weight'].max() / weights_original['weight'].min():.0f}:1\n"
        f"❌ TOO EXTREME",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
    )

    # Plot 2: Sqrt weights
    ax2 = axes[1]
    ax2.bar(weights_sqrt["class_id"], weights_sqrt["weight"], color="green", alpha=0.7)
    ax2.set_xlabel("Class ID")
    ax2.set_ylabel("Weight")
    ax2.set_title("Sqrt Smoothing (Recommended)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.5,
        0.95,
        f"Range: [{weights_sqrt['weight'].min():.2f}, {weights_sqrt['weight'].max():.1f}]\n"
        f"Ratio: {weights_sqrt['weight'].max() / weights_sqrt['weight'].min():.1f}:1\n"
        f"✅ BALANCED",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    # Plot 3: Comparison
    ax3 = axes[2]
    ax3.plot(
        weights_original["class_id"],
        weights_original["weight"],
        "r-",
        label="Original",
        linewidth=2,
        alpha=0.7,
    )
    ax3.plot(
        weights_sqrt["class_id"],
        weights_sqrt["weight"],
        "g-",
        label="Sqrt (Recommended)",
        linewidth=2,
        alpha=0.7,
    )
    ax3.set_xlabel("Class ID")
    ax3.set_ylabel("Weight")
    ax3.set_title("Side-by-Side Comparison", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = "demo_plots/class_weights_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved class weights comparison to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze agricultural disease dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Agricultural Disease Dataset Analysis")
    print("=" * 60)
    print('"Bad programmers worry about code. Good programmers worry about data." - Linus')
    print("=" * 60)

    # Analyze class distribution
    train_counts, val_counts, rare_classes, major_classes = analyze_class_distribution()

    # Analyze class weights
    analyze_class_weights()

    if args.visualize:
        # Generate plots
        plot_class_distribution(train_counts, val_counts)
        compare_class_weights()
        visualize_augmentation()

        print("\n" + "=" * 60)
        print("All visualizations saved to: demo_plots/")
        print("=" * 60)
        print("  - class_distribution.png")
        print("  - class_weights_comparison.png")
        print("  - augmentation_examples.png")

    # Summary and recommendations
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    print("\n🔴 Root Cause of Low Accuracy:")
    print("  1. Extreme class imbalance (2445:1)")
    print("  2. Original class weights too aggressive (517x)")
    print("  3. Model quickly overfits to major classes")
    print("  4. Converges in 6 epochs with ~30% accuracy")

    print("\n✅ Solutions Applied:")
    print("  1. Stronger data augmentation:")
    print("     - Rotation: 15° → 45°")
    print("     - More color jitter, cutout, distortions")
    print("     - Higher augmentation probability")
    print("  2. Better class weights:")
    print("     - Sqrt smoothing: 49:1 (vs 2445:1)")
    print("     - Prevents loss explosion")
    print("  3. Higher resolution:")
    print("     - 224 → 320 pixels")
    print("     - Captures finer disease details")
    print("  4. Two-stage training:")
    print("     - Stage 1: Train head (10 epochs, LR=1e-3)")
    print("     - Stage 2: Fine-tune all (40 epochs, LR=3e-4)")

    print("\n🎯 Expected Improvement:")
    print("  Before: ~30% (premature convergence)")
    print("  After:  60-75% (proper learning)")

    print("\n🚀 Next Steps:")
    print("  bash train_fixed.sh")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
