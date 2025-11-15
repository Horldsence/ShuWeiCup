"""
Create Balanced Dataset
========================

"Premature optimization is the root of all evil." - Knuth
But training on 31k samples when you can train on 12k is just common sense.

This script creates a balanced training set with 200 samples per class.
- Classes with >200 samples: randomly sample 200
- Classes with <200 samples: keep all (augmentation will handle it)

Usage:
    python create_balanced_dataset.py
    python create_balanced_dataset.py --samples-per-class 300
"""

import argparse
import random
from pathlib import Path

import pandas as pd


def create_balanced_dataset(
    input_meta_path: str,
    output_meta_path: str,
    samples_per_class: int = 200,
    seed: int = 42,
):
    """
    Create balanced dataset by sampling from original.

    Args:
        input_meta_path: Path to original metadata CSV
        output_meta_path: Path to save balanced metadata CSV
        samples_per_class: Target samples per class
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print("=" * 60)
    print("Creating Balanced Dataset")
    print("=" * 60)

    # Load original metadata
    df = pd.read_csv(input_meta_path)
    print(f"\nOriginal dataset: {len(df):,} samples")

    # Group by class
    class_groups = df.groupby("label_61")
    print(f"Classes: {len(class_groups)}")

    # Sample from each class
    balanced_samples = []
    stats = {
        "kept_all": 0,
        "downsampled": 0,
        "rare_classes": [],
    }

    for class_id, group in class_groups:
        n_samples = len(group)

        if n_samples >= samples_per_class:
            # Downsample: randomly select samples_per_class samples
            sampled = group.sample(n=samples_per_class, random_state=seed)
            balanced_samples.append(sampled)
            stats["downsampled"] += 1
        else:
            # Keep all samples (too few to downsample)
            balanced_samples.append(group)
            stats["kept_all"] += 1
            stats["rare_classes"].append((class_id, n_samples))

    # Combine all samples
    balanced_df = pd.concat(balanced_samples, ignore_index=True)

    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save
    output_path = Path(output_meta_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(output_path, index=False)

    print(f"\nBalanced dataset: {len(balanced_df):,} samples")
    print(f"Reduction: {100 * (1 - len(balanced_df) / len(df)):.1f}%")
    print(f"\nSampling statistics:")
    print(f"  Downsampled classes: {stats['downsampled']}")
    print(f"  Kept all samples: {stats['kept_all']}")

    if stats["rare_classes"]:
        print(f"\nRare classes (<{samples_per_class} samples):")
        for class_id, n_samples in sorted(stats["rare_classes"], key=lambda x: x[1]):
            print(f"  Class {class_id:2d}: {n_samples:3d} samples (kept all)")

    print(f"\nSaved to: {output_path}")

    # Class distribution summary
    new_counts = balanced_df["label_61"].value_counts().sort_index()
    print(f"\nNew class distribution:")
    print(f"  Min: {new_counts.min()}")
    print(f"  Max: {new_counts.max()}")
    print(f"  Mean: {new_counts.mean():.1f}")
    print(f"  Target: {samples_per_class}")

    return balanced_df


def verify_balance(metadata_path: str):
    """Verify the balanced dataset."""
    df = pd.read_csv(metadata_path)

    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    counts = df["label_61"].value_counts().sort_index()

    print(f"\nTotal samples: {len(df):,}")
    print(f"Classes: {len(counts)}")
    print(f"\nPer-class distribution:")
    print(f"  Min: {counts.min()}")
    print(f"  Max: {counts.max()}")
    print(f"  Mean: {counts.mean():.1f}")
    print(f"  Std: {counts.std():.1f}")

    # Show distribution
    import numpy as np

    bins = [0, 50, 100, 150, 200, 250, 300]
    hist, _ = np.histogram(counts.values, bins=bins)

    print(f"\nSample distribution:")
    for i in range(len(bins) - 1):
        print(f"  {bins[i]:3d}-{bins[i + 1]:3d}: {hist[i]:2d} classes")


def main():
    parser = argparse.ArgumentParser(description="Create balanced training dataset")

    parser.add_argument(
        "--input",
        type=str,
        default="data/cleaned/metadata/train_metadata.csv",
        help="Input metadata CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cleaned/metadata/train_metadata_balanced.csv",
        help="Output metadata CSV",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=200,
        help="Target samples per class",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create balanced dataset
    balanced_df = create_balanced_dataset(
        input_meta_path=args.input,
        output_meta_path=args.output,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    # Verify
    verify_balance(args.output)

    print("\n" + "=" * 60)
    print("✅ Balanced dataset created successfully!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Use balanced dataset for training:")
    print(f"     bash train_fast.sh")
    print(f"\n  2. Compare training time:")
    print(f"     Original: ~31k samples")
    print(f"     Balanced: ~{len(balanced_df):,} samples")
    print(f"     Speedup: ~{31541 / len(balanced_df):.1f}x per epoch")


if __name__ == "__main__":
    main()
