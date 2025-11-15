"""
Training Visualization Tool
============================

"A picture is worth a thousand words." - But only if it's a GOOD picture.

This script visualizes training progress from checkpoint files.
Simple, standalone, no dependencies on training code.

Usage:
    python visualize_training.py --checkpoint checkpoints/task1_improved/best.pth
    python visualize_training.py --checkpoint-dir checkpoints/task1_improved/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import torch


def load_checkpoint_history(checkpoint_path: str):
    """
    Load training history from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        history: Dict with training metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "history" not in checkpoint:
        raise ValueError(f"No training history found in {checkpoint_path}")

    history = checkpoint["history"]

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epochs: {len(history['epoch'])}")
    print(f"  Best Val Acc: {max(history['val_acc']):.2f}%")

    return history, checkpoint


def plot_training_curves(history, save_path: str, title: str = "Training Progress"):
    """
    Plot comprehensive training curves.

    Good taste: show all important metrics in one clear visualization.
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=18, fontweight="bold")

    epochs = history["epoch"]

    # ============================================================
    # Plot 1: Loss curves
    # ============================================================
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2, alpha=0.8)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2, alpha=0.8)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curves", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Add min loss marker
    min_val_loss_idx = history["val_loss"].index(min(history["val_loss"]))
    min_val_loss_epoch = history["epoch"][min_val_loss_idx]
    min_val_loss = history["val_loss"][min_val_loss_idx]
    ax1.plot(min_val_loss_epoch, min_val_loss, "g*", markersize=15)
    ax1.annotate(
        f"Min: {min_val_loss:.4f}",
        xy=(min_val_loss_epoch, min_val_loss),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
    )

    # ============================================================
    # Plot 2: Accuracy curves
    # ============================================================
    ax2 = axes[0, 1]
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2, alpha=0.8)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2, alpha=0.8)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy Curves", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11, loc="lower right")
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Add best accuracy marker
    best_val_acc_idx = history["val_acc"].index(max(history["val_acc"]))
    best_val_acc_epoch = history["epoch"][best_val_acc_idx]
    best_val_acc = history["val_acc"][best_val_acc_idx]
    ax2.plot(best_val_acc_epoch, best_val_acc, "g*", markersize=15)
    ax2.annotate(
        f"Best: {best_val_acc:.2f}%\n@ Epoch {best_val_acc_epoch}",
        xy=(best_val_acc_epoch, best_val_acc),
        xytext=(10, -30),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7),
    )

    # ============================================================
    # Plot 3: Learning rate schedule
    # ============================================================
    ax3 = axes[1, 0]
    ax3.plot(epochs, history["learning_rate"], "g-", linewidth=2.5, alpha=0.8)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Learning Rate", fontsize=12)
    ax3.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3, linestyle="--")

    # Add warmup/decay markers
    if len(epochs) > 0:
        # Find warmup end (when LR stops increasing)
        lrs = history["learning_rate"]
        for i in range(1, len(lrs)):
            if lrs[i] < lrs[i - 1]:  # LR started decreasing
                ax3.axvline(epochs[i], color="orange", linestyle="--", linewidth=2, alpha=0.7)
                ax3.text(
                    epochs[i],
                    max(lrs) * 0.5,
                    "Warmup End",
                    rotation=90,
                    verticalalignment="center",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.5),
                )
                break

    # ============================================================
    # Plot 4: Overfitting analysis (train-val gap)
    # ============================================================
    ax4 = axes[1, 1]

    # Calculate gaps
    acc_gap = [train - val for train, val in zip(history["train_acc"], history["val_acc"])]

    ax4.plot(epochs, acc_gap, "purple", linewidth=2, alpha=0.8, label="Train-Val Gap")
    ax4.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax4.fill_between(epochs, 0, acc_gap, alpha=0.3, color="purple")
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("Accuracy Gap (%)", fontsize=12)
    ax4.set_title("Overfitting Analysis", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=11, loc="upper left")
    ax4.grid(True, alpha=0.3, linestyle="--")

    # Add interpretation text
    final_gap = acc_gap[-1]
    if final_gap < 5:
        status = "✅ Good fit"
        color = "green"
    elif final_gap < 10:
        status = "⚠️ Slight overfitting"
        color = "orange"
    else:
        status = "❌ Overfitting"
        color = "red"

    ax4.text(
        0.5,
        0.95,
        f"{status}\nFinal Gap: {final_gap:.2f}%",
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3),
    )

    # ============================================================
    # Add summary statistics box
    # ============================================================
    summary_text = (
        f"📊 Training Summary\n"
        f"{'=' * 30}\n"
        f"Total Epochs: {len(epochs)}\n"
        f"Best Val Acc: {best_val_acc:.2f}% @ Epoch {best_val_acc_epoch}\n"
        f"Final Train Acc: {history['train_acc'][-1]:.2f}%\n"
        f"Final Val Acc: {history['val_acc'][-1]:.2f}%\n"
        f"Min Val Loss: {min_val_loss:.4f} @ Epoch {min_val_loss_epoch}\n"
        f"Final LR: {history['learning_rate'][-1]:.2e}"
    )

    fig.text(
        0.02,
        0.02,
        summary_text,
        fontsize=10,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.97])

    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Visualization saved to {save_path}")

    plt.close()


def plot_comparison(histories: dict, save_path: str):
    """
    Compare multiple training runs.

    Args:
        histories: Dict mapping run names to history dicts
        save_path: Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training Runs Comparison", fontsize=18, fontweight="bold")

    colors = ["blue", "red", "green", "orange", "purple", "brown"]

    for idx, (name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        epochs = history["epoch"]

        # Plot losses
        axes[0, 0].plot(
            epochs,
            history["train_loss"],
            color=color,
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            label=f"{name} (train)",
        )
        axes[0, 1].plot(
            epochs,
            history["val_loss"],
            color=color,
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            label=f"{name} (val)",
        )

        # Plot accuracies
        axes[1, 0].plot(
            epochs,
            history["train_acc"],
            color=color,
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            label=f"{name} (train)",
        )
        axes[1, 1].plot(
            epochs,
            history["val_acc"],
            color=color,
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            label=f"{name} (val)",
        )

    axes[0, 0].set_title("Train Loss", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Val Loss", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Train Accuracy", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Val Accuracy", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Comparison plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize training progress")

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing checkpoints (will plot best.pth)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        help="Compare multiple checkpoints",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Training Visualization Tool")
    print("=" * 60)

    if args.compare:
        # Compare multiple runs
        print(f"\nComparing {len(args.compare)} training runs...")

        histories = {}
        for ckpt_path in args.compare:
            name = Path(ckpt_path).parent.name
            history, _ = load_checkpoint_history(ckpt_path)
            histories[name] = history

        output_path = args.output or "training_comparison.png"
        plot_comparison(histories, output_path)

    else:
        # Single checkpoint visualization
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        elif args.checkpoint_dir:
            checkpoint_path = Path(args.checkpoint_dir) / "best.pth"
        else:
            raise ValueError("Must provide --checkpoint or --checkpoint-dir")

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\nLoading checkpoint: {checkpoint_path}")
        history, checkpoint = load_checkpoint_history(checkpoint_path)

        # Generate title
        checkpoint_dir = Path(checkpoint_path).parent.name
        title = f"Training Progress: {checkpoint_dir}"

        # Output path
        if args.output:
            output_path = args.output
        else:
            output_path = Path(checkpoint_path).parent / "training_visualization.png"

        plot_training_curves(history, str(output_path), title)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
