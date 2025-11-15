"""
Visualization Demo Script
==========================

"Talk is cheap. Show me the code." - Linus Torvalds

This script demonstrates the training visualization functionality
with synthetic data. Run this to see what the plots look like.

Usage:
    python demo_visualization.py
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_synthetic_history(scenario: str = "good"):
    """
    Generate synthetic training history for demo.

    Args:
        scenario: 'good', 'overfitting', or 'underfitting'

    Returns:
        history: Dict with training metrics
    """
    epochs = list(range(50))

    if scenario == "good":
        # Good training: convergence, small gap
        train_loss = [2.0 * np.exp(-0.1 * e) + 0.1 + np.random.normal(0, 0.02) for e in epochs]
        val_loss = [2.0 * np.exp(-0.1 * e) + 0.15 + np.random.normal(0, 0.03) for e in epochs]
        train_acc = [10 + 85 * (1 - np.exp(-0.12 * e)) + np.random.normal(0, 0.5) for e in epochs]
        val_acc = [10 + 80 * (1 - np.exp(-0.12 * e)) + np.random.normal(0, 1.0) for e in epochs]

    elif scenario == "overfitting":
        # Overfitting: train continues improving, val plateaus
        train_loss = [2.0 * np.exp(-0.15 * e) + 0.05 + np.random.normal(0, 0.02) for e in epochs]
        val_loss = [
            2.0 * np.exp(-0.1 * e) + 0.3 + np.random.normal(0, 0.03)
            if e < 20
            else 0.5 + 0.01 * (e - 20) + np.random.normal(0, 0.05)
            for e in epochs
        ]
        train_acc = [10 + 88 * (1 - np.exp(-0.15 * e)) + np.random.normal(0, 0.5) for e in epochs]
        val_acc = [
            10 + 70 * (1 - np.exp(-0.12 * e)) + np.random.normal(0, 1.0)
            if e < 20
            else 75 - 0.1 * (e - 20) + np.random.normal(0, 1.5)
            for e in epochs
        ]

    else:  # underfitting
        # Underfitting: both train and val low
        train_loss = [1.5 * np.exp(-0.05 * e) + 0.8 + np.random.normal(0, 0.03) for e in epochs]
        val_loss = [1.5 * np.exp(-0.05 * e) + 0.85 + np.random.normal(0, 0.04) for e in epochs]
        train_acc = [10 + 40 * (1 - np.exp(-0.08 * e)) + np.random.normal(0, 1.0) for e in epochs]
        val_acc = [10 + 35 * (1 - np.exp(-0.08 * e)) + np.random.normal(0, 1.5) for e in epochs]

    # Learning rate with warmup + cosine decay
    warmup = 5
    lr = []
    for e in epochs:
        if e < warmup:
            lr.append(1e-5 + (5e-4 - 1e-5) * (e / warmup))
        else:
            # Cosine decay
            progress = (e - warmup) / (len(epochs) - warmup)
            lr.append(1e-6 + (5e-4 - 1e-6) * 0.5 * (1 + np.cos(np.pi * progress)))

    return {
        "epoch": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "learning_rate": lr,
    }


def plot_training_curves(history, save_path: str, title: str = "Training Progress"):
    """
    Plot comprehensive training curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=18, fontweight="bold")

    epochs = history["epoch"]

    # Plot 1: Loss curves
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

    # Plot 2: Accuracy curves
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

    # Plot 3: Learning rate schedule
    ax3 = axes[1, 0]
    ax3.plot(epochs, history["learning_rate"], "g-", linewidth=2.5, alpha=0.8)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Learning Rate", fontsize=12)
    ax3.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3, linestyle="--")

    # Add warmup marker
    lrs = history["learning_rate"]
    for i in range(1, len(lrs)):
        if lrs[i] < lrs[i - 1]:
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

    # Plot 4: Overfitting analysis
    ax4 = axes[1, 1]
    acc_gap = [train - val for train, val in zip(history["train_acc"], history["val_acc"])]
    ax4.plot(epochs, acc_gap, "purple", linewidth=2, alpha=0.8, label="Train-Val Gap")
    ax4.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax4.fill_between(epochs, 0, acc_gap, alpha=0.3, color="purple")
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("Accuracy Gap (%)", fontsize=12)
    ax4.set_title("Overfitting Analysis", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=11, loc="upper left")
    ax4.grid(True, alpha=0.3, linestyle="--")

    # Add interpretation
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

    # Summary statistics
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Training Visualization Demo")
    print("=" * 60)
    print("\nGenerating synthetic training scenarios...")

    output_dir = Path("demo_plots")
    output_dir.mkdir(exist_ok=True)

    scenarios = ["good", "overfitting", "underfitting"]

    for scenario in scenarios:
        print(f"\n📊 Generating '{scenario}' scenario...")
        history = generate_synthetic_history(scenario)
        output_path = output_dir / f"demo_{scenario}.png"
        plot_training_curves(history, str(output_path), f"Demo: {scenario.capitalize()} Training")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print(f"\nGenerated plots in: {output_dir}/")
    print("  - demo_good.png          : Well-trained model")
    print("  - demo_overfitting.png   : Overfitting example")
    print("  - demo_underfitting.png  : Underfitting example")
    print("\nThese plots show what to expect during real training.")


if __name__ == "__main__":
    main()
