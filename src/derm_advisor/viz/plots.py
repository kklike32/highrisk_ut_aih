from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(metrics_json: str | Path, out_dir: str | Path) -> list[Path]:
    metrics_json = Path(metrics_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(metrics_json.read_text())
    history = data.get("history", [])
    if not history:
        raise ValueError("No training history found in metrics.json")

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    val_acc = [r["val_acc"] for r in history]
    val_bal = [r["val_balanced_acc"] for r in history]

    paths: list[Path] = []

    # Loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training vs validation loss")
    plt.legend()
    loss_path = out_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=180)
    plt.close()
    paths.append(loss_path)

    # Accuracy curve
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, val_acc, label="val_acc")
    plt.plot(epochs, val_bal, label="val_balanced_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Validation accuracy")
    plt.ylim(0, 1)
    plt.legend()
    acc_path = out_dir / "val_accuracy_curve.png"
    plt.tight_layout()
    plt.savefig(acc_path, dpi=180)
    plt.close()
    paths.append(acc_path)

    return paths

