from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from derm_advisor.viz.plots import plot_training_curves


def _load_metrics(metrics_json: str | Path) -> dict:
    return json.loads(Path(metrics_json).read_text(encoding="utf-8"))


def _load_predictions(metrics: dict) -> pd.DataFrame:
    predictions_path = Path(metrics["predictions_csv"])
    return pd.read_csv(predictions_path)


def plot_workflow_diagram(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")

    boxes = [
        (0.05, "HAM10000 download"),
        (0.25, "Lesion-level split"),
        (0.45, "ConvNeXt training"),
        (0.65, "Evaluation bundle"),
        (0.85, "Streamlit + ADK"),
    ]
    for xpos, label in boxes:
        rect = plt.Rectangle((xpos - 0.08, 0.35), 0.16, 0.28, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(xpos, 0.49, label, ha="center", va="center", fontsize=11)

    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate(
            "",
            xy=(end[0] - 0.1, 0.49),
            xytext=(start[0] + 0.1, 0.49),
            arrowprops={"arrowstyle": "->", "linewidth": 1.8},
        )

    ax.set_title("ISIC / HAM10000 workflow", fontsize=14, pad=14)
    out_path = out_dir / "workflow_diagram.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_val_macro_f1_curve(metrics_json: str | Path, out_dir: str | Path) -> Path | None:
    data = _load_metrics(metrics_json)
    history = data.get("history", [])
    if not history or "val_macro_f1" not in history[0]:
        return None

    epochs = [row["epoch"] for row in history]
    macro_f1 = [row["val_macro_f1"] for row in history]

    out_dir = Path(out_dir)
    out_path = out_dir / "val_macro_f1_curve.png"
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, macro_f1, label="val_macro_f1")
    plt.xlabel("epoch")
    plt.ylabel("macro F1")
    plt.title("Validation macro F1")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_class_distribution(metrics_json: str | Path, out_dir: str | Path) -> Path:
    metrics = _load_metrics(metrics_json)
    split_summary = metrics.get("split_summary")
    dataset_summary = metrics.get("dataset_summary", {})
    rows: list[dict[str, object]] = []

    source = split_summary.get("splits") if isinstance(split_summary, dict) else None
    if source:
        for split_name, split_info in source.items():
            for label, count in split_info.get("class_counts", {}).items():
                rows.append({"split": split_name, "label": label, "count": int(count)})
    else:
        for split_name, split_info in dataset_summary.items():
            for label, count in split_info.get("class_counts", {}).items():
                rows.append({"split": split_name, "label": label, "count": int(count)})

    frame = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_path = out_dir / "class_distribution.png"
    plt.figure(figsize=(10, 5))
    sns.barplot(data=frame, x="label", y="count", hue="split")
    plt.title("Class distribution by split")
    plt.xlabel("class")
    plt.ylabel("images")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_confusion_matrix(metrics_json: str | Path, out_dir: str | Path) -> Path:
    metrics = _load_metrics(metrics_json)
    class_names = metrics.get("class_names", [])
    matrix = np.asarray(metrics["test_confusion_matrix_normalized"], dtype=float)

    out_dir = Path(out_dir)
    out_path = out_dir / "confusion_matrix.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Normalized test confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_per_class_metrics(metrics_json: str | Path, out_dir: str | Path) -> Path:
    metrics = _load_metrics(metrics_json)
    frame = pd.DataFrame(metrics.get("test_per_class", []))
    melted = frame.melt(
        id_vars=["label"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )

    out_dir = Path(out_dir)
    out_path = out_dir / "per_class_metrics.png"
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="label", y="value", hue="metric")
    plt.ylim(0, 1)
    plt.title("Per-class test metrics")
    plt.xlabel("class")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_reliability_diagram(metrics_json: str | Path, out_dir: str | Path) -> Path:
    metrics = _load_metrics(metrics_json)
    frame = _load_predictions(metrics)
    confidences = frame["confidence"].to_numpy(dtype=float)
    correctness = frame["correct"].astype(float).to_numpy()
    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_acc: list[float] = []
    bin_conf: list[float] = []

    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lower) & (confidences < upper)
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        if not np.any(mask):
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            continue
        bin_acc.append(float(np.mean(correctness[mask])))
        bin_conf.append(float(np.mean(confidences[mask])))

    out_dir = Path(out_dir)
    out_path = out_dir / "reliability_diagram.png"
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
    plt.plot(bin_centers, bin_acc, marker="o", label="accuracy")
    plt.plot(bin_centers, bin_conf, marker="s", label="confidence")
    plt.xlabel("confidence bin")
    plt.ylabel("value")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_confidence_histogram(metrics_json: str | Path, out_dir: str | Path) -> Path:
    metrics = _load_metrics(metrics_json)
    frame = _load_predictions(metrics)

    out_dir = Path(out_dir)
    out_path = out_dir / "confidence_histogram.png"
    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=frame,
        x="confidence",
        hue="correct",
        bins=10,
        stat="density",
        common_norm=False,
        element="step",
    )
    plt.title("Confidence by correctness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def _plot_one_vs_rest_curve(
    metrics_json: str | Path,
    out_dir: str | Path,
    *,
    positive_label: str,
    curve_kind: str,
) -> Path | None:
    metrics = _load_metrics(metrics_json)
    frame = _load_predictions(metrics)
    score_col = f"prob__{positive_label}"
    if score_col not in frame.columns:
        return None

    y_true = (frame["true_label"] == positive_label).astype(int).to_numpy()
    y_score = frame[score_col].to_numpy(dtype=float)
    if np.min(y_true) == np.max(y_true):
        return None

    out_dir = Path(out_dir)
    suffix = "roc" if curve_kind == "roc" else "pr"
    out_path = out_dir / f"{positive_label}_{suffix}_curve.png"
    plt.figure(figsize=(6, 5))
    if curve_kind == "roc":
        RocCurveDisplay.from_predictions(y_true, y_score)
        plt.title(f"{positive_label} one-vs-rest ROC")
    else:
        PrecisionRecallDisplay.from_predictions(y_true, y_score)
        plt.title(f"{positive_label} one-vs-rest PR")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def plot_error_gallery(metrics_json: str | Path, out_dir: str | Path, limit: int = 8) -> Path | None:
    metrics = _load_metrics(metrics_json)
    frame = _load_predictions(metrics)
    if "image_path" not in frame.columns:
        return None

    errors = frame[~frame["correct"].astype(bool)].copy()
    if errors.empty:
        return None

    errors = errors.sort_values("confidence", ascending=False).head(limit)
    rows = int(np.ceil(len(errors) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, 4)

    for axis in axes.ravel():
        axis.axis("off")

    for axis, (_, row) in zip(axes.ravel(), errors.iterrows()):
        image_path = Path(row["image_path"])
        if not image_path.exists():
            continue
        axis.imshow(Image.open(image_path).convert("RGB"))
        axis.set_title(
            f"true={row['true_label']}\npred={row['pred_label']}\nconf={row['confidence']:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    fig.suptitle("Top confident test errors", fontsize=14)
    out_dir = Path(out_dir)
    out_path = out_dir / "error_gallery.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def generate_isic_report_assets(
    metrics_json: str | Path,
    *,
    positive_label: str = "mel",
) -> list[Path]:
    metrics_json = Path(metrics_json)
    out_dir = metrics_json.parent
    generated = [plot_workflow_diagram(out_dir)]
    generated.extend(plot_training_curves(metrics_json, out_dir))

    maybe_paths = [
        plot_val_macro_f1_curve(metrics_json, out_dir),
        plot_class_distribution(metrics_json, out_dir),
        plot_confusion_matrix(metrics_json, out_dir),
        plot_per_class_metrics(metrics_json, out_dir),
        plot_reliability_diagram(metrics_json, out_dir),
        plot_confidence_histogram(metrics_json, out_dir),
        _plot_one_vs_rest_curve(metrics_json, out_dir, positive_label=positive_label, curve_kind="roc"),
        _plot_one_vs_rest_curve(metrics_json, out_dir, positive_label=positive_label, curve_kind="pr"),
        plot_error_gallery(metrics_json, out_dir),
    ]

    for maybe_path in maybe_paths:
        if maybe_path is not None:
            generated.append(maybe_path)
    return generated
