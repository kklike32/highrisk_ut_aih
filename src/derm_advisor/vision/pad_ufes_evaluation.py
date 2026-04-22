from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

from derm_advisor.vision.pad_ufes_dataset import PADUFESImageSample


@dataclass(frozen=True)
class PADUFESEvalArtifacts:
    summary: dict[str, Any]
    per_class: list[dict[str, Any]]
    confusion_matrix: list[list[int]]
    confusion_matrix_normalized: list[list[float]]
    predictions: list[dict[str, Any]]


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _safe_auc(binary_true: np.ndarray, scores: np.ndarray) -> float | None:
    if binary_true.min() == binary_true.max():
        return None
    return float(roc_auc_score(binary_true, scores))


def _safe_average_precision(binary_true: np.ndarray, scores: np.ndarray) -> float | None:
    if binary_true.min() == binary_true.max():
        return None
    return float(average_precision_score(binary_true, scores))


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    confidences = np.max(y_prob, axis=1)
    correctness = (y_true == y_pred).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(correctness[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        ece += (float(np.sum(mask)) / float(len(confidences))) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes, dtype=float)[y_true]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def compute_class_weights(samples: list[PADUFESImageSample], num_classes: int) -> list[float]:
    counts = np.bincount([sample.label for sample in samples], minlength=num_classes).astype(float)
    if np.any(counts == 0):
        raise ValueError("Cannot compute class weights with a missing class in the training split.")

    weights = counts.sum() / (num_classes * counts)
    weights = weights / np.mean(weights)
    return weights.astype(float).tolist()


def summarize_samples(
    samples: list[PADUFESImageSample],
    class_names: list[str],
) -> dict[str, Any]:
    counts = np.bincount([sample.label for sample in samples], minlength=len(class_names))
    class_counts = {class_names[idx]: int(counts[idx]) for idx in range(len(class_names))}
    return {
        "num_images": int(len(samples)),
        "class_counts": class_counts,
    }


def build_prediction_records(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    image_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_idx, true_label in enumerate(y_true):
        row_probs = y_prob[row_idx]
        record: dict[str, Any] = {
            "image_path": image_paths[row_idx] if image_paths is not None else None,
            "true_index": int(true_label),
            "true_label": class_names[int(true_label)],
            "pred_index": int(y_pred[row_idx]),
            "pred_label": class_names[int(y_pred[row_idx])],
            "confidence": float(np.max(row_probs)),
            "correct": bool(true_label == y_pred[row_idx]),
        }
        for class_idx, class_name in enumerate(class_names):
            record[f"prob__{class_name}"] = float(row_probs[class_idx])
        records.append(record)
    return records


def save_prediction_records(records: list[dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(out_path, index=False)
    return out_path


# pylint: disable=too-many-arguments
def evaluate_predictions(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    loss: float | None = None,
    image_paths: list[str] | None = None,
) -> PADUFESEvalArtifacts:
    num_classes = len(class_names)
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    conf_norm = conf.astype(float)
    row_sums = conf_norm.sum(axis=1, keepdims=True)
    np.divide(conf_norm, row_sums, out=conf_norm, where=row_sums != 0)

    y_true_one_hot = np.eye(num_classes, dtype=float)[y_true]
    topk = 1.0 if num_classes <= 2 else float(
        top_k_accuracy_score(y_true, y_prob, k=min(2, num_classes), labels=labels)
    )
    confidences = np.max(y_prob, axis=1)
    correct_mask = y_true == y_pred

    per_class: list[dict[str, Any]] = []
    auc_values: list[float] = []
    auprc_values: list[float] = []
    for class_idx, class_name in enumerate(class_names):
        auroc = _safe_auc(y_true_one_hot[:, class_idx], y_prob[:, class_idx])
        auprc = _safe_average_precision(y_true_one_hot[:, class_idx], y_prob[:, class_idx])
        if auroc is not None:
            auc_values.append(auroc)
        if auprc is not None:
            auprc_values.append(auprc)

        per_class.append(
            {
                "label": class_name,
                "precision": float(precision[class_idx]),
                "recall": float(recall[class_idx]),
                "f1": float(f1[class_idx]),
                "support": int(support[class_idx]),
                "auroc": auroc,
                "average_precision": auprc,
            }
        )

    summary = {
        "loss": None if loss is None else float(loss),
        "num_samples": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_acc": topk,
        "macro_auroc": _mean_or_none(np.asarray(auc_values, dtype=float)),
        "macro_average_precision": _mean_or_none(np.asarray(auprc_values, dtype=float)),
        "mean_confidence": float(np.mean(confidences)),
        "mean_confidence_correct": _mean_or_none(confidences[correct_mask]),
        "mean_confidence_incorrect": _mean_or_none(confidences[~correct_mask]),
        "ece": float(expected_calibration_error(y_true, y_pred, y_prob)),
        "brier_score": float(multiclass_brier_score(y_true, y_prob, num_classes)),
    }

    predictions = build_prediction_records(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        image_paths=image_paths,
    )
    return PADUFESEvalArtifacts(
        summary=summary,
        per_class=per_class,
        confusion_matrix=conf.astype(int).tolist(),
        confusion_matrix_normalized=conf_norm.astype(float).tolist(),
        predictions=predictions,
    )
