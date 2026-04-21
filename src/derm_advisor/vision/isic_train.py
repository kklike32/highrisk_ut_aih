from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from derm_advisor.config import Paths
from derm_advisor.vision.isic_dataset import (
    ISICPathLabelDataset,
    load_isic_split_from_imagefolder,
)
from derm_advisor.vision.isic_evaluation import (
    ISICEvalArtifacts,
    compute_class_weights,
    evaluate_predictions,
    save_prediction_records,
    summarize_samples,
)
from derm_advisor.vision.model import ModelConfig, create_model
from derm_advisor.vision.transforms import build_eval_tfms, build_train_tfms


@dataclass(frozen=True)
class ISICTrainConfig:
    dataset_root: str
    image_size: int = 224
    backbone: str = "convnext_tiny.fb_in22k"
    pretrained: bool = True
    epochs: int = 15
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    device: str = "cuda"
    seed: int = 42
    run_name: str = "isic_convnext_tiny"
    artifacts_dir: str | None = None
    reports_dir: str | None = None
    checkpoint_name: str = "isic_convnext_tiny_best.pt"
    use_class_weights: bool = True
    save_last_checkpoint: bool = True


@dataclass(frozen=True)
class ISICOutputPaths:
    artifact_dir: Path
    report_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metrics_json: Path
    predictions_csv: Path


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_device(preferred: str) -> str:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_output_paths(cfg: ISICTrainConfig, paths: Paths) -> ISICOutputPaths:
    artifact_root = (
        Path(cfg.artifacts_dir).expanduser().resolve()
        if cfg.artifacts_dir
        else paths.artifacts_dir
    )
    report_root = Path(cfg.reports_dir).expanduser().resolve() if cfg.reports_dir else paths.reports_dir

    artifact_dir = artifact_root / cfg.run_name
    report_dir = report_root / cfg.run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return ISICOutputPaths(
        artifact_dir=artifact_dir,
        report_dir=report_dir,
        best_checkpoint=artifact_dir / cfg.checkpoint_name,
        last_checkpoint=artifact_dir / "last_model.pt",
        metrics_json=report_dir / "metrics.json",
        predictions_csv=report_dir / "test_predictions.csv",
    )


def _checkpoint_payload(
    model: nn.Module,
    *,
    class_names: list[str],
    cfg: ISICTrainConfig,
    model_cfg: ModelConfig,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "train_config": asdict(cfg),
        "model_config": asdict(model_cfg),
        **extra,
    }


def _load_split_summary(dataset_root: Path) -> dict[str, Any] | None:
    summary_path = dataset_root / "split_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _verify_class_names(
    train_class_names: list[str],
    val_class_names: list[str],
    test_class_names: list[str],
) -> None:
    if train_class_names != val_class_names or train_class_names != test_class_names:
        raise ValueError("Train/val/test class folders do not match. Rebuild the ISIC dataset splits.")


def _unpack_batch(batch):
    if len(batch) == 3:
        x, y, image_paths = batch
        return x, y, list(image_paths)
    x, y = batch
    return x, y, None


def _run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    class_names: list[str],
    loss_fn: nn.Module,
) -> ISICEvalArtifacts:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    losses: list[float] = []
    image_paths: list[str] = []

    for batch in loader:
        x, y, batch_paths = _unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)
        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            loss = loss_fn(logits, y)
            pred = torch.argmax(probs, dim=-1)
        losses.append(float(loss.detach().cpu()))
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().tolist())
        if batch_paths is not None:
            image_paths.extend(batch_paths)

    image_paths_arg = image_paths if image_paths else None
    return evaluate_predictions(
        y_true=np.asarray(y_true, dtype=int),
        y_pred=np.asarray(y_pred, dtype=int),
        y_prob=np.asarray(y_prob, dtype=float),
        class_names=class_names,
        loss=float(np.mean(losses)) if losses else None,
        image_paths=image_paths_arg,
    )


def train_isic_model(cfg: ISICTrainConfig) -> Path:
    _seed_all(cfg.seed)
    paths = Paths.default()
    output_paths = _resolve_output_paths(cfg, paths)
    dataset_root = Path(cfg.dataset_root).expanduser().resolve()

    train_samples, train_class_names = load_isic_split_from_imagefolder(dataset_root, "train")
    val_samples, val_class_names = load_isic_split_from_imagefolder(dataset_root, "val")
    test_samples, test_class_names = load_isic_split_from_imagefolder(dataset_root, "test")
    _verify_class_names(train_class_names, val_class_names, test_class_names)
    class_names = train_class_names

    train_ds = ISICPathLabelDataset(train_samples, transform=build_train_tfms(cfg.image_size))
    val_ds = ISICPathLabelDataset(val_samples, transform=build_eval_tfms(cfg.image_size))
    test_ds = ISICPathLabelDataset(
        test_samples,
        transform=build_eval_tfms(cfg.image_size),
        return_path=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    device = _pick_device(cfg.device)
    model_cfg = ModelConfig(
        backbone=cfg.backbone,
        num_classes=len(class_names),
        pretrained=cfg.pretrained,
    )
    model = create_model(model_cfg).to(device)

    class_weights = None
    if cfg.use_class_weights:
        class_weights = compute_class_weights(train_samples, len(class_names))
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
    else:
        loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict[str, Any]] = []
    best_val = -1.0
    split_summary = _load_split_summary(dataset_root)
    dataset_summary = {
        "train": summarize_samples(train_samples, class_names),
        "val": summarize_samples(val_samples, class_names),
        "test": summarize_samples(test_samples, class_names),
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        train_losses: list[float] = []
        for batch in pbar:
            x, y, _ = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=float(np.mean(train_losses)))

        val_eval = _run_eval(model, val_loader, device, class_names, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else None,
                "val_loss": val_eval.summary["loss"],
                "val_acc": val_eval.summary["acc"],
                "val_balanced_acc": val_eval.summary["balanced_acc"],
                "val_macro_f1": val_eval.summary["macro_f1"],
                "val_weighted_f1": val_eval.summary["weighted_f1"],
                "val_top2_acc": val_eval.summary["top2_acc"],
                "val_ece": val_eval.summary["ece"],
            }
        )

        if val_eval.summary["balanced_acc"] > best_val:
            best_val = float(val_eval.summary["balanced_acc"])
            torch.save(
                _checkpoint_payload(
                    model,
                    class_names=class_names,
                    cfg=cfg,
                    model_cfg=model_cfg,
                    extra={
                        "best_val_balanced_acc": best_val,
                        "class_weights": class_weights,
                    },
                ),
                output_paths.best_checkpoint,
            )

    if cfg.save_last_checkpoint:
        torch.save(
            _checkpoint_payload(
                model,
                class_names=class_names,
                cfg=cfg,
                model_cfg=model_cfg,
                extra={
                    "best_val_balanced_acc": best_val,
                    "class_weights": class_weights,
                },
            ),
            output_paths.last_checkpoint,
        )

    ckpt = torch.load(output_paths.best_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_eval = _run_eval(model, test_loader, device, class_names, loss_fn)
    save_prediction_records(test_eval.predictions, output_paths.predictions_csv)

    output = {
        "history": history,
        "test": test_eval.summary,
        "test_per_class": test_eval.per_class,
        "test_confusion_matrix": test_eval.confusion_matrix,
        "test_confusion_matrix_normalized": test_eval.confusion_matrix_normalized,
        "class_names": ckpt["class_names"],
        "best_checkpoint": str(output_paths.best_checkpoint),
        "last_checkpoint": str(output_paths.last_checkpoint) if cfg.save_last_checkpoint else None,
        "predictions_csv": str(output_paths.predictions_csv),
        "dataset_root": str(dataset_root),
        "dataset_summary": dataset_summary,
        "split_summary": split_summary,
        "train_config": ckpt["train_config"],
        "model_config": ckpt["model_config"],
        "class_weights": class_weights,
    }

    output_paths.metrics_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_paths.metrics_json
