from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from derm_advisor.config import Paths
from derm_advisor.vision.dataset import PathLabelDataset, load_split_from_imagefolder
from derm_advisor.vision.model import ModelConfig, create_model
from derm_advisor.vision.transforms import build_eval_tfms, build_train_tfms


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str  # expects train/ val/ test/ subfolders
    image_size: int = 224
    backbone: str = "convnext_tiny.fb_in22k"
    pretrained: bool = True
    num_classes: int = 7
    epochs: int = 5
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    device: str = "mps"  # "cuda" or "cpu"
    seed: int = 42


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_eval(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    loss_fn = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.inference_mode():
            logits = model(x)
            loss = loss_fn(logits, y)
            pred = torch.argmax(logits, dim=-1)
        losses.append(float(loss.detach().cpu()))
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def train(cfg: TrainConfig) -> Path:
    _seed_all(cfg.seed)
    paths = Paths.default()
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(cfg.dataset_root).expanduser().resolve()

    train_samples, class_names = load_split_from_imagefolder(dataset_root, "train")
    val_samples, _ = load_split_from_imagefolder(dataset_root, "val")
    test_samples, _ = load_split_from_imagefolder(dataset_root, "test")

    train_ds = PathLabelDataset(train_samples, transform=build_train_tfms(cfg.image_size))
    val_ds = PathLabelDataset(val_samples, transform=build_eval_tfms(cfg.image_size))
    test_ds = PathLabelDataset(test_samples, transform=build_eval_tfms(cfg.image_size))

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

    device = cfg.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model_cfg = ModelConfig(
        backbone=cfg.backbone,
        num_classes=len(class_names),
        pretrained=cfg.pretrained,
    )
    model = create_model(model_cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict] = []
    best_val = -1.0
    best_path = paths.artifacts_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        train_losses: list[float] = []
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=float(np.mean(train_losses)))

        val_metrics = _run_eval(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_balanced_acc": val_metrics["balanced_acc"],
        }
        history.append(row)

        if val_metrics["balanced_acc"] > best_val:
            best_val = float(val_metrics["balanced_acc"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "train_config": asdict(cfg),
                    "model_config": asdict(model_cfg),
                },
                best_path,
            )

    # final test metrics using best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = _run_eval(model, test_loader, device)

    out = {
        "history": history,
        "test": test_metrics,
        "class_names": ckpt["class_names"],
        "best_checkpoint": str(best_path),
        "train_config": ckpt["train_config"],
        "model_config": ckpt["model_config"],
    }

    metrics_path = paths.reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2))
    return metrics_path

