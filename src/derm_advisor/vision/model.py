from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "convnext_tiny.fb_in22k"
    num_classes: int = 7
    pretrained: bool = True
    dropout: float = 0.2


def create_model(cfg: ModelConfig) -> nn.Module:
    m = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        drop_rate=cfg.dropout,
    )
    return m


@torch.inference_mode()
def predict_proba(model: nn.Module, image_tensor: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)
    logits = model(x)
    return torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()

