from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from derm_advisor.vision.model import ModelConfig, create_model, predict_proba
from derm_advisor.vision.transforms import build_eval_tfms


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


def _pick_device(preferred: str = "mps") -> str:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(ckpt_path: str | Path, device: str = "mps"):
    device = _pick_device(device)
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["model_config"]
    model = create_model(ModelConfig(**model_cfg))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, ckpt, device


def classify_image(image_path: str | Path, ckpt_path: str | Path) -> ClassificationResult:
    model, ckpt, device = load_checkpoint(ckpt_path, device="mps")
    class_names: list[str] = ckpt["class_names"]
    image_size = int(ckpt["train_config"].get("image_size", 224))

    tfm = build_eval_tfms(image_size)
    img = Image.open(Path(image_path)).convert("RGB")
    arr = np.array(img)
    t = tfm(image=arr)["image"]
    proba = predict_proba(model, t, device=device)

    top_idx = int(torch.argmax(proba).item())
    label = class_names[top_idx]
    confidence = float(proba[top_idx].item())
    probs = {class_names[i]: float(proba[i].item()) for i in range(len(class_names))}
    return ClassificationResult(label=label, confidence=confidence, probabilities=probs)

