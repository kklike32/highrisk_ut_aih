from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int


class PathLabelDataset(Dataset):
    """
    Minimal dataset that works with arbitrary folder layouts.

    It expects a list of (image_path, label_index). Augmentations are passed in
    as a callable that receives/returns a dict with "image" key (Albumentations style).
    """

    def __init__(self, samples: list[ImageSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        image = np.array(img)
        if self.transform is not None:
            out = self.transform(image=image)
            image = out["image"]
        # Albumentations ToTensorV2 yields torch.Tensor already; otherwise convert.
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, int(s.label)


def load_split_from_imagefolder(root: Path, split: Split) -> tuple[list[ImageSample], list[str]]:
    """
    Loads samples from:
      root/train/<class>/*.jpg
      root/val/<class>/*.jpg
      root/test/<class>/*.jpg
    Returns (samples, class_names).
    """
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples: list[ImageSample] = []
    for class_name in class_names:
        for p in sorted((split_dir / class_name).rglob("*")):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                samples.append(ImageSample(path=p, label=class_to_idx[class_name]))

    if not samples:
        raise RuntimeError(f"No images found under {split_dir}")

    return samples, class_names

