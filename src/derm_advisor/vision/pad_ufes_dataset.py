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
class PADUFESImageSample:
    path: Path
    label: int


class PADUFESPathLabelDataset(Dataset):
    """
    ImageFolder-style dataset used by the PAD-UFES-20 pipeline.

    This lives alongside the partner-owned HAM10000 dataset code so the new
    project direction can evolve independently.
    """

    def __init__(self, samples: list[PADUFESImageSample], transform=None, return_path: bool = False):
        self.samples = samples
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        image_array = np.array(image)
        if self.transform is not None:
            image_array = self.transform(image=image_array)["image"]
        if not isinstance(image_array, torch.Tensor):
            image_array = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        if self.return_path:
            return image_array, int(sample.label), str(sample.path)
        return image_array, int(sample.label)


def load_pad_ufes_split_from_imagefolder(
    root: Path,
    split: Split,
) -> tuple[list[PADUFESImageSample], list[str]]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    class_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
    class_names = [path.name for path in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples: list[PADUFESImageSample] = []
    for class_name in class_names:
        class_dir = split_dir / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                samples.append(PADUFESImageSample(path=path, label=class_to_idx[class_name]))

    if not samples:
        raise RuntimeError(f"No images found under {split_dir}")

    return samples, class_names
