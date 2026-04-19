from __future__ import annotations

import argparse

from derm_advisor.config import Paths
from derm_advisor.vision.train import TrainConfig, train
from derm_advisor.viz.plots import plot_training_curves


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True, help="Path containing train/val/test folders.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--backbone", default="convnext_tiny.fb_in22k")
    p.add_argument("--device", default="mps", help='One of: "mps", "cuda", "cpu"')
    args = p.parse_args()

    metrics_path = train(
        TrainConfig(
            dataset_root=args.dataset_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            backbone=args.backbone,
            device=args.device,
        )
    )

    paths = Paths.default()
    plot_training_curves(metrics_path, paths.reports_dir)
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote plots to: {paths.reports_dir}")


if __name__ == "__main__":
    main()

