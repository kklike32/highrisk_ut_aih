from __future__ import annotations

import argparse

from derm_advisor.vision.pad_ufes_train import PADUFESTrainConfig, train_pad_ufes_model
from derm_advisor.viz.pad_ufes_reports import generate_pad_ufes_report_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Path containing train/val/test folders.")
    parser.add_argument("--run-name", default="pad_ufes20_efficientnetv2_s")
    parser.add_argument("--checkpoint-name", default="pad_ufes20_efficientnetv2_s_best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", default="tf_efficientnetv2_s.in21k")
    parser.add_argument("--device", default="cuda", help='One of: "cuda", "mps", "cpu"')
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-class", default="MEL", help="Class to use for ROC/PR report assets.")
    parser.add_argument("--artifacts-dir", default=None, help="Optional artifact output root.")
    parser.add_argument("--reports-dir", default=None, help="Optional reports output root.")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted cross entropy for this run.",
    )
    parser.add_argument(
        "--no-save-last-checkpoint",
        action="store_true",
        help="Skip writing a final last_model.pt checkpoint.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained image weights (useful for offline smoke tests).",
    )
    args = parser.parse_args()

    metrics_path = train_pad_ufes_model(
        PADUFESTrainConfig(
            dataset_root=args.dataset_root,
            run_name=args.run_name,
            checkpoint_name=args.checkpoint_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            backbone=args.backbone,
            device=args.device,
            num_workers=args.num_workers,
            seed=args.seed,
            artifacts_dir=args.artifacts_dir,
            reports_dir=args.reports_dir,
            use_class_weights=not args.no_class_weights,
            save_last_checkpoint=not args.no_save_last_checkpoint,
            pretrained=not args.no_pretrained,
        )
    )

    generated = generate_pad_ufes_report_assets(metrics_path, positive_label=args.positive_class)
    print(f"Wrote metrics: {metrics_path}")
    for path in generated:
        print(f"Wrote report asset: {path}")


if __name__ == "__main__":
    main()
