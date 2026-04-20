from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _find_images(root: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png"}
    imgs: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            imgs[p.stem] = p
    return imgs


def main() -> None:
    """
    Prepare HAM10000 into ImageFolder layout with train/val/test splits:
      out/train/<dx>/*.jpg
      out/val/<dx>/*.jpg
      out/test/<dx>/*.jpg

    This script copies files by default (safe everywhere). Use --symlink if you prefer symlinks.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ham-root",
        required=True,
        help="Directory containing HAM10000 images and HAM10000_metadata.csv",
    )
    p.add_argument("--out", required=True, help="Output dataset root (will create train/val/test).")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symlink", action="store_true", help="Symlink instead of copy (mac/linux).")
    args = p.parse_args()

    ham_root = Path(args.ham_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    meta_path = ham_root / "HAM10000_metadata.csv"
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata CSV: {meta_path}")

    df = pd.read_csv(meta_path)
    if "image_id" not in df.columns or "dx" not in df.columns:
        raise SystemExit("Expected columns image_id and dx in HAM10000_metadata.csv")

    images = _find_images(ham_root)
    df = df[df["image_id"].isin(images.keys())].copy()
    if df.empty:
        raise SystemExit("No images matched metadata image_id values. Check your download contents.")

    y = df["dx"].astype(str).values
    splitter_1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_val_idx, test_idx = next(splitter_1.split(df, y))
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_tv = df_train_val["dx"].astype(str).values
    val_frac_of_tv = args.val_size / (1.0 - args.test_size)
    splitter_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_tv, random_state=args.seed)
    train_idx, val_idx = next(splitter_2.split(df_train_val, y_tv))
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    def materialize(split: str, frame: pd.DataFrame) -> None:
        for _, row in frame.iterrows():
            image_id = str(row["image_id"])
            label = str(row["dx"])
            src = images[image_id]
            dst_dir = out_root / split / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if dst.exists():
                continue
            if args.symlink:
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)

    materialize("train", df_train)
    materialize("val", df_val)
    materialize("test", df_test)

    print("Prepared dataset:")
    print(f"  train: {len(df_train)}")
    print(f"  val:   {len(df_val)}")
    print(f"  test:  {len(df_test)}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
