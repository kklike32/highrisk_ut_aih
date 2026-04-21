from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _find_images(root: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png"}
    images: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            images[path.stem] = path
    return images


def _build_group_frame(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("lesion_id")
        .agg(
            dx=("dx", "first"),
            num_images=("image_id", "count"),
        )
        .reset_index()
    )
    lesion_dx_counts = df.groupby("lesion_id")["dx"].nunique()
    if int(lesion_dx_counts.max()) != 1:
        raise SystemExit("Found lesion_id values mapped to multiple diagnosis labels.")
    return grouped


def _stratified_group_split(
    grouped: pd.DataFrame,
    *,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    labels = grouped["dx"].astype(str).to_numpy()
    lesion_ids = grouped["lesion_id"].astype(str).to_numpy()

    outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(outer.split(grouped, labels))
    train_val = grouped.iloc[train_val_idx].reset_index(drop=True)
    test_ids = set(lesion_ids[test_idx].tolist())

    inner_labels = train_val["dx"].astype(str).to_numpy()
    inner_ids = train_val["lesion_id"].astype(str).to_numpy()
    val_frac_of_train_val = val_size / (1.0 - test_size)
    inner = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_train_val, random_state=seed)
    train_idx, val_idx = next(inner.split(train_val, inner_labels))

    train_ids = set(inner_ids[train_idx].tolist())
    val_ids = set(inner_ids[val_idx].tolist())
    return train_ids, val_ids, test_ids


def _materialize_split(
    frame: pd.DataFrame,
    *,
    split: str,
    out_root: Path,
    images: dict[str, Path],
    symlink: bool,
) -> None:
    for _, row in frame.iterrows():
        image_id = str(row["image_id"])
        label = str(row["dx"])
        src = images[image_id]
        dst_dir = out_root / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists():
            continue
        if symlink:
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)


def _split_summary(frame: pd.DataFrame) -> dict[str, object]:
    class_counts = frame["dx"].value_counts().sort_index()
    return {
        "num_images": int(len(frame)),
        "num_lesions": int(frame["lesion_id"].nunique()),
        "class_counts": {label: int(count) for label, count in class_counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ham-root",
        required=True,
        help="Directory containing HAM10000 images and HAM10000_metadata.csv",
    )
    parser.add_argument("--out", required=True, help="Output dataset root (train/val/test).")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copy (macOS/Linux).")
    args = parser.parse_args()

    ham_root = Path(args.ham_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    meta_path = ham_root / "HAM10000_metadata.csv"
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata CSV: {meta_path}")

    df = pd.read_csv(meta_path)
    required_columns = {"image_id", "dx", "lesion_id"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise SystemExit(f"Missing required columns in HAM10000_metadata.csv: {missing}")

    images = _find_images(ham_root)
    df = df[df["image_id"].isin(images.keys())].copy()
    if df.empty:
        raise SystemExit("No images matched metadata image_id values. Check your download contents.")

    grouped = _build_group_frame(df)
    train_ids, val_ids, test_ids = _stratified_group_split(
        grouped,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    df_train = df[df["lesion_id"].isin(train_ids)].reset_index(drop=True)
    df_val = df[df["lesion_id"].isin(val_ids)].reset_index(drop=True)
    df_test = df[df["lesion_id"].isin(test_ids)].reset_index(drop=True)

    _materialize_split(df_train, split="train", out_root=out_root, images=images, symlink=args.symlink)
    _materialize_split(df_val, split="val", out_root=out_root, images=images, symlink=args.symlink)
    _materialize_split(df_test, split="test", out_root=out_root, images=images, symlink=args.symlink)

    overlap_exists = bool(
        train_ids.intersection(val_ids) or train_ids.intersection(test_ids) or val_ids.intersection(test_ids)
    )
    summary = {
        "source_metadata": str(meta_path),
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "symlink": bool(args.symlink),
        "splits": {
            "train": _split_summary(df_train),
            "val": _split_summary(df_val),
            "test": _split_summary(df_test),
        },
        "leakage_check": {
            "lesion_overlap": overlap_exists,
        },
    }
    (out_root / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Prepared lesion-level dataset:")
    print(f"  train images: {len(df_train)}  lesions: {df_train['lesion_id'].nunique()}")
    print(f"  val images:   {len(df_val)}  lesions: {df_val['lesion_id'].nunique()}")
    print(f"  test images:  {len(df_test)}  lesions: {df_test['lesion_id'].nunique()}")
    print(f"  lesion overlap across splits: {overlap_exists}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
