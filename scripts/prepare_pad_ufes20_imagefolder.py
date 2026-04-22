from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


SUPPORTED_DIAGNOSTICS = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
REQUIRED_COLUMNS = {"img_id", "lesion_id", "diagnostic"}


def _find_metadata_csv(root: Path) -> Path:
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path, nrows=5)
        except Exception:  # pragma: no cover - defensive for arbitrary user files
            continue
        normalized = {str(column).strip().lower() for column in frame.columns}
        if REQUIRED_COLUMNS.issubset(normalized):
            return path
    raise SystemExit(
        "Could not find a PAD-UFES-20 metadata CSV with columns "
        "`img_id`, `lesion_id`, and `diagnostic` under the provided root."
    )


def _find_images(root: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            images[path.stem] = path
    return images


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    missing_columns = REQUIRED_COLUMNS.difference(normalized.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise SystemExit(f"Missing required columns in PAD-UFES-20 metadata: {missing}")

    normalized["img_id"] = normalized["img_id"].astype(str).str.strip()
    normalized["lesion_id"] = normalized["lesion_id"].astype(str).str.strip()
    normalized["diagnostic"] = normalized["diagnostic"].astype(str).str.strip().str.upper()
    normalized = normalized[
        normalized["img_id"].ne("")
        & normalized["lesion_id"].ne("")
        & normalized["diagnostic"].isin(SUPPORTED_DIAGNOSTICS)
    ].copy()
    if normalized.empty:
        supported = ", ".join(SUPPORTED_DIAGNOSTICS)
        raise SystemExit(
            "No valid PAD-UFES-20 rows remained after filtering for supported diagnostics: "
            f"{supported}"
        )
    return normalized


def _build_group_frame(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby("lesion_id")
        .agg(
            diagnostic=("diagnostic", "first"),
            num_images=("img_id", "count"),
        )
        .reset_index()
    )
    lesion_dx_counts = frame.groupby("lesion_id")["diagnostic"].nunique()
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
    labels = grouped["diagnostic"].astype(str).to_numpy()
    lesion_ids = grouped["lesion_id"].astype(str).to_numpy()

    outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(outer.split(grouped, labels))
    train_val = grouped.iloc[train_val_idx].reset_index(drop=True)
    test_ids = set(lesion_ids[test_idx].tolist())

    inner_labels = train_val["diagnostic"].astype(str).to_numpy()
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
        img_id = str(row["img_id"])
        label = str(row["diagnostic"])
        src = images[img_id]
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
    class_counts = frame["diagnostic"].value_counts().sort_index()
    summary: dict[str, object] = {
        "num_images": int(len(frame)),
        "num_lesions": int(frame["lesion_id"].nunique()),
        "class_counts": {label: int(count) for label, count in class_counts.items()},
    }
    if "patient_id" in frame.columns:
        patient_ids = frame["patient_id"].dropna().astype(str).str.strip()
        patient_ids = patient_ids[patient_ids.ne("")]
        summary["num_patients"] = int(patient_ids.nunique())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Extracted PAD-UFES-20 directory containing images and metadata CSV(s).",
    )
    parser.add_argument("--out", required=True, help="Output dataset root (train/val/test).")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copy (macOS/Linux).")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    metadata_path = _find_metadata_csv(dataset_root)
    images = _find_images(dataset_root)

    frame = pd.read_csv(metadata_path)
    frame = _normalize_frame(frame)
    frame = frame[frame["img_id"].isin(images.keys())].copy()
    if frame.empty:
        raise SystemExit("No PAD-UFES-20 images matched metadata img_id values. Check the extraction path.")

    grouped = _build_group_frame(frame)
    train_ids, val_ids, test_ids = _stratified_group_split(
        grouped,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    split_frames = {
        "train": frame[frame["lesion_id"].isin(train_ids)].reset_index(drop=True),
        "val": frame[frame["lesion_id"].isin(val_ids)].reset_index(drop=True),
        "test": frame[frame["lesion_id"].isin(test_ids)].reset_index(drop=True),
    }

    for split_name, split_frame in split_frames.items():
        _materialize_split(
            split_frame,
            split=split_name,
            out_root=out_root,
            images=images,
            symlink=args.symlink,
        )

    metadata_dir = out_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_frame in split_frames.items():
        split_frame.to_csv(metadata_dir / f"{split_name}_metadata.csv", index=False)

    overlap_exists = bool(
        train_ids.intersection(val_ids)
        or train_ids.intersection(test_ids)
        or val_ids.intersection(test_ids)
    )
    summary = {
        "source_metadata": str(metadata_path),
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "symlink": bool(args.symlink),
        "supported_diagnostics": list(SUPPORTED_DIAGNOSTICS),
        "splits": {
            split_name: _split_summary(split_frame)
            for split_name, split_frame in split_frames.items()
        },
        "leakage_check": {
            "lesion_overlap": overlap_exists,
        },
    }
    (out_root / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Prepared PAD-UFES-20 lesion-level dataset:")
    for split_name, split_frame in split_frames.items():
        print(
            f"  {split_name}: images={len(split_frame)} "
            f"lesions={split_frame['lesion_id'].nunique()}"
        )
    print(f"  lesion overlap across splits: {overlap_exists}")
    print(f"  metadata CSV: {metadata_path}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
