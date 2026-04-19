from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """
    Thin wrapper around Kaggle API.

    Examples:
      python scripts/download_kaggle_dataset.py --dataset "kmader/skin-cancer-mnist-ham10000" --out data/kaggle
    """
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help='Kaggle dataset slug like "kmader/skin-cancer-mnist-ham10000"')
    p.add_argument("--out", required=True, help="Output directory (will be created).")
    args = p.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Missing kaggle package. Run: pip install -r requirements.txt\n"
            f"Original import error: {e}"
        )

    # Uses ~/.kaggle/kaggle.json (see README for setup)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(args.dataset, path=str(out_dir), unzip=True)
    print(f"Downloaded+unzipped Kaggle dataset to: {out_dir}")


if __name__ == "__main__":
    main()

