from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_dir: Path
    artifacts_dir: Path
    reports_dir: Path

    @staticmethod
    def default() -> "Paths":
        repo_root = Path(__file__).resolve().parents[2]
        return Paths(
            repo_root=repo_root,
            data_dir=repo_root / "data",
            artifacts_dir=repo_root / "artifacts",
            reports_dir=repo_root / "reports",
        )

