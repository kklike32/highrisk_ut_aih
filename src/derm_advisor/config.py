"""Repository paths and local-model configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LOCAL_LLM_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Paths:
    """Common repository output locations."""

    repo_root: Path
    data_dir: Path
    artifacts_dir: Path
    reports_dir: Path

    @staticmethod
    def default() -> "Paths":
        """Return the default repo-relative output directories."""
        repo_root = Path(__file__).resolve().parents[2]
        return Paths(
            repo_root=repo_root,
            data_dir=repo_root / "data",
            artifacts_dir=repo_root / "artifacts",
            reports_dir=repo_root / "reports",
        )


def local_llm_model() -> str:
    """Return the LiteLLM selector for the configured local model."""
    raw_value = os.getenv("DERM_ADVISOR_MODEL", DEFAULT_LOCAL_LLM_MODEL).strip()
    model_name = raw_value or DEFAULT_LOCAL_LLM_MODEL
    if "/" in model_name:
        return model_name
    return f"ollama_chat/{model_name}"


def local_llm_model_name() -> str:
    """Return the configured local model name without rewriting it."""
    raw_value = os.getenv("DERM_ADVISOR_MODEL", DEFAULT_LOCAL_LLM_MODEL).strip()
    return raw_value or DEFAULT_LOCAL_LLM_MODEL


def local_llm_kwargs() -> dict[str, object]:
    """Return provider-specific kwargs for the configured local model."""
    raw_value = os.getenv("DERM_ADVISOR_DISABLE_THINKING", "true").strip().lower()
    if raw_value in _FALSE_VALUES:
        disable_thinking = False
    else:
        disable_thinking = raw_value in _TRUE_VALUES or raw_value == ""

    model_name = local_llm_model_name().lower()
    if disable_thinking and model_name.startswith("qwen3:"):
        return {"think": False}
    return {}


def ollama_api_base() -> str:
    """Return the configured Ollama HTTP endpoint."""
    raw_value = os.getenv("OLLAMA_API_BASE", DEFAULT_OLLAMA_API_BASE).strip()
    return raw_value or DEFAULT_OLLAMA_API_BASE
