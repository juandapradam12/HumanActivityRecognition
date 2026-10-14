"""Persist and load trained HAR models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .data import ACTIVITY_LABELS


DEFAULT_MODEL_DIR = Path("artifacts") / "models"


def save_model(
    model: Any,
    path: str | Path,
    *,
    meta: dict | None = None,
) -> Path:
    """Serialize a fitted estimator and optional JSON sidecar metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    payload = {
        "activity_labels": ACTIVITY_LABELS,
        **(meta or {}),
    }
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(payload, indent=2))
    return path


def load_model(path: str | Path):
    """Load a joblib-serialized estimator."""
    return joblib.load(Path(path))


def load_metadata(path: str | Path) -> dict:
    """Load the JSON sidecar written by :func:`save_model`."""
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())
