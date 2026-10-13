"""Data loading helpers for the UCI Human Activity Recognition dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def _default_data_dir() -> Path:
    """Resolve the repository root that holds the UCI text files."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "X_train.txt").exists() and (parent / "features.txt").exists():
            return parent
    return Path.cwd()


def _unique_feature_names(raw_names: list[str]) -> list[str]:
    """Make feature names unique (UCI HAR duplicates a few column labels)."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in raw_names:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    return unique


def load_feature_names(path: str | Path | None = None) -> list[str]:
    """Load and uniquify the 561 sensor feature names."""
    path = Path(path) if path else _default_data_dir() / "features.txt"
    feats = pd.read_csv(path, sep=r"\s+", header=None, names=["idx", "name"])
    return _unique_feature_names(feats["name"].tolist())


def load_activity_labels(path: str | Path | None = None) -> dict[int, str]:
    """Load activity id → name mapping, falling back to the built-in map."""
    path = Path(path) if path else _default_data_dir() / "activity_labels.txt"
    if not path.exists():
        return dict(ACTIVITY_LABELS)
    table = pd.read_csv(path, sep=r"\s+", header=None, names=["id", "name"])
    return dict(zip(table["id"].astype(int), table["name"].astype(str)))


def load_har_dataset(
    data_dir: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the training features and labels shipped with this repository.

    Returns
    -------
    X : pd.DataFrame
        Shape (n_samples, 561) with named sensor features.
    y : pd.Series
        Integer activity labels in {1..6}.
    """
    root = Path(data_dir) if data_dir else _default_data_dir()
    feature_names = load_feature_names(root / "features.txt")

    X = pd.read_csv(
        root / "X_train.txt",
        sep=r"\s+",
        header=None,
        names=feature_names,
    )
    y = pd.read_csv(root / "y_train.txt", header=None, names=["label"])["label"]
    y.name = "label"
    return X, y


def label_names(y: pd.Series | list[int], mapping: dict[int, str] | None = None) -> pd.Series:
    """Map integer labels to human-readable activity names."""
    mapping = mapping or ACTIVITY_LABELS
    return pd.Series(y).map(mapping)
