"""Data loading helpers for the UCI Human Activity Recognition dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

SITTING_STANDING_LABELS = {4: "SITTING", 5: "STANDING"}


@dataclass(frozen=True)
class HarSplit:
    """A labeled HAR partition with optional subject identifiers."""

    X: pd.DataFrame
    y: pd.Series
    subjects: pd.Series | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "y", self.y.rename("label"))
        if self.subjects is not None:
            object.__setattr__(self, "subjects", self.subjects.rename("subject"))


def _default_data_dir() -> Path:
    """Resolve a directory that holds the UCI text files."""
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        candidates.extend(
            [
                parent,
                parent / "data" / "uci_har",
            ]
        )
    candidates.append(Path.cwd())
    candidates.append(Path.cwd() / "data" / "uci_har")

    for path in candidates:
        if (path / "features.txt").exists() and (
            (path / "X_train.txt").exists()
            or (path / "X_test.txt").exists()
        ):
            return path
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


def _read_feature_matrix(path: Path, feature_names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=feature_names)


def _read_labels(path: Path) -> pd.Series:
    y = pd.read_csv(path, header=None, names=["label"])["label"]
    y.name = "label"
    return y


def _read_subjects(path: Path) -> pd.Series:
    subjects = pd.read_csv(path, header=None, names=["subject"])["subject"]
    subjects.name = "subject"
    return subjects


def _resolve_file(root: Path, filename: str) -> Path:
    """Find a data file in the root or the nested data/uci_har folder."""
    direct = root / filename
    if direct.exists():
        return direct
    nested = root / "data" / "uci_har" / filename
    if nested.exists():
        return nested
    # When root is already data/uci_har, also allow parent repo root.
    parent_hit = root.parent.parent / filename if root.name == "uci_har" else None
    if parent_hit is not None and parent_hit.exists():
        return parent_hit
    raise FileNotFoundError(
        f"Missing {filename}. Run: python scripts/download_data.py"
    )


def load_har_dataset(
    data_dir: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the training features and labels.

    Returns
    -------
    X : pd.DataFrame
        Shape (n_samples, 561) with named sensor features.
    y : pd.Series
        Integer activity labels in {1..6}.
    """
    split = load_split("train", data_dir=data_dir, with_subjects=False)
    return split.X, split.y


def load_split(
    split: str = "train",
    *,
    data_dir: str | Path | None = None,
    with_subjects: bool = True,
) -> HarSplit:
    """
    Load the official UCI ``train`` or ``test`` partition.

    Parameters
    ----------
    split:
        ``"train"`` or ``"test"``.
    with_subjects:
        When True, also load ``subject_*.txt`` (required for LOSO).
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    root = Path(data_dir) if data_dir else _default_data_dir()
    # Prefer repo root when features live there but subjects were downloaded
    # into data/uci_har — resolve each file independently via helpers that
    # search both locations.
    search_root = root
    if not (search_root / "features.txt").exists() and (
        Path.cwd() / "features.txt"
    ).exists():
        search_root = Path.cwd()

    feature_names = load_feature_names(_resolve_file(search_root, "features.txt"))
    X = _read_feature_matrix(
        _resolve_file(search_root, f"X_{split}.txt"), feature_names
    )
    y = _read_labels(_resolve_file(search_root, f"y_{split}.txt"))

    subjects = None
    if with_subjects:
        subject_path = None
        for candidate in (
            search_root / f"subject_{split}.txt",
            search_root / "data" / "uci_har" / f"subject_{split}.txt",
            Path.cwd() / f"subject_{split}.txt",
            Path.cwd() / "data" / "uci_har" / f"subject_{split}.txt",
        ):
            if candidate.exists():
                subject_path = candidate
                break
        if subject_path is None:
            raise FileNotFoundError(
                f"Missing subject_{split}.txt. Run: python scripts/download_data.py"
            )
        subjects = _read_subjects(subject_path)
        if len(subjects) != len(X):
            raise ValueError(
                f"Subject count ({len(subjects)}) != feature rows ({len(X)})"
            )

    if len(y) != len(X):
        raise ValueError(f"Label count ({len(y)}) != feature rows ({len(X)})")
    return HarSplit(X=X, y=y, subjects=subjects)


def load_official_splits(
    data_dir: str | Path | None = None,
    *,
    with_subjects: bool = True,
) -> Tuple[HarSplit, HarSplit]:
    """Load official UCI train and test partitions."""
    return (
        load_split("train", data_dir=data_dir, with_subjects=with_subjects),
        load_split("test", data_dir=data_dir, with_subjects=with_subjects),
    )


def sitting_standing_mask(y: pd.Series | Iterable[int]) -> pd.Series:
    """Boolean mask selecting only sitting (4) and standing (5) windows."""
    y = pd.Series(y)
    return y.isin(SITTING_STANDING_LABELS)


def label_names(
    y: pd.Series | list[int], mapping: dict[int, str] | None = None
) -> pd.Series:
    """Map integer labels to human-readable activity names."""
    mapping = mapping or ACTIVITY_LABELS
    return pd.Series(y).map(mapping)


def available_data_files(data_dir: str | Path | None = None) -> dict[str, bool]:
    """Report which expected UCI files are present."""
    root = Path(data_dir) if data_dir else _default_data_dir()
    names = [
        "features.txt",
        "activity_labels.txt",
        "X_train.txt",
        "y_train.txt",
        "X_test.txt",
        "y_test.txt",
        "subject_train.txt",
        "subject_test.txt",
    ]
    result = {}
    for name in names:
        try:
            _resolve_file(root, name)
            result[name] = True
        except FileNotFoundError:
            result[name] = False
    return result
