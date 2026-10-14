"""Load raw UCI HAR inertial signal windows (9 × 128)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import HarSplit, _default_data_dir, _read_labels, _read_subjects

SIGNAL_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)

N_TIMESTEPS = 128
N_CHANNELS = len(SIGNAL_CHANNELS)


@dataclass(frozen=True)
class RawHarSplit:
    """Raw IMU windows aligned with labels / subjects."""

    X: np.ndarray  # (n, n_channels, n_timesteps)
    y: pd.Series
    subjects: pd.Series | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.X.shape)  # type: ignore[return-value]


def _signals_dir(data_dir: str | Path | None = None) -> Path:
    root = Path(data_dir) if data_dir else _default_data_dir()
    candidates = [
        root / "data" / "inertial_signals",
        root / "inertial_signals",
        Path.cwd() / "data" / "inertial_signals",
    ]
    # When root is repo root from _default_data_dir
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "data" / "inertial_signals")
    for path in candidates:
        probe = path / "body_acc_x_train.txt"
        if probe.exists():
            return path
    raise FileNotFoundError(
        "Raw inertial signals not found. Run:\n"
        "  python scripts/download_data.py --with-raw"
    )


def load_raw_split(
    split: str = "train",
    *,
    data_dir: str | Path | None = None,
    with_subjects: bool = True,
) -> RawHarSplit:
    """
    Load raw 9-channel IMU windows for ``train`` or ``test``.

    Returns
    -------
    RawHarSplit
        ``X`` has shape ``(n_windows, 9, 128)``.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    sig_dir = _signals_dir(data_dir)
    channels = []
    for name in SIGNAL_CHANNELS:
        path = sig_dir / f"{name}_{split}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run scripts/download_data.py --with-raw")
        channels.append(np.loadtxt(path, dtype=np.float32))

    # Stack to (n, channels, timesteps)
    X = np.stack(channels, axis=1)
    if X.ndim != 3 or X.shape[1:] != (N_CHANNELS, N_TIMESTEPS):
        raise ValueError(f"Unexpected raw shape {X.shape}")

    root = Path(data_dir) if data_dir else _default_data_dir()
    # Labels live next to engineered features at repo root
    from .data import _resolve_file

    y = _read_labels(_resolve_file(root, f"y_{split}.txt"))
    if len(y) != len(X):
        raise ValueError(f"Label/raw length mismatch: {len(y)} vs {len(X)}")

    subjects = None
    if with_subjects:
        subjects = _read_subjects(_resolve_file(root, f"subject_{split}.txt"))
        if len(subjects) != len(X):
            raise ValueError("Subject/raw length mismatch")

    return RawHarSplit(X=X, y=y, subjects=subjects)


def load_raw_official_splits(
    data_dir: str | Path | None = None,
    *,
    with_subjects: bool = True,
) -> tuple[RawHarSplit, RawHarSplit]:
    """Load raw train and test partitions."""
    return (
        load_raw_split("train", data_dir=data_dir, with_subjects=with_subjects),
        load_raw_split("test", data_dir=data_dir, with_subjects=with_subjects),
    )


def labels_to_index(y: pd.Series | np.ndarray) -> np.ndarray:
    """Map UCI labels {1..6} → {0..5} for CrossEntropyLoss."""
    arr = np.asarray(y, dtype=np.int64)
    return arr - 1


def index_to_labels(idx: np.ndarray) -> np.ndarray:
    """Map {0..5} → UCI labels {1..6}."""
    return np.asarray(idx, dtype=np.int64) + 1
