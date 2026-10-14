"""Dataset loading tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from har.data import (
    ACTIVITY_LABELS,
    available_data_files,
    load_har_dataset,
    load_official_splits,
    load_split,
    sitting_standing_mask,
)


def test_load_train_shape():
    X, y = load_har_dataset()
    assert X.shape == (7352, 561)
    assert len(y) == 7352
    assert set(y.unique()) == set(ACTIVITY_LABELS)


def test_official_test_available_or_skip():
    files = available_data_files()
    if not files.get("X_test.txt"):
        pytest.skip("X_test.txt not present — run scripts/download_data.py")
    train, test = load_official_splits(with_subjects=True)
    assert train.X.shape == (7352, 561)
    assert test.X.shape == (2947, 561)
    assert train.subjects is not None
    assert test.subjects.nunique() == 9
    assert train.subjects.nunique() == 21


def test_sitting_standing_mask():
    import pandas as pd

    y = pd.Series([1, 4, 5, 6, 4])
    mask = sitting_standing_mask(y)
    assert mask.tolist() == [False, True, True, False, True]


def test_load_split_requires_known_name():
    with pytest.raises(ValueError):
        load_split("validation")
