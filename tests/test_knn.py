"""Parity tests for the from-scratch KNN classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from har.data import load_har_dataset
from har.knn import CustomKNN, majority_vote


@pytest.fixture(scope="module")
def tiny_split():
    X, y = load_har_dataset()
    return train_test_split(X, y, test_size=0.3, random_state=24, stratify=y)


def test_majority_vote_tie_breaks_to_nearest():
    # labels ordered nearest → farthest; 1 and 2 tied → prefer first (1)
    assert majority_vote([1, 2, 1, 2, 3], distances=[0.1, 0.2, 0.3, 0.4, 0.5]) == 1


def test_custom_knn_matches_sklearn(tiny_split):
    X_train, X_test, y_train, y_test = tiny_split
    n = 150
    custom = CustomKNN(5).fit(X_train, y_train)
    sk = KNeighborsClassifier(5).fit(X_train, y_train)
    assert np.all(custom.predict(X_test.iloc[:n]) == sk.predict(X_test.iloc[:n]))
