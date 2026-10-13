#!/usr/bin/env python3
"""Fast sanity checks for the HAR package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from har.data import ACTIVITY_LABELS, load_har_dataset
from har.knn import CustomKNN
from har.models import build_model_zoo


def main() -> None:
    X, y = load_har_dataset()
    assert X.shape == (7352, 561), X.shape
    assert not X.isnull().any().any()
    assert set(y.unique()) == set(ACTIVITY_LABELS)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=24, stratify=y
    )

    custom = CustomKNN(5).fit(X_train, y_train)
    sk = KNeighborsClassifier(5).fit(X_train, y_train)

    n = 200
    c_pred = custom.predict(X_test.iloc[:n])
    s_pred = sk.predict(X_test.iloc[:n])
    assert np.all(c_pred == s_pred), "Custom KNN must match sklearn"
    assert accuracy_score(y_test.iloc[:n], c_pred) > 0.9

    zoo = build_model_zoo()
    # Train a fast strong model
    model = zoo["Extra Trees"]
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc > 0.97, acc

    print("smoke_test OK")
    print(f"  custom/sklearn KNN agreement on {n}: perfect")
    print(f"  Extra Trees accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
