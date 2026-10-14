#!/usr/bin/env python3
"""Fast sanity checks for the HAR package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from har.data import ACTIVITY_LABELS, available_data_files, load_har_dataset, load_split
from har.knn import CustomKNN
from har.models import CascadedHARClassifier, build_model_zoo


def main() -> None:
    files = available_data_files()
    assert files["X_train.txt"] and files["y_train.txt"], files

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

    model = build_model_zoo(include_custom_knn=False)["Extra Trees"]
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc > 0.97, acc

    if files.get("X_test.txt") and files.get("subject_train.txt"):
        train = load_split("train", with_subjects=True)
        test = load_split("test", with_subjects=True)
        assert len(train.X) == 7352
        assert len(test.X) == 2947
        assert train.subjects is not None and test.subjects is not None
        cascaded = CascadedHARClassifier(random_state=24)
        # Fit on a subject-held subset for speed
        mask = train.subjects.isin([1, 3, 5, 7, 8])
        cascaded.fit(train.X.loc[mask], train.y.loc[mask])
        pred = cascaded.predict(test.X.iloc[:100])
        assert pred.shape == (100,)

    print("smoke_test OK")
    print(f"  custom/sklearn KNN agreement on {n}: perfect")
    print(f"  Extra Trees accuracy: {acc:.4f}")
    print(f"  data files: {sum(files.values())}/{len(files)} present")


if __name__ == "__main__":
    main()
