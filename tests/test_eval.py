"""Evaluation helper tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from har.data import available_data_files, load_split
from har.evaluate import leave_one_subject_out
from har.models import (
    CascadedHARClassifier,
    evaluate_sitting_standing_specialist,
    sitting_standing_error_rate,
)


def test_sitting_standing_error_rate():
    y_true = [4, 4, 5, 5, 1]
    y_pred = [4, 5, 5, 4, 1]
    # 2 confused out of 4 posture windows
    assert sitting_standing_error_rate(y_true, y_pred) == 0.5


@pytest.mark.slow
def test_loso_two_subjects():
    files = available_data_files()
    if not files.get("subject_train.txt"):
        pytest.skip("subject files missing")
    train = load_split("train", with_subjects=True)
    # Tiny model + two subjects for speed
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    subjects = [1, 3]
    mask = train.subjects.isin(subjects)
    result = leave_one_subject_out(
        model,
        train.X.loc[mask],
        train.y.loc[mask],
        train.subjects.loc[mask],
        subject_ids=subjects,
    )
    assert len(result) == 2
    assert "overall" in result.attrs
    assert 0.0 <= result.attrs["overall"]["accuracy"] <= 1.0


def test_cascaded_and_specialist_on_official_slice():
    files = available_data_files()
    if not (files.get("X_test.txt") and files.get("subject_train.txt")):
        pytest.skip("official split missing")
    train = load_split("train", with_subjects=True)
    test = load_split("test", with_subjects=False)
    # Fit on a few subjects for speed
    mask = train.subjects.isin([1, 3, 5, 6, 7, 8])
    clf = CascadedHARClassifier(random_state=24)
    clf.fit(train.X.loc[mask], train.y.loc[mask])
    pred = clf.predict(test.X.iloc[:50])
    assert pred.shape == (50,)

    specialist = evaluate_sitting_standing_specialist(
        train.X.loc[mask],
        train.y.loc[mask],
        test.X,
        test.y,
        random_state=24,
    )
    assert specialist["n_test"] > 0
    assert specialist["accuracy"] > 0.8
