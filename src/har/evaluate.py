"""Evaluation protocols: official UCI test split and leave-one-subject-out."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from .data import ACTIVITY_LABELS, HarSplit, load_official_splits
from .models import confusion, detailed_report, sitting_standing_error_rate


def metric_row(
    y_true,
    y_pred,
    *,
    model: str,
    protocol: str,
    extra: Optional[dict] = None,
) -> dict:
    """Standard metrics dictionary for leaderboards."""
    row = {
        "model": model,
        "protocol": protocol,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "sitting_standing_error": float(
            sitting_standing_error_rate(y_true, y_pred)
        ),
        "n_test": int(len(y_true)),
    }
    if extra:
        row.update(extra)
    return row


def evaluate_on_official_test(
    models: Dict[str, Any],
    *,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """
    Train on the official UCI train partition; score on the official test partition.

    This is the protocol used in the original HAR papers and avoids random
    window leakage within the training volunteers.
    """
    train, test = load_official_splits(data_dir=data_dir, with_subjects=True)
    rows: List[dict] = []
    fitted: Dict[str, Any] = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        estimator = clone(model)
        estimator.fit(train.X, train.y)
        pred = estimator.predict(test.X)
        fitted[name] = estimator
        predictions[name] = np.asarray(pred)
        rows.append(
            metric_row(
                test.y,
                pred,
                model=name,
                protocol="official_test",
                extra={
                    "n_train": len(train.y),
                    "n_subjects_train": int(train.subjects.nunique()),
                    "n_subjects_test": int(test.subjects.nunique()),
                },
            )
        )

    results = (
        pd.DataFrame(rows)
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )
    results.attrs["fitted_models"] = fitted
    results.attrs["predictions"] = predictions
    results.attrs["y_test"] = test.y
    results.attrs["train"] = train
    results.attrs["test"] = test
    return results


def leave_one_subject_out(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    subjects: pd.Series,
    *,
    subject_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Leave-one-subject-out cross-validation.

    For each held-out volunteer, train on everyone else and predict their windows.
    """
    if subjects is None:
        raise ValueError("subjects are required for LOSO evaluation")

    subjects = pd.Series(subjects).reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    unique_subjects = (
        sorted(subject_ids)
        if subject_ids is not None
        else sorted(subjects.unique())
    )

    rows: List[dict] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for subject in unique_subjects:
        test_mask = subjects == subject
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue

        estimator = clone(model)
        estimator.fit(X.loc[train_mask], y.loc[train_mask])
        pred = estimator.predict(X.loc[test_mask])
        y_true = y.loc[test_mask]

        all_true.append(np.asarray(y_true))
        all_pred.append(np.asarray(pred))
        rows.append(
            metric_row(
                y_true,
                pred,
                model=getattr(model, "__class__", type(model)).__name__,
                protocol="loso",
                extra={"subject": int(subject)},
            )
        )

    per_subject = pd.DataFrame(rows)
    y_all = np.concatenate(all_true)
    p_all = np.concatenate(all_pred)
    summary = metric_row(
        y_all,
        p_all,
        model=per_subject["model"].iloc[0] if len(per_subject) else "model",
        protocol="loso_overall",
        extra={
            "n_subjects": int(per_subject["subject"].nunique())
            if "subject" in per_subject
            else 0,
            "accuracy_mean": float(per_subject["accuracy"].mean())
            if len(per_subject)
            else float("nan"),
            "accuracy_std": float(per_subject["accuracy"].std(ddof=0))
            if len(per_subject)
            else float("nan"),
        },
    )
    per_subject.attrs["overall"] = summary
    per_subject.attrs["y_true"] = y_all
    per_subject.attrs["y_pred"] = p_all
    per_subject.attrs["report"] = detailed_report(y_all, p_all)
    per_subject.attrs["confusion"] = confusion(y_all, p_all)
    return per_subject


def loso_on_train_partition(
    models: Dict[str, Any],
    *,
    data_dir: str | None = None,
    max_models: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run LOSO for each model on the official training volunteers only.

    Returns a summary table (one row per model) with mean±std subject accuracy.
    """
    train = load_official_splits(data_dir=data_dir, with_subjects=True)[0]
    items = list(models.items())
    if max_models is not None:
        items = items[:max_models]

    summaries: List[dict] = []
    details: Dict[str, pd.DataFrame] = {}

    for name, model in items:
        per_subject = leave_one_subject_out(
            model, train.X, train.y, train.subjects
        )
        overall = dict(per_subject.attrs["overall"])
        overall["model"] = name
        summaries.append(overall)
        details[name] = per_subject

    table = (
        pd.DataFrame(summaries)
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )
    table.attrs["details"] = details
    return table


def combine_train_test(train: HarSplit, test: HarSplit) -> HarSplit:
    """Stack official partitions — useful for full-subject LOSO over all 30 people."""
    X = pd.concat([train.X, test.X], axis=0, ignore_index=True)
    y = pd.concat([train.y, test.y], axis=0, ignore_index=True)
    subjects = None
    if train.subjects is not None and test.subjects is not None:
        subjects = pd.concat(
            [train.subjects, test.subjects], axis=0, ignore_index=True
        )
    return HarSplit(X=X, y=y, subjects=subjects)
