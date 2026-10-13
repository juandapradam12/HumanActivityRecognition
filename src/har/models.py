"""Model zoo and evaluation helpers for Human Activity Recognition."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .data import ACTIVITY_LABELS
from .knn import CustomKNN


def build_model_zoo(random_state: int = 24) -> Dict[str, Any]:
    """
    Return a curated set of classifiers for smartphone HAR.

    Includes the pedagogical custom k-NN plus strong classical baselines.
    """
    return {
        "Custom KNN (k=5)": CustomKNN(n_neighbors=5),
        "sklearn KNN (k=5)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "sklearn KNN (k=7, distance)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(n_neighbors=7, weights="distance"),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(max_iter=5000, dual=False)),
            ]
        ),
        "RBF SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=10, gamma="scale")),
            ]
        ),
        "PCA(100) + RBF SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=100, random_state=random_state)),
                ("clf", SVC(kernel="rbf", C=10, gamma="scale")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200,
            random_state=random_state,
        ),
    }


def evaluate_models(
    models: Dict[str, Any],
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    label_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Fit each model and return a ranked metrics table."""
    label_names = label_names or ACTIVITY_LABELS
    rows: List[dict] = []
    fitted: Dict[str, Any] = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = model
        predictions[name] = pred
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "f1_weighted": f1_score(y_test, pred, average="weighted"),
                "f1_macro": f1_score(y_test, pred, average="macro"),
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )
    results.attrs["fitted_models"] = fitted
    results.attrs["predictions"] = predictions
    results.attrs["label_names"] = label_names
    return results


def detailed_report(
    y_true,
    y_pred,
    label_names: Optional[Dict[int, str]] = None,
) -> str:
    """Classification report with readable activity names."""
    label_names = label_names or ACTIVITY_LABELS
    labels = sorted(label_names)
    target_names = [label_names[i] for i in labels]
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
    )


def confusion(
    y_true,
    y_pred,
    label_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Confusion matrix as a labeled DataFrame."""
    label_names = label_names or ACTIVITY_LABELS
    labels = sorted(label_names)
    names = [label_names[i] for i in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=names, columns=names)


def top_features(
    X: pd.DataFrame,
    y,
    n: int = 15,
    random_state: int = 24,
) -> pd.Series:
    """Rank features with an Extra Trees importance score."""
    model = ExtraTreesClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(n)
    )
