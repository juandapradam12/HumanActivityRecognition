"""Model zoo and evaluation helpers for Human Activity Recognition."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .data import ACTIVITY_LABELS, SITTING_STANDING_LABELS, sitting_standing_mask
from .knn import CustomKNN


def build_model_zoo(
    random_state: int = 24,
    *,
    include_custom_knn: bool = True,
    include_calibrated: bool = True,
) -> Dict[str, Any]:
    """
    Return a curated set of classifiers for smartphone HAR.

    Includes the pedagogical custom k-NN plus strong classical baselines.
    """
    zoo: Dict[str, Any] = {}
    if include_custom_knn:
        zoo["Custom KNN (k=5)"] = CustomKNN(n_neighbors=5)

    zoo.update(
        {
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
    )

    if include_calibrated:
        # Probability-calibrated logistic regression — useful for demos / thresholds.
        zoo["Calibrated Logistic Regression"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LogisticRegression(max_iter=2000),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        )
    return zoo


def build_default_classifier(random_state: int = 24):
    """Best default multiclass model for export / demo."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        random_state=random_state,
    )


def build_sitting_standing_specialist(random_state: int = 24) -> Pipeline:
    """
    Binary specialist for the hard sitting vs standing pair.

    Uses a calibrated RBF SVM on scaled features — strong on this subspace and
    returns usable probabilities for cascading with a multiclass model.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    SVC(kernel="rbf", C=10, gamma="scale"),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )


class CascadedHARClassifier(BaseEstimator, ClassifierMixin):
    """
    Multiclass model with a sitting/standing specialist override.

    1. Predict with the base multiclass estimator.
    2. When the base predicts sitting or standing, re-decide with the binary
       specialist trained only on those two classes.
    """

    def __init__(
        self,
        base_estimator=None,
        specialist=None,
        *,
        random_state: int = 24,
    ):
        self.base_estimator = base_estimator
        self.specialist = specialist
        self.random_state = random_state

    def _base(self):
        return self.base_estimator or build_default_classifier(
            random_state=self.random_state
        )

    def _specialist(self):
        return self.specialist or build_sitting_standing_specialist(
            random_state=self.random_state
        )

    def fit(self, X, y):
        X_arr, y_arr = check_X_y(X, y, dtype=np.float64, ensure_all_finite=True)
        y_series = pd.Series(y_arr)
        self.base_estimator_ = clone(self._base())
        self.specialist_ = clone(self._specialist())
        self.base_estimator_.fit(X_arr, y_arr)

        mask = sitting_standing_mask(y_series).to_numpy()
        if mask.sum() < 10:
            raise ValueError("Not enough sitting/standing samples to fit specialist")
        self.specialist_.fit(X_arr[mask], y_arr[mask])
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["base_estimator_", "specialist_"])
        X_arr = check_array(X, dtype=np.float64, ensure_all_finite=True)
        base_pred = np.asarray(self.base_estimator_.predict(X_arr))
        pred = base_pred.copy()
        mask = np.isin(base_pred, list(SITTING_STANDING_LABELS))
        if mask.any():
            pred[mask] = self.specialist_.predict(X_arr[mask])
        return pred

    def predict_proba_specialist(self, X):
        """Sitting/standing probabilities from the specialist (2 columns)."""
        check_is_fitted(self, ["specialist_"])
        X_arr = check_array(X, dtype=np.float64, ensure_all_finite=True)
        return self.specialist_.predict_proba(X_arr)


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
                "sitting_standing_error": sitting_standing_error_rate(
                    y_test, pred
                ),
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


def sitting_standing_error_rate(y_true, y_pred) -> float:
    """
    Fraction of true sitting/standing windows confused with the other posture.

    Returns 0.0 when the test slice has no sitting/standing samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isin(y_true, [4, 5])
    if not mask.any():
        return 0.0
    true = y_true[mask]
    pred = y_pred[mask]
    confused = ((true == 4) & (pred == 5)) | ((true == 5) & (pred == 4))
    return float(confused.mean())


def evaluate_sitting_standing_specialist(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    random_state: int = 24,
) -> dict:
    """Train/evaluate the binary specialist on sitting vs standing only."""
    train_mask = sitting_standing_mask(y_train)
    test_mask = sitting_standing_mask(y_test)
    clf = build_sitting_standing_specialist(random_state=random_state)
    clf.fit(
        X_train.loc[train_mask] if hasattr(X_train, "loc") else X_train[train_mask],
        pd.Series(y_train)[train_mask],
    )
    X_te = X_test.loc[test_mask] if hasattr(X_test, "loc") else X_test[test_mask]
    y_te = pd.Series(y_test)[test_mask]
    pred = clf.predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1_weighted": float(f1_score(y_te, pred, average="weighted")),
        "n_test": int(len(y_te)),
        "report": classification_report(
            y_te,
            pred,
            target_names=["SITTING", "STANDING"],
            digits=4,
        ),
        "model": clf,
        "predictions": pred,
        "y_true": y_te,
    }


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
