"""From-scratch k-Nearest Neighbors with a fast NumPy backend."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def euclidean_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Pairwise squared-Euclidean distances between query and reference rows.

    Uses the identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b so a full
    distance matrix can be built with one matrix multiply.
    """
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    xx = np.sum(query * query, axis=1, keepdims=True)
    yy = np.sum(reference * reference, axis=1, keepdims=True).T
    return np.maximum(xx + yy - 2.0 * query @ reference.T, 0.0)


def majority_vote(labels: Iterable, distances: Iterable[float] | None = None):
    """
    Majority vote with nearest-neighbor tie-break.

    If several labels share the top vote count, prefer the one that appears
    first among the nearest neighbors (same convention as the original notebook).
    """
    labels = list(labels)
    if not labels:
        raise ValueError("Cannot vote on an empty label list.")

    counts = Counter(labels).most_common()
    if len(counts) == 1 or counts[0][1] > counts[1][1]:
        return counts[0][0]

    top = counts[0][1]
    tied = {label for label, count in counts if count == top}

    if distances is not None:
        order = np.argsort(np.asarray(distances, dtype=float))
        ordered = [labels[i] for i in order]
    else:
        ordered = labels

    for label in ordered:
        if label in tied:
            return label
    return ordered[0]


class CustomKNN(BaseEstimator, ClassifierMixin):
    """
    Educational k-NN classifier.

    Matches scikit-learn's Euclidean KNeighborsClassifier on the HAR dataset
    while remaining easy to read and inspect.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64, ensure_all_finite=True)
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors cannot exceed the training set size")

        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        self.n_features_in_ = self.X_train_.shape[1]
        return self

    def _predict_one(self, point: np.ndarray):
        diff = self.X_train_ - point
        dists = np.linalg.norm(diff, axis=1)
        nearest = np.argpartition(dists, self.n_neighbors - 1)[: self.n_neighbors]
        return majority_vote(self.y_train_[nearest], dists[nearest])

    def predict(self, X):
        check_is_fitted(self, ["X_train_", "y_train_"])
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)

        # Fast path: vectorized distance matrix + argpartition.
        d2 = euclidean_distances(X, self.X_train_)
        nn_idx = np.argpartition(d2, kth=self.n_neighbors - 1, axis=1)[
            :, : self.n_neighbors
        ]

        preds = []
        for i in range(X.shape[0]):
            neighbors = nn_idx[i]
            preds.append(
                majority_vote(self.y_train_[neighbors], d2[i, neighbors])
            )
        return np.asarray(preds)

    def predict_slow(self, X):
        """Row-by-row prediction — useful for teaching the algorithm step by step."""
        check_is_fitted(self, ["X_train_", "y_train_"])
        X = check_array(X, dtype=np.float64, ensure_all_finite=True)
        return np.asarray([self._predict_one(row) for row in X])


def custom_knn_predict(
    point,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    n_neighbors: int = 5,
):
    """Single-point helper kept for notebook / teaching compatibility."""
    model = CustomKNN(n_neighbors=n_neighbors).fit(X_train, y_train)
    point = np.asarray(point, dtype=np.float64).reshape(1, -1)
    return model.predict(point)[0]
