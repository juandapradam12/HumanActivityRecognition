"""Plotting helpers for exploratory analysis and model comparison."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import ACTIVITY_LABELS


def plot_feature_correlation(
    X: pd.DataFrame,
    n_features: int = 20,
    ax: Optional[plt.Axes] = None,
):
    """Lower-triangle correlation heatmap for the first *n_features* columns."""
    subset = X.iloc[:, :n_features]
    corr = subset.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax,
    )
    ax.set_title(f"Correlation — first {n_features} features")
    return ax


def plot_activity_counts(
    y,
    label_names=None,
    ax: Optional[plt.Axes] = None,
):
    """Bar chart of activity class frequencies."""
    label_names = label_names or ACTIVITY_LABELS
    counts = pd.Series(y).value_counts().sort_index()
    counts.index = [label_names.get(i, str(i)) for i in counts.index]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", color="#2c7fb8", ax=ax, rot=30)
    ax.set_ylabel("Windows")
    ax.set_title("Samples per activity")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_model_comparison(
    results: pd.DataFrame,
    metric: str = "accuracy",
    ax: Optional[plt.Axes] = None,
):
    """Horizontal bar chart ranking models by a metric column."""
    ordered = results.sort_values(metric, ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["model"], ordered[metric], color="#41b6c4")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title("Model comparison")
    ax.set_xlim(max(0.9, ordered[metric].min() - 0.02), 1.0)
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_confusion_matrix(
    cm: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Confusion matrix",
):
    """Annotated confusion-matrix heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax
