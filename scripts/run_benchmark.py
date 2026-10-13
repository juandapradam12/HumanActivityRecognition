#!/usr/bin/env python3
"""Train and compare HAR classifiers; print a ranked leaderboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import train_test_split

from har.data import ACTIVITY_LABELS, load_har_dataset
from har.models import (
    build_model_zoo,
    confusion,
    detailed_report,
    evaluate_models,
    top_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Hold-out fraction for evaluation (default: 0.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=24,
        help="Random seed for the train/test split",
    )
    parser.add_argument(
        "--skip-custom-knn",
        action="store_true",
        help="Drop the from-scratch KNN from the zoo (faster smoke runs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y = load_har_dataset()
    print(f"Loaded {X.shape[0]} windows × {X.shape[1]} features")
    print("Class balance:")
    for label, count in y.value_counts().sort_index().items():
        print(f"  {label} {ACTIVITY_LABELS[label]:20s} {count}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    models = build_model_zoo(random_state=args.seed)
    if args.skip_custom_knn:
        models.pop("Custom KNN (k=5)", None)

    print("\nTraining models…")
    results = evaluate_models(models, X_train, y_train, X_test, y_test)
    print("\nLeaderboard")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_name = results.iloc[0]["model"]
    best_pred = results.attrs["predictions"][best_name]
    print(f"\nBest model: {best_name}")
    print(detailed_report(y_test, best_pred))
    print("Confusion matrix")
    print(confusion(y_test, best_pred))

    print("\nTop sensor features (Extra Trees importance)")
    print(top_features(X_train, y_train).to_string())


if __name__ == "__main__":
    main()
