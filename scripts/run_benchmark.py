#!/usr/bin/env python3
"""Train and compare HAR classifiers under multiple evaluation protocols."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import train_test_split

from har.data import ACTIVITY_LABELS, load_har_dataset, load_official_splits
from har.evaluate import evaluate_on_official_test, loso_on_train_partition
from har.models import (
    CascadedHARClassifier,
    build_model_zoo,
    confusion,
    detailed_report,
    evaluate_models,
    evaluate_sitting_standing_specialist,
    top_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("holdout", "official", "loso", "all"),
        default="official",
        help="Evaluation protocol (default: official UCI test split)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Hold-out fraction when --protocol=holdout",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=24,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-custom-knn",
        action="store_true",
        help="Drop the from-scratch KNN from the zoo",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller zoo for faster smoke runs",
    )
    return parser.parse_args()


def select_models(args: argparse.Namespace):
    zoo = build_model_zoo(
        random_state=args.seed,
        include_custom_knn=not args.skip_custom_knn,
        include_calibrated=not args.quick,
    )
    if args.quick:
        keep = [
            "sklearn KNN (k=5)",
            "Logistic Regression",
            "Extra Trees",
            "Hist Gradient Boosting",
        ]
        zoo = {k: zoo[k] for k in keep if k in zoo}
    zoo["Cascaded HGB + Sit/Stand"] = CascadedHARClassifier(random_state=args.seed)
    return zoo


def run_holdout(models, args) -> None:
    X, y = load_har_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    print("\n=== Random stratified hold-out (optimistic) ===")
    results = evaluate_models(models, X_train, y_train, X_test, y_test)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    _print_best(results, y_test)


def run_official(models, args) -> None:
    print("\n=== Official UCI train → test ===")
    results = evaluate_on_official_test(models)
    cols = [
        "model",
        "accuracy",
        "f1_weighted",
        "f1_macro",
        "sitting_standing_error",
        "n_test",
    ]
    print(results[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    _print_best(results, results.attrs["y_test"])

    train, test = load_official_splits(with_subjects=True)
    print("\n=== Sitting vs standing specialist (official test) ===")
    specialist = evaluate_sitting_standing_specialist(
        train.X, train.y, test.X, test.y, random_state=args.seed
    )
    print(
        f"accuracy={specialist['accuracy']:.4f}  "
        f"f1={specialist['f1_weighted']:.4f}  n={specialist['n_test']}"
    )
    print(specialist["report"])


def run_loso(models, args) -> None:
    print("\n=== Leave-one-subject-out on training volunteers ===")
    # LOSO is expensive (21 folds). Skip calibrated cascade by default —
    # nested CV inside each fold is prohibitively slow.
    preferred = (
        "Logistic Regression",
        "Extra Trees",
        "Hist Gradient Boosting",
    )
    compact = {k: models[k] for k in preferred if k in models}
    if not compact:
        compact = {
            k: v
            for k, v in models.items()
            if "Cascaded" not in k and "Calibrated" not in k
        }
    table = loso_on_train_partition(compact)
    cols = [
        "model",
        "accuracy",
        "accuracy_mean",
        "accuracy_std",
        "f1_weighted",
        "sitting_standing_error",
        "n_subjects",
    ]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = table.iloc[0]["model"]
    detail = table.attrs["details"][best]
    print(f"\nBest LOSO model: {best}")
    print(detail.attrs["report"])


def _print_best(results, y_test) -> None:
    best_name = results.iloc[0]["model"]
    best_pred = results.attrs["predictions"][best_name]
    print(f"\nBest model: {best_name}")
    print(detailed_report(y_test, best_pred))
    print("Confusion matrix")
    print(confusion(y_test, best_pred))


def main() -> None:
    args = parse_args()
    models = select_models(args)
    print(f"Models: {', '.join(models)}")

    train_probe = load_har_dataset()
    print(
        f"Loaded train features: {train_probe[0].shape[0]} windows × "
        f"{train_probe[0].shape[1]} features"
    )
    print("Class balance (train):")
    for label, count in train_probe[1].value_counts().sort_index().items():
        print(f"  {label} {ACTIVITY_LABELS[label]:20s} {count}")

    protocols = (
        ["holdout", "official", "loso"]
        if args.protocol == "all"
        else [args.protocol]
    )
    for protocol in protocols:
        if protocol == "holdout":
            run_holdout(models, args)
        elif protocol == "official":
            run_official(models, args)
        elif protocol == "loso":
            run_loso(models, args)

    print("\nTop sensor features (Extra Trees importance on train)")
    X, y = load_har_dataset()
    print(top_features(X, y).to_string())


if __name__ == "__main__":
    main()
