#!/usr/bin/env python3
"""Train the default HAR model on the official train split and export it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import accuracy_score

from har.data import load_official_splits
from har.export import save_model
from har.models import CascadedHARClassifier, build_default_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "models" / "har_cascaded.joblib",
        help="Output joblib path",
    )
    parser.add_argument(
        "--model",
        choices=("cascaded", "hgb"),
        default="cascaded",
        help="Which estimator to export",
    )
    parser.add_argument("--seed", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, test = load_official_splits(with_subjects=False)

    if args.model == "cascaded":
        model = CascadedHARClassifier(random_state=args.seed)
        name = "Cascaded HGB + Sit/Stand specialist"
    else:
        model = build_default_classifier(random_state=args.seed)
        name = "Hist Gradient Boosting"

    model.fit(train.X, train.y)
    pred = model.predict(test.X)
    acc = float(accuracy_score(test.y, pred))
    path = save_model(
        model,
        args.out,
        meta={
            "model_name": name,
            "official_test_accuracy": acc,
            "n_train": int(len(train.y)),
            "n_test": int(len(test.y)),
            "feature_names": list(train.X.columns),
        },
    )
    print(f"Saved {name} → {path}")
    print(f"Official test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
