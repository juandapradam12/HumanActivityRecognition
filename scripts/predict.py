#!/usr/bin/env python3
"""Predict activity labels for a feature matrix using an exported model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from har.data import ACTIVITY_LABELS, load_feature_names
from har.export import load_metadata, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "models" / "har_cascaded.joblib",
        help="Path to a joblib model from scripts/export_model.py",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Whitespace-delimited feature matrix (561 cols) or a single row",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path for predictions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "Train one with: python scripts/export_model.py"
        )

    model = load_model(args.model)
    meta = load_metadata(args.model)
    feature_names = meta.get("feature_names") or load_feature_names()

    X = pd.read_csv(args.input, sep=r"\s+", header=None)
    if X.shape[1] != len(feature_names):
        raise SystemExit(
            f"Expected {len(feature_names)} features, got {X.shape[1]}"
        )
    X.columns = feature_names

    preds = model.predict(X)
    names = [ACTIVITY_LABELS.get(int(p), str(p)) for p in preds]
    frame = pd.DataFrame({"label": preds, "activity": names})
    print(frame.to_string(index=True))

    if args.out:
        frame.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
