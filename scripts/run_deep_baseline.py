#!/usr/bin/env python3
"""Compare handcrafted-feature models vs a raw-IMU 1D CNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from har.cnn import evaluate_predictions, loso_cnn, predict_cnn, train_eval_official
from har.data import load_official_splits
from har.evaluate import leave_one_subject_out
from har.models import CascadedHARClassifier, confusion, detailed_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--loso-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip-loso",
        action="store_true",
        help="Only run official train→test comparison",
    )
    parser.add_argument(
        "--loso-max-subjects",
        type=int,
        default=None,
        help="Limit LOSO subjects (smoke / CI). Default: all 21 train subjects",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "deep_baseline",
        help="Directory for metrics JSON / CSV",
    )
    return parser.parse_args()


def feature_models(seed: int) -> dict:
    return {
        "Logistic Regression (features)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        ),
        "Extra Trees (features)": ExtraTreesClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "Hist Gradient Boosting (features)": HistGradientBoostingClassifier(
            max_iter=200, random_state=seed
        ),
        "Cascaded HGB + Sit/Stand (features)": CascadedHARClassifier(
            random_state=seed
        ),
    }


def evaluate_feature_official(seed: int) -> pd.DataFrame:
    train, test = load_official_splits(with_subjects=True)
    rows = []
    preds = {}
    for name, model in feature_models(seed).items():
        model.fit(train.X, train.y)
        pred = model.predict(test.X)
        m = evaluate_predictions(test.y, pred)
        m["model"] = name
        m["protocol"] = "official_test"
        m["input"] = "handcrafted_561"
        rows.append(m)
        preds[name] = pred
    frame = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    frame.attrs["predictions"] = preds
    frame.attrs["y_test"] = test.y
    return frame


def evaluate_feature_loso(seed: int, max_subjects: int | None) -> pd.DataFrame:
    train, _ = load_official_splits(with_subjects=True)
    subjects = sorted(train.subjects.unique())
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
        mask = train.subjects.isin(subjects)
        X, y, sub = train.X.loc[mask], train.y.loc[mask], train.subjects.loc[mask]
    else:
        X, y, sub = train.X, train.y, train.subjects

    rows = []
    for name, model in feature_models(seed).items():
        # Skip cascade under full LOSO — nested calibration is very slow.
        if "Cascaded" in name and max_subjects is None:
            continue
        per = leave_one_subject_out(
            model, X, y, sub, subject_ids=subjects
        )
        overall = dict(per.attrs["overall"])
        overall["model"] = name
        overall["input"] = "handcrafted_561"
        rows.append(overall)
        print(
            f"LOSO features {name}: acc={overall['accuracy']:.4f} "
            f"mean±std={overall['accuracy_mean']:.4f}±{overall['accuracy_std']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Official test: handcrafted features ===", flush=True)
    feat_official = evaluate_feature_official(args.seed)
    print(
        feat_official[
            ["model", "accuracy", "f1_weighted", "sitting_standing_error"]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        flush=True,
    )

    print("\n=== Official test: 1D CNN on raw IMU ===", flush=True)
    cnn_result, cnn_metrics, cnn_pred = train_eval_official(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        verbose=True,
    )
    cnn_metrics["input"] = "raw_imu_9x128"
    print(
        f"{cnn_metrics['model']}: acc={cnn_metrics['accuracy']:.4f}  "
        f"f1={cnn_metrics['f1_weighted']:.4f}  "
        f"sit↔stand_err={cnn_metrics['sitting_standing_error']:.4f}",
        flush=True,
    )
    print(detailed_report(feat_official.attrs["y_test"], cnn_pred))
    print(confusion(feat_official.attrs["y_test"], cnn_pred))

    official = pd.concat(
        [feat_official.assign(input="handcrafted_561"), pd.DataFrame([cnn_metrics])],
        ignore_index=True,
    ).sort_values("accuracy", ascending=False)
    print("\n=== Side-by-side (official UCI test) ===", flush=True)
    print(
        official[
            ["model", "input", "accuracy", "f1_weighted", "sitting_standing_error"]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        flush=True,
    )
    official.to_csv(args.out_dir / "official_comparison.csv", index=False)

    # Persist CNN weights
    import torch

    weights_path = args.out_dir / "har_cnn1d.pt"
    torch.save(
        {
            "state_dict": cnn_result.model.state_dict(),
            "metrics": cnn_metrics,
            "history": cnn_result.history,
        },
        weights_path,
    )
    print(f"Saved CNN weights → {weights_path}", flush=True)

    if not args.skip_loso:
        print("\n=== LOSO: handcrafted features ===", flush=True)
        feat_loso = evaluate_feature_loso(args.seed, args.loso_max_subjects)

        print("\n=== LOSO: 1D CNN on raw IMU ===", flush=True)
        _, cnn_loso = loso_cnn(
            epochs=args.loso_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            max_subjects=args.loso_max_subjects,
            verbose=True,
        )
        cnn_loso["input"] = "raw_imu_9x128"
        loso = pd.concat(
            [feat_loso, pd.DataFrame([cnn_loso])], ignore_index=True
        ).sort_values("accuracy", ascending=False)
        print("\n=== Side-by-side (LOSO) ===", flush=True)
        cols = [
            c
            for c in (
                "model",
                "input",
                "accuracy",
                "accuracy_mean",
                "accuracy_std",
                "sitting_standing_error",
                "n_subjects",
            )
            if c in loso.columns
        ]
        print(loso[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        loso.to_csv(args.out_dir / "loso_comparison.csv", index=False)
        (args.out_dir / "loso_cnn_overall.json").write_text(
            json.dumps(cnn_loso, indent=2)
        )

    (args.out_dir / "official_cnn_metrics.json").write_text(
        json.dumps(cnn_metrics, indent=2)
    )
    print(f"\nWrote comparison tables under {args.out_dir}")


if __name__ == "__main__":
    main()
