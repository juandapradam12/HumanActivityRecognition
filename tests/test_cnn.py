"""Tests for the raw-IMU 1D CNN baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")

from har.cnn import HARCNN1D, evaluate_predictions, predict_cnn, train_cnn
from har.raw_data import N_CHANNELS, N_TIMESTEPS, labels_to_index


def test_cnn_forward_shape():
    model = HARCNN1D()
    x = torch.randn(4, N_CHANNELS, N_TIMESTEPS)
    logits = model(x)
    assert logits.shape == (4, 6)


def test_train_cnn_overfits_tiny_synthetic():
    rng = np.random.default_rng(0)
    # Two easy classes with different mean channels
    X0 = rng.normal(0.0, 0.1, size=(40, N_CHANNELS, N_TIMESTEPS)).astype(np.float32)
    X1 = rng.normal(1.0, 0.1, size=(40, N_CHANNELS, N_TIMESTEPS)).astype(np.float32)
    X = np.concatenate([X0, X1], axis=0)
    y = np.array([1] * 40 + [2] * 40)  # UCI-style labels
    result = train_cnn(X, y, epochs=8, batch_size=16, verbose=False)
    pred = predict_cnn(result.model, X)
    assert evaluate_predictions(y, pred)["accuracy"] >= 0.9


def test_labels_to_index():
    assert labels_to_index([1, 6, 3]).tolist() == [0, 5, 2]


@pytest.mark.slow
def test_raw_split_loads_when_present():
    signals = ROOT / "data" / "inertial_signals" / "body_acc_x_train.txt"
    if not signals.exists():
        pytest.skip("raw signals not downloaded")
    from har.raw_data import load_raw_split

    train = load_raw_split("train", with_subjects=True)
    assert train.X.shape == (7352, 9, 128)
    assert train.subjects is not None
