"""1D CNN baseline on raw UCI HAR inertial windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "PyTorch is required for the raw-signal CNN baseline.\n"
        "Install with: pip install 'har-smartphone[deep]'   # or: pip install torch"
    ) from exc

from sklearn.metrics import accuracy_score, f1_score

from .models import sitting_standing_error_rate
from .raw_data import (
    N_CHANNELS,
    N_TIMESTEPS,
    index_to_labels,
    labels_to_index,
)


class HARCNN1D(nn.Module):
    """
    Compact 1D CNN for 9 × 128 IMU windows.

    Architecture (intentionally small / CPU-friendly):
    Conv(9→64) → Conv(64→128) → Conv(128→128) → AdaptiveAvgPool → Linear(6)
    """

    def __init__(self, n_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class CNNTrainResult:
    model: HARCNN1D
    history: list[dict]
    device: str


def _device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    batch_size: int,
):
    def make(X, y, shuffle):
        xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        return DataLoader(
            TensorDataset(xt, yt),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_loader = make(X_train, y_train, True)
    val_loader = make(X_val, y_val, False) if X_val is not None else None
    return train_loader, val_loader


def train_cnn(
    X_train: np.ndarray,
    y_train,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val=None,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 24,
    device: str = "auto",
    verbose: bool = True,
) -> CNNTrainResult:
    """
    Train :class:`HARCNN1D` on raw windows.

    ``y_*`` may be UCI labels {1..6}; they are mapped to {0..5} internally.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device(device)

    y_tr = labels_to_index(y_train)
    y_va = labels_to_index(y_val) if y_val is not None else None

    model = HARCNN1D().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = _loaders(X_train, y_tr, X_val, y_va, batch_size)

    history: list[dict] = []
    best_state = None
    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
            n += len(xb)
        row = {"epoch": epoch, "train_loss": total_loss / max(n, 1)}

        if val_loader is not None:
            acc = _accuracy_loader(model, val_loader, dev)
            row["val_accuracy"] = acc
            if acc >= best_val:
                best_val = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            extra = f"  val_acc={row.get('val_accuracy', float('nan')):.4f}" if "val_accuracy" in row else ""
            print(f"epoch {epoch:02d}  loss={row['train_loss']:.4f}{extra}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return CNNTrainResult(model=model, history=history, device=str(dev))


@torch.no_grad()
def _accuracy_loader(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += len(yb)
    return correct / max(total, 1)


@torch.no_grad()
def predict_cnn(model: HARCNN1D, X: np.ndarray, *, device: str = "auto", batch_size: int = 256) -> np.ndarray:
    """Return UCI labels {1..6}."""
    dev = _device(device)
    model = model.to(dev)
    model.eval()
    preds = []
    xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    for i in range(0, len(xt), batch_size):
        logits = model(xt[i : i + batch_size].to(dev))
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return index_to_labels(np.concatenate(preds))


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "sitting_standing_error": float(sitting_standing_error_rate(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def train_eval_official(
    *,
    epochs: int = 25,
    batch_size: int = 64,
    seed: int = 24,
    device: str = "auto",
    verbose: bool = True,
) -> tuple[CNNTrainResult, dict, np.ndarray]:
    """Train on official raw train split; evaluate on official raw test split."""
    from .raw_data import load_raw_official_splits

    train, test = load_raw_official_splits(with_subjects=True)
    # Hold out 10% of train windows for model selection (stratified-ish by shuffling seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train.X))
    n_val = max(1, int(0.1 * len(idx)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    result = train_cnn(
        train.X[tr_idx],
        train.y.iloc[tr_idx],
        X_val=train.X[val_idx],
        y_val=train.y.iloc[val_idx],
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        verbose=verbose,
    )
    pred = predict_cnn(result.model, test.X, device=device)
    metrics = evaluate_predictions(test.y, pred)
    metrics["model"] = "1D CNN (raw IMU)"
    metrics["protocol"] = "official_test"
    return result, metrics, pred


def loso_cnn(
    *,
    epochs: int = 12,
    batch_size: int = 64,
    seed: int = 24,
    device: str = "auto",
    max_subjects: Optional[int] = None,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """
    Leave-one-subject-out on raw training volunteers.

    Uses fewer epochs by default (CPU-friendly). Set ``max_subjects`` to
    subsample subjects for a quicker estimate.
    """
    from .raw_data import load_raw_split

    train = load_raw_split("train", with_subjects=True)
    subjects = sorted(train.subjects.unique())
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    rows = []
    all_true = []
    all_pred = []
    for i, subject in enumerate(subjects, 1):
        if verbose:
            print(f"[LOSO {i}/{len(subjects)}] hold out subject {subject}", flush=True)
        test_mask = (train.subjects == subject).to_numpy()
        train_mask = ~test_mask
        result = train_cnn(
            train.X[train_mask],
            train.y.iloc[train_mask],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            verbose=False,
        )
        pred = predict_cnn(result.model, train.X[test_mask], device=device)
        y_true = train.y.iloc[test_mask]
        m = evaluate_predictions(y_true, pred)
        m["subject"] = int(subject)
        m["model"] = "1D CNN (raw IMU)"
        m["protocol"] = "loso"
        rows.append(m)
        all_true.append(np.asarray(y_true))
        all_pred.append(pred)
        if verbose:
            print(f"  subject {subject}: acc={m['accuracy']:.4f}", flush=True)

    y_all = np.concatenate(all_true)
    p_all = np.concatenate(all_pred)
    overall = evaluate_predictions(y_all, p_all)
    overall.update(
        {
            "model": "1D CNN (raw IMU)",
            "protocol": "loso_overall",
            "n_subjects": len(subjects),
            "accuracy_mean": float(np.mean([r["accuracy"] for r in rows])),
            "accuracy_std": float(np.std([r["accuracy"] for r in rows])),
        }
    )
    return rows, overall
