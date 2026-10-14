"""Human Activity Recognition utilities."""

from .data import ACTIVITY_LABELS, load_har_dataset, load_official_splits, load_split
from .knn import CustomKNN
from .models import CascadedHARClassifier, build_model_zoo, evaluate_models

__all__ = [
    "ACTIVITY_LABELS",
    "CascadedHARClassifier",
    "CustomKNN",
    "build_model_zoo",
    "evaluate_models",
    "load_har_dataset",
    "load_official_splits",
    "load_split",
]

__version__ = "1.2.0"
