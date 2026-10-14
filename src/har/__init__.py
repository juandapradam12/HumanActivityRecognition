"""Human Activity Recognition utilities."""

from .data import ACTIVITY_LABELS, load_har_dataset
from .knn import CustomKNN
from .models import build_model_zoo, evaluate_models

__all__ = [
    "ACTIVITY_LABELS",
    "CustomKNN",
    "build_model_zoo",
    "evaluate_models",
    "load_har_dataset",
]

__version__ = "1.1.0"
