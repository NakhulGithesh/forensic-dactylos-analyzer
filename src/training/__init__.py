"""
Stage 6 Training & Evaluation Package.
"""

from src.training.losses import (
    CrossEntropyWithLabelSmoothing,
    AmbiguityAwareCrossEntropyLoss,
)
from src.training.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)
from src.training.trainer import (
    FingerprintTrainer,
)

__all__ = [
    "CrossEntropyWithLabelSmoothing",
    "AmbiguityAwareCrossEntropyLoss",
    "ClassificationMetrics",
    "compute_classification_metrics",
    "FingerprintTrainer",
]
