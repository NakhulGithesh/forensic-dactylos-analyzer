"""
Stage 9: Robustness, External Evaluation & Alteration Stress Testing Module.
Evaluates model and pipeline stability, representation drift, feature degradation,
and quality gating effectiveness across SOCOFing Real and Altered cohorts.
"""

from .robustness import (
    FingerprintRobustnessEvaluator,
    PairEvaluation,
    compute_robustness_metrics,
)

__all__ = [
    "FingerprintRobustnessEvaluator",
    "PairEvaluation",
    "compute_robustness_metrics",
]
