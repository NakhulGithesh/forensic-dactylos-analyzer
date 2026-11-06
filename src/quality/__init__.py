"""
Biometric Quality Assessment & Damage Detection Package.
Provides standardized biometric image quality assessment (IQA), defect segmentation,
and forensic quality reporting.
"""

from .assessment import (
    BiometricQualityResult,
    assess_biometric_quality,
    compute_local_ridge_clarity,
    compute_frequency_continuity,
    compute_sharpness_and_blur,
    compute_dynamic_range_saturation,
    compute_composite_quality_score,
    classify_quality_tier,
)
from .damage import (
    DamageSegmentationResult,
    segment_damage_and_defects,
    detect_ridge_flow_disruptions,
    categorize_defects,
)
from .analyzer import (
    QualityReport,
    analyze_fingerprint_quality,
)

__all__ = [
    # Unified Pipeline
    "analyze_fingerprint_quality",
    "QualityReport",
    # Assessment
    "BiometricQualityResult",
    "assess_biometric_quality",
    "compute_local_ridge_clarity",
    "compute_frequency_continuity",
    "compute_sharpness_and_blur",
    "compute_dynamic_range_saturation",
    "compute_composite_quality_score",
    "classify_quality_tier",
    # Damage & Defects
    "DamageSegmentationResult",
    "segment_damage_and_defects",
    "detect_ridge_flow_disruptions",
    "categorize_defects",
]
