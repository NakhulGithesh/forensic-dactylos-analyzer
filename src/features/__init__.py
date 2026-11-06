"""
Fingerprint Feature Detection & Geometric Analysis Package.
Includes:
- Ridge Orientation Field & Coherence Estimation (Bazen & Gerez / Rao)
- Singular Point Extraction (Poincaré Index, Complex Rotational Symmetry Filter)
- Non-Maximum Suppression, Dipole Suppression, and Confidence Scoring
- Ridge Frequency Estimation and Streamline Trajectory Analysis
- Loop Directionality and Anatomical (Ulnar vs. Radial) Classification
"""

from .orientation import (
    OrientationResult,
    compute_orientation_field,
    compute_coherence,
    compute_orientation_curvature,
    segment_foreground,
    to_grayscale_numpy,
)
from .singular_points import (
    SingularPoint,
    SingularPointsResult,
    detect_singular_points,
    compute_poincare_index_map,
    compute_complex_phase_filters,
    compute_singular_point_confidence,
    cluster_singular_points,
    cancel_core_delta_dipoles,
    predict_delta_truncation,
)
from .geometry import (
    GeometryResult,
    estimate_ridge_frequency,
    trace_ridge_streamlines,
    trace_single_streamline,
    detect_loop_directionality,
    extract_fingerprint_geometry,
)

__all__ = [
    # Orientation
    "OrientationResult",
    "compute_orientation_field",
    "compute_coherence",
    "compute_orientation_curvature",
    "segment_foreground",
    "to_grayscale_numpy",
    # Singular points
    "SingularPoint",
    "SingularPointsResult",
    "detect_singular_points",
    "compute_poincare_index_map",
    "compute_complex_phase_filters",
    "compute_singular_point_confidence",
    "cluster_singular_points",
    "cancel_core_delta_dipoles",
    "predict_delta_truncation",
    # Geometry
    "GeometryResult",
    "estimate_ridge_frequency",
    "trace_ridge_streamlines",
    "trace_single_streamline",
    "detect_loop_directionality",
    "extract_fingerprint_geometry",
    # Multi-channel transforms
    "OrientationAuxiliaryTransform",
]

from .transforms import OrientationAuxiliaryTransform
