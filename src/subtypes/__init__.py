"""
Stage 7: Subtype Classification & 20-Label Hierarchical Dermatoglyphic Engine.
Unifies Stage 6 Neural Network Pattern Classification with Stage 5 Geometric Analysis.
"""

from .taxonomy import (
    PrimaryCategory,
    ForensicVerifiability,
    SubtypeLabel,
    SubtypeDefinition,
    TAXONOMY_REGISTRY,
    get_subtypes_for_category,
    get_subtypes_for_nist_class,
    parse_subtype_label,
    normalize_subtype_str,
    validate_taxonomy_integrity,
)
from .rules import (
    RuleDecision,
    LoopRuleEngine,
    ArchRuleEngine,
    WhorlRuleEngine,
    compute_core_streamline_geometry,
    compute_ridge_count,
)
from .classifier import (
    SubtypeClassificationResult,
    HierarchicalSubtypeClassifier,
)

__all__ = [
    # Taxonomy
    "PrimaryCategory",
    "ForensicVerifiability",
    "SubtypeLabel",
    "SubtypeDefinition",
    "TAXONOMY_REGISTRY",
    "get_subtypes_for_category",
    "get_subtypes_for_nist_class",
    "parse_subtype_label",
    "normalize_subtype_str",
    "validate_taxonomy_integrity",
    # Rules
    "RuleDecision",
    "LoopRuleEngine",
    "ArchRuleEngine",
    "WhorlRuleEngine",
    "compute_core_streamline_geometry",
    "compute_ridge_count",
    # Classifier
    "SubtypeClassificationResult",
    "HierarchicalSubtypeClassifier",
]
