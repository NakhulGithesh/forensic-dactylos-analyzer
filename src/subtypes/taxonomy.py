"""
Dermatoglyphic 20-Subtype Taxonomy & Formal Specification.
Formalizes the 4 major categories and 20 dermatoglyphic labels established
in reference biometric standards and dermatoglyphic taxonomy:

Categories:
1. LOOP (4 subtypes):
   - Ulnar Loop
   - Radial Loop
   - Ulnar Falling Loop
   - Radial Falling Loop
2. ARCH (4 subtypes):
   - Simple Arch
   - Tented Arch
   - Tented Arch with Ulnar Loop
   - Tented Arch with Radial Loop
3. WHORL (11 subtypes):
   - Target Centric Whorl
   - Spiral Whorl
   - Elongated Whorl – Target Centric
   - Elongated Whorl – Spiral
   - Double Loop Whorl
   - Composite Whorl
   - Imploding Whorl
   - Ulnar Target Centric Peacock's Eye
   - Radial Target Centric Peacock's Eye
   - Ulnar Spiral Peacock's Eye
   - Radial Spiral Peacock's Eye
4. ACCIDENTAL (1 subtype):
   - Accidental

Every subtype is rigorously defined with required singular points, geometric signatures,
mathematical criteria, and forensic verifiability levels.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union


class PrimaryCategory(str, Enum):
    """The four macro pattern categories."""
    LOOP = "Loop"
    ARCH = "Arch"
    WHORL = "Whorl"
    ACCIDENTAL = "Accidental"


class ForensicVerifiability(str, Enum):
    """
    Forensic distinction level of subtype classification:
    - MATHEMATICALLY_VERIFIABLE: Rigorous, deterministic geometric calculation
      (e.g., Ulnar/Radial slant + hand, curvature thrust, closed contour metric).
    - SEMI_QUANTITATIVE: Quantifiable via calibrated metric thresholds, but sensitive
      to sensor field-of-view margins or edge truncation (e.g. falling loop downward slope,
      tented arch ridge count < 9, peacock delta proximity).
    - EXPERT_ADJUDICATED: High morphological complexity / fusion patterns
      (e.g. Composite / Imploding / Accidental) where algorithmic rules formulate
      probabilistic hypotheses that warrant expert human examiner verification.
    """
    MATHEMATICALLY_VERIFIABLE = "mathematically_verifiable"
    SEMI_QUANTITATIVE = "semi_quantitative"
    EXPERT_ADJUDICATED = "expert_adjudicated"


class SubtypeLabel(str, Enum):
    """The canonical 20 dermatoglyphic subtype labels."""
    # Loops (4)
    ULNAR_LOOP = "Ulnar Loop"
    RADIAL_LOOP = "Radial Loop"
    ULNAR_FALLING_LOOP = "Ulnar Falling Loop"
    RADIAL_FALLING_LOOP = "Radial Falling Loop"

    # Arches (4)
    SIMPLE_ARCH = "Simple Arch"
    TENTED_ARCH = "Tented Arch"
    TENTED_ARCH_ULNAR_LOOP = "Tented Arch with Ulnar Loop"
    TENTED_ARCH_RADIAL_LOOP = "Tented Arch with Radial Loop"

    # Whorls (11)
    TARGET_CENTRIC_WHORL = "Target Centric Whorl"
    SPIRAL_WHORL = "Spiral Whorl"
    ELONGATED_WHORL_TARGET = "Elongated Whorl – Target Centric"
    ELONGATED_WHORL_SPIRAL = "Elongated Whorl – Spiral"
    DOUBLE_LOOP_WHORL = "Double Loop Whorl"
    COMPOSITE_WHORL = "Composite Whorl"
    IMPLODING_WHORL = "Imploding Whorl"
    ULNAR_TARGET_CENTRIC_PEACOCK = "Ulnar Target Centric Peacock's Eye"
    RADIAL_TARGET_CENTRIC_PEACOCK = "Radial Target Centric Peacock's Eye"
    ULNAR_SPIRAL_PEACOCK = "Ulnar Spiral Peacock's Eye"
    RADIAL_SPIRAL_PEACOCK = "Radial Spiral Peacock's Eye"

    # Accidental (1)
    ACCIDENTAL = "Accidental"


@dataclass
class SubtypeDefinition:
    """Formal dermatoglyphic and geometric specification of a fingerprint subtype."""
    label: SubtypeLabel
    category: PrimaryCategory
    nist_classes: List[str]                  # Compatible NIST SD04 classes (e.g. ['L', 'R'], ['W'])
    min_cores: int                           # Expected minimum core singular points
    max_cores: int                           # Expected maximum core singular points
    min_deltas: int                          # Expected minimum delta singular points
    max_deltas: int                          # Expected maximum delta singular points
    description: str                         # Concise morphological definition
    forensic_verifiability: ForensicVerifiability
    geometric_signature: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label.value
        d["category"] = self.category.value
        d["forensic_verifiability"] = self.forensic_verifiability.value
        return d


# ---------------------------------------------------------------------------
# CANONICAL 20-SUBTYPE DEFINITION REGISTRY
# ---------------------------------------------------------------------------

TAXONOMY_REGISTRY: Dict[SubtypeLabel, SubtypeDefinition] = {
    # ------------------ LOOPS (4) ------------------
    SubtypeLabel.ULNAR_LOOP: SubtypeDefinition(
        label=SubtypeLabel.ULNAR_LOOP,
        category=PrimaryCategory.LOOP,
        nist_classes=["L", "R"],
        min_cores=1,
        max_cores=1,
        min_deltas=1,
        max_deltas=1,
        description="Standard single-core loop pattern flowing towards the little finger (ulna bone) with lateral exit.",
        forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
        geometric_signature={
            "flow_target": "ulna",
            "exit_trajectory": "lateral",
            "exit_angle_max_deg": 35.0,
            "min_ridge_count": 9,
        },
    ),
    SubtypeLabel.RADIAL_LOOP: SubtypeDefinition(
        label=SubtypeLabel.RADIAL_LOOP,
        category=PrimaryCategory.LOOP,
        nist_classes=["L", "R"],
        min_cores=1,
        max_cores=1,
        min_deltas=1,
        max_deltas=1,
        description="Standard single-core loop pattern flowing towards the thumb (radius bone) with lateral exit.",
        forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
        geometric_signature={
            "flow_target": "radius",
            "exit_trajectory": "lateral",
            "exit_angle_max_deg": 35.0,
            "min_ridge_count": 9,
        },
    ),
    SubtypeLabel.ULNAR_FALLING_LOOP: SubtypeDefinition(
        label=SubtypeLabel.ULNAR_FALLING_LOOP,
        category=PrimaryCategory.LOOP,
        nist_classes=["L", "R"],
        min_cores=1,
        max_cores=1,
        min_deltas=1,
        max_deltas=1,
        description="Loop flowing towards ulna whose recurring ridges droop and exit downward toward the basal flexion crease.",
        forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
        geometric_signature={
            "flow_target": "ulna",
            "exit_trajectory": "downward",
            "exit_angle_min_deg": 35.0,
            "high_delta_elevation": True,
        },
    ),
    SubtypeLabel.RADIAL_FALLING_LOOP: SubtypeDefinition(
        label=SubtypeLabel.RADIAL_FALLING_LOOP,
        category=PrimaryCategory.LOOP,
        nist_classes=["L", "R"],
        min_cores=1,
        max_cores=1,
        min_deltas=1,
        max_deltas=1,
        description="Loop flowing towards radius whose recurring ridges droop and exit downward toward the basal flexion crease.",
        forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
        geometric_signature={
            "flow_target": "radius",
            "exit_trajectory": "downward",
            "exit_angle_min_deg": 35.0,
            "high_delta_elevation": True,
        },
    ),

    # ------------------ ARCHES (4) ------------------
    SubtypeLabel.SIMPLE_ARCH: SubtypeDefinition(
        label=SubtypeLabel.SIMPLE_ARCH,
        category=PrimaryCategory.ARCH,
        nist_classes=["A"],
        min_cores=0,
        max_cores=0,
        min_deltas=0,
        max_deltas=0,
        description="Simple continuous wave-like ridge pattern crossing smoothly from one side to the other without vertical thrust or triradius.",
        forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
        geometric_signature={
            "vertical_thrust": False,
            "max_curvature_threshold": 0.25,
            "cores": 0,
            "deltas": 0,
        },
    ),
    SubtypeLabel.TENTED_ARCH: SubtypeDefinition(
        label=SubtypeLabel.TENTED_ARCH,
        category=PrimaryCategory.ARCH,
        nist_classes=["T", "A"],
        min_cores=0,
        max_cores=1,
        min_deltas=0,
        max_deltas=1,
        description="Arch pattern possessing an acute vertical thrust (tent pole), sharp angle >= 45 deg, or central triradius.",
        forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
        geometric_signature={
            "vertical_thrust": True,
            "min_curvature_peak": 0.28,
            "spine_angle_near_90": True,
        },
    ),
    SubtypeLabel.TENTED_ARCH_ULNAR_LOOP: SubtypeDefinition(
        label=SubtypeLabel.TENTED_ARCH_ULNAR_LOOP,
        category=PrimaryCategory.ARCH,
        nist_classes=["T", "R", "L"],
        min_cores=1,
        max_cores=1,
        min_deltas=1,
        max_deltas=1,
        description="Tented arch presenting a pseudo-loop formation towards ulna but with ridge count between delta and core < 9.",
        forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
        geometric_signature={
            "flow_target": "ulna",
            "max_ridge_count": 8,
            "has_recurve": True,
        },
    ),
    SubtypeLabel.TENTED_ARCH_RADIAL_LOOP: SubtypeDefinition(
        label=SubtypeLabel.TENTED_ARCH_RADIAL_LOOP,
        category=PrimaryCategory.ARCH,
        nist_classes=["T", "L", "R"],
        min_cores=1,
        max_cores=1,
        min_deltas=1,
        max_deltas=1,
        description="Tented arch presenting a pseudo-loop formation towards radius but with ridge count between delta and core < 9.",
        forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
        geometric_signature={
            "flow_target": "radius",
            "max_ridge_count": 8,
            "has_recurve": True,
        },
    ),

    # ------------------ WHORLS (11) ------------------
    SubtypeLabel.TARGET_CENTRIC_WHORL: SubtypeDefinition(
        label=SubtypeLabel.TARGET_CENTRIC_WHORL,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=1,
        max_cores=2,
        min_deltas=2,
        max_deltas=2,
        description="Concentric closed circular contours around a bullseye center flanked by two deltas.",
        forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
        geometric_signature={
            "topology": "concentric_closed",
            "max_aspect_ratio": 1.45,
            "max_eccentricity": 0.72,
            "deltas_expected": 2,
        },
    ),
    SubtypeLabel.SPIRAL_WHORL: SubtypeDefinition(
        label=SubtypeLabel.SPIRAL_WHORL,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=1,
        max_cores=1,
        min_deltas=2,
        max_deltas=2,
        description="Continuous Archimedean or logarithmic spiral ridge line evolving outward around a single center flanked by two deltas.",
        forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
        geometric_signature={
            "topology": "spiral_open",
            "max_aspect_ratio": 1.45,
            "max_eccentricity": 0.72,
            "deltas_expected": 2,
        },
    ),
    SubtypeLabel.ELONGATED_WHORL_TARGET: SubtypeDefinition(
        label=SubtypeLabel.ELONGATED_WHORL_TARGET,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=1,
        max_cores=2,
        min_deltas=2,
        max_deltas=2,
        description="Stretched oval target centric whorl with central closed rings having aspect ratio >= 1.45 or eccentricity >= 0.72.",
        forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
        geometric_signature={
            "topology": "concentric_closed",
            "min_aspect_ratio": 1.45,
            "min_eccentricity": 0.72,
            "deltas_expected": 2,
        },
    ),
    SubtypeLabel.ELONGATED_WHORL_SPIRAL: SubtypeDefinition(
        label=SubtypeLabel.ELONGATED_WHORL_SPIRAL,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=1,
        max_cores=2,
        min_deltas=2,
        max_deltas=2,
        description="Stretched oval spiral whorl with central open spiral having aspect ratio >= 1.45 or eccentricity >= 0.72.",
        forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
        geometric_signature={
            "topology": "spiral_open",
            "min_aspect_ratio": 1.45,
            "min_eccentricity": 0.72,
            "deltas_expected": 2,
        },
    ),
    SubtypeLabel.DOUBLE_LOOP_WHORL: SubtypeDefinition(
        label=SubtypeLabel.DOUBLE_LOOP_WHORL,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=2,
        max_cores=2,
        min_deltas=2,
        max_deltas=2,
        description="Two distinct separate loop formations interlocking in an S-curve with opposing core apertures and 2 deltas.",
        forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
        geometric_signature={
            "dual_opposing_cores": True,
            "interlocking_recurve": True,
            "deltas_expected": 2,
        },
    ),
    SubtypeLabel.COMPOSITE_WHORL: SubtypeDefinition(
        label=SubtypeLabel.COMPOSITE_WHORL,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=1,
        max_cores=2,
        min_deltas=2,
        max_deltas=2,
        description="Composite whorl pattern combining two distinct morphological types or having severely asymmetric delta elevations.",
        forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
        geometric_signature={
            "asymmetric_delta_elevation": True,
            "heterogeneous_core_flow": True,
        },
    ),
    SubtypeLabel.IMPLODING_WHORL: SubtypeDefinition(
        label=SubtypeLabel.IMPLODING_WHORL,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=1,
        max_cores=2,
        min_deltas=2,
        max_deltas=2,
        description="Center ridges collapse inward into a tight constricted core with high radial gradient and enclosed spiral pocket.",
        forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
        geometric_signature={
            "inward_collapse": True,
            "min_core_curvature": 0.38,
        },
    ),
    SubtypeLabel.ULNAR_TARGET_CENTRIC_PEACOCK: SubtypeDefinition(
        label=SubtypeLabel.ULNAR_TARGET_CENTRIC_PEACOCK,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=1,
        max_cores=1,
        min_deltas=2,
        max_deltas=2,
        description="Whorl with concentric eye structure and proximate delta (< 5 ridges / < 45 px) oriented towards ulna.",
        forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
        geometric_signature={
            "topology": "concentric_closed",
            "proximate_delta": True,
            "max_proximate_delta_dist_px": 45.0,
            "flow_target": "ulna",
        },
    ),
    SubtypeLabel.RADIAL_TARGET_CENTRIC_PEACOCK: SubtypeDefinition(
        label=SubtypeLabel.RADIAL_TARGET_CENTRIC_PEACOCK,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=1,
        max_cores=1,
        min_deltas=2,
        max_deltas=2,
        description="Whorl with concentric eye structure and proximate delta (< 5 ridges / < 45 px) oriented towards radius.",
        forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
        geometric_signature={
            "topology": "concentric_closed",
            "proximate_delta": True,
            "max_proximate_delta_dist_px": 45.0,
            "flow_target": "radius",
        },
    ),
    SubtypeLabel.ULNAR_SPIRAL_PEACOCK: SubtypeDefinition(
        label=SubtypeLabel.ULNAR_SPIRAL_PEACOCK,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=1,
        max_cores=1,
        min_deltas=2,
        max_deltas=2,
        description="Whorl with spiral eye structure and proximate delta (< 5 ridges / < 45 px) oriented towards ulna.",
        forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
        geometric_signature={
            "topology": "spiral_open",
            "proximate_delta": True,
            "max_proximate_delta_dist_px": 45.0,
            "flow_target": "ulna",
        },
    ),
    SubtypeLabel.RADIAL_SPIRAL_PEACOCK: SubtypeDefinition(
        label=SubtypeLabel.RADIAL_SPIRAL_PEACOCK,
        category=PrimaryCategory.WHORL,
        nist_classes=["W"],
        min_cores=1,
        max_cores=1,
        min_deltas=2,
        max_deltas=2,
        description="Whorl with spiral eye structure and proximate delta (< 5 ridges / < 45 px) oriented towards radius.",
        forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
        geometric_signature={
            "topology": "spiral_open",
            "proximate_delta": True,
            "max_proximate_delta_dist_px": 45.0,
            "flow_target": "radius",
        },
    ),

    # ------------------ ACCIDENTAL (1) ------------------
    SubtypeLabel.ACCIDENTAL: SubtypeDefinition(
        label=SubtypeLabel.ACCIDENTAL,
        category=PrimaryCategory.ACCIDENTAL,
        nist_classes=["W"],
        min_cores=2,
        max_cores=4,
        min_deltas=3,
        max_deltas=5,
        description="Complex irregular pattern possessing 3 or more deltas or fusion of two or more distinct non-standard pattern types.",
        forensic_verifiability=ForensicVerifiability.EXPERT_ADJUDICATED,
        geometric_signature={
            "min_deltas": 3,
            "irregular_multi_pattern": True,
        },
    ),
}


def normalize_subtype_str(name: str) -> str:
    """Normalize subtype string handling en-dashes, hyphens, and whitespace."""
    s = str(name).strip().replace("\u2013", "-").replace("—", "-")
    # Collapse multiple spaces
    s = " ".join(s.split())
    return s


def parse_subtype_label(val: Union[str, SubtypeLabel]) -> Optional[SubtypeLabel]:
    """Robust parser for SubtypeLabel from enum or string representation."""
    if isinstance(val, SubtypeLabel):
        return val
    norm = normalize_subtype_str(val).lower()
    for lbl, defn in TAXONOMY_REGISTRY.items():
        if norm == normalize_subtype_str(lbl.value).lower():
            return lbl
    return None


def get_subtypes_for_category(category: Union[PrimaryCategory, str]) -> List[SubtypeDefinition]:
    """Retrieve all subtype definitions belonging to a primary category."""
    cat_val = category.value if isinstance(category, PrimaryCategory) else category
    return [defn for defn in TAXONOMY_REGISTRY.values() if defn.category.value.lower() == str(cat_val).lower()]


def get_subtypes_for_nist_class(nist_class: str) -> List[SubtypeDefinition]:
    """Retrieve all compatible subtype definitions for a given NIST pattern code (A, T, W, L, R)."""
    code = str(nist_class).strip().upper()
    return [defn for defn in TAXONOMY_REGISTRY.values() if code in defn.nist_classes]


def validate_taxonomy_integrity() -> bool:
    """Validate that exactly 20 subtypes are defined across the 4 categories."""
    assert len(TAXONOMY_REGISTRY) == 20, f"Expected 20 subtypes, found {len(TAXONOMY_REGISTRY)}"
    counts = {
        PrimaryCategory.LOOP: len(get_subtypes_for_category(PrimaryCategory.LOOP)),
        PrimaryCategory.ARCH: len(get_subtypes_for_category(PrimaryCategory.ARCH)),
        PrimaryCategory.WHORL: len(get_subtypes_for_category(PrimaryCategory.WHORL)),
        PrimaryCategory.ACCIDENTAL: len(get_subtypes_for_category(PrimaryCategory.ACCIDENTAL)),
    }
    assert counts[PrimaryCategory.LOOP] == 4, f"Expected 4 loops, got {counts[PrimaryCategory.LOOP]}"
    assert counts[PrimaryCategory.ARCH] == 4, f"Expected 4 arches, got {counts[PrimaryCategory.ARCH]}"
    assert counts[PrimaryCategory.WHORL] == 11, f"Expected 11 whorls, got {counts[PrimaryCategory.WHORL]}"
    assert counts[PrimaryCategory.ACCIDENTAL] == 1, f"Expected 1 accidental, got {counts[PrimaryCategory.ACCIDENTAL]}"
    return True
