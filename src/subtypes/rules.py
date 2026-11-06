"""
Dermatoglyphic Geometric Subtype Discriminators & Rule Engine.
Implements quantifiable geometric decision rules to resolve the 20 subtypes:

1. Loops (4):
   - Ulnar vs. Radial: Hand metadata + continuous flow exit direction.
   - Standard vs. Falling Loop: Delta-to-core elevation and downward exit angle.
2. Arches (4):
   - Simple vs. Tented: Ridge curvature peak, vertical thrust / spine analysis.
   - Tented Arch with Loop: Localized recurve / pseudo-core with ridge count < 9.
3. Whorls (11):
   - Target Centric vs. Spiral: Streamline contour closure vs. open radial drift.
   - Elongated (Target Centric vs. Spiral): Core region aspect ratio and eccentricity.
   - Double Loop: Dual opposing cores with interlocking recurve trajectories.
   - Peacock's Eye (4 types): Proximate delta (< 45 px / <= 5 ridges) to core eye.
   - Composite & Imploding: Irregular multi-delta topologies or collapsing spiral core.
4. Accidental (1):
   - Irregular multi-delta topologies (>= 3 deltas) or multi-pattern fusion.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np

from src.features.orientation import OrientationResult
from src.features.singular_points import SingularPointsResult, SingularPoint
from src.features.geometry import GeometryResult, trace_single_streamline
from src.models.predict import infer_hand
from src.subtypes.taxonomy import (
    SubtypeLabel,
    PrimaryCategory,
    ForensicVerifiability,
    TAXONOMY_REGISTRY,
)


@dataclass
class RuleDecision:
    """Outcome of a category-specific geometric discrimination rule."""
    subtype: SubtypeLabel
    category: PrimaryCategory
    confidence: float
    is_uncertain: bool
    reason: Optional[str]
    forensic_verifiability: ForensicVerifiability
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtype": self.subtype.value,
            "category": self.category.value,
            "confidence": float(round(self.confidence, 4)),
            "is_uncertain": self.is_uncertain,
            "reason": self.reason,
            "forensic_verifiability": self.forensic_verifiability.value,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# HELPER GEOMETRIC FUNCTIONS
# ---------------------------------------------------------------------------

def compute_core_streamline_geometry(
    core: SingularPoint,
    orientation_res: OrientationResult,
    wavelength: float = 10.0,
) -> Dict[str, Any]:
    """
    Trace streamlines around core to compute contour closure, radial drift,
    aspect ratio, eccentricity, and distinguish concentric vs. spiral vs. imploding topology.
    """
    cx, cy = core.x, core.y
    theta = orientation_res.theta
    mask = orientation_res.mask
    coherence = orientation_res.coherence

    wl = max(wavelength, 7.0)
    probe_radii = [wl * 1.5, wl * 2.5]
    probe_angles = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]

    dr_dthetas: List[float] = []
    stream_pts: List[np.ndarray] = []
    total_rotations: List[float] = []

    for r in probe_radii:
        for phi in probe_angles:
            sx = cx + r * np.cos(phi)
            sy = cy + r * np.sin(phi)
            if 0 <= int(sy) < theta.shape[0] and 0 <= int(sx) < theta.shape[1] and mask[int(sy), int(sx)]:
                sl = trace_single_streamline(
                    sx, sy,
                    theta_field=theta,
                    mask=mask,
                    coherence=coherence,
                    step_size=1.5,
                    max_steps=160,
                )
                if len(sl) >= 20:
                    stream_pts.append(sl)
                    dists = np.hypot(sl[:, 0] - sx, sl[:, 1] - sy)
                    start_idx = int(np.argmin(dists))
                    fwd = sl[start_idx:]
                    if len(fwd) >= 25:
                        dx = fwd[:, 0] - cx
                        dy = fwd[:, 1] - cy
                        radii = np.hypot(dx, dy)
                        angles = np.unwrap(np.arctan2(dy, dx))
                        tot_rot = float(angles[-1] - angles[0])
                        total_rotations.append(abs(tot_rot))
                        if abs(tot_rot) >= 1.0 * np.pi:
                            drift_rate = float((radii[-1] - radii[0]) / abs(tot_rot))
                            dr_dthetas.append(drift_rate)

    # Compute Aspect Ratio & Eccentricity from central streamline points
    aspect_ratio = 1.0
    eccentricity = 0.0
    all_pts = np.vstack(stream_pts) if stream_pts else np.empty((0, 2))

    if len(all_pts) >= 20:
        cov = np.cov(all_pts, rowvar=False)
        evals = np.linalg.eigvalsh(cov)
        val_min = max(float(evals[0]), 1e-4)
        val_max = max(float(evals[1]), 1e-4)
        aspect_ratio = float(np.sqrt(val_max / val_min))
        eccentricity = float(np.sqrt(max(0.0, 1.0 - val_min / val_max)))

    # Compute radial drift per full revolution (2pi radians)
    if dr_dthetas:
        median_dr_dtheta = float(np.median(dr_dthetas))
        delta_r_2pi = median_dr_dtheta * 2.0 * np.pi
    else:
        # Fallback to mean radial projection
        y_coords, x_coords = np.ogrid[:theta.shape[0], :theta.shape[1]]
        dx_grid = x_coords - cx
        dy_grid = y_coords - cy
        dist_grid = np.sqrt(dx_grid**2 + dy_grid**2)
        core_mask = (dist_grid >= 12.0) & (dist_grid <= 42.0) & (mask > 0)
        mean_proj = 0.0
        if np.any(core_mask):
            rx = dx_grid / (dist_grid + 1e-6)
            ry = dy_grid / (dist_grid + 1e-6)
            tx = np.cos(theta)
            ty = np.sin(theta)
            mean_proj = float(np.mean(np.abs(tx * rx + ty * ry)[core_mask]))
        median_dr_dtheta = (mean_proj - 0.40) * 5.0
        delta_r_2pi = median_dr_dtheta * 2.0 * np.pi

    # Topological classification
    drift_threshold = max(0.65 * wl, 6.5)
    is_imploding = (delta_r_2pi < -drift_threshold)
    is_spiral = (delta_r_2pi > drift_threshold)
    is_elongated = (aspect_ratio >= 1.42) or (eccentricity >= 0.70)

    return {
        "aspect_ratio": float(round(aspect_ratio, 3)),
        "eccentricity": float(round(eccentricity, 3)),
        "dr_dtheta": float(round(median_dr_dtheta, 3)),
        "delta_r_2pi": float(round(delta_r_2pi, 2)),
        "is_imploding": is_imploding,
        "is_spiral": is_spiral,
        "is_elongated": is_elongated,
        "num_streamlines_analyzed": len(stream_pts),
        "num_drift_samples": len(dr_dthetas),
    }


def compute_ridge_count(
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    wavelength: float,
) -> float:
    """Calculate estimated anatomical ridge count along Euclidean segment."""
    dist = float(np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]))
    return dist / max(wavelength, 4.0)


# ---------------------------------------------------------------------------
# CATEGORY RULE ENGINES
# ---------------------------------------------------------------------------

class LoopRuleEngine:
    """
    Discriminates Loop Subtypes:
    - Ulnar Loop
    - Radial Loop
    - Ulnar Falling Loop
    - Radial Falling Loop
    """
    @staticmethod
    def evaluate(
        primary_class: str,
        primary_conf: float,
        orientation_res: OrientationResult,
        singular_points_res: SingularPointsResult,
        geometry_res: GeometryResult,
        hand: Optional[str] = None,
        finger_num: Optional[int] = None,
        digit_code: Optional[str] = None,
    ) -> RuleDecision:
        resolved_hand = infer_hand(hand=hand, finger_num=finger_num, digit_code=digit_code)
        flow_asym = geometry_res.flow_asymmetry
        flow_dir = geometry_res.flow_direction

        # 1. Determine Slant Direction (Right vs. Left)
        # Trust Stage 6 Neural Network primary prediction when confident
        if primary_class.upper() in ("L", "R") and primary_conf >= 0.50:
            is_right_slant = (primary_class.upper() == "R")
        elif abs(flow_asym) >= 0.05:
            is_right_slant = (flow_asym > 0)
        elif flow_dir in ("right", "left"):
            is_right_slant = (flow_dir == "right")
        else:
            is_right_slant = (primary_class.upper() == "R")

        slant_str = "right" if is_right_slant else "left"

        # 2. Map Anatomical Direction (Ulnar vs. Radial)
        # Right hand: right slant -> Ulnar (pinky), left slant -> Radial (thumb)
        # Left hand: left slant -> Ulnar (pinky), right slant -> Radial (thumb)
        is_ulnar = True
        hand_uncertain = False
        if resolved_hand == "Right":
            is_ulnar = is_right_slant
        elif resolved_hand == "Left":
            is_ulnar = not is_right_slant
        else:
            hand_uncertain = True
            # Default hypothesis based on standard ~94% population ulnar prior
            is_ulnar = True

        # 3. Detect Falling Loop Trajectory
        # A falling loop's recurring ridges droop and exit downward toward the basal crease
        # Exit angle is measured in degrees: 0 is right, 90 is straight down, 180 is left
        exit_angle = geometry_res.exit_angle_deg
        # Downward deviation from horizontal:
        if is_right_slant:
            # Right loop exiting towards right-down: angle in [0, 90]
            downward_dev = exit_angle if exit_angle > 0.0 else 0.0
        else:
            # Left loop exiting towards left-down: angle in (90, 180]
            downward_dev = (180.0 - exit_angle) if exit_angle > 90.0 else 0.0

        # Delta-to-core elevation check
        core = singular_points_res.cores[0] if singular_points_res.cores else None
        delta = singular_points_res.deltas[0] if singular_points_res.deltas else None
        delta_high_elevation = False
        vert_dist_px = 0.0
        if core is not None and delta is not None:
            # In standard loop, delta is well below core: y_delta - y_core > 50 px
            # In falling loop, delta is near horizontal core level: y_delta - y_core < 42 px
            vert_dist_px = float(delta.y - core.y)
            if vert_dist_px < 42.0:
                delta_high_elevation = True

        is_falling = (downward_dev >= 35.0) and (delta_high_elevation or downward_dev >= 45.0)

        # 4. Resolve Final Subtype Label
        if is_falling:
            label = SubtypeLabel.ULNAR_FALLING_LOOP if is_ulnar else SubtypeLabel.RADIAL_FALLING_LOOP
            verifiability = ForensicVerifiability.SEMI_QUANTITATIVE
        else:
            label = SubtypeLabel.ULNAR_LOOP if is_ulnar else SubtypeLabel.RADIAL_LOOP
            verifiability = ForensicVerifiability.MATHEMATICALLY_VERIFIABLE

        # 5. Uncertainty and Confidence Scoring
        confidence = float(np.clip(primary_conf * 0.70 + geometry_res.confidence * 0.30, 0.40, 0.98))
        is_uncertain = False
        reason = None

        if hand_uncertain:
            is_uncertain = True
            reason = "Hand metadata required to resolve Ulnar vs. Radial anatomical orientation"
            confidence *= 0.75
        elif singular_points_res.delta_truncated and is_falling:
            is_uncertain = True
            reason = "Insufficient evidence: delta truncated, cannot verify elevation for falling loop"
            confidence *= 0.80

        evidence = {
            "resolved_hand": resolved_hand,
            "slant_direction": slant_str,
            "flow_asymmetry": float(round(flow_asym, 3)),
            "exit_angle_deg": float(round(exit_angle, 2)),
            "downward_deviation_deg": float(round(downward_dev, 2)),
            "is_falling": is_falling,
            "delta_high_elevation": delta_high_elevation,
            "core_delta_vert_dist_px": float(round(vert_dist_px, 1)),
            "delta_truncated": singular_points_res.delta_truncated,
        }

        return RuleDecision(
            subtype=label,
            category=PrimaryCategory.LOOP,
            confidence=confidence,
            is_uncertain=is_uncertain,
            reason=reason,
            forensic_verifiability=verifiability,
            evidence=evidence,
        )


class ArchRuleEngine:
    """
    Discriminates Arch Subtypes:
    - Simple Arch
    - Tented Arch
    - Tented Arch with Ulnar Loop
    - Tented Arch with Radial Loop
    """
    @staticmethod
    def evaluate(
        primary_class: str,
        primary_conf: float,
        orientation_res: OrientationResult,
        singular_points_res: SingularPointsResult,
        geometry_res: GeometryResult,
        hand: Optional[str] = None,
        finger_num: Optional[int] = None,
        digit_code: Optional[str] = None,
    ) -> RuleDecision:
        resolved_hand = infer_hand(hand=hand, finger_num=finger_num, digit_code=digit_code)
        h, w = orientation_res.theta.shape
        num_cores = singular_points_res.num_cores
        num_deltas = singular_points_res.num_deltas

        # 1. Check for Tented Arch with Loop (localized recurve with ridge count < 9)
        wavelength = max(geometry_res.median_wavelength, 7.0)
        has_pseudo_loop = False
        ridge_count = 0.0
        recurve_slant = "right"

        if num_cores >= 1 and num_deltas >= 1:
            core = singular_points_res.cores[0]
            closest_d = min(singular_points_res.deltas, key=lambda d: np.hypot(core.x - d.x, core.y - d.y))
            ridge_count = compute_ridge_count((core.x, core.y), (closest_d.x, closest_d.y), wavelength)
            # FBI/Dermatoglyphic criterion: pseudo-loop with ridge count < 9
            if ridge_count < 9.0:
                has_pseudo_loop = True
                recurve_slant = "right" if closest_d.x < core.x else "left"

        if has_pseudo_loop:
            is_ulnar = True
            hand_uncertain = False
            if resolved_hand == "Right":
                is_ulnar = (recurve_slant == "right")
            elif resolved_hand == "Left":
                is_ulnar = (recurve_slant == "left")
            else:
                hand_uncertain = True

            label = SubtypeLabel.TENTED_ARCH_ULNAR_LOOP if is_ulnar else SubtypeLabel.TENTED_ARCH_RADIAL_LOOP
            confidence = float(np.clip(primary_conf * 0.65 + 0.30, 0.45, 0.92))
            is_uncertain = hand_uncertain
            reason = "Hand metadata required to resolve Ulnar vs. Radial orientation" if hand_uncertain else None

            return RuleDecision(
                subtype=label,
                category=PrimaryCategory.ARCH,
                confidence=confidence,
                is_uncertain=is_uncertain,
                reason=reason,
                forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
                evidence={
                    "has_pseudo_loop": True,
                    "ridge_count": float(round(ridge_count, 2)),
                    "recurve_slant": recurve_slant,
                    "resolved_hand": resolved_hand,
                    "num_cores": num_cores,
                    "num_deltas": num_deltas,
                },
            )

        # 2. Simple Arch vs. Tented Arch Discriminator
        cen_y_min, cen_y_max = int(0.20 * h), int(0.80 * h)
        cen_x_min, cen_x_max = int(0.20 * w), int(0.80 * w)
        cen_theta = orientation_res.theta[cen_y_min:cen_y_max, cen_x_min:cen_x_max]
        cen_mask = orientation_res.mask[cen_y_min:cen_y_max, cen_x_min:cen_x_max]

        # Ridges oriented vertically: |theta - pi/2| < 20 deg (0.35 rad)
        vert_ridges = (np.abs(cen_theta - np.pi / 2.0) < 0.35) & (cen_mask > 0)
        vert_ratio = float(np.mean(vert_ridges)) if np.any(cen_mask) else 0.0

        # Orientation curvature 95th percentile (robust to single-pixel noise spikes)
        curv_map = singular_points_res.curvature_map
        cen_curv_pts = curv_map[cen_y_min:cen_y_max, cen_x_min:cen_x_max] if curv_map is not None else None
        p95_curv = float(np.percentile(cen_curv_pts[cen_mask > 0], 95)) if (cen_curv_pts is not None and np.any(cen_mask)) else 0.0

        # Filter confident central singular points (central tent pole / triradius)
        central_cores = [
            c for c in singular_points_res.cores
            if c.confidence >= 0.65 and cen_x_min <= c.x <= cen_x_max and cen_y_min <= c.y <= cen_y_max
        ]
        central_deltas = [
            d for d in singular_points_res.deltas
            if d.confidence >= 0.65 and cen_x_min <= d.x <= cen_x_max and cen_y_min <= d.y <= cen_y_max
        ]
        has_central_singularity = (len(central_cores) > 0 or len(central_deltas) > 0)

        # Primary neural network prediction
        is_nn_tented = (primary_class.upper() == "T")

        # Tented arch decision: vertical thrust OR strong curvature peak OR presence of central spire/triradius
        is_tented = (
            (vert_ratio >= 0.04) or
            (p95_curv >= 0.80) or
            has_central_singularity or
            (is_nn_tented and (vert_ratio >= 0.015 or p95_curv >= 0.55))
        )

        if is_tented:
            label = SubtypeLabel.TENTED_ARCH
            confidence = float(np.clip(primary_conf * 0.60 + min(vert_ratio * 2.0, 0.35) + 0.05, 0.50, 0.96))
        else:
            label = SubtypeLabel.SIMPLE_ARCH
            confidence = float(np.clip(primary_conf * 0.65 + (1.0 - min(vert_ratio * 4.0, 0.50)) * 0.35, 0.50, 0.96))

        return RuleDecision(
            subtype=label,
            category=PrimaryCategory.ARCH,
            confidence=confidence,
            is_uncertain=False,
            reason=None,
            forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
            evidence={
                "vertical_ridge_ratio": float(round(vert_ratio, 4)),
                "p95_central_curvature": float(round(p95_curv, 4)),
                "num_cores": num_cores,
                "num_deltas": num_deltas,
                "has_central_singularity": has_central_singularity,
                "primary_class": primary_class,
            },
        )


class WhorlRuleEngine:
    """
    Discriminates Whorl Subtypes (11) & Accidental (1):
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
    - Accidental
    """
    @staticmethod
    def evaluate(
        primary_class: str,
        primary_conf: float,
        orientation_res: OrientationResult,
        singular_points_res: SingularPointsResult,
        geometry_res: GeometryResult,
        hand: Optional[str] = None,
        finger_num: Optional[int] = None,
        digit_code: Optional[str] = None,
    ) -> RuleDecision:
        resolved_hand = infer_hand(hand=hand, finger_num=finger_num, digit_code=digit_code)
        h, w = orientation_res.theta.shape
        wavelength = max(geometry_res.median_wavelength, 7.0)

        # 1. Filter interior cores
        margin_x = 0.08 * w
        margin_y = 0.08 * h

        # Active foreground bounds for basilar crease boundary suppression
        fg_y, _ = np.nonzero(orientation_res.mask)
        if fg_y.size > 0:
            y_min_fg = float(np.min(fg_y))
            y_max_fg = float(np.max(fg_y))
            crease_cutoff_y = y_min_fg + 0.80 * (y_max_fg - y_min_fg)
        else:
            crease_cutoff_y = 0.80 * h

        # Preliminary reference core elevation for delta filtering from non-crease cores
        prelim_cores = [
            c for c in singular_points_res.cores
            if c.y < crease_cutoff_y and margin_y < c.y < (h - margin_y) and margin_x < c.x < (w - margin_x)
        ]
        cy_ref = min([c.y for c in prelim_cores]) if prelim_cores else (0.40 * h)

        # 2. Filter deltas: must be in interior pattern area (below upper margin, y >= cy_ref - 30)
        pattern_deltas = [
            d for d in singular_points_res.deltas
            if d.confidence >= 0.60 and margin_x < d.x < (w - margin_x) and (cy_ref - 30.0) < d.y < (h - margin_y)
        ]
        if not pattern_deltas and singular_points_res.deltas:
            pattern_deltas = [
                d for d in singular_points_res.deltas
                if d.confidence >= 0.50 and margin_x < d.x < (w - margin_x) and (cy_ref - 30.0) < d.y < (h - margin_y)
            ]

        # Remove clustered / duplicate deltas within 35 px of each other
        distinct_deltas: List[SingularPoint] = []
        for d in pattern_deltas:
            if not any(np.hypot(d.x - od.x, d.y - od.y) < 35.0 for od in distinct_deltas):
                distinct_deltas.append(d)

        deltas = distinct_deltas
        num_deltas = len(deltas)

        ref_deltas = deltas if deltas else singular_points_res.deltas
        max_delta_y = max((d.y for d in ref_deltas), default=None)

        def is_peacock_eye_core(c: SingularPoint, d_list: List[SingularPoint]) -> bool:
            """Preserve valid central core in Peacock's Eye topology."""
            if len(d_list) < 2:
                return False
            d_dists = sorted(float(np.hypot(c.x - d.x, c.y - d.y)) for d in d_list)
            min_d, max_d = d_dists[0], d_dists[-1]
            return (
                (min_d <= 7.0 * wavelength or min_d <= 80.0)
                and (max_d / max(min_d, 1.0) >= 1.75)
                and (max_d >= 10.0 * wavelength or max_d >= 120.0)
            )

        # Spatial pattern area & crease filtering applied across candidate cores
        valid_cores: List[SingularPoint] = []
        for c in singular_points_res.cores:
            if not (margin_x < c.x < (w - margin_x) and margin_y < c.y < (h - margin_y)):
                continue
            if c.y >= crease_cutoff_y:
                continue
            if max_delta_y is not None and c.y > max_delta_y:
                continue
            if any(np.hypot(c.x - d.x, c.y - d.y) < 45.0 for d in ref_deltas):
                if not (len(prelim_cores) <= 1 and is_peacock_eye_core(c, ref_deltas)):
                    continue
            valid_cores.append(c)

        # Select confident cores with fallback so genuine cores are never dropped
        cores = [c for c in valid_cores if c.confidence >= 0.60]
        if len(cores) < 2:
            additional = [c for c in valid_cores if c.confidence >= 0.50 and c not in cores]
            cores.extend(additional)

        num_cores = len(cores)

        # 3. ACCIDENTAL PATTERN GATING
        # Strictly requires >= 3 distinct, confident interior deltas
        if num_deltas >= 3:
            return RuleDecision(
                subtype=SubtypeLabel.ACCIDENTAL,
                category=PrimaryCategory.ACCIDENTAL,
                confidence=float(np.clip(primary_conf * 0.50 + 0.35, 0.45, 0.90)),
                is_uncertain=False,
                reason=None,
                forensic_verifiability=ForensicVerifiability.EXPERT_ADJUDICATED,
                evidence={
                    "num_deltas": num_deltas,
                    "num_cores": num_cores,
                    "trigger": "multi_delta_irregularity",
                },
            )

        # 4. DOUBLE LOOP WHORL (S-Family, 2 distinct cores with opposing loop orientations)
        if num_cores >= 2:
            found_dl = False
            best_pair = None
            best_ang_diff = 0.0

            for i in range(len(cores)):
                for j in range(i + 1, len(cores)):
                    c1, c2 = cores[i], cores[j]
                    core_dist = float(np.hypot(c1.x - c2.x, c1.y - c2.y))

                    both_in_pattern = (
                        margin_x < c1.x < (w - margin_x) and margin_x < c2.x < (w - margin_x) and
                        margin_y < c1.y < (h - margin_y) and margin_y < c2.y < (h - margin_y) and
                        c1.y < crease_cutoff_y and c2.y < crease_cutoff_y and
                        (max_delta_y is None or (c1.y <= max_delta_y and c2.y <= max_delta_y)) and
                        not any(np.hypot(c1.x - d.x, c1.y - d.y) < 45.0 for d in ref_deltas) and
                        not any(np.hypot(c2.x - d.x, c2.y - d.y) < 45.0 for d in ref_deltas)
                    )

                    if 20.0 <= core_dist <= 0.65 * w and both_in_pattern:
                        y1, x1 = int(np.clip(c1.y, 0, h - 1)), int(np.clip(c1.x, 0, w - 1))
                        y2, x2 = int(np.clip(c2.y, 0, h - 1)), int(np.clip(c2.x, 0, w - 1))
                        th1 = float(np.rad2deg(orientation_res.theta[y1, x1]))
                        th2 = float(np.rad2deg(orientation_res.theta[y2, x2]))
                        diff_deg = abs(th1 - th2)
                        ang_diff = min(diff_deg, 180.0 - diff_deg)

                        if ang_diff >= 35.0 and ang_diff > best_ang_diff:
                            best_ang_diff = ang_diff
                            best_pair = (c1, c2, core_dist, ang_diff, th1, th2)
                            found_dl = True

            if found_dl and best_pair is not None:
                c1, c2, core_dist, ang_diff, th1, th2 = best_pair
                return RuleDecision(
                    subtype=SubtypeLabel.DOUBLE_LOOP_WHORL,
                    category=PrimaryCategory.WHORL,
                    confidence=float(np.clip(primary_conf * 0.60 + 0.32, 0.55, 0.94)),
                    is_uncertain=False,
                    reason=None,
                    forensic_verifiability=ForensicVerifiability.MATHEMATICALLY_VERIFIABLE,
                    evidence={
                        "num_cores": num_cores,
                        "core_distance_px": float(round(core_dist, 1)),
                        "core_angle_difference_deg": float(round(ang_diff, 1)),
                        "opposing_core_angles": [float(round(th1, 1)), float(round(th2, 1))],
                    },
                )

        # 5. PEACOCK'S EYE (4 Subtypes: Central Pocket Loop Whorl)
        # Proximate delta close to core (<= 7 ridges or <= 80 px) and distal delta far away (ratio >= 1.75)
        if num_cores >= 1 and num_deltas >= 2:
            primary_core = cores[0]
            delta_dists = [
                (d, float(np.hypot(primary_core.x - d.x, primary_core.y - d.y)))
                for d in deltas
            ]
            delta_dists.sort(key=lambda t: t[1])
            closest_delta, min_d_dist = delta_dists[0]
            farthest_delta, max_d_dist = delta_dists[-1]

            min_ridge_count = min_d_dist / wavelength

            is_peacock_distance = (
                (min_d_dist <= 7.0 * wavelength or min_d_dist <= 80.0)
                and (max_d_dist / max(min_d_dist, 1.0) >= 1.75)
                and (max_d_dist >= 10.0 * wavelength or max_d_dist >= 120.0)
            )

            if is_peacock_distance:
                core_geom = compute_core_streamline_geometry(primary_core, orientation_res, wavelength)
                is_spiral = core_geom["is_spiral"]

                # Eye slants towards distal delta (opposite of proximate delta)
                eye_slant_right = (closest_delta.x < primary_core.x)
                is_ulnar = True
                hand_uncertain = False
                if resolved_hand == "Right":
                    is_ulnar = eye_slant_right
                elif resolved_hand == "Left":
                    is_ulnar = not eye_slant_right
                else:
                    hand_uncertain = True

                if is_spiral:
                    peacock_label = (
                        SubtypeLabel.ULNAR_SPIRAL_PEACOCK
                        if is_ulnar
                        else SubtypeLabel.RADIAL_SPIRAL_PEACOCK
                    )
                else:
                    peacock_label = (
                        SubtypeLabel.ULNAR_TARGET_CENTRIC_PEACOCK
                        if is_ulnar
                        else SubtypeLabel.RADIAL_TARGET_CENTRIC_PEACOCK
                    )

                return RuleDecision(
                    subtype=peacock_label,
                    category=PrimaryCategory.WHORL,
                    confidence=float(np.clip(primary_conf * 0.65 + 0.25, 0.45, 0.90)),
                    is_uncertain=hand_uncertain,
                    reason="Hand metadata required to resolve Ulnar vs. Radial Peacock's Eye" if hand_uncertain else None,
                    forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
                    evidence={
                        "proximate_delta_dist_px": float(round(min_d_dist, 1)),
                        "proximate_ridge_count": float(round(min_ridge_count, 2)),
                        "farthest_delta_dist_px": float(round(max_d_dist, 1)),
                        "delta_distance_ratio": float(round(max_d_dist / max(min_d_dist, 1.0), 2)),
                        "is_spiral_eye": is_spiral,
                        "resolved_hand": resolved_hand,
                        "core_geom": core_geom,
                    },
                )

        # 6. COMPOSITE WHORL (Severely asymmetric delta elevations)
        if num_deltas >= 2:
            d1, d2 = deltas[0], deltas[1]
            delta_vert_diff = abs(d1.y - d2.y)
            if delta_vert_diff >= 65.0:
                return RuleDecision(
                    subtype=SubtypeLabel.COMPOSITE_WHORL,
                    category=PrimaryCategory.WHORL,
                    confidence=float(np.clip(primary_conf * 0.60 + 0.28, 0.50, 0.88)),
                    is_uncertain=False,
                    reason=None,
                    forensic_verifiability=ForensicVerifiability.SEMI_QUANTITATIVE,
                    evidence={
                        "delta_vert_diff_px": float(round(delta_vert_diff, 1)),
                        "num_deltas": num_deltas,
                    },
                )

        # 7. CORE GEOMETRY & TOPOLOGY EVALUATION
        if num_cores >= 1:
            core_ref = cores[0]
        else:
            m_pts = np.nonzero(orientation_res.mask)
            my = float(np.mean(m_pts[0])) if m_pts[0].size > 0 else 128.0
            mx = float(np.mean(m_pts[1])) if m_pts[1].size > 0 else 128.0
            core_ref = SingularPoint(
                x=mx, y=my, kind="core", confidence=0.5,
                poincare_index=0.5, filter_response=0.5,
                is_truncated=False, feature_visible=True,
            )

        core_geom = compute_core_streamline_geometry(core_ref, orientation_res, wavelength)
        is_imploding = core_geom["is_imploding"]
        is_spiral = core_geom["is_spiral"]
        is_elongated = core_geom["is_elongated"]

        if is_imploding:
            label = SubtypeLabel.IMPLODING_WHORL
            verifiability = ForensicVerifiability.SEMI_QUANTITATIVE
        elif is_elongated:
            label = (
                SubtypeLabel.ELONGATED_WHORL_SPIRAL
                if is_spiral
                else SubtypeLabel.ELONGATED_WHORL_TARGET
            )
            verifiability = ForensicVerifiability.MATHEMATICALLY_VERIFIABLE
        else:
            label = (
                SubtypeLabel.SPIRAL_WHORL
                if is_spiral
                else SubtypeLabel.TARGET_CENTRIC_WHORL
            )
            verifiability = ForensicVerifiability.MATHEMATICALLY_VERIFIABLE

        confidence = float(np.clip(primary_conf * 0.70 + 0.25, 0.50, 0.95))
        is_uncertain = False
        reason = None

        if num_deltas < 2 and singular_points_res.delta_truncated:
            is_uncertain = True
            reason = "Insufficient evidence: delta truncated outside sensor FOV"
            confidence *= 0.82

        return RuleDecision(
            subtype=label,
            category=PrimaryCategory.WHORL,
            confidence=confidence,
            is_uncertain=is_uncertain,
            reason=reason,
            forensic_verifiability=verifiability,
            evidence={
                "core_geom": core_geom,
                "num_cores": num_cores,
                "num_deltas": num_deltas,
                "delta_truncated": singular_points_res.delta_truncated,
            },
        )
