"""
Damage and Defect Segmentation Engine for Fingerprint Recognition.
Detects and localizes physical or synthetic degradation:
1. Ridge Flow Disruption: sudden orientation discontinuities, shear seams, and low-coherence patches.
2. Defect Categorization:
   - 'Scar / Crease' (narrow linear disruptions).
   - 'Obliteration / Abrasion' (amorphous degraded regions).
   - 'Rotation / Distortion Boundary' (sharp circular or shear seams matching CR/Zcut).
   - 'Sensor Truncation' (ridge flow truncated at sensor boundaries).
3. Usable Foreground Area %: ratio of valid, undamaged foreground ridges to impression bounding box.
4. Binary Damage Mask (0 = intact, 1 = damaged / defective).
Optimized for Apple Silicon with vectorized NumPy and OpenCV routines.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
import numpy as np

from src.features.orientation import (
    OrientationResult,
    compute_orientation_field,
    compute_orientation_curvature,
    to_grayscale_numpy,
)


@dataclass
class DamageSegmentationResult:
    """
    Structured container for damage segmentation and defect classification.
    """
    damage_mask: np.ndarray             # 2D binary numpy array {0, 1} (H, W), dtype uint8
    usable_area_percentage: float      # Ratio of undamaged foreground to bounding box in [0.0, 100.0]
    usable_foreground_ratio: float     # Ratio of undamaged foreground to total foreground in [0.0, 1.0]
    damage_detected: bool              # True if meaningful damage / defects detected
    damage_types: List[str]            # List of detected defect strings
    damaged_pixels: int                # Number of damaged foreground pixels
    undamaged_pixels: int              # Number of intact foreground pixels
    total_foreground_pixels: int       # Total foreground impression pixels
    bounding_box_area: int             # Area of foreground bounding box
    defect_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to serializable dictionary."""
        return {
            "usable_area_percentage": float(round(self.usable_area_percentage, 2)),
            "usable_foreground_ratio": float(round(self.usable_foreground_ratio, 4)),
            "damage_detected": self.damage_detected,
            "damage_types": self.damage_types,
            "damaged_pixels": self.damaged_pixels,
            "undamaged_pixels": self.undamaged_pixels,
            "total_foreground_pixels": self.total_foreground_pixels,
            "bounding_box_area": self.bounding_box_area,
            "defect_details": self.defect_details,
        }


def detect_ridge_flow_disruptions(
    img: np.ndarray,
    orientation_res: OrientationResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Detect sudden orientation discontinuities, shear seams, and low-coherence patches.
    Uses multi-scale orientation fields (fine vs. coarse smoothing) and orientation curvature.

    Returns:
        Tuple:
            - raw_disruption_mask: np.ndarray (H, W) binary {0, 1}
            - low_coh_mask: np.ndarray (H, W) binary {0, 1}
            - seam_mask: np.ndarray (H, W) binary {0, 1}
            - interior_mask: np.ndarray (H, W) binary {0, 1}
            - angle_diff: np.ndarray (H, W) float32 in radians
            - local_std: np.ndarray (H, W) float32 local standard deviation
            - fg_std: float 75th percentile interior local std
    """
    gray = to_grayscale_numpy(img)
    h, w = gray.shape
    total_area = h * w
    fg = orientation_res.mask
    coh = orientation_res.coherence

    zeros_u8 = np.zeros((h, w), dtype=np.uint8)
    zeros_f32 = np.zeros((h, w), dtype=np.float32)

    if np.count_nonzero(fg) == 0:
        return zeros_u8, zeros_u8, zeros_u8, zeros_u8, zeros_f32, zeros_f32, 0.0

    # Local mean and standard deviation filter
    k_box = (15, 15)
    local_mean = cv2.boxFilter(gray, cv2.CV_32F, k_box)
    local_sq_mean = cv2.boxFilter(gray**2, cv2.CV_32F, k_box)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0.0))

    # Detect blank background paper and untouched sensor glass (low variance & extreme intensity)
    is_bg = (local_std < 16.0) & ((local_mean > 165.0) | (local_mean < 30.0))
    active_fg = (fg > 0) & (~is_bg)

    if np.count_nonzero(active_fg) == 0:
        return zeros_u8, zeros_u8, zeros_u8, zeros_u8, zeros_f32, zeros_f32, 0.0

    # Interior mask: exclude boundary margins where ridges naturally taper into the sensor edge
    margin_px = max(4, int(min(h, w) * 0.05))
    dist = cv2.distanceTransform(active_fg.astype(np.uint8), cv2.DIST_L2, 3)
    interior = ((dist >= margin_px) & (active_fg > 0)).astype(np.uint8)

    # Exclude extreme frame borders (e.g. 512x512 NIST header text / margins or sensor chip rims)
    if h >= 400 and w >= 400:
        border_y_top = max(48, int(h * 0.10))
        border_y_bot = max(65, int(h * 0.13))
        border_x = max(42, int(w * 0.085))
        interior[:border_y_top, :] = 0
        interior[-border_y_bot:, :] = 0
        interior[:, :border_x] = 0
        interior[:, -border_x:] = 0
    else:
        border_x = max(10, int(w * 0.08))
        border_y = max(10, int(h * 0.08))
        interior[:border_y, :] = 0
        interior[-border_y:, :] = 0
        interior[:, :border_x] = 0
        interior[:, -border_x:] = 0

    if np.count_nonzero(interior) == 0:
        return zeros_u8, zeros_u8, zeros_u8, interior, zeros_f32, local_std, 0.0

    # Baseline coherence of healthy ridges in interior impression
    p75_coh = float(np.percentile(coh[interior > 0], 75))
    fg_std = float(np.percentile(local_std[interior > 0], 75)) if np.any(interior > 0) else 40.0

    # 1. Low-coherence or collapsed-contrast degradation (Obliteration, abrasion, severe smudging)
    low_std_mask = (local_std < min(14.0, fg_std * 0.25)) & (interior > 0)
    coh_threshold = min(0.32, p75_coh * 0.45)
    low_coh_mask = (((coh < coh_threshold) & (interior > 0)) | low_std_mask).astype(np.uint8)

    # 2. Orientation Discontinuity (Cuts, scars, central rotation seams, Z-cut shear seams)
    coarse_sigma = max(4.0, float(h) / 80.0)
    vx_coarse = cv2.GaussianBlur(orientation_res.vx, (0, 0), sigmaX=coarse_sigma, sigmaY=coarse_sigma)
    vy_coarse = cv2.GaussianBlur(orientation_res.vy, (0, 0), sigmaX=coarse_sigma, sigmaY=coarse_sigma)
    theta_coarse = np.mod(0.5 * np.arctan2(vy_coarse, vx_coarse) + np.pi / 2.0, np.pi)

    d2 = 2.0 * (orientation_res.theta - theta_coarse)
    angle_diff = 0.5 * np.abs(np.arctan2(np.sin(d2), np.cos(d2)))

    seam_angle_thresh = np.deg2rad(28.0)
    seam_coh_thresh = p75_coh * 0.85
    seam_mask = (
        (angle_diff > seam_angle_thresh)
        & (coh < seam_coh_thresh)
        & (interior > 0)
    ).astype(np.uint8)

    raw_disruption = cv2.bitwise_or(low_coh_mask, seam_mask)

    # Bridge small gaps along seams and scars with morphological closing
    # Note: Avoid MORPH_OPEN which erases thin lines/scars and rotation boundaries
    k_size = 3 if total_area < 30000 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    closed_disruption = cv2.morphologyEx(raw_disruption, cv2.MORPH_CLOSE, kernel)

    # Filter out isolated salt-and-pepper noise components (< 8 px) without destroying thin lines
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(closed_disruption)
    clean_disruption = np.zeros((h, w), dtype=np.uint8)
    for lbl in range(1, num_l):
        if stats[lbl, cv2.CC_STAT_AREA] >= 8:
            clean_disruption[labels == lbl] = 1

    return clean_disruption, low_coh_mask, seam_mask, interior, angle_diff, local_std, fg_std


def categorize_defects(
    disruption_mask: np.ndarray,
    img: np.ndarray,
    orientation_res: OrientationResult,
    interior_mask: np.ndarray,
    angle_diff: np.ndarray,
    local_std: Optional[np.ndarray] = None,
    fg_std: float = 40.0,
) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """
    Classify connected components of ridge disruptions into standardized defect categories:
    - 'Scar / Crease' (narrow linear disruptions).
    - 'Obliteration / Abrasion' (amorphous degraded regions).
    - 'Rotation / Distortion Boundary' (sharp circular or shear seams matching CR/Zcut).
    - 'Sensor Truncation' (ridge flow truncated at sensor boundaries).

    Returns:
        Tuple:
            - damage_types: List[str]
            - verified_damage_mask: np.ndarray (H, W) binary {0, 1}
            - defect_details: Dict[str, Any]
    """
    h, w = img.shape
    total_area = h * w
    fg = orientation_res.mask
    coh = orientation_res.coherence

    damage_types: List[str] = []
    verified_mask = np.zeros((h, w), dtype=np.uint8)
    details: Dict[str, Any] = {
        "num_defect_components": 0,
        "scar_crease_components": 0,
        "obliteration_components": 0,
        "rotation_distortion_components": 0,
        "sensor_truncated_sides": [],
    }

    # 1. Check Sensor Truncation at Image Margins (active high-contrast ridge flow hitting frame borders)
    top_active = float(np.mean((fg[:2, :] > 0) & (local_std[:2, :] > 16.0))) if local_std is not None else float(np.mean(fg[:2, :] > 0))
    bottom_active = float(np.mean((fg[-2:, :] > 0) & (local_std[-2:, :] > 16.0))) if local_std is not None else float(np.mean(fg[-2:, :] > 0))
    left_active = float(np.mean((fg[:, :2] > 0) & (local_std[:, :2] > 16.0))) if local_std is not None else float(np.mean(fg[:, :2] > 0))
    right_active = float(np.mean((fg[:, -2:] > 0) & (local_std[:, -2:] > 16.0))) if local_std is not None else float(np.mean(fg[:, -2:] > 0))

    truncated_sides = []
    if top_active >= 0.15:
        truncated_sides.append("top")
    if bottom_active >= 0.15:
        truncated_sides.append("bottom")
    if left_active >= 0.15:
        truncated_sides.append("left")
    if right_active >= 0.15:
        truncated_sides.append("right")

    border_active_ratio = (
        np.sum((fg[0, :] > 0) & (local_std[0, :] > 16.0 if local_std is not None else True))
        + np.sum((fg[-1, :] > 0) & (local_std[-1, :] > 16.0 if local_std is not None else True))
        + np.sum((fg[:, 0] > 0) & (local_std[:, 0] > 16.0 if local_std is not None else True))
        + np.sum((fg[:, -1] > 0) & (local_std[:, -1] > 16.0 if local_std is not None else True))
    ) / max(2 * (h + w), 1)

    if len(truncated_sides) > 0 or border_active_ratio >= 0.14:
        damage_types.append("Sensor Truncation")
        details["sensor_truncated_sides"] = truncated_sides

    # 2. Connected Component Analysis of Disruption Mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(disruption_mask)

    # Scale-adaptive minimum area thresholds
    min_comp_area = max(12, int(total_area * 0.002))
    min_obl_area = max(50, int(total_area * 0.006))

    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    scar_count = 0
    obl_count = 0
    rot_count = 0

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_comp_area:
            continue

        comp_mask = (labels == lbl)
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        min_dim = min(bw, bh)

        mean_coh_comp = float(np.mean(coh[comp_mask]))
        mean_adiff_comp = float(np.rad2deg(np.mean(angle_diff[comp_mask])))
        p80_adiff_comp = float(np.rad2deg(np.percentile(angle_diff[comp_mask], 80))) if np.any(comp_mask) else 0.0
        mean_std_comp = float(np.mean(local_std[comp_mask])) if local_std is not None else 100.0

        # Boundary ring surrounding candidate defect (must be surrounded by ridges)
        dil = cv2.morphologyEx(comp_mask.astype(np.uint8), cv2.MORPH_DILATE, k_dil)
        ring = (dil - comp_mask.astype(np.uint8)) & fg
        ring_coh = float(np.mean(coh[ring > 0])) if np.count_nonzero(ring) > 0 else 0.0

        is_defect = False

        # Category A: Scar / Crease (narrow linear disruptions traversing ridges)
        max_thickness = max(8, int(h * 0.035))
        if aspect_ratio >= 2.5 and min_dim <= max_thickness and (mean_adiff_comp > 20.0 or (mean_std_comp < min(25.0, fg_std * 0.45) and ring_coh >= 0.28)):
            scar_count += 1
            is_defect = True

        # Category B: Obliteration / Abrasion (amorphous degraded regions surrounded by ridges)
        if area >= min_obl_area and ring_coh >= 0.28:
            if (mean_std_comp < min(20.0, fg_std * 0.35)) or (mean_coh_comp < 0.28 and mean_std_comp < fg_std * 0.60):
                obl_count += 1
                is_defect = True

        # Category C: Rotation / Distortion Boundary (CR circular seams and Zcut shear boundaries)
        if (mean_adiff_comp > 20.0 or p80_adiff_comp > 28.0) and (aspect_ratio >= 1.5 or area >= min_obl_area):
            rot_count += 1
            is_defect = True

        if is_defect:
            verified_mask[comp_mask] = 1

    details["num_defect_components"] = scar_count + obl_count + rot_count
    details["scar_crease_components"] = scar_count
    details["obliteration_components"] = obl_count
    details["rotation_distortion_components"] = rot_count

    if scar_count > 0 and "Scar / Crease" not in damage_types:
        damage_types.append("Scar / Crease")
    if obl_count > 0 and "Obliteration / Abrasion" not in damage_types:
        damage_types.append("Obliteration / Abrasion")
    if rot_count > 0 and "Rotation / Distortion Boundary" not in damage_types:
        damage_types.append("Rotation / Distortion Boundary")

    return damage_types, verified_mask, details


def segment_damage_and_defects(
    img: Union[np.ndarray, Any],
    orientation_res: Optional[OrientationResult] = None,
    block_size: int = 16,
) -> DamageSegmentationResult:
    """
    End-to-end defect segmentation and usable foreground area quantification.

    Returns:
        DamageSegmentationResult containing binary damage mask, defect categories, and metrics.
    """
    gray = to_grayscale_numpy(img)
    h, w = gray.shape

    if orientation_res is None:
        orientation_res = compute_orientation_field(gray, block_size=block_size)

    fg = orientation_res.mask
    fg_pixels = int(np.count_nonzero(fg))

    if fg_pixels == 0:
        return DamageSegmentationResult(
            damage_mask=np.zeros((h, w), dtype=np.uint8),
            usable_area_percentage=0.0,
            usable_foreground_ratio=0.0,
            damage_detected=False,
            damage_types=[],
            damaged_pixels=0,
            undamaged_pixels=0,
            total_foreground_pixels=0,
            bounding_box_area=h * w,
            defect_details={"warning": "Empty foreground impression"},
        )

    # Bounding box of impression foreground
    ys, xs = np.where(fg > 0)
    bbox_h = int(np.max(ys) - np.min(ys) + 1)
    bbox_w = int(np.max(xs) - np.min(xs) + 1)
    bbox_area = max(bbox_h * bbox_w, 1)

    # 1. Detect ridge disruptions
    clean_disruption, low_coh_mask, seam_mask, interior, angle_diff, local_std, fg_std = detect_ridge_flow_disruptions(
        gray, orientation_res
    )

    # 2. Categorize defects and filter verified damage mask
    damage_types, damage_mask, defect_details = categorize_defects(
        disruption_mask=clean_disruption,
        img=gray,
        orientation_res=orientation_res,
        interior_mask=interior,
        angle_diff=angle_diff,
        local_std=local_std,
        fg_std=fg_std,
    )

    # Damage mask must strictly lie within foreground
    damage_mask = (damage_mask & fg).astype(np.uint8)

    damaged_pixels = int(np.count_nonzero(damage_mask))
    undamaged_pixels = int(np.count_nonzero((fg > 0) & (damage_mask == 0)))

    # Usable Foreground Area % (relative to total impression bounding box)
    usable_area_percentage = float(np.clip(100.0 * undamaged_pixels / bbox_area, 0.0, 100.0))
    # Usable Foreground Ratio (relative to actual foreground mask)
    usable_foreground_ratio = float(undamaged_pixels / max(fg_pixels, 1))

    # Damage detected flag: True if verified damage pixels exceed noise threshold or defect types found
    min_damage_threshold = max(20, int(gray.size * 0.002))
    damage_detected = bool(
        (damaged_pixels >= min_damage_threshold and len(damage_types) > 0)
        or any(t in damage_types for t in ["Obliteration / Abrasion", "Rotation / Distortion Boundary", "Scar / Crease"])
    )

    return DamageSegmentationResult(
        damage_mask=damage_mask,
        usable_area_percentage=usable_area_percentage,
        usable_foreground_ratio=usable_foreground_ratio,
        damage_detected=damage_detected,
        damage_types=damage_types,
        damaged_pixels=damaged_pixels,
        undamaged_pixels=undamaged_pixels,
        total_foreground_pixels=fg_pixels,
        bounding_box_area=bbox_area,
        defect_details=defect_details,
    )
