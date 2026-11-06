"""
Unified Biometric Quality and Damage Detection Engine.
Integrates Biometric Image Quality Assessment (IQA) and Damage & Defect Segmentation:
- Composite Overall Quality Score Q in [0, 100].
- 4-Tier classification: 'Excellent' (>=80), 'Good' (60-79), 'Fair' (40-59), 'Poor' (<40).
- Usable Foreground Area % and Defect Categorization.
- Binary Damage Mask and forensic quality warning generation.
- Acceptability gate for downstream classification (Q >= 40 and Usable Area >= 40%).
Optimized for Apple Silicon (macOS) with target latency < 50ms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import os
import cv2
import numpy as np
import torch

from src.features.orientation import (
    OrientationResult,
    compute_orientation_field,
    to_grayscale_numpy,
)
from .assessment import (
    BiometricQualityResult,
    assess_biometric_quality,
    classify_quality_tier,
)
from .damage import (
    DamageSegmentationResult,
    segment_damage_and_defects,
)


@dataclass
class QualityReport:
    """
    Unified forensic quality and damage report for a fingerprint impression.
    """
    overall_score: float                     # Composite Quality Score Q in [0.0, 100.0]
    quality_tier: str                        # 'Excellent' | 'Good' | 'Fair' | 'Poor'
    usable_area_percentage: float            # Undamaged foreground / impression bounding box in [0.0, 100.0]
    metrics: Dict[str, float]                # Dictionary of individual biometric quality metrics
    damage_detected: bool                    # True if verified physical/synthetic damage detected
    damage_types: List[str]                  # List of defect categories detected
    damage_mask: np.ndarray                  # 2D binary numpy array (H, W), dtype uint8 {0, 1}
    quality_warning: str                     # Forensic summary warning string
    is_acceptable_for_classification: bool   # False if overall_score < 40 or usable_area_percentage < 40%
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def damaged_pixels(self) -> int:
        """Number of defective / damaged pixels in damage_mask."""
        return int(np.count_nonzero(self.damage_mask))

    @property
    def undamaged_pixels(self) -> int:
        """Number of intact foreground pixels."""
        fg_mask = self.details.get("foreground_mask")
        if fg_mask is not None:
            return int(np.count_nonzero((fg_mask > 0) & (self.damage_mask == 0)))
        return int(np.count_nonzero(self.damage_mask == 0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert quality report to serializable dictionary."""
        return {
            "overall_score": float(round(self.overall_score, 2)),
            "quality_tier": self.quality_tier,
            "usable_area_percentage": float(round(self.usable_area_percentage, 2)),
            "damaged_pixels": self.damaged_pixels,
            "undamaged_pixels": self.undamaged_pixels,
            "metrics": {k: float(round(v, 4)) for k, v in self.metrics.items()},
            "damage_detected": self.damage_detected,
            "damage_types": self.damage_types,
            "quality_warning": self.quality_warning,
            "is_acceptable_for_classification": self.is_acceptable_for_classification,
            "details": self.details,
        }

    def create_damage_overlay(
        self,
        image_input: Union[np.ndarray, Any],
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Render a 3-channel RGB overlay showing:
        - Original grayscale impression as background.
        - Segmented damage mask highlighted in vibrant red/yellow.
        - Usable foreground area boundary in cyan.

        Returns:
            np.ndarray: RGB image (H, W, 3), dtype uint8.
        """
        gray = to_grayscale_numpy(image_input).astype(np.uint8)
        h, w = gray.shape

        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = color_img.copy()

        # Highlight damage mask in vibrant red (RGB: [255, 30, 30])
        dmg = self.damage_mask > 0
        if np.any(dmg):
            overlay[dmg] = [235, 45, 45]
            color_img = cv2.addWeighted(overlay, alpha, color_img, 1.0 - alpha, 0)

            # Draw thin yellow contour around damage patches
            dmg_u8 = dmg.astype(np.uint8) * 255
            contours, _ = cv2.findContours(dmg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color_img, contours, -1, (255, 220, 0), 1)

        # Draw cyan contour around foreground impression boundary
        fg_mask = self.details.get("foreground_mask")
        if fg_mask is not None and np.any(fg_mask > 0):
            fg_u8 = (fg_mask > 0).astype(np.uint8) * 255
            fg_contours, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color_img, fg_contours, -1, (0, 220, 255), 1)

        return color_img


def _generate_quality_warning(
    overall_score: float,
    usable_area_percentage: float,
    damage_detected: bool,
    damage_types: List[str],
    metrics: Dict[str, float],
    is_acceptable: bool,
) -> str:
    """Generate concise, actionable forensic summary warning string."""
    warnings: List[str] = []

    if not is_acceptable:
        if overall_score < 40.0 and usable_area_percentage < 40.0:
            warnings.append("Severe overall degradation (score < 40 and usable area < 40%)")
        elif overall_score < 40.0:
            warnings.append(f"Low composite biometric quality (score {overall_score:.1f} < 40)")
        else:
            warnings.append(f"Insufficient usable foreground area ({usable_area_percentage:.1f}% < 40%)")

    # Specific defect warnings
    if damage_detected:
        if "Obliteration / Abrasion" in damage_types:
            warnings.append("High obliteration / abrasion detected in foreground region; ridge flow destroyed")
        if "Rotation / Distortion Boundary" in damage_types:
            warnings.append("Sharp rotation / distortion boundary seam detected matching altered print")
        if "Scar / Crease" in damage_types:
            warnings.append("Scar or crease disruptions traversing ridge streamlines")
        if "Sensor Truncation" in damage_types and not ("Obliteration / Abrasion" in damage_types or "Rotation / Distortion Boundary" in damage_types):
            warnings.append("Sensor truncation at boundary margins")

    # Illumination / Focus warnings
    if metrics.get("is_dry", False):
        warnings.append("Dry impression detected; faint ridges with low dynamic range")
    elif metrics.get("is_wet", False):
        warnings.append("Wet impression / ink flooding detected; valleys saturated dark")

    if metrics.get("sharpness", 100.0) < 35.0:
        warnings.append("Severe blur detected; ridge boundaries indistinct")

    if not warnings:
        return "None"
    return "; ".join(warnings)


def analyze_fingerprint_quality(
    image_input: Union[str, np.ndarray, torch.Tensor, Any],
    block_size: int = 16,
) -> QualityReport:
    """
    Unified Quality and Damage Detection Pipeline for Fingerprint Recognition.

    Args:
        image_input: Image path, 2D/3D numpy array, PyTorch tensor, or PIL Image.
        block_size: Grid stride for block-level orientation and clarity analysis (default: 16).

    Returns:
        QualityReport: Structured dataclass with composite score, tier, metrics, and defect mask.
    """
    # 1. Resolve image input to grayscale numpy array
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Fingerprint image file not found: {image_input}")
        gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Failed to read image from path: {image_input}")
        gray = gray.astype(np.float32)
    else:
        gray = to_grayscale_numpy(image_input)

    h, w = gray.shape

    # 2. Degenerate Input Guard (uniform white, uniform black, zero variance)
    img_var = float(np.var(gray))
    if img_var < 1e-3 or h < 8 or w < 8:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        metrics = {
            "clarity": 0.0,
            "coherence": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "sharpness": 0.0,
            "frequency_continuity": 0.0,
            "usable_area": 0.0,
            "dynamic_range": 0.0,
            "dryness_ratio": 1.0 if float(np.mean(gray)) > 128 else 0.0,
            "wetness_ratio": 1.0 if float(np.mean(gray)) <= 128 else 0.0,
            "laplacian_variance": 0.0,
            "mean_gradient": 0.0,
        }
        return QualityReport(
            overall_score=0.0,
            quality_tier="Poor",
            usable_area_percentage=0.0,
            metrics=metrics,
            damage_detected=False,
            damage_types=[],
            damage_mask=empty_mask,
            quality_warning="Degenerate or blank input image with zero variance",
            is_acceptable_for_classification=False,
            details={"foreground_mask": empty_mask},
        )

    # 3. Continuous Orientation Field & Foreground Segmentation
    orientation_res = compute_orientation_field(gray, block_size=block_size)

    # 4. Defect and Damage Segmentation
    damage_res = segment_damage_and_defects(
        gray, orientation_res=orientation_res, block_size=block_size
    )

    # 5. Biometric Quality Assessment (IQA)
    quality_res = assess_biometric_quality(
        gray,
        orientation_res=orientation_res,
        usable_area_percentage=damage_res.usable_area_percentage,
        block_size=block_size,
    )

    # 6. Quality Metrics Aggregation
    metrics = {
        "clarity": quality_res.clarity_score,
        "coherence": quality_res.coherence_score,
        "contrast": quality_res.contrast_score,
        "saturation": quality_res.contrast_score,
        "sharpness": quality_res.sharpness_score,
        "frequency_continuity": quality_res.frequency_continuity_score,
        "usable_area": damage_res.usable_area_percentage,
        "usable_foreground_ratio": damage_res.usable_foreground_ratio,
        "dynamic_range": quality_res.dynamic_range,
        "dryness_ratio": quality_res.dryness_ratio,
        "wetness_ratio": quality_res.wetness_ratio,
        "laplacian_variance": quality_res.laplacian_variance,
        "mean_gradient": quality_res.mean_gradient,
    }

    # 7. Acceptability Gate
    # Image is unacceptable if overall score < 40 or usable area < 40%
    is_acceptable = bool(
        (quality_res.overall_score >= 40.0) and (damage_res.usable_area_percentage >= 40.0)
    )

    # 8. Generate Forensic Quality Warning
    warning_str = _generate_quality_warning(
        overall_score=quality_res.overall_score,
        usable_area_percentage=damage_res.usable_area_percentage,
        damage_detected=damage_res.damage_detected,
        damage_types=damage_res.damage_types,
        metrics=metrics,
        is_acceptable=is_acceptable,
    )

    details = {
        "foreground_mask": orientation_res.mask,
        "clarity_map": quality_res.clarity_map,
        "freq_map": quality_res.freq_map,
        "defect_details": damage_res.defect_details,
        "is_dry": quality_res.is_dry,
        "is_wet": quality_res.is_wet,
    }

    return QualityReport(
        overall_score=quality_res.overall_score,
        quality_tier=quality_res.quality_tier,
        usable_area_percentage=damage_res.usable_area_percentage,
        metrics=metrics,
        damage_detected=damage_res.damage_detected,
        damage_types=damage_res.damage_types,
        damage_mask=damage_res.damage_mask,
        quality_warning=warning_str,
        is_acceptable_for_classification=is_acceptable,
        details=details,
    )
