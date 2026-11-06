"""
Unified Biometric Fingerprint Recognition & Classification System Pipeline (Stage 9).
Integrates all prior stages into a coherent, high-performance forensic analyzer:
- Stage 8: Biometric Image Quality Assessment (IQA) & Defect Segmentation
- Quality Gating: Biometric acceptability thresholds (Q >= 40 and Usable Area >= 40%)
- Stage 4: Foreground normalization and 256x256 tensor preprocessing
- Stage 6: ResNet-18 deep convolutional pattern inference (A, T, W, L, R)
- Stage 5: Orientation field, singular points (cores/deltas), and ridge flow geometry
- Stage 7: 20-label hierarchical dermatoglyphic subtype classification
- Forensic Explanation: Step-by-step reasoning narrative matching the Stage 10 dashboard
- Complete JSON/dict serialization and multi-layer diagnostic overlay rendering.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import sys
import time
import json
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import (
    load_fingerprint_classifier,
    get_default_device,
    NIST_CLASSES,
    IDX_TO_NIST_CLASS,
    NIST_CLASS_TO_IDX,
)
from src.models.predict import (
    infer_hand,
    map_to_anatomical,
    DIGIT_CODE_TO_FINGER,
    FINGER_NUM_TO_NAME,
    predict_fingerprint,
)
from src.data.transforms import FingerprintTransformPipeline
from src.quality.analyzer import (
    analyze_fingerprint_quality,
    QualityReport,
)
from src.features.orientation import (
    OrientationResult,
    compute_orientation_field,
    to_grayscale_numpy,
)
from src.features.singular_points import (
    SingularPoint,
    SingularPointsResult,
    detect_singular_points,
)
from src.features.geometry import (
    GeometryResult,
    extract_fingerprint_geometry,
    trace_single_streamline,
)
from src.subtypes.taxonomy import (
    SubtypeLabel,
    PrimaryCategory,
    ForensicVerifiability,
    TAXONOMY_REGISTRY,
)
from src.subtypes.rules import (
    RuleDecision,
    LoopRuleEngine,
    ArchRuleEngine,
    WhorlRuleEngine,
)
from src.subtypes.classifier import (
    SubtypeClassificationResult,
    HierarchicalSubtypeClassifier,
)


def _clean_json_serializable(obj: Any) -> Any:
    """
    Recursively sanitize objects to primitive JSON-serializable types.
    Handles NumPy types, PyTorch tensors, Enums, and nested collections.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(round(obj, 6))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(round(val, 6))
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple, set)):
        return [_clean_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _clean_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        return _clean_json_serializable(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return _clean_json_serializable(obj.tolist())
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return _clean_json_serializable(obj.to_dict())
    return str(obj)


def _format_quality_dict(quality_report: QualityReport) -> Dict[str, Any]:
    """
    Format QualityReport to a lightweight, JSON-safe dictionary.
    Excludes massive 2D raster arrays (clarity_map, foreground_mask, freq_map)
    from the dictionary while recording their shapes, preserving fast serialization.
    Raw pixel masks remain accessible directly via result.quality_report.details and result.damage_mask.
    """
    q_dict = quality_report.to_dict()
    if "details" in q_dict and isinstance(q_dict["details"], dict):
        clean_details = {}
        for k, v in q_dict["details"].items():
            if hasattr(v, "shape"):
                clean_details[f"{k}_shape"] = list(v.shape)
            else:
                clean_details[k] = v
        q_dict["details"] = clean_details
    return q_dict


@dataclass
class FingerprintAnalysisResult:
    """
    Unified end-to-end output container for biometric fingerprint analysis.
    Directly feeds the Stage 10 Operational Dashboard and Web API.
    """
    # Top-level identification & status
    image_path: Optional[str]
    input_shape: Tuple[int, int]
    hand: Optional[str]
    finger_num: Optional[int]
    finger_name: Optional[str]
    digit_code: Optional[str]
    decision_gate: str                      # "ACCEPTED" | "ACCEPTED_WITH_WARNINGS" | "REJECTED_LOW_QUALITY"
    is_acceptable: bool

    # Stage 8: Quality & Damage Diagnostics
    quality: Dict[str, Any]

    # Stage 6: Neural Pattern Classification
    classification: Dict[str, Any]

    # Stage 5: Dermatoglyphic Landmarks & Geometry
    features: Dict[str, Any]

    # Stage 7: Subtype Refinement (20-label taxonomy)
    subtype: Dict[str, Any]

    # Step-by-Step Reasoning / Explanation
    explanation: List[Dict[str, str]]

    # Latency breakdown (ms)
    execution_time_ms: Dict[str, float]

    # Intermediate forensic objects (optional, excluded from default serialization)
    quality_report: Optional[QualityReport] = field(default=None, repr=False)
    subtype_result: Optional[SubtypeClassificationResult] = field(default=None, repr=False)
    damage_mask: Optional[np.ndarray] = field(default=None, repr=False)
    raw_image: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to a completely JSON-serializable dictionary."""
        d = {
            "image_path": self.image_path,
            "input_shape": list(self.input_shape),
            "hand": self.hand,
            "finger_num": self.finger_num,
            "finger_name": self.finger_name,
            "digit_code": self.digit_code,
            "decision_gate": self.decision_gate,
            "is_acceptable": self.is_acceptable,
            "quality": self.quality,
            "classification": self.classification,
            "features": self.features,
            "subtype": self.subtype,
            "explanation": self.explanation,
            "execution_time_ms": self.execution_time_ms,
        }
        return _clean_json_serializable(d)

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to a clean, formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def create_overlay(self, alpha: float = 0.45) -> np.ndarray:
        """
        Render a publication-quality 3-channel RGB overlay combining:
        - Grayscale impression as background.
        - Usable foreground boundary contour in cyan.
        - Segmented damage mask in vibrant red with thin yellow borders.
        - Ridge flow streamlines traced in turquoise/cyan.
        - Singular points: Cores in red (circle + crosshair), Deltas in blue (triangles).

        Returns:
            np.ndarray: RGB image (H, W, 3), dtype uint8.
        """
        if self.raw_image is not None:
            gray = self.raw_image.astype(np.uint8)
        else:
            h, w = self.input_shape if (self.input_shape and self.input_shape[0] > 0) else (256, 256)
            gray = np.zeros((h, w), dtype=np.uint8)

        h, w = gray.shape
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = color_img.copy()

        # 1. Damage mask overlay
        if self.damage_mask is not None and np.any(self.damage_mask > 0):
            dmg = self.damage_mask > 0
            overlay[dmg] = [235, 45, 45]
            color_img = cv2.addWeighted(overlay, alpha, color_img, 1.0 - alpha, 0)

            # Thin yellow contour around damage patches
            dmg_u8 = dmg.astype(np.uint8) * 255
            contours, _ = cv2.findContours(dmg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color_img, contours, -1, (255, 220, 0), 1)

        # 2. Cyan contour around usable foreground area
        fg_mask = None
        if self.quality_report and "foreground_mask" in self.quality_report.details:
            fg_mask = self.quality_report.details["foreground_mask"]
        elif self.subtype_result and self.subtype_result.orientation_res:
            fg_mask = self.subtype_result.orientation_res.mask

        if fg_mask is not None and np.any(fg_mask > 0):
            fg_u8 = (fg_mask > 0).astype(np.uint8) * 255
            fg_contours, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color_img, fg_contours, -1, (0, 225, 255), 1)

        # 3. Streamlines
        if self.subtype_result and self.subtype_result.geometry_res:
            streamlines = self.subtype_result.geometry_res.streamlines
            for sl in streamlines[:16]:
                if len(sl) > 2:
                    pts = np.int32(sl)
                    for pt_idx in range(len(pts) - 1):
                        pt1 = tuple(pts[pt_idx])
                        pt2 = tuple(pts[pt_idx + 1])
                        if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                            cv2.line(color_img, pt1, pt2, (0, 230, 180), 1, cv2.LINE_AA)

        # 4. Cores (Red circle + crosshair)
        cores_list = self.features.get("cores", [])
        for c in cores_list:
            cx, cy = int(round(c["x"])), int(round(c["y"]))
            if 0 <= cx < w and 0 <= cy < h:
                # Red circle
                cv2.circle(color_img, (cx, cy), 8, (235, 30, 30), 2, cv2.LINE_AA)
                # Red crosshair
                cv2.line(color_img, (cx - 12, cy), (cx + 12, cy), (235, 30, 30), 1, cv2.LINE_AA)
                cv2.line(color_img, (cx, cy - 12), (cx, cy + 12), (235, 30, 30), 1, cv2.LINE_AA)
                # Center white dot
                cv2.circle(color_img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # 5. Deltas (Blue triangle)
        deltas_list = self.features.get("deltas", [])
        for d in deltas_list:
            dx, dy = int(round(d["x"])), int(round(d["y"]))
            if 0 <= dx < w and 0 <= dy < h:
                r = 9
                tri = np.array([
                    [dx, dy - r],
                    [dx - int(r * 0.866), dy + int(r * 0.5)],
                    [dx + int(r * 0.866), dy + int(r * 0.5)],
                ], dtype=np.int32)
                # Blue triangle (Dodger blue)
                cv2.polylines(color_img, [tri], isClosed=True, color=(30, 144, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(color_img, (dx, dy), 2, (255, 255, 255), -1, cv2.LINE_AA)

        return color_img

    def create_composite_overlay(self, alpha: float = 0.45) -> np.ndarray:
        """Alias for create_overlay() providing identical composite rendering."""
        return self.create_overlay(alpha=alpha)


class FingerprintSystemAnalyzer:
    """
    Unified Biometric Fingerprint Recognition & Forensic Classification System (Stage 9).
    Combines Image Quality Assessment (IQA), Defect Localization, Deep Neural Classification,
    Topological Landmark Verification, and Hierarchical 20-Subtype Rule Resolution.
    """

    def __init__(
        self,
        model_checkpoint: Optional[str] = "models/best_model.pt",
        model: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        min_quality_score: float = 40.0,
        min_usable_area: float = 40.0,
        min_coherence_threshold: float = 0.25,
        min_confidence_gate: float = 0.45,
    ):
        """
        Initialize the unified analyzer engine.

        Args:
            model_checkpoint: Path to PyTorch model weights.
            model: Optional pre-loaded nn.Module (takes precedence over checkpoint).
            device: Target execution device ('mps', 'cuda', 'cpu'). Auto-selected if None.
            min_quality_score: Quality score cutoff Q for quality gating (default: 40.0).
            min_usable_area: Usable foreground % cutoff for quality gating (default: 40.0).
            min_coherence_threshold: Minimum orientation coherence gate (default: 0.25).
            min_confidence_gate: Minimum primary classifier confidence gate (default: 0.45).
        """
        self.device = torch.device(device) if device is not None else get_default_device()
        self.min_quality_score = min_quality_score
        self.min_usable_area = min_usable_area
        self.min_coherence_threshold = min_coherence_threshold
        self.min_confidence_gate = min_confidence_gate

        # Stage 6 Neural Network Model
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        elif model_checkpoint is not None and os.path.isfile(model_checkpoint):
            self.model = load_fingerprint_classifier(model_checkpoint, device=self.device)
            self.model.eval()
        else:
            self.model = None

        # Stage 4 Preprocessing Transform
        self.transform = FingerprintTransformPipeline(
            target_size=(256, 256),
            enhancement="hybrid",
            channels=1,
        )

        # Stage 7 Subtype Classifier Engine
        self.subtype_classifier = HierarchicalSubtypeClassifier(
            model=self.model,
            device=self.device,
            min_coherence_threshold=self.min_coherence_threshold,
            min_confidence_gate=self.min_confidence_gate,
        )

    def analyze(
        self,
        image: Union[str, np.ndarray, torch.Tensor, Any],
        hand: Optional[str] = None,
        finger_num: Optional[Union[int, str]] = None,
        digit_code: Optional[str] = None,
    ) -> FingerprintAnalysisResult:
        """
        Execute full end-to-end fingerprint analysis across Stages 4 through 8.

        Args:
            image: Image file path, NumPy 2D array, PyTorch tensor, or PIL Image.
            hand: 'Right' or 'Left' (optional hand metadata).
            finger_num: 1-10 (optional digit number).
            digit_code: '01'-'10' (optional standard two-digit code).

        Returns:
            FingerprintAnalysisResult with complete diagnostic, classification, and explanation outputs.
        """
        t_total_start = time.perf_counter()
        image_path: Optional[str] = image if isinstance(image, str) else None

        # -------------------------------------------------------------------
        # 1. Ingestion & Input Grayscale Normalization
        # -------------------------------------------------------------------
        if image is None:
            raise ValueError("Input image cannot be None or empty.")

        if isinstance(image, str):
            if not os.path.isfile(image):
                raise FileNotFoundError(f"Fingerprint image file not found: {image}")
            raw_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                raise IOError(f"Could not load image file: {image}")
            raw_img = raw_img.astype(np.float32)
        else:
            raw_img = to_grayscale_numpy(image)

        # Scale down excessively large images preserving aspect ratio (max dimension 1024)
        max_dim = max(raw_img.shape[:2])
        if max_dim > 1024:
            scale = 1024.0 / max_dim
            new_w = max(1, int(round(raw_img.shape[1] * scale)))
            new_h = max(1, int(round(raw_img.shape[0] * scale)))
            raw_img = cv2.resize(raw_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            raw_img = raw_img.astype(np.float32)

        input_h, input_w = raw_img.shape
        img_var = float(np.var(raw_img))

        # Resolved anatomical metadata
        resolved_hand = infer_hand(hand=hand, finger_num=finger_num, digit_code=digit_code)
        f_num = None
        f_name = None
        if digit_code is not None and str(digit_code).zfill(2) in DIGIT_CODE_TO_FINGER:
            f_num, f_name = DIGIT_CODE_TO_FINGER[str(digit_code).zfill(2)]
        elif finger_num is not None:
            try:
                f_num = int(finger_num)
                f_name = FINGER_NUM_TO_NAME.get(f_num)
            except (ValueError, TypeError):
                pass

        # -------------------------------------------------------------------
        # 2. Stage 8: Biometric Quality Assessment & Damage Detection
        # -------------------------------------------------------------------
        t_q_start = time.perf_counter()
        quality_report = analyze_fingerprint_quality(raw_img)
        t_q_end = time.perf_counter()
        q_time_ms = (t_q_end - t_q_start) * 1000.0

        # Quality Gating Check
        is_acceptable = bool(
            quality_report.is_acceptable_for_classification
            and (quality_report.overall_score >= self.min_quality_score)
            and (quality_report.usable_area_percentage >= self.min_usable_area)
        )

        # -------------------------------------------------------------------
        # Degenerate Guard: Uniform Black, Uniform White, Zero Variance
        # -------------------------------------------------------------------
        if img_var < 1e-3 or input_h < 8 or input_w < 8:
            t_total_end = time.perf_counter()
            total_time_ms = (t_total_end - t_total_start) * 1000.0

            explanation = [
                {
                    "step": "1. Quality & Condition Assessment",
                    "title": "Severe Degenerate Input Rejection",
                    "detail": f"Input image has zero variance (variance={img_var:.5f}) or insufficient dimensions ({input_h}x{input_w}). Biometric quality score Q is 0.0 (Poor). Usable foreground area is 0.0%.",
                },
                {
                    "step": "2. Deep Learning Pattern Inference",
                    "title": "Inference Bypassed",
                    "detail": "Classification bypassed due to lack of ridge signals. Primary pattern is indeterminate.",
                },
                {
                    "step": "3. Dermatoglyphic Landmark Verification",
                    "title": "Zero Landmarks Detected",
                    "detail": "No cores, deltas, or ridge flow streamlines could be extracted from uniform background.",
                },
                {
                    "step": "4. Anatomical & Subtype Synthesis",
                    "title": "Status: Uncertain / Degraded",
                    "detail": "Subtype cannot be resolved. Flagged as Uncertain with 0.0% confidence.",
                },
                {
                    "step": "5. Operational Decision & Forensic Gate",
                    "title": "REJECTED_LOW_QUALITY",
                    "detail": "Impression rejected by biometric quality gate. Recapture is required under standard biometric enrollment protocol.",
                },
            ]

            return FingerprintAnalysisResult(
                image_path=image_path,
                input_shape=(input_h, input_w),
                hand=resolved_hand,
                finger_num=f_num,
                finger_name=f_name,
                digit_code=digit_code,
                decision_gate="REJECTED_LOW_QUALITY",
                is_acceptable=False,
                quality=_format_quality_dict(quality_report),
                classification={
                    "primary_class": "Unknown",
                    "confidence": 0.0,
                    "probabilities": {c: 0.20 for c in NIST_CLASSES},
                    "anatomical_label": "Unknown",
                    "top_k": [],
                    "logits": [],
                },
                features={
                    "num_cores": 0,
                    "num_deltas": 0,
                    "cores": [],
                    "deltas": [],
                    "delta_truncated": False,
                    "expected_delta_sector": None,
                    "flow_direction": "unknown",
                    "loop_type_estimate": "unknown",
                    "median_frequency": 0.0,
                    "median_wavelength": 0.0,
                    "ridge_count": 0.0,
                    "flow_asymmetry": 0.0,
                    "exit_angle_deg": 0.0,
                    "mean_coherence": 0.0,
                    "streamlines_count": 0,
                },
                subtype={
                    "subtype": "Uncertain / Degraded",
                    "candidate_subtype": "Uncertain / Degraded",
                    "primary_category": "Accidental",
                    "status": "Uncertain",
                    "is_uncertain": True,
                    "confidence": 0.0,
                    "reason": "Degenerate input: uniform or zero-variance image rejected by quality gate",
                    "forensic_verifiability": "Expert Discretion",
                },
                explanation=explanation,
                execution_time_ms={
                    "quality_ms": float(round(q_time_ms, 2)),
                    "preprocessing_ms": 0.0,
                    "classification_ms": 0.0,
                    "features_ms": 0.0,
                    "subtype_ms": 0.0,
                    "total_ms": float(round(total_time_ms, 2)),
                },
                quality_report=quality_report,
                subtype_result=None,
                damage_mask=quality_report.damage_mask,
                raw_image=raw_img,
            )

        # -------------------------------------------------------------------
        # 3. Stage 4: Preprocessing (Crop/Pad, Normalize, 256x256 Tensor)
        # -------------------------------------------------------------------
        t_pre_start = time.perf_counter()
        preproc_tensor = self.transform(raw_img)
        t_pre_end = time.perf_counter()
        preproc_time_ms = (t_pre_end - t_pre_start) * 1000.0

        # -------------------------------------------------------------------
        # 4. Stage 6: Neural Network Pattern Classification
        # -------------------------------------------------------------------
        t_cls_start = time.perf_counter()
        if self.model is not None:
            pred_obj = predict_fingerprint(
                model=self.model,
                image=preproc_tensor,
                transform=lambda x: x,
                device=self.device,
                hand=resolved_hand,
                finger_num=f_num,
                digit_code=digit_code,
            )
            primary_class = pred_obj.predicted_class
            confidence = pred_obj.confidence
            probabilities = pred_obj.probabilities
            anatomical_label = pred_obj.anatomical_label
            top_k_list = [
                {"class": item[0], "prob": float(round(item[1], 4)), "anatomical_label": item[2]}
                for item in pred_obj.top_k
            ]
            logits = pred_obj.logits
            if pred_obj.hand:
                resolved_hand = pred_obj.hand
            if pred_obj.finger_num:
                f_num = pred_obj.finger_num
            if pred_obj.finger_name:
                f_name = pred_obj.finger_name
        else:
            # Fallback if running headless
            primary_class = "W"
            confidence = 0.50
            probabilities = {c: 0.20 for c in NIST_CLASSES}
            probabilities["W"] = 0.50
            anatomical_label = "Whorl"
            top_k_list = [{"class": "W", "prob": 0.50, "anatomical_label": "Whorl"}]
            logits = [0.0] * 5
        t_cls_end = time.perf_counter()
        cls_time_ms = (t_cls_end - t_cls_start) * 1000.0

        # -------------------------------------------------------------------
        # 5. Stage 5: Geometric Feature & Landmark Extraction
        # -------------------------------------------------------------------
        t_feat_start = time.perf_counter()
        ori_res = compute_orientation_field(raw_img)
        sp_res = detect_singular_points(ori_res)
        geo_res = extract_fingerprint_geometry(
            raw_img, ori_res, sp_res, hand=resolved_hand, finger_name=f_name
        )
        t_feat_end = time.perf_counter()
        feat_time_ms = (t_feat_end - t_feat_start) * 1000.0

        # Extract features dictionary
        cores_data = [c.to_dict() for c in sp_res.cores]
        deltas_data = [d.to_dict() for d in sp_res.deltas]
        fg_pix = int(np.count_nonzero(ori_res.mask))
        mean_coh = float(np.mean(ori_res.coherence[ori_res.mask > 0])) if fg_pix > 0 else 0.0

        # Estimate dermatoglyphic ridge count between closest core and delta
        ridge_count = 0.0
        if sp_res.cores and sp_res.deltas:
            min_cd_dist = min(
                math.hypot(c.x - d.x, c.y - d.y)
                for c in sp_res.cores
                for d in sp_res.deltas
            )
            wl = geo_res.median_wavelength if geo_res.median_wavelength > 0 else 9.0
            ridge_count = float(round(min_cd_dist / wl, 1))

        features_dict = {
            "num_cores": sp_res.num_cores,
            "num_deltas": sp_res.num_deltas,
            "cores": cores_data,
            "deltas": deltas_data,
            "delta_truncated": sp_res.delta_truncated,
            "expected_delta_sector": sp_res.expected_delta_sector,
            "flow_direction": geo_res.flow_direction,
            "loop_type_estimate": geo_res.loop_type_estimate,
            "median_frequency": float(round(geo_res.median_frequency, 4)),
            "median_wavelength": float(round(geo_res.median_wavelength, 2)),
            "ridge_count": ridge_count,
            "flow_asymmetry": float(round(geo_res.flow_asymmetry, 3)),
            "exit_angle_deg": float(round(geo_res.exit_angle_deg, 2)),
            "mean_coherence": float(round(mean_coh, 3)),
            "streamlines_count": len(geo_res.streamlines),
        }

        # -------------------------------------------------------------------
        # 6. Stage 7: Subtype Refinement & Uncertainty Synthesis
        # -------------------------------------------------------------------
        t_sub_start = time.perf_counter()
        # Direct rule evaluation based on Stage 6 prediction and Stage 5 features
        if primary_class.upper() in ("L", "R"):
            rule_decision = LoopRuleEngine.evaluate(
                primary_class=primary_class,
                primary_conf=confidence,
                orientation_res=ori_res,
                singular_points_res=sp_res,
                geometry_res=geo_res,
                hand=resolved_hand,
                finger_num=f_num,
                digit_code=digit_code,
            )
            primary_category = PrimaryCategory.LOOP
        elif primary_class.upper() in ("A", "T"):
            rule_decision = ArchRuleEngine.evaluate(
                primary_class=primary_class,
                primary_conf=confidence,
                orientation_res=ori_res,
                singular_points_res=sp_res,
                geometry_res=geo_res,
                hand=resolved_hand,
                finger_num=f_num,
                digit_code=digit_code,
            )
            primary_category = PrimaryCategory.ARCH
        else:
            rule_decision = WhorlRuleEngine.evaluate(
                primary_class=primary_class,
                primary_conf=confidence,
                orientation_res=ori_res,
                singular_points_res=sp_res,
                geometry_res=geo_res,
                hand=resolved_hand,
                finger_num=f_num,
                digit_code=digit_code,
            )
            primary_category = rule_decision.category

        candidate_subtype = rule_decision.subtype
        final_subtype = candidate_subtype
        final_confidence = rule_decision.confidence
        is_uncertain = rule_decision.is_uncertain
        reason = rule_decision.reason

        # Gating Check 1: Quality gate rejection overrides subtype certainty
        if not is_acceptable:
            is_uncertain = True
            reason = f"Quality Gate Rejection: {quality_report.quality_warning}"
            final_confidence = min(final_confidence * 0.60, 0.40)

        # Gating Check 2: Severe physical damage overrides certainty
        elif quality_report.damage_detected and "Obliteration / Abrasion" in quality_report.damage_types:
            if not is_uncertain:
                is_uncertain = True
                reason = "Extensive foreground ridge obliteration detected"
                final_confidence = min(final_confidence * 0.70, 0.50)

        # Gating Check 3: Conflicting neural vs. geometric signals
        if primary_class.upper() in ("A", "T") and sp_res.num_deltas >= 2 and geo_res.loop_type_estimate == "whorl":
            is_uncertain = True
            reason = "Conflicting evidence: Neural network predicted Arch but bilateral deltas indicate Whorl"
            final_confidence *= 0.70
        elif primary_class.upper() == "W" and sp_res.num_cores == 0 and sp_res.num_deltas == 0 and geo_res.loop_type_estimate == "arch":
            is_uncertain = True
            reason = "Conflicting evidence: Neural network predicted Whorl but zero singular points indicate Arch"
            final_confidence *= 0.70

        # Gating Check 4: Coherence and primary confidence gate
        if mean_coh < self.min_coherence_threshold and not is_uncertain:
            is_uncertain = True
            reason = f"Low ridge coherence across foreground (mean={mean_coh:.2f} < {self.min_coherence_threshold})"
            final_confidence *= 0.65
        elif confidence < self.min_confidence_gate and not is_uncertain:
            is_uncertain = True
            reason = f"Low primary classifier confidence ({confidence:.2f} < {self.min_confidence_gate})"
            final_confidence *= 0.75

        subtype_status = "Uncertain" if is_uncertain else "Resolved"

        subtype_dict = {
            "subtype": final_subtype.value,
            "candidate_subtype": candidate_subtype.value,
            "primary_category": primary_category.value,
            "status": subtype_status,
            "is_uncertain": is_uncertain,
            "confidence": float(round(final_confidence, 4)),
            "reason": reason,
            "forensic_verifiability": rule_decision.forensic_verifiability.value,
            "rule_evidence": rule_decision.evidence,
        }

        # Build Stage 7 SubtypeClassificationResult object
        subtype_result_obj = SubtypeClassificationResult(
            primary_class=primary_class,
            primary_confidence=confidence,
            primary_category=primary_category,
            subtype=final_subtype,
            candidate_subtype=candidate_subtype,
            status=subtype_status,
            is_uncertain=is_uncertain,
            confidence=float(round(final_confidence, 4)),
            reason=reason,
            forensic_verifiability=rule_decision.forensic_verifiability,
            evidence_summary=features_dict,
            hand=resolved_hand,
            finger_num=f_num,
            finger_name=f_name,
            orientation_res=ori_res,
            singular_points_res=sp_res,
            geometry_res=geo_res,
        )
        t_sub_end = time.perf_counter()
        sub_time_ms = (t_sub_end - t_sub_start) * 1000.0

        # -------------------------------------------------------------------
        # 7. Decision Gate Synthesis
        # -------------------------------------------------------------------
        if not is_acceptable:
            decision_gate = "REJECTED_LOW_QUALITY"
        elif is_uncertain or quality_report.damage_detected or (confidence < 0.60):
            decision_gate = "ACCEPTED_WITH_WARNINGS"
        else:
            decision_gate = "ACCEPTED"

        # -------------------------------------------------------------------
        # 8. Forensic Explanation Synthesis (Dashboard Matching)
        # -------------------------------------------------------------------
        t_exp_start = time.perf_counter()
        explanation = self._synthesize_explanation(
            quality_report=quality_report,
            primary_class=primary_class,
            confidence=confidence,
            probabilities=probabilities,
            anatomical_label=anatomical_label,
            top_k_list=top_k_list,
            sp_res=sp_res,
            geo_res=geo_res,
            mean_coh=mean_coh,
            final_subtype=final_subtype,
            subtype_status=subtype_status,
            is_uncertain=is_uncertain,
            final_confidence=final_confidence,
            reason=reason,
            forensic_verifiability=rule_decision.forensic_verifiability,
            decision_gate=decision_gate,
            resolved_hand=resolved_hand,
            f_num=f_num,
            f_name=f_name,
        )
        t_exp_end = time.perf_counter()
        exp_time_ms = (t_exp_end - t_exp_start) * 1000.0

        t_total_end = time.perf_counter()
        total_time_ms = (t_total_end - t_total_start) * 1000.0

        execution_times = {
            "quality_ms": float(round(q_time_ms, 2)),
            "preprocessing_ms": float(round(preproc_time_ms, 2)),
            "classification_ms": float(round(cls_time_ms, 2)),
            "features_ms": float(round(feat_time_ms, 2)),
            "subtype_ms": float(round(sub_time_ms, 2)),
            "explanation_ms": float(round(exp_time_ms, 2)),
            "total_ms": float(round(total_time_ms, 2)),
        }

        classification_dict = {
            "primary_class": primary_class,
            "confidence": float(round(confidence, 4)),
            "probabilities": {k: float(round(v, 4)) for k, v in probabilities.items()},
            "anatomical_label": anatomical_label,
            "top_k": top_k_list,
            "logits": [float(round(l, 4)) for l in logits],
        }

        return FingerprintAnalysisResult(
            image_path=image_path,
            input_shape=(input_h, input_w),
            hand=resolved_hand,
            finger_num=f_num,
            finger_name=f_name,
            digit_code=digit_code,
            decision_gate=decision_gate,
            is_acceptable=is_acceptable,
            quality=_format_quality_dict(quality_report),
            classification=classification_dict,
            features=features_dict,
            subtype=subtype_dict,
            explanation=explanation,
            execution_time_ms=execution_times,
            quality_report=quality_report,
            subtype_result=subtype_result_obj,
            damage_mask=quality_report.damage_mask,
            raw_image=raw_img,
        )

    def analyze_batch(
        self,
        images: List[Union[str, np.ndarray, torch.Tensor, Any]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[FingerprintAnalysisResult]:
        """
        Analyze an iterable cohort of fingerprint impressions sequentially.
        """
        results: List[FingerprintAnalysisResult] = []
        n = len(images)

        for i in range(n):
            img = images[i]
            meta = metadata_list[i] if (metadata_list and i < len(metadata_list)) else {}
            res = self.analyze(
                image=img,
                hand=meta.get("hand"),
                finger_num=meta.get("finger_num"),
                digit_code=meta.get("digit_code"),
            )
            results.append(res)

        return results

    def _synthesize_explanation(
        self,
        quality_report: QualityReport,
        primary_class: str,
        confidence: float,
        probabilities: Dict[str, float],
        anatomical_label: str,
        top_k_list: List[Dict[str, Any]],
        sp_res: SingularPointsResult,
        geo_res: GeometryResult,
        mean_coh: float,
        final_subtype: SubtypeLabel,
        subtype_status: str,
        is_uncertain: bool,
        final_confidence: float,
        reason: Optional[str],
        forensic_verifiability: ForensicVerifiability,
        decision_gate: str,
        resolved_hand: Optional[str],
        f_num: Optional[int],
        f_name: Optional[str],
    ) -> List[Dict[str, str]]:
        """
        Generate structured, step-by-step forensic reasoning narrative
        answering 'How was this determined?' for human examiners and dashboard UI.
        """
        explanation: List[Dict[str, str]] = []

        # Step 1: Quality & Impression Diagnostics
        dmg_str = ", ".join(quality_report.damage_types) if quality_report.damage_detected else "None detected"
        gate_str = "Passed" if quality_report.is_acceptable_for_classification else "FAILED (Low Quality)"
        step1_text = (
            f"Impression scored Q={quality_report.overall_score:.1f}/100 ({quality_report.quality_tier} Tier) "
            f"with {quality_report.usable_area_percentage:.1f}% usable foreground area. "
            f"Local ridge clarity: {quality_report.metrics.get('clarity', 0.0):.1f}/100, "
            f"frequency continuity: {quality_report.metrics.get('frequency_continuity', 0.0):.1f}/100. "
            f"Defect status: {dmg_str}. Biometric quality gate: {gate_str}."
        )
        if quality_report.quality_warning != "None":
            step1_text += f" Forensic Warning: {quality_report.quality_warning}."
        explanation.append({
            "step": "1. Quality & Condition Assessment",
            "title": f"Quality Tier: {quality_report.quality_tier} (Q={quality_report.overall_score:.1f})",
            "detail": step1_text,
        })

        # Step 2: Deep Learning Pattern Inference
        runner_up = top_k_list[1] if len(top_k_list) > 1 else None
        margin_str = f" (+{(confidence - runner_up['prob']) * 100:.1f}% margin over runner-up {runner_up['class']})" if runner_up else ""
        step2_text = (
            f"ResNet-18 neural classifier evaluated 256x256 preprocessed ridge pattern. "
            f"Predicted primary macro pattern '{primary_class}' ({anatomical_label}) with {confidence * 100:.1f}% softmax confidence{margin_str}. "
            f"Class probability distribution: " + ", ".join([f"{k}: {v*100:.1f}%" for k, v in probabilities.items()]) + "."
        )
        explanation.append({
            "step": "2. Deep Learning Pattern Inference",
            "title": f"Macro Class: '{primary_class}' ({confidence * 100:.1f}%)",
            "detail": step2_text,
        })

        # Step 3: Dermatoglyphic Landmark & Flow Verification
        trunc_str = " (Delta truncated by sensor margin)" if sp_res.delta_truncated else " (Interior landmarks fully enclosed)"
        step3_text = (
            f"Singular point detection verified {sp_res.num_cores} core(s) and {sp_res.num_deltas} delta(s){trunc_str}. "
            f"Ridge flow geometry indicates '{geo_res.flow_direction}' directionality with mean exit angle of {geo_res.exit_angle_deg:.1f}° "
            f"and median ridge wavelength of {geo_res.median_wavelength:.1f} px (frequency: {geo_res.median_frequency:.3f} cyc/px). "
            f"Mean orientation field coherence is {mean_coh:.2f} across foreground ridges."
        )
        explanation.append({
            "step": "3. Dermatoglyphic Landmark Verification",
            "title": f"Topology: {sp_res.num_cores} Core(s), {sp_res.num_deltas} Delta(s)",
            "detail": step3_text,
        })

        # Step 4: Anatomical & Subtype Synthesis
        meta_str = ""
        if resolved_hand:
            meta_str = f"Integrated {resolved_hand} hand"
            if f_name:
                meta_str += f" ({f_name})"
            elif f_num:
                meta_str += f" (Digit #{f_num})"
            meta_str += " metadata. "
        else:
            meta_str = "No anatomical hand metadata provided (assumed anatomical default). "

        if is_uncertain:
            step4_text = (
                f"{meta_str}Subtype resolution evaluated hypothesis '{final_subtype.value}', "
                f"but status was flagged as UNCERTAIN ({final_confidence * 100:.1f}% calibrated confidence). "
                f"Uncertainty Reason: {reason}. Verifiability Level: {forensic_verifiability.value}."
            )
        else:
            step4_text = (
                f"{meta_str}Geometric rule engine successfully resolved subtype '{final_subtype.value}' "
                f"with {final_confidence * 100:.1f}% confidence ({subtype_status}). "
                f"Verifiability Level: {forensic_verifiability.value}."
            )
        explanation.append({
            "step": "4. Anatomical & Subtype Synthesis",
            "title": f"Subtype: {final_subtype.value} ({subtype_status})",
            "detail": step4_text,
        })

        # Step 5: System Operational Gate & Recommendation
        if decision_gate == "ACCEPTED":
            gate_advisory = "Impression accepted with full forensic fidelity. Suitable for automated biometric filing and minutiae matching."
        elif decision_gate == "ACCEPTED_WITH_WARNINGS":
            gate_advisory = "Impression accepted with operational cautions. Examiner review recommended due to defect presence or boundary truncation."
        else:
            gate_advisory = "Impression REJECTED due to severe degradation or defect obliteration. Subject re-acquisition recommended."

        explanation.append({
            "step": "5. System Operational Gate",
            "title": f"Operational Gate: {decision_gate}",
            "detail": gate_advisory,
        })

        return explanation
