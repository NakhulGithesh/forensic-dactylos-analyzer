"""
Hierarchical Subtype Classification Engine (Stage 7).
Unifies Stage 6 Neural Network Pattern Classification with Stage 5 Geometric
Dermatoglyphic Analysis to determine the 20 dermatoglyphic subtypes:

Pipeline Flow:
1. Preprocessed image (Stage 4) -> Stage 6 Model predicts primary class (A, T, W, L, R).
2. Extract Stage 5 features: orientation field, coherence, cores, deltas, streamlines, truncation flags.
3. Check degradation & sensor truncation gates (coherence, missing deltas, boundary margin).
4. Apply category-specific geometric subtype discriminators from `rules.py`.
5. Calibrate overall subtype confidence and enforce explicit "Uncertain / Insufficient Evidence" status.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import (
    get_default_device,
    load_fingerprint_classifier,
)
from src.models.predict import (
    predict_fingerprint,
    predict_batch,
    infer_hand,
    DIGIT_CODE_TO_FINGER,
    FINGER_NUM_TO_NAME,
)
from src.data.transforms import FingerprintTransformPipeline
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
)
from src.subtypes.taxonomy import (
    SubtypeLabel,
    PrimaryCategory,
    ForensicVerifiability,
    TAXONOMY_REGISTRY,
    parse_subtype_label,
)
from src.subtypes.rules import (
    RuleDecision,
    LoopRuleEngine,
    ArchRuleEngine,
    WhorlRuleEngine,
)


@dataclass
class SubtypeClassificationResult:
    """Structured output container for Stage 7 Hierarchical Subtype Classification."""
    primary_class: str                       # NIST class 'A', 'T', 'W', 'L', 'R'
    primary_confidence: float                # Neural network primary pattern confidence [0, 1]
    primary_category: PrimaryCategory        # Macro pattern category (Loop, Arch, Whorl, Accidental)
    subtype: SubtypeLabel                    # Resolved or candidate subtype label
    candidate_subtype: SubtypeLabel          # Primary hypothesis before uncertainty gating
    status: str                              # "Resolved" or "Uncertain"
    is_uncertain: bool                       # True if evidence is ambiguous, degraded, or truncated
    confidence: float                        # Calibrated subtype confidence [0.0, 1.0]
    reason: Optional[str]                    # Specific explanation if uncertain
    forensic_verifiability: ForensicVerifiability # Mathematical vs semi-quantitative vs expert
    evidence_summary: Dict[str, Any]         # Geometric evidence parameters
    hand: Optional[str]                      # Resolved hand ('Right', 'Left', or None)
    finger_num: Optional[int]                # Digit 1-10
    finger_name: Optional[str]               # Thumb, Index, Middle, Ring, Little
    orientation_res: Optional[OrientationResult] = None
    singular_points_res: Optional[SingularPointsResult] = None
    geometry_res: Optional[GeometryResult] = None

    def to_dict(self, include_features: bool = False) -> Dict[str, Any]:
        """Convert result to clean dictionary for logging, benchmarking, and reporting."""
        d = {
            "primary_class": self.primary_class,
            "primary_confidence": float(round(self.primary_confidence, 4)),
            "primary_category": self.primary_category.value,
            "subtype": self.subtype.value,
            "candidate_subtype": self.candidate_subtype.value,
            "status": self.status,
            "is_uncertain": self.is_uncertain,
            "confidence": float(round(self.confidence, 4)),
            "reason": self.reason,
            "forensic_verifiability": self.forensic_verifiability.value,
            "hand": self.hand,
            "finger_num": self.finger_num,
            "finger_name": self.finger_name,
            "evidence_summary": self.evidence_summary,
        }
        if include_features:
            if self.singular_points_res:
                d["num_cores"] = self.singular_points_res.num_cores
                d["num_deltas"] = self.singular_points_res.num_deltas
                d["cores"] = [core_pt.to_dict() for core_pt in self.singular_points_res.cores]
                d["deltas"] = [delta_pt.to_dict() for delta_pt in self.singular_points_res.deltas]
                d["delta_truncated"] = self.singular_points_res.delta_truncated
            if self.geometry_res:
                d["geometry"] = self.geometry_res.to_dict()
        return d


class HierarchicalSubtypeClassifier:
    """
    Hierarchical Hybrid Subtype Classifier (Stage 7).
    Integrates Stage 6 Deep Learning Inference with Stage 5 Dermatoglyphic Geometry.
    """
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_checkpoint: Optional[str] = "models/best_model.pt",
        device: Optional[Union[str, torch.device]] = None,
        min_coherence_threshold: float = 0.25,
        min_confidence_gate: float = 0.45,
    ):
        self.device = torch.device(device) if device is not None else get_default_device()
        self.min_coherence_threshold = min_coherence_threshold
        self.min_confidence_gate = min_confidence_gate

        # Load Stage 6 neural network classifier
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        elif model_checkpoint is not None and os.path.isfile(model_checkpoint):
            self.model = load_fingerprint_classifier(model_checkpoint, device=self.device)
            self.model.eval()
        else:
            self.model = None

        self.transform = FingerprintTransformPipeline(
            target_size=(256, 256),
            enhancement="hybrid",
            channels=1,
        )

    def classify(
        self,
        image: Union[str, np.ndarray, torch.Tensor],
        hand: Optional[str] = None,
        finger_num: Optional[Union[int, str]] = None,
        digit_code: Optional[str] = None,
    ) -> SubtypeClassificationResult:
        """
        Classify a single fingerprint image into one of the 20 dermatoglyphic subtypes.

        Args:
            image: Image file path (str), NumPy 2D grayscale array, or preprocessed Tensor.
            hand: 'Right' or 'Left' (optional).
            finger_num: 1-10 (optional).
            digit_code: '01'-'10' (optional).

        Returns:
            SubtypeClassificationResult with complete hierarchical evidence.
        """
        # 1. Load Grayscale NumPy Image for Feature Extraction
        if isinstance(image, str):
            if not os.path.isfile(image):
                raise FileNotFoundError(f"Image not found at path: {image}")
            raw_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                raise IOError(f"Could not read image from {image}")
        elif isinstance(image, np.ndarray):
            raw_img = to_grayscale_numpy(image)
        elif isinstance(image, torch.Tensor):
            raw_img = to_grayscale_numpy(image.detach().cpu())
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 2. Stage 5 Geometric Feature Extraction (computed first for robust feature fusion)
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

        ori_res = compute_orientation_field(raw_img)
        sp_res = detect_singular_points(ori_res)
        geo_res = extract_fingerprint_geometry(
            raw_img, ori_res, sp_res, hand=resolved_hand, finger_name=f_name
        )

        # 3. Stage 6 Primary Pattern Inference (A, T, W, L, R)
        if self.model is not None:
            pred = predict_fingerprint(
                model=self.model,
                image=image,
                transform=self.transform,
                device=self.device,
                hand=hand,
                finger_num=finger_num,
                digit_code=digit_code,
            )
            primary_class = pred.predicted_class
            primary_conf = pred.confidence
            if pred.hand:
                resolved_hand = pred.hand
            if pred.finger_num:
                f_num = pred.finger_num
            if pred.finger_name:
                f_name = pred.finger_name
        else:
            # Intelligent fallback from Stage 5 geometric flow if running headless
            l_est = geo_res.loop_type_estimate
            if sp_res.num_cores == 0 and sp_res.num_deltas == 0:
                primary_class = "A"
            elif l_est == "right_loop":
                primary_class = "R"
            elif l_est == "left_loop":
                primary_class = "L"
            elif l_est == "arch":
                primary_class = "A"
            elif l_est == "tented_arch":
                primary_class = "T"
            else:
                primary_class = "W"
            primary_conf = float(geo_res.confidence)

        # 4. Gating Check: Coherence & Foreground Quality
        fg_pixels = np.count_nonzero(ori_res.mask)
        mean_coh = float(np.mean(ori_res.coherence[ori_res.mask > 0])) if fg_pixels > 0 else 0.0

        is_degraded = (fg_pixels < 800) or (mean_coh < self.min_coherence_threshold)

        # 5. Apply Category-Specific Geometric Rules
        if primary_class.upper() in ("L", "R"):
            rule_decision = LoopRuleEngine.evaluate(
                primary_class=primary_class,
                primary_conf=primary_conf,
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
                primary_conf=primary_conf,
                orientation_res=ori_res,
                singular_points_res=sp_res,
                geometry_res=geo_res,
                hand=resolved_hand,
                finger_num=f_num,
                digit_code=digit_code,
            )
            primary_category = PrimaryCategory.ARCH

        else:  # 'W' Whorl
            rule_decision = WhorlRuleEngine.evaluate(
                primary_class=primary_class,
                primary_conf=primary_conf,
                orientation_res=ori_res,
                singular_points_res=sp_res,
                geometry_res=geo_res,
                hand=resolved_hand,
                finger_num=f_num,
                digit_code=digit_code,
            )
            primary_category = rule_decision.category

        # 6. Synthesize Uncertainty and Final Confidence
        candidate_subtype = rule_decision.subtype
        final_confidence = rule_decision.confidence
        is_uncertain = rule_decision.is_uncertain
        reason = rule_decision.reason

        # Check for macro-class / geometry contradiction
        is_conflicting = False
        if primary_class.upper() in ("A", "T") and sp_res.num_deltas >= 2 and geo_res.loop_type_estimate == "whorl":
            is_conflicting = True
            is_uncertain = True
            reason = "Conflicting evidence: neural network predicted Arch but bilateral deltas indicate Whorl"
            final_confidence *= 0.70
        elif primary_class.upper() == "W" and sp_res.num_cores == 0 and sp_res.num_deltas == 0 and geo_res.loop_type_estimate == "arch":
            is_conflicting = True
            is_uncertain = True
            reason = "Conflicting evidence: neural network predicted Whorl but zero singular points indicate Arch"
            final_confidence *= 0.70

        if is_degraded:
            is_uncertain = True
            reason = f"Insufficient evidence: degraded impression / low ridge coherence (mean={mean_coh:.2f})"
            final_confidence *= 0.60
        elif primary_conf < self.min_confidence_gate and not is_uncertain:
            is_uncertain = True
            reason = f"Insufficient evidence: low primary classifier confidence ({primary_conf:.2f})"
            final_confidence *= 0.75

        status = "Uncertain" if is_uncertain else "Resolved"

        evidence_summary = {
            "mean_coherence": float(round(mean_coh, 3)),
            "foreground_pixels": int(fg_pixels),
            "num_cores": sp_res.num_cores,
            "num_deltas": sp_res.num_deltas,
            "delta_truncated": sp_res.delta_truncated,
            "expected_delta_sector": sp_res.expected_delta_sector,
            "rule_evidence": rule_decision.evidence,
        }

        return SubtypeClassificationResult(
            primary_class=primary_class,
            primary_confidence=primary_conf,
            primary_category=primary_category,
            subtype=candidate_subtype,
            candidate_subtype=candidate_subtype,
            status=status,
            is_uncertain=is_uncertain,
            confidence=float(round(final_confidence, 4)),
            reason=reason,
            forensic_verifiability=rule_decision.forensic_verifiability,
            evidence_summary=evidence_summary,
            hand=resolved_hand,
            finger_num=f_num,
            finger_name=f_name,
            orientation_res=ori_res,
            singular_points_res=sp_res,
            geometry_res=geo_res,
        )

    def classify_batch(
        self,
        images: Union[List[str], List[np.ndarray], torch.Tensor],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SubtypeClassificationResult]:
        """
        Classify a collection of fingerprint images efficiently.
        """
        results: List[SubtypeClassificationResult] = []
        n = len(images) if isinstance(images, (list, tuple)) else images.shape[0]

        for i in range(n):
            img = images[i]
            meta = metadata_list[i] if (metadata_list and i < len(metadata_list)) else {}
            res = self.classify(
                image=img,
                hand=meta.get("hand"),
                finger_num=meta.get("finger_num"),
                digit_code=meta.get("digit_code"),
            )
            results.append(res)

        return results
