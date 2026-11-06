"""
Robustness & Alteration Stress Testing Engine (Stage 9).
Evaluates system robustness, representation drift, feature degradation, and quality gating
across SOCOFing Real baseline and Altered cohorts (Easy, Medium, Hard across CR, Obl, Zcut).

CRITICAL CONSTRAINTS:
- Do NOT calculate pattern classification accuracy on SOCOFing (lacks pattern ground truth).
- Use SOCOFing strictly for: robustness testing, quality evaluation, representation drift
  under alterations, and graceful uncertainty handling.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import sys
import time
import json
import csv
import random
from collections import defaultdict

import numpy as np

from src.pipeline.analyzer import (
    FingerprintSystemAnalyzer,
    FingerprintAnalysisResult,
    _clean_json_serializable,
)


SOCOFING_FINGER_MAP = {
    ("Right", "thumb"): 1,
    ("Right", "index"): 2,
    ("Right", "middle"): 3,
    ("Right", "ring"): 4,
    ("Right", "little"): 5,
    ("Left", "thumb"): 6,
    ("Left", "index"): 7,
    ("Left", "middle"): 8,
    ("Left", "ring"): 9,
    ("Left", "little"): 10,
}


def _calc_corr(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    arr_x = np.array(x, dtype=np.float64)
    arr_y = np.array(y, dtype=np.float64)
    std_x = float(np.std(arr_x))
    std_y = float(np.std(arr_y))
    if std_x < 1e-8 or std_y < 1e-8:
        return 0.0
    r = float(np.corrcoef(arr_x, arr_y)[0, 1])
    return float(round(r, 4))


@dataclass
class PairEvaluation:
    """
    Comparison container for a single Real vs. Altered fingerprint pair.
    """
    finger_id: str
    subject_id: str
    gender: str
    hand: str
    finger: str
    finger_num: Optional[int]
    real_path: str
    altered_path: str
    alteration_tier: str       # "Easy", "Medium", "Hard"
    alteration_type: str       # "CR", "Obl", "Zcut"

    # Primary Macro Prediction Stability
    real_primary_class: str
    altered_primary_class: str
    is_stable: bool            # True if real_primary_class == altered_primary_class

    # Confidence Drift
    real_confidence: float
    altered_confidence: float
    confidence_delta: float

    # Quality & Defect Drift
    real_quality_score: float
    altered_quality_score: float
    quality_delta: float
    real_usable_area: float
    altered_usable_area: float
    usable_area_delta: float
    damage_detected: bool
    damage_types: List[str]
    quality_warning_triggered: bool

    # Geometric Feature Drift
    real_coherence: float
    altered_coherence: float
    coherence_delta: float
    real_cores: int
    altered_cores: int
    cores_delta: int
    real_deltas: int
    altered_deltas: int
    deltas_delta: int

    # Subtype & Gating Response
    real_is_uncertain: bool
    altered_is_uncertain: bool
    decision_gate: str
    graceful_handling: bool    # Protected against silent overconfident error

    def to_dict(self) -> Dict[str, Any]:
        return _clean_json_serializable(asdict(self))


def compute_robustness_metrics(pair_evaluations: List[PairEvaluation]) -> Dict[str, Any]:
    """
    Compute structured statistical metrics across a collection of PairEvaluations.
    """
    if not pair_evaluations:
        return {"error": "No evaluations provided"}

    total_evals = len(pair_evaluations)
    unique_fingers = len(set(p.finger_id for p in pair_evaluations))

    tiers = ["Easy", "Medium", "Hard"]
    types = ["CR", "Obl", "Zcut"]

    # 1. Macro Stability Rates
    stable_count = sum(1 for p in pair_evaluations if p.is_stable)
    overall_stability = (stable_count / total_evals) * 100.0

    stability_by_tier: Dict[str, Dict[str, Any]] = {}
    for t in tiers:
        sub = [p for p in pair_evaluations if p.alteration_tier == t]
        c = len(sub)
        s = sum(1 for p in sub if p.is_stable)
        rate = (s / max(c, 1)) * 100.0 if c > 0 else 0.0
        stability_by_tier[t] = {"count": c, "stable": s, "rate_pct": float(round(rate, 2))}

    stability_by_type: Dict[str, Dict[str, Any]] = {}
    for ty in types:
        sub = [p for p in pair_evaluations if p.alteration_type == ty]
        c = len(sub)
        s = sum(1 for p in sub if p.is_stable)
        rate = (s / max(c, 1)) * 100.0 if c > 0 else 0.0
        stability_by_type[ty] = {"count": c, "stable": s, "rate_pct": float(round(rate, 2))}

    # 3x3 Stability Matrix
    matrix_3x3: Dict[str, Dict[str, float]] = {}
    for t in tiers:
        matrix_3x3[t] = {}
        for ty in types:
            sub = [p for p in pair_evaluations if p.alteration_tier == t and p.alteration_type == ty]
            c = len(sub)
            s = sum(1 for p in sub if p.is_stable)
            rate = (s / max(c, 1)) * 100.0 if c > 0 else 0.0
            matrix_3x3[t][ty] = float(round(rate, 2))

    # 2. Quality and Feature Drift
    quality_deltas = [p.quality_delta for p in pair_evaluations]
    usable_deltas = [p.usable_area_delta for p in pair_evaluations]
    coherence_deltas = [p.coherence_delta for p in pair_evaluations]
    conf_deltas = [p.confidence_delta for p in pair_evaluations]

    cores_changed = sum(1 for p in pair_evaluations if p.cores_delta != 0)
    deltas_changed = sum(1 for p in pair_evaluations if p.deltas_delta != 0)

    # 3. Quality Gate & Uncertainty Effectiveness
    quality_warnings_count = sum(1 for p in pair_evaluations if p.quality_warning_triggered)
    rejection_count = sum(1 for p in pair_evaluations if p.decision_gate == "REJECTED_LOW_QUALITY")
    uncertainty_count = sum(1 for p in pair_evaluations if p.altered_is_uncertain)

    # Graceful Handling: For unstable predictions, what % triggered uncertainty or warnings?
    unstable_pairs = [p for p in pair_evaluations if not p.is_stable]
    unstable_total = len(unstable_pairs)
    unstable_graceful = sum(1 for p in unstable_pairs if p.graceful_handling)
    graceful_handling_rate = (unstable_graceful / max(unstable_total, 1)) * 100.0 if unstable_total > 0 else 100.0

    # 4. Statistical Correlations
    q_scores = [p.altered_quality_score for p in pair_evaluations]
    uncertainty_flags = [1.0 if p.altered_is_uncertain else 0.0 for p in pair_evaluations]
    stability_flags = [1.0 if p.is_stable else 0.0 for p in pair_evaluations]
    usable_areas = [p.altered_usable_area for p in pair_evaluations]

    corr_q_uncertainty = _calc_corr(q_scores, uncertainty_flags)
    corr_q_stability = _calc_corr(q_scores, stability_flags)
    corr_usable_uncertainty = _calc_corr(usable_areas, uncertainty_flags)

    # Breakdown of uncertainty by Quality Tier
    tier_uncertainty: Dict[str, Dict[str, Any]] = {}
    for tier_name, q_min, q_max in [
        ("Excellent", 80.0, 100.0),
        ("Good", 60.0, 79.999),
        ("Fair", 40.0, 59.999),
        ("Poor", 0.0, 39.999),
    ]:
        tier_sub = [p for p in pair_evaluations if q_min <= p.altered_quality_score <= q_max]
        cnt = len(tier_sub)
        unc_cnt = sum(1 for p in tier_sub if p.altered_is_uncertain)
        rate = (unc_cnt / max(cnt, 1)) * 100.0 if cnt > 0 else 0.0
        tier_uncertainty[tier_name] = {
            "count": cnt,
            "uncertain_count": unc_cnt,
            "uncertainty_rate_pct": float(round(rate, 2)),
        }

    return {
        "summary": {
            "unique_fingers_evaluated": unique_fingers,
            "total_pair_evaluations": total_evals,
            "overall_stability_rate_pct": float(round(overall_stability, 2)),
            "graceful_handling_rate_pct": float(round(graceful_handling_rate, 2)),
            "quality_warning_trigger_rate_pct": float(round((quality_warnings_count / total_evals) * 100.0, 2)),
            "uncertainty_rate_pct": float(round((uncertainty_count / total_evals) * 100.0, 2)),
            "rejection_rate_pct": float(round((rejection_count / total_evals) * 100.0, 2)),
        },
        "macro_stability": {
            "by_tier": stability_by_tier,
            "by_type": stability_by_type,
            "matrix_3x3": matrix_3x3,
        },
        "feature_degradation": {
            "mean_quality_score_delta": float(round(np.mean(quality_deltas), 2)),
            "mean_usable_area_delta_pct": float(round(np.mean(usable_deltas), 2)),
            "mean_coherence_delta": float(round(np.mean(coherence_deltas), 4)),
            "mean_confidence_delta": float(round(np.mean(conf_deltas), 4)),
            "cores_changed_pct": float(round((cores_changed / total_evals) * 100.0, 2)),
            "deltas_changed_pct": float(round((deltas_changed / total_evals) * 100.0, 2)),
        },
        "correlations": {
            "pearson_r_quality_vs_uncertainty": corr_q_uncertainty,
            "pearson_r_quality_vs_stability": corr_q_stability,
            "pearson_r_usable_area_vs_uncertainty": corr_usable_uncertainty,
            "uncertainty_by_quality_tier": tier_uncertainty,
        },
    }


class FingerprintRobustnessEvaluator:
    """
    Comprehensive evaluator executing stress tests across SOCOFing Real and Altered cohorts.
    """

    def __init__(
        self,
        analyzer: Optional[FingerprintSystemAnalyzer] = None,
        real_manifest: str = "manifests/socofing_real.csv",
        altered_manifest: str = "manifests/socofing_altered.csv",
        project_root: Optional[str] = None,
    ):
        self.project_root = project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.real_manifest = os.path.join(self.project_root, real_manifest) if not os.path.isabs(real_manifest) else real_manifest
        self.altered_manifest = os.path.join(self.project_root, altered_manifest) if not os.path.isabs(altered_manifest) else altered_manifest

        self.analyzer = analyzer or FingerprintSystemAnalyzer()
        self.paired_cohorts: Dict[str, Dict[str, Any]] = {}
        self._load_and_pair_manifests()

    def _load_and_pair_manifests(self):
        """Index and match Real images to all their Altered variants."""
        if not os.path.isfile(self.real_manifest) or not os.path.isfile(self.altered_manifest):
            raise FileNotFoundError(
                f"Manifest files not found: {self.real_manifest} or {self.altered_manifest}"
            )

        # 1. Load Real impressions
        real_dict: Dict[str, Dict[str, Any]] = {}
        with open(self.real_manifest, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub_id = row["subject_id"].strip()
                gender = row["gender"].strip()
                hand = row["hand"].strip()
                finger = row["finger"].strip().lower()
                finger_key = f"{sub_id}__{gender}_{hand}_{finger}"

                real_dict[finger_key] = {
                    "finger_id": finger_key,
                    "subject_id": sub_id,
                    "gender": gender,
                    "hand": hand,
                    "finger": finger,
                    "real_path": row["image_path"].strip(),
                    "altered": {},
                }

        # 2. Match Altered impressions
        with open(self.altered_manifest, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub_id = row["subject_id"].strip()
                gender = row["gender"].strip()
                hand = row["hand"].strip()
                finger = row["finger"].strip().lower()
                finger_key = f"{sub_id}__{gender}_{hand}_{finger}"

                if finger_key in real_dict:
                    tier = row["alteration_tier"].strip()
                    alt_type = row["alteration_type"].strip()
                    real_dict[finger_key]["altered"][(tier, alt_type)] = row["image_path"].strip()

        # Keep only fingers that have complete 9-variant altered suite (Easy, Med, Hard across CR, Obl, Zcut)
        complete_cohorts = {
            k: v for k, v in real_dict.items() if len(v["altered"]) == 9
        }
        self.paired_cohorts = complete_cohorts

    def evaluate_pair(
        self,
        finger_info: Dict[str, Any],
        tier: str,
        alt_type: str,
    ) -> PairEvaluation:
        """
        Evaluate a single finger's Real print vs. its specific altered variant.
        """
        finger_id = finger_info["finger_id"]
        hand = finger_info["hand"]
        finger = finger_info["finger"]
        f_num = SOCOFING_FINGER_MAP.get((hand, finger), None)

        real_rel_path = finger_info["real_path"]
        alt_rel_path = finger_info["altered"][(tier, alt_type)]

        real_abs_path = os.path.join(self.project_root, real_rel_path)
        alt_abs_path = os.path.join(self.project_root, alt_rel_path)

        # Run Real analysis
        res_real = self.analyzer.analyze(real_abs_path, hand=hand, finger_num=f_num)

        # Run Altered analysis
        res_alt = self.analyzer.analyze(alt_abs_path, hand=hand, finger_num=f_num)

        # Compute Deltas & Stability
        is_stable = bool(res_real.classification["primary_class"] == res_alt.classification["primary_class"])
        conf_delta = float(res_alt.classification["confidence"] - res_real.classification["confidence"])
        q_delta = float(res_alt.quality["overall_score"] - res_real.quality["overall_score"])
        usable_delta = float(res_alt.quality["usable_area_percentage"] - res_real.quality["usable_area_percentage"])
        coh_delta = float(res_alt.features["mean_coherence"] - res_real.features["mean_coherence"])
        cores_delta = int(res_alt.features["num_cores"] - res_real.features["num_cores"])
        deltas_delta = int(res_alt.features["num_deltas"] - res_real.features["num_deltas"])

        warning_triggered = bool(res_alt.quality.get("quality_warning", "None") != "None")
        dmg_detected = bool(res_alt.quality.get("damage_detected", False))
        dmg_types = list(res_alt.quality.get("damage_types", []))

        # Graceful handling check:
        # If prediction was stable, it's successful.
        # If unstable, it is safely handled if: flagged uncertain, warning triggered, or confidence dropped >= 0.10.
        graceful = is_stable or res_alt.subtype["is_uncertain"] or warning_triggered or (conf_delta < -0.10)

        return PairEvaluation(
            finger_id=finger_id,
            subject_id=finger_info["subject_id"],
            gender=finger_info["gender"],
            hand=hand,
            finger=finger,
            finger_num=f_num,
            real_path=real_rel_path,
            altered_path=alt_rel_path,
            alteration_tier=tier,
            alteration_type=alt_type,
            real_primary_class=res_real.classification["primary_class"],
            altered_primary_class=res_alt.classification["primary_class"],
            is_stable=is_stable,
            real_confidence=float(res_real.classification["confidence"]),
            altered_confidence=float(res_alt.classification["confidence"]),
            confidence_delta=float(round(conf_delta, 4)),
            real_quality_score=float(res_real.quality["overall_score"]),
            altered_quality_score=float(res_alt.quality["overall_score"]),
            quality_delta=float(round(q_delta, 2)),
            real_usable_area=float(res_real.quality["usable_area_percentage"]),
            altered_usable_area=float(res_alt.quality["usable_area_percentage"]),
            usable_area_delta=float(round(usable_delta, 2)),
            damage_detected=dmg_detected,
            damage_types=dmg_types,
            quality_warning_triggered=warning_triggered,
            real_coherence=float(res_real.features["mean_coherence"]),
            altered_coherence=float(res_alt.features["mean_coherence"]),
            coherence_delta=float(round(coh_delta, 4)),
            real_cores=int(res_real.features["num_cores"]),
            altered_cores=int(res_alt.features["num_cores"]),
            cores_delta=cores_delta,
            real_deltas=int(res_real.features["num_deltas"]),
            altered_deltas=int(res_alt.features["num_deltas"]),
            deltas_delta=deltas_delta,
            real_is_uncertain=bool(res_real.subtype["is_uncertain"]),
            altered_is_uncertain=bool(res_alt.subtype["is_uncertain"]),
            decision_gate=res_alt.decision_gate,
            graceful_handling=bool(graceful),
        )

    def evaluate_cohort(
        self,
        cohort_size: int = 100,
        seed: int = 42,
    ) -> Tuple[List[PairEvaluation], Dict[str, Any]]:
        """
        Evaluate full cohort of fingers across all 9 alteration modalities.

        Args:
            cohort_size: Number of unique fingers to evaluate (default: 100, yields 900 altered pairs).
            seed: Deterministic random seed for reproducible sampling.

        Returns:
            Tuple[List[PairEvaluation], Dict[str, Any]]: Raw evaluations and aggregated summary metrics.
        """
        keys = sorted(list(self.paired_cohorts.keys()))
        rng = random.Random(seed)
        rng.shuffle(keys)

        sampled_keys = keys[:cohort_size]
        tiers = ["Easy", "Medium", "Hard"]
        types = ["CR", "Obl", "Zcut"]

        pair_evals: List[PairEvaluation] = []

        for key in sampled_keys:
            finger_info = self.paired_cohorts[key]
            for tier in tiers:
                for ty in types:
                    pe = self.evaluate_pair(finger_info, tier=tier, alt_type=ty)
                    pair_evals.append(pe)

        metrics = compute_robustness_metrics(pair_evals)
        return pair_evals, metrics

    def save_results(
        self,
        pair_evals: List[PairEvaluation],
        metrics: Dict[str, Any],
        output_json_path: str,
    ):
        """Save benchmark results to structured JSON file."""
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        data = {
            "metrics": metrics,
            "evaluations_sample": [p.to_dict() for p in pair_evals[:20]],
            "total_evaluations": len(pair_evals),
        }
        with open(output_json_path, mode="w", encoding="utf-8") as f:
            json.dump(_clean_json_serializable(data), f, indent=2)
