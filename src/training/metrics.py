"""
Comprehensive Metrics and Evaluation Engine for Fingerprint Pattern Classification.
Computes P-Accuracy, Permissive PC-Accuracy, 5x5 Confusion Matrix,
disaggregated clean vs. ambiguous breakdowns, and macro/per-class Precision, Recall, and F1.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ClassificationMetrics:
    """Detailed evaluation metrics container."""
    p_accuracy: float
    pc_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]
    normalized_confusion_matrix: List[List[float]]
    clean_p_accuracy: float
    clean_pc_accuracy: float
    clean_count: int
    ambiguous_p_accuracy: float
    ambiguous_pc_accuracy: float
    ambiguous_count: int
    total_count: int
    class_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_classification_metrics(
    y_pred: Union[np.ndarray, List[int]],
    y_true_primary: Union[np.ndarray, List[int]],
    y_true_secondary: Optional[Union[np.ndarray, List[int]]] = None,
    is_ambiguous: Optional[Union[np.ndarray, List[bool]]] = None,
    class_names: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """
    Compute full classification metrics suite without external scikit-learn dependency.
    """
    if class_names is None:
        class_names = ["A", "T", "W", "L", "R"]
    num_classes = len(class_names)

    y_p = np.asarray(y_pred, dtype=int)
    y_t = np.asarray(y_true_primary, dtype=int)
    n = len(y_p)

    if n == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    if len(y_p) != len(y_t):
        raise ValueError(f"Mismatched lengths: y_pred has {len(y_p)}, y_true_primary has {len(y_t)}")

    if y_true_secondary is not None:
        y_s = np.asarray(y_true_secondary, dtype=int)
        if len(y_s) != n:
            raise ValueError(f"Mismatched lengths: y_pred has {n}, y_true_secondary has {len(y_s)}")
    else:
        y_s = np.full(n, -1, dtype=int)

    if is_ambiguous is not None:
        amb = np.asarray(is_ambiguous, dtype=bool)
        if len(amb) != n:
            raise ValueError(f"Mismatched lengths: y_pred has {n}, is_ambiguous has {len(amb)}")
    else:
        amb = (y_s >= 0) & (y_s != y_t)

    # 1. Primary Accuracy (P-Accuracy)
    p_matches = (y_p == y_t)
    p_acc = float(np.mean(p_matches))

    # 2. Permissive Dual-Class Accuracy (PC-Accuracy)
    pc_matches = p_matches | ((y_s >= 0) & (y_p == y_s))
    pc_acc = float(np.mean(pc_matches))

    # 3. 5x5 Confusion Matrix (based on primary ground truth)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_idx, pred_idx in zip(y_t, y_p):
        if 0 <= true_idx < num_classes and 0 <= pred_idx < num_classes:
            cm[true_idx, pred_idx] += 1

    # Row-normalized confusion matrix (recall per class)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    # 4. Per-Class and Macro Precision, Recall, F1
    per_class_prec = {}
    per_class_rec = {}
    per_class_f1 = {}

    precs = []
    recs = []
    f1s = []

    for c in range(num_classes):
        c_name = class_names[c]
        tp = float(cm[c, c])
        col_sum = float(cm[:, c].sum())
        row_sum = float(cm[c, :].sum())

        prec = (tp / col_sum) if col_sum > 0 else 0.0
        rec = (tp / row_sum) if row_sum > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_prec[c_name] = float(prec)
        per_class_rec[c_name] = float(rec)
        per_class_f1[c_name] = float(f1)

        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    macro_prec = float(np.mean(precs))
    macro_rec = float(np.mean(recs))
    macro_f1 = float(np.mean(f1s))

    # 5. Disaggregated Subsets (Clean vs Ambiguous)
    clean_indices = np.where(~amb)[0]
    amb_indices = np.where(amb)[0]

    clean_count = int(len(clean_indices))
    amb_count = int(len(amb_indices))

    if clean_count > 0:
        clean_p_acc = float(np.mean(p_matches[clean_indices]))
        clean_pc_acc = float(np.mean(pc_matches[clean_indices]))
    else:
        clean_p_acc, clean_pc_acc = 0.0, 0.0

    if amb_count > 0:
        amb_p_acc = float(np.mean(p_matches[amb_indices]))
        amb_pc_acc = float(np.mean(pc_matches[amb_indices]))
    else:
        amb_p_acc, amb_pc_acc = 0.0, 0.0

    return ClassificationMetrics(
        p_accuracy=p_acc,
        pc_accuracy=pc_acc,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        macro_f1=macro_f1,
        per_class_precision=per_class_prec,
        per_class_recall=per_class_rec,
        per_class_f1=per_class_f1,
        confusion_matrix=cm.tolist(),
        normalized_confusion_matrix=norm_cm.tolist(),
        clean_p_accuracy=clean_p_acc,
        clean_pc_accuracy=clean_pc_acc,
        clean_count=clean_count,
        ambiguous_p_accuracy=amb_p_acc,
        ambiguous_pc_accuracy=amb_pc_acc,
        ambiguous_count=amb_count,
        total_count=n,
        class_names=class_names,
    )
