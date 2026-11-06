# Stage 6 — Main Pattern Classification Report (5 NIST Classes)

**Date:** November 2026  
**Hardware Platform:** Apple Silicon (macOS, Apple M5 Pro, 64 GB Unified RAM)  
**Execution Device:** Native Apple Silicon GPU (`mps`)  
**Status:** Complete — Audited, Verified & Benchmark-Validated  
**Project:** AI Fingerprint Recognition & Classification System  

---

## Executive Summary & Performance Gates

Stage 6 implements, trains, and empirically validates the 5-class fingerprint pattern classification engine in `src/models/` and `src/training/`.
The classifier categorizes 256x256 preprocessed fingerprint impressions into the five canonical NIST biometric classes:
1. **Plain Arch (A)**
2. **Tented Arch (T)**
3. **Whorl (W)**
4. **Left-slant Loop (L)**
5. **Right-slant Loop (R)**

The model further incorporates domain-informed dermatoglyphic translation in `src/models/predict.py`, resolving slant loop categories into unambiguous anatomical classes (**Ulnar Loop** vs. **Radial Loop**) via hand and digit anatomical metadata.

### Verification Gate Summary

| Acceptance Gate / Metric | Requirement | Measured Result | Status |
| :--- | :--- | :--- | :---: |
| **Model Backbones** | ResNet-18 (custom/adapted), ResNet-34, EfficientNet-B0 | All 4 backbones supported; resnet18 deployed | **PASSED** |
| **Primary Accuracy (P-Accuracy)** | Strict match against primary ground truth | **92.33%** (554 / 600) | **PASSED** |
| **Permissive Accuracy (PC-Accuracy)** | Match against primary OR secondary ground truth | **95.67%** (574 / 600) | **PASSED** |
| **Clean Prints Accuracy** | Unambiguous single-class subset | **95.38%** (475 / 498) | **PASSED** |
| **Ambiguous Prints PC-Accuracy** | Permissive match on dual-class subset | **97.06%** (99 / 102) | **PASSED** |
| **Macro F1-Score** | Macro-averaged F1 across 5 classes | **92.30%** | **PASSED** |
| **MPS Hardware Latency** | Batch inference throughput on Apple Silicon M5 Pro | **936.7 imgs/sec** (1.07 ms/img) | **PASSED** |
| **Regression Test Suite** | All Stages 3, 4, 5 tests passing | 80 / 80 Existing Tests + Stage 6 Suite | **PASSED** |

---

## 1. Quantitative Classification Results on NIST SD04 Test Set (N = 600)

The test partition consists of 600 independent fingerprint impressions (300 mated rolled pairs, exactly 120 per class) with zero subject overlap with the training or validation splits.

### 1.1 Overall Metrics Breakdown

- **Total Test Images:** 600
- **Strict Primary Accuracy (P-Accuracy):** **92.33%**
- **Permissive Dual-Class Accuracy (PC-Accuracy):** **95.67%**
- **Macro-Averaged Precision:** **92.29%**
- **Macro-Averaged Recall:** **92.33%**
- **Macro-Averaged F1-Score:** **92.30%**

### 1.2 Disaggregated Performance: Clean vs. Ambiguous Fingerprints

NIST SD04 contains approximately 17% ambiguous fingerprints where human fingerprint examiners registered secondary classifications (e.g., Tented Arch vs. Arch, or Loop vs. Arch). Disaggregating performance validates model behavior across both categories:

| Sub-cohort | Count | Strict P-Accuracy | Permissive PC-Accuracy | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Prints (Single Class)** | 498 | **95.38%** | **95.38%** | Unambiguous ridge patterns |
| **Ambiguous Prints (Cross-Referenced)** | 102 | **77.45%** | **97.06%** | Borderline topological cases |
| **Total Test Cohort** | 600 | **92.33%** | **95.67%** | Balanced (120 per class) |

### 1.3 Per-Class Precision, Recall, and F1-Scores

| Class Code | Pattern Name | Test Support | Precision (%) | Recall (%) | F1-Score (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **A** | Plain Arch | 120 | 90.83% | 90.83% | 90.83% |
| **T** | Tented Arch | 120 | 82.76% | 80.00% | 81.36% |
| **W** | Whorl | 120 | 100.00% | 99.17% | 99.58% |
| **L** | Left Loop | 120 | 92.06% | 96.67% | 94.31% |
| **R** | Right Loop | 120 | 95.80% | 95.00% | 95.40% |

---

## 2. 5x5 Confusion Matrix Analysis

### 2.1 Raw Prediction Counts

| True \ Pred | A (Arch) | T (Tented) | W (Whorl) | L (Left) | R (Right) | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A (Plain Arch)** | 109 | 10 | 0 | 1 | 0 | 120 |
| **T (Tented Arch)** | 10 | 96 | 0 | 9 | 5 | 120 |
| **W (Whorl)** | 0 | 1 | 119 | 0 | 0 | 120 |
| **L (Left Loop)** | 0 | 4 | 0 | 116 | 0 | 120 |
| **R (Right Loop)** | 1 | 5 | 0 | 0 | 114 | 120 |

### 2.2 Row-Normalized Recall (%)

| True \ Pred | A (Arch) | T (Tented) | W (Whorl) | L (Left) | R (Right) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A (Plain Arch)** | 90.8% | 8.3% | 0.0% | 0.8% | 0.0% |
| **T (Tented Arch)** | 8.3% | 80.0% | 0.0% | 7.5% | 4.2% |
| **W (Whorl)** | 0.0% | 0.8% | 99.2% | 0.0% | 0.0% |
| **L (Left Loop)** | 0.0% | 3.3% | 0.0% | 96.7% | 0.0% |
| **R (Right Loop)** | 0.8% | 4.2% | 0.0% | 0.0% | 95.0% |

### 2.3 Topological & Morphological Error Dynamics
1. **Whorls (W):** Near-perfect classification recall. Two core/delta topological structures provide unmistakable circular symmetry signatures.
2. **Left vs. Right Loops (L vs. R):** Zero cross-over confusion between Left and Right Loops, confirming the strict ban on horizontal flipping in data loading.
3. **Arch vs. Tented Arch (A vs. T):** The primary source of human and machine discordance in NIST SD04. Fingerprints with low-amplitude vertical thrust (< 45°) frequently straddle the threshold between Plain Arch and Tented Arch, explaining the substantial boost in PC-Accuracy on ambiguous prints.

---

## 3. Anatomical Dermatoglyphic Translation Verification

In biometrics and dermatoglyphics, loop classification is physiologically defined relative to forearm anatomy:
- **Ulnar Loop:** Ridge flow exits toward the little finger (ulna bone).
- **Radial Loop:** Ridge flow exits toward the thumb (radius bone).

`src/models/predict.py` executes this deterministic mapping:
- Right Hand: `R` → **Ulnar Loop**, `L` → **Radial Loop**
- Left Hand:  `L` → **Ulnar Loop**, `R` → **Radial Loop**
- Non-loops (`A`, `T`, `W`): Invariant to hand (`Plain Arch`, `Tented Arch`, `Whorl`).

Automated verification in `tests/test_stage6_models.py` validates that 100% of test predictions properly resolve into dermatoglyphic anatomical labels when hand or digit indices (`01`-`10`) are supplied.

---

## 4. Hardware Latency & Unified Memory Benchmarks (M5 Pro)

- **Device:** Apple Silicon GPU (`mps`) via Unified Memory Architecture.
- **Batch Size:** 64
- **Throughput:** **936.7 images/sec**
- **Inference Latency:** **1.07 ms/image**
- **Checkpoint Size:** **127.99 MB** (`models/best_model.pt`)

---

## 5. Visual Artifact Index

All generated visual artifacts have been compiled and verified in `investigation_output/stage6/`:
1. `investigation_output/stage6/loss_accuracy_curves.png`: Training/Validation Loss and P-Accuracy / PC-Accuracy progression curves.
2. `investigation_output/stage6/confusion_matrix.png`: Dual-panel raw counts and normalized % recall 5x5 confusion matrix heatmaps.
3. `investigation_output/stage6/sample_predictions_contact_sheet.png`: 3x4 structured contact sheet exhibiting high-confidence correct classifications, edge-case ambiguous prints with dual annotations, and top neural misclassifications.
