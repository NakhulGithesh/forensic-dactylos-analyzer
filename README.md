# Forensic Dactylos Analyzer 🔍

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%2F%20CUDA%20%2F%20CPU-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-206%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-gray.svg)]()

**Forensic Dactylos Analyzer** is an end-to-end AI-powered biometric fingerprint intelligence workstation. It combines deep neural network inference, classical orientation field mathematics, dermatoglyphic taxonomy rules, and image quality assessment (IQA) into an examiner-grade interactive web application.

---

## 🏛️ System Architecture & Capabilities

```
                       FINGERPRINT IMPRESSION (RAW)
                                   │
      ┌────────────────────────────┴────────────────────────────┐
      ▼                                                         ▼
[Stage 4: Preprocessing]                              [Stage 8: Biometric IQA]
Lanczos-4, Hybrid CLAHE 4x4,                           RCI Clarity Index, Frequency
256x256 Tensor Normalization                           Defect Mask (Scars, Obliteration)
      │                                                         │
      ▼                                                         │
[Stage 6: ResNet-18 Classifier]                                 │
Apple Silicon MPS Device Acceleration                           │
NIST 5-Class Prediction (W, L, R, A, T)                          │
      │                                                         │
      └────────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
              [Stage 5: Geometric & Landmark Extractor]
              - Continuous Squared Gradient Orientation Field
              - Poincaré Index + Complex Phase Symmetry Filters
              - Dipole Cancellation & Basilar Crease Suppression
              - Core & Delta Coordinates, Streamline Tracing
                                   │
                                   ▼
             [Stage 7: 20-Label Dermatoglyphic Hierarchy]
             - Whorls: Target Centric, Spiral, Double Loop,
               Elongated (Target/Spiral), Peacock's Eye
             - Loops: Ulnar vs. Radial, Standard vs. Falling
             - Arches: Simple vs. Tented (Spire, Triradius, Loop)
             - Accidental: Multi-pattern & Complex Fusion
                                   │
                                   ▼
              [Stage 10: Interactive Forensic Workstation]
              - FastAPI Backend (MPS Hardware Acceleration)
              - Multi-Layer Vector HTML5 Canvas (0ms Latency)
              - Cursor-Centered Pan & Zoom Lightbox Viewer
              - Human-in-the-Loop Correction Vault & Review
```

### 1. Neural Pattern Classifier (`src/models/`)
- Adapted **ResNet-18** architecture with single-channel grayscale input trained on NIST Special Database 4 (SD04).
- **92.33% Primary Accuracy (P-Acc)** and **95.67% Permissive Accuracy (PC-Acc)** across ambiguous FBI examiner cross-referenced impressions.
- **99.58% Whorl F1 Score**, with sub-millisecond inference latency (1.07 ms) on Apple Silicon Metal Performance Shaders (`mps`).

### 2. Landmark & Geometry Extraction (`src/features/`)
- Continuous ridge orientation fields using normalized squared gradients with Gaussian tensor smoothing.
- Hybrid Poincaré Index and complex rotational symmetry filters (`m = 1` cores, `m = -1` deltas).
- **Basilar Crease & Peripheral Suppression**: Discards pseudo-cores near the interphalangeal flexion crease (bottom 20% of foreground) and below the delta baseline.
- Deterministic streamline integration computing continuous exit drift angles and core-to-delta ridge counts.

### 3. 20-Label Dermatoglyphic Subtype Engine (`src/subtypes/`)
- Rigorous hierarchical taxonomy resolving 20 clinical and forensic subtypes without fabricated ground-truth training labels.
- Evaluates concentric vs. spiral streamline closure, core eccentricity, opposing loop flows, and proximate delta ratios.
- Calibrated uncertainty gates flag ambiguous or sensor-clipped impressions as `Uncertain` rather than forcing inaccurate predictions.

### 4. Biometric Quality Assessment & Defect Segmentation (`src/quality/`)
- Multi-component biometric quality engine: Local Ridge Clarity Index (RCI), frequency continuity, foreground area percentage, and saturation checks.
- Spatial defect segmentation localizing physical alterations: abraded obliterations, linear scars, and rotational boundary seams.

### 5. Interactive Forensic Workstation (`src/app/`)
- **FastAPI** async backend serving REST endpoints (`/api/analyze`, `/api/health`, `/api/feedback`).
- **HTML5 Canvas Multi-Layer Vector Engine**: Instant client-side toggling for Cores (Red), Deltas (Blue), Ridge Flow (Cyan), and Defect Masks.
- **High-Resolution Inspection Lightbox**: Click to open full-screen modal with cursor-centered mouse-wheel zoom, trackpad pinch gestures, and click-and-drag panning.
- **False Prediction Reporting & Correction Vault**: Accidental-click protected feedback flow logging misclassifications with thumbnails, examiner notes, and side-by-side verification comparisons.

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14 on macOS Apple Silicon).
- PyTorch 2.0+ (supports `mps`, `cuda`, or `cpu`).

### 1. Clone & Install
```bash
git clone https://github.com/NakhulGithesh/forensic-dactylos-analyzer.git
cd forensic-dactylos-analyzer

# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Checkpoints
Download `best_model.pt` from [GitHub Releases](https://github.com/NakhulGithesh/forensic-dactylos-analyzer/releases/tag/v1.0.0) and place it in the `models/` directory:
```bash
# Verify model placement
ls -lh models/best_model.pt
```
*(Note: If no model weights are present, the system operates in geometric rule-fallback mode).*

### 3. Launch Workstation
```bash
python3 run_app.py
```
- Open browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
- Swagger / OpenAPI Docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
- Health Probe: **[http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)**

---

## 🧪 Automated Test Suite

The system includes a comprehensive regression test suite covering all modules:

```bash
# Run complete test suite (206 tests)
pytest tests/ -v

# Run Stage 10 Web Application tests
pytest tests/test_stage10_app.py -v
```

```
======================== 206 passed, 1 warning in 12.36s ========================
```

---

## 📂 Repository Structure

```
forensic-dactylos-analyzer/
├── src/
│   ├── app/                 # FastAPI server, static dashboard, canvas logic
│   │   ├── main.py          # REST API endpoints & feedback queue
│   │   └── static/          # HTML5, forensic CSS, vector JS engine
│   ├── data/                # Preprocessing, aspect-ratio padding, transforms
│   ├── features/            # Orientation fields, Poincaré, singular points
│   ├── models/              # ResNet-18 backbone, inference wrappers
│   ├── subtypes/            # 20-label dermatoglyphic taxonomy & rule engine
│   ├── quality/             # Ridge clarity, NFIQ metrics, damage detection
│   ├── pipeline/            # FingerprintSystemAnalyzer end-to-end integration
│   ├── training/            # PyTorch losses, trainer, and device loaders
│   └── evaluation/          # Robustness stress testing on altered prints
├── tests/                   # 206 unit, integration, and API tests
├── manifests/               # Data splits & evaluation manifests
├── reports/                 # Stage-by-stage engineering audit reports
├── scripts/                 # Benchmarks and evaluation utilities
├── data/
│   └── feedback_queue/      # Human-in-the-loop correction storage
├── models/                  # Checkpoint directory (best_model.pt)
├── run_app.py               # Top-level application server launcher
└── pytest.ini               # Pytest configuration
```

---

## 📜 Standards & Compliance
- **FPS Standards**: Resolution calibrated for standard 500 DPI biometric sensors (~8–10 px ridge wavelength).
- **NIST Special Database 4 (SD04)**: Standard 8-bit grayscale 512×512 fingerprint benchmark.
- **SOCOFing**: Synthetic Fingerprint Alteration benchmark (Obliteration, Central Rotation, Z-cut).
- **Henry System & FBI FD-249**: Anatomical loop orientation definitions (Ulnar vs. Radial).

---

## ⚖️ License
MIT License. See [LICENSE](LICENSE) for details.
