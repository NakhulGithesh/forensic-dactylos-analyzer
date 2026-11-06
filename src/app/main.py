"""
FastAPI Biometric Fingerprint Recognition & Forensic Classification Server (Stage 10).
Provides RESTful APIs for unified fingerprint analysis, sample gallery quick-loading,
health monitoring with Apple Silicon MPS GPU detection, and serves the interactive dashboard.
"""

import base64
import datetime
import io
import json
import os
import sys
import threading
import uuid
from typing import Dict, List, Optional, Any, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.models.backbones import get_default_device
from src.pipeline.analyzer import (
    FingerprintAnalysisResult,
    FingerprintSystemAnalyzer,
)

# Root directory of the repository
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FEEDBACK_DIR = os.path.join(BASE_DIR, "data", "feedback_queue")
FEEDBACK_IMAGES_DIR = os.path.join(FEEDBACK_DIR, "images")
FEEDBACK_JSON_PATH = os.path.join(FEEDBACK_DIR, "corrections.json")

os.makedirs(FEEDBACK_IMAGES_DIR, exist_ok=True)
if not os.path.isfile(FEEDBACK_JSON_PATH):
    with open(FEEDBACK_JSON_PATH, "w", encoding="utf-8") as _f:
        json.dump([], _f)

_feedback_lock = threading.Lock()


def _load_corrections() -> List[Dict[str, Any]]:
    """Thread-safe retrieval of logged feedback corrections."""
    if not os.path.isfile(FEEDBACK_JSON_PATH):
        return []
    try:
        with open(FEEDBACK_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_corrections(records: List[Dict[str, Any]]) -> None:
    """Thread-safe atomic persistence of feedback corrections to disk."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    temp_path = f"{FEEDBACK_JSON_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    os.replace(temp_path, FEEDBACK_JSON_PATH)

# Initialize FastAPI application
app = FastAPI(
    title="AI Fingerprint Forensic Classification API",
    description="Unified Biometric Fingerprint Recognition, Defect Detection, and Subtype Taxonomy API",
    version="1.0.0",
)

# CORS middleware for open local accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Curated Sample Gallery Registry
# ---------------------------------------------------------------------------
SAMPLE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nist_whorl": {
        "id": "nist_whorl",
        "name": "NIST Clean: Double Loop Whorl (f0001_01)",
        "category": "NIST Clean",
        "filename": "f0001_01.png",
        "rel_path": "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0001_01.png",
        "hand": "Right",
        "digit_code": "01",
        "finger_name": "Thumb",
        "expected_class": "Whorl",
        "description": "High-contrast clean whorl pattern featuring dual cores and bilateral delta triradii.",
    },
    "nist_ulnar_loop": {
        "id": "nist_ulnar_loop",
        "name": "NIST Clean: Ulnar Loop (f0002_05)",
        "category": "NIST Clean",
        "filename": "f0002_05.png",
        "rel_path": "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0002_05.png",
        "hand": "Right",
        "digit_code": "05",
        "finger_name": "Little",
        "expected_class": "Ulnar Loop",
        "description": "Right hand little finger impression with ridge flow looping toward the ulnar margin.",
    },
    "nist_plain_arch": {
        "id": "nist_plain_arch",
        "name": "NIST Clean: Plain Arch (f0005_03)",
        "category": "NIST Clean",
        "filename": "f0005_03.png",
        "rel_path": "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0005_03.png",
        "hand": "Right",
        "digit_code": "03",
        "finger_name": "Middle",
        "expected_class": "Plain Arch",
        "description": "Smooth continuous horizontal ridge waves with zero cores and zero deltas.",
    },
    "nist_tented_arch": {
        "id": "nist_tented_arch",
        "name": "NIST Clean: Tented Arch (f0021_07)",
        "category": "NIST Clean",
        "filename": "f0021_07.png",
        "rel_path": "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0021_07.png",
        "hand": "Left",
        "digit_code": "07",
        "finger_name": "Index",
        "expected_class": "Tented Arch",
        "description": "Sharp angular center spine with steep vertical ridge elevation (>45°).",
    },
    "nist_radial_loop": {
        "id": "nist_radial_loop",
        "name": "NIST Clean: Radial Loop (f0148_02)",
        "category": "NIST Clean",
        "filename": "f0148_02.png",
        "rel_path": "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0148_02.png",
        "hand": "Right",
        "digit_code": "02",
        "finger_name": "Index",
        "expected_class": "Radial Loop",
        "description": "Right index finger impression with ridge exit curving toward the radial thumb margin.",
    },
    "nist_ambiguous": {
        "id": "nist_ambiguous",
        "name": "NIST Ambiguous: Arch/Loop Cross-Reference (s0008_10)",
        "category": "NIST Ambiguous",
        "filename": "s0008_10.png",
        "rel_path": "Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/s0008_10.png",
        "hand": "Left",
        "digit_code": "10",
        "finger_name": "Little",
        "expected_class": "Tented Arch / Loop",
        "description": "Challenging borderline topology showing ambiguous delta formation and subtle recurve.",
    },
    "socofing_real": {
        "id": "socofing_real",
        "name": "SOCOFing Real: Clean Sensor Acquisition (100__M_Left_index)",
        "category": "SOCOFing Real",
        "filename": "100__M_Left_index_finger.BMP",
        "rel_path": "Fingerprint Data/SOCOFing/Real/100__M_Left_index_finger.BMP",
        "hand": "Left",
        "digit_code": "07",
        "finger_name": "Index",
        "expected_class": "Real Unaltered",
        "description": "Intact optical sensor acquisition without physical defect or alteration.",
    },
    "socofing_altered_obl": {
        "id": "socofing_altered_obl",
        "name": "SOCOFing Altered: Severe Obliteration (100__M_Left_index_Obl)",
        "category": "SOCOFing Altered",
        "filename": "100__M_Left_index_finger_Obl.BMP",
        "rel_path": "Fingerprint Data/SOCOFing/Altered/Altered-Hard/100__M_Left_index_finger_Obl.BMP",
        "hand": "Left",
        "digit_code": "07",
        "finger_name": "Index",
        "expected_class": "Altered (Obliteration)",
        "description": "Extensive chemical or mechanical abrasion destroying central core ridges.",
    },
    "socofing_altered_cr": {
        "id": "socofing_altered_cr",
        "name": "SOCOFing Altered: Central Rotation (100__M_Left_index_CR)",
        "category": "SOCOFing Altered",
        "filename": "100__M_Left_index_finger_CR.BMP",
        "rel_path": "Fingerprint Data/SOCOFing/Altered/Altered-Hard/100__M_Left_index_finger_CR.BMP",
        "hand": "Left",
        "digit_code": "07",
        "finger_name": "Index",
        "expected_class": "Altered (Central Rotation)",
        "description": "Surgically simulated core tissue rotation defect producing boundary discontinuities.",
    },
    "socofing_altered_zcut": {
        "id": "socofing_altered_zcut",
        "name": "SOCOFing Altered: Z-cut Scar (100__M_Left_index_Zcut)",
        "category": "SOCOFing Altered",
        "filename": "100__M_Left_index_finger_Zcut.BMP",
        "rel_path": "Fingerprint Data/SOCOFing/Altered/Altered-Hard/100__M_Left_index_finger_Zcut.BMP",
        "hand": "Left",
        "digit_code": "07",
        "finger_name": "Index",
        "expected_class": "Altered (Z-cut Scar)",
        "description": "Tri-directional deep scar laceration cutting through primary pattern ridges.",
    },
}

# ---------------------------------------------------------------------------
# Global Analyzer Engine Singleton
# ---------------------------------------------------------------------------
_analyzer_instance: Optional[FingerprintSystemAnalyzer] = None


def get_analyzer() -> FingerprintSystemAnalyzer:
    """Retrieve or lazily initialize the unified biometric analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        model_checkpoint = os.path.join(BASE_DIR, "models", "best_model.pt")
        ckpt = model_checkpoint if os.path.isfile(model_checkpoint) else None
        _analyzer_instance = FingerprintSystemAnalyzer(
            model_checkpoint=ckpt,
            min_quality_score=40.0,
            min_usable_area=40.0,
        )
    return _analyzer_instance


def _image_to_base64_png(img: Optional[np.ndarray]) -> str:
    """Encode a 2D grayscale or 3D RGB uint8/float numpy array to a Base64 PNG data URL."""
    if img is None or img.size == 0:
        return ""
    if img.dtype != np.uint8:
        # Scale float or int appropriately
        if np.max(img) <= 1.0 and np.min(img) >= 0.0:
            arr = (img * 255.0).astype(np.uint8)
        else:
            arr = np.clip(img, 0, 255).astype(np.uint8)
    else:
        arr = img

    # OpenCV expects BGR for multi-channel encoding
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", bgr)
    else:
        success, buffer = cv2.imencode(".png", arr)

    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _damage_mask_to_base64_png(damage_mask: Optional[np.ndarray], height: int, width: int) -> str:
    """Encode segmented defect mask as a transparent RGBA PNG with red fill and yellow boundary."""
    if damage_mask is None or not np.any(damage_mask > 0):
        return ""

    if damage_mask.shape[:2] != (height, width):
        dmg_arr = cv2.resize(damage_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        dmg_arr = damage_mask

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    dmg = (dmg_arr > 0)
    # Red semi-transparent fill
    rgba[dmg, 0] = 239  # R
    rgba[dmg, 1] = 68   # G
    rgba[dmg, 2] = 68   # B
    rgba[dmg, 3] = 150  # Alpha

    # Thin yellow contour around defect regions
    dmg_u8 = dmg.astype(np.uint8) * 255
    contours, _ = cv2.findContours(dmg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgba, contours, -1, (255, 230, 0, 255), 2)

    # Convert to BGRA for cv2 imencode
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    success, buffer = cv2.imencode(".png", bgra)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """
    Biometric system health and hardware acceleration status.
    Verifies Apple Silicon MPS GPU readiness and deep model checkpoint status.
    """
    device = get_default_device()
    mps_active = (device.type == "mps") and torch.backends.mps.is_available()
    analyzer = get_analyzer()

    return {
        "status": "healthy",
        "model_loaded": analyzer.model is not None,
        "device": str(device),
        "mps_active": mps_active,
        "pipeline_ready": True,
        "version": "1.0.0 (Stage 10)",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@app.get("/api/sample-gallery")
async def get_sample_gallery() -> Dict[str, Any]:
    """
    Returns curated sample fingerprints bundled in the project
    covering clean NIST prints, ambiguous cross-referenced prints, SOCOFing Real,
    and Altered prints (Obliteration, Central Rotation, Z-cut Scar).
    """
    samples = []
    for sid, data in SAMPLE_REGISTRY.items():
        abs_path = os.path.join(BASE_DIR, data["rel_path"])
        exists = os.path.isfile(abs_path)
        samples.append({
            "id": data["id"],
            "name": data["name"],
            "category": data["category"],
            "filename": data["filename"],
            "hand": data["hand"],
            "digit_code": data["digit_code"],
            "finger_name": data["finger_name"],
            "expected_class": data["expected_class"],
            "description": data["description"],
            "available": exists,
        })
    return {
        "samples": samples,
        "total": len(samples),
    }


@app.get("/api/sample-image/{sample_id}")
async def get_sample_image(sample_id: str):
    """Serve the raw fingerprint image for a given sample ID."""
    if sample_id not in SAMPLE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found in registry.")

    rel_path = SAMPLE_REGISTRY[sample_id]["rel_path"]
    abs_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f"Sample image file not found on disk: {rel_path}")

    media_type = "image/png" if abs_path.lower().endswith(".png") else "image/bmp"
    return FileResponse(abs_path, media_type=media_type)


@app.post("/api/analyze")
async def analyze_fingerprint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    hand: Optional[str] = Form(None),
    digit_code: Optional[str] = Form(None),
    finger_num: Optional[int] = Form(None),
):
    """
    Execute full unified forensic fingerprint analysis.
    Accepts either an uploaded image (multipart/form-data) OR a sample_id from the gallery.
    Optional parameters: hand ('Right'/'Left'), digit_code ('01'-'10'), finger_num (1-10).

    Returns:
        Complete JSON analysis result including quality diagnostics, deep neural class,
        geometric landmarks (cores, deltas, streamlines), 20-label subtype,
        step-by-step forensic reasoning narrative, and Base64 overlay assets.
    """
    # Also support JSON body if called programmatically
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            sample_id = body.get("sample_id", sample_id)
            hand = body.get("hand", hand)
            digit_code = body.get("digit_code", digit_code)
            finger_num = body.get("finger_num", finger_num)
        except Exception:
            pass

    # Also support query parameters
    query_params = request.query_params
    if not sample_id and "sample_id" in query_params:
        sample_id = query_params["sample_id"]
    if not hand and "hand" in query_params:
        hand = query_params["hand"]
    if not digit_code and "digit_code" in query_params:
        digit_code = query_params["digit_code"]

    image_source = None
    input_filename = "upload.png"

    # 1. Resolve image source from upload file or sample_id
    if file is not None and file.filename:
        input_filename = file.filename
        try:
            contents = await file.read()
            if len(contents) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
            nparr = np.frombuffer(contents, np.uint8)
            decoded_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if decoded_img is None:
                raise HTTPException(status_code=400, detail="Uploaded file could not be decoded as an image.")
            # Scale down large smartphone / camera uploads preserving aspect ratio (max dimension 1024)
            max_dim = max(decoded_img.shape[:2])
            if max_dim > 1024:
                scale = 1024.0 / max_dim
                new_w = max(1, int(round(decoded_img.shape[1] * scale)))
                new_h = max(1, int(round(decoded_img.shape[0] * scale)))
                decoded_img = cv2.resize(decoded_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            image_source = decoded_img
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {str(e)}")

    elif sample_id:
        if sample_id not in SAMPLE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found in registry.")

        sample_meta = SAMPLE_REGISTRY[sample_id]
        abs_path = os.path.join(BASE_DIR, sample_meta["rel_path"])
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail=f"Sample file not found on disk: {sample_meta['rel_path']}")

        image_source = abs_path
        input_filename = sample_meta["filename"]

        # Default hand and digit code from sample registry if not specified
        if not hand:
            hand = sample_meta.get("hand")
        if not digit_code:
            digit_code = sample_meta.get("digit_code")

    else:
        raise HTTPException(
            status_code=400,
            detail="No input provided. Please provide an uploaded image file or a valid 'sample_id'.",
        )

    # 2. Execute analysis pipeline
    analyzer = get_analyzer()
    try:
        result: FingerprintAnalysisResult = analyzer.analyze(
            image=image_source,
            hand=hand,
            finger_num=finger_num,
            digit_code=digit_code,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensic analysis failed: {str(e)}")

    # 3. Build unified JSON dictionary response
    res_dict = result.to_dict()
    res_dict["filename"] = input_filename

    # Ensure damage_percentage is explicitly provided in quality
    if "damaged_pixels" in res_dict.get("quality", {}):
        dmg_px = res_dict["quality"].get("damaged_pixels", 0)
        undmg_px = res_dict["quality"].get("undamaged_pixels", 0)
        total_fg = dmg_px + undmg_px
        res_dict["quality"]["damage_percentage"] = (
            round((dmg_px / total_fg * 100.0), 2) if total_fg > 0 else 0.0
        )

    # 4. Extract streamline coordinates for vector Canvas rendering
    streamlines_coords: List[List[List[float]]] = []
    if result.subtype_result and result.subtype_result.geometry_res:
        raw_streamlines = result.subtype_result.geometry_res.streamlines
        for sl in raw_streamlines:
            if len(sl) > 2:
                streamlines_coords.append(
                    [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in sl]
                )

    # 5. Attach Base64 visual assets
    h, w = result.input_shape if (result.input_shape and result.input_shape[0] > 0) else (256, 256)
    res_dict["visualizations"] = {
        "image_base64": _image_to_base64_png(result.raw_image),
        "original": _image_to_base64_png(result.raw_image),
        "composite_overlay_base64": _image_to_base64_png(result.create_overlay()),
        "damage_mask_base64": _damage_mask_to_base64_png(result.damage_mask, h, w),
        "cores": result.features.get("cores", []),
        "deltas": result.features.get("deltas", []),
        "streamlines": streamlines_coords,
    }

    return JSONResponse(content=res_dict)


# ---------------------------------------------------------------------------
# Review Vault & False Prediction Feedback APIs
# ---------------------------------------------------------------------------

class FeedbackPayload(BaseModel):
    image_base64: Optional[str] = None
    filename: Optional[str] = "upload.png"
    ai_prediction: Dict[str, Any] = Field(default_factory=dict)
    user_correction: str
    category: Optional[str] = "Other"
    notes: Optional[str] = ""
    metrics: Optional[Dict[str, Any]] = None


@app.post("/api/feedback")
async def submit_feedback(payload: FeedbackPayload) -> Dict[str, Any]:
    """
    Log an examiner misclassification correction to the review vault archive.
    Saves image snapshot and structured comparison telemetry.
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    ts_str = now_utc.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    corr_id = f"corr_{ts_str}_{short_uuid}"

    img_filename = None
    if payload.image_base64:
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(raw_b64)
            img_filename = f"{corr_id}.png"
            img_path = os.path.join(FEEDBACK_IMAGES_DIR, img_filename)
            with open(img_path, "wb") as f:
                f.write(img_bytes)
        except Exception:
            img_filename = None

    record = {
        "id": corr_id,
        "timestamp": now_utc.isoformat(),
        "filename": payload.filename or "upload.png",
        "image_filename": img_filename,
        "image_url": f"/api/feedback/image/{img_filename}" if img_filename else None,
        "ai_prediction": payload.ai_prediction,
        "user_correction": payload.user_correction,
        "category": payload.category or "Other",
        "notes": payload.notes or "",
        "metrics": payload.metrics or {},
    }

    with _feedback_lock:
        corrections = _load_corrections()
        corrections.append(record)
        _save_corrections(corrections)
        total_count = len(corrections)

    return {
        "status": "success",
        "id": corr_id,
        "total_corrections": total_count,
    }


@app.get("/api/feedback")
async def get_feedback(category: Optional[str] = Query(None)) -> Dict[str, Any]:
    """
    Retrieve logged correction records sorted descending by timestamp.
    Supports optional category filtering.
    """
    with _feedback_lock:
        records = _load_corrections()

    if category and category.strip() and category.strip().lower() != "all":
        cat = category.strip().lower()
        filtered = []
        for r in records:
            r_cat = str(r.get("category", "")).strip().lower()
            r_corr = str(r.get("user_correction", "")).strip().lower()
            matched = False
            if r_cat:
                if cat == r_cat or cat in r_cat or r_cat in cat:
                    matched = True
            if not matched and r_corr:
                if cat == r_corr or cat in r_corr:
                    matched = True
            if matched:
                filtered.append(r)
        records = filtered

    # Sort descending by timestamp
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return {"corrections": records}


@app.get("/api/feedback/count")
async def get_feedback_count() -> Dict[str, Any]:
    """Return the total count of logged corrections in the archive."""
    with _feedback_lock:
        records = _load_corrections()
    return {"count": len(records)}


@app.get("/api/feedback/image/{filename}")
async def get_feedback_image(filename: str):
    """Serve the stored fingerprint impression thumbnail or raw image for a correction."""
    safe_name = os.path.basename(filename)
    if not safe_name.endswith(".png"):
        safe_name = f"{safe_name}.png"
    img_path = os.path.join(FEEDBACK_IMAGES_DIR, safe_name)
    if not os.path.isfile(img_path):
        orig_path = os.path.join(FEEDBACK_IMAGES_DIR, os.path.basename(filename))
        if os.path.isfile(orig_path):
            return FileResponse(orig_path, media_type="image/png")
        raise HTTPException(status_code=404, detail=f"Correction image '{filename}' not found.")
    return FileResponse(img_path, media_type="image/png")


@app.get("/api/feedback/{feedback_id}")
async def get_feedback_record(feedback_id: str) -> Dict[str, Any]:
    """Retrieve single correction record by unique ID."""
    with _feedback_lock:
        records = _load_corrections()
    for r in records:
        if r.get("id") == feedback_id:
            return r
    raise HTTPException(status_code=404, detail=f"Correction '{feedback_id}' not found.")


# ---------------------------------------------------------------------------
# Static Web Dashboard Frontend Mounting
# ---------------------------------------------------------------------------
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serve the interactive web frontend dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return JSONResponse({
        "system": "AI Fingerprint Recognition & Forensic Classification System",
        "status": "API active, frontend index.html pending creation",
        "api_docs": "/docs",
        "endpoints": ["/api/health", "/api/sample-gallery", "/api/analyze"],
    })
