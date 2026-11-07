"""
Automated Integration Tests for Stage 10 Web Application & API.
Verifies FastAPI endpoints: /api/health, /api/sample-gallery, /api/analyze, /api/sample-image,
as well as static frontend serving, defect detection, and full forensic JSON response schema.
"""

import io
import json
import os
import zipfile
import pytest
from fastapi.testclient import TestClient

from src.app.main import (
    app,
    BASE_DIR,
    SAMPLE_REGISTRY,
    FEEDBACK_IMAGES_DIR,
    FEEDBACK_JSON_PATH,
    _save_corrections,
)


@pytest.fixture(scope="module")
def client():
    """Create FastAPI TestClient fixture."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def clean_feedback_queue():
    """Ensure feedback queue is cleanly reset after test module finishes."""
    yield
    _save_corrections([])
    if os.path.isdir(FEEDBACK_IMAGES_DIR):
        for fname in os.listdir(FEEDBACK_IMAGES_DIR):
            if fname == ".gitkeep":
                continue
            fpath = os.path.join(FEEDBACK_IMAGES_DIR, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass


class TestStage10ApplicationAPI:
    """Test suite for Stage 10 FastAPI application."""

    def test_health_endpoint(self, client):
        """Verify /api/health returns 200 and valid system status."""
        response = client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["pipeline_ready"] is True
        assert "device" in data
        assert "mps_active" in data
        assert "version" in data
        assert "timestamp" in data

    def test_sample_gallery_endpoint(self, client):
        """Verify /api/sample-gallery returns curated test images and valid metadata."""
        response = client.get("/api/sample-gallery")
        assert response.status_code == 200

        data = response.json()
        assert "samples" in data
        assert "total" in data
        assert data["total"] == 10
        assert len(data["samples"]) == 10

        # Verify expected sample IDs exist
        sample_ids = {s["id"] for s in data["samples"]}
        expected_ids = {
            "nist_whorl",
            "nist_ulnar_loop",
            "nist_plain_arch",
            "nist_tented_arch",
            "nist_radial_loop",
            "nist_ambiguous",
            "socofing_real",
            "socofing_altered_obl",
            "socofing_altered_cr",
            "socofing_altered_zcut",
        }
        assert expected_ids.issubset(sample_ids)

        # Ensure all sample files actually exist on disk
        for sample in data["samples"]:
            assert sample["available"] is True, f"Sample {sample['id']} missing from disk"
            assert sample["name"]
            assert sample["category"]
            assert sample["expected_class"]

    def test_sample_image_endpoint(self, client):
        """Verify /api/sample-image/{sample_id} serves image files."""
        # Test clean NIST PNG
        res_png = client.get("/api/sample-image/nist_whorl")
        assert res_png.status_code == 200
        assert res_png.headers["content-type"] == "image/png"
        assert len(res_png.content) > 1000

        # Test SOCOFing BMP
        res_bmp = client.get("/api/sample-image/socofing_real")
        assert res_bmp.status_code == 200
        assert res_bmp.headers["content-type"] == "image/bmp"
        assert len(res_bmp.content) > 1000

        # Test invalid sample ID returns 404
        res_404 = client.get("/api/sample-image/unknown_sample_id")
        assert res_404.status_code == 404

    def test_analyze_via_multipart_file_upload(self, client):
        """Verify /api/analyze successfully processes a real NIST image uploaded as multipart/form-data."""
        sample_rel_path = SAMPLE_REGISTRY["nist_whorl"]["rel_path"]
        abs_path = os.path.join(BASE_DIR, sample_rel_path)
        assert os.path.isfile(abs_path)

        with open(abs_path, "rb") as f:
            files = {"file": ("f0001_01.png", f, "image/png")}
            data = {"hand": "Right", "digit_code": "01"}
            response = client.post("/api/analyze", files=files, data=data)

        assert response.status_code == 200
        res = response.json()

        # Identification & Geometry
        assert res["filename"] == "f0001_01.png"
        assert res["input_shape"] == [512, 512]
        assert res["hand"] == "Right"
        assert res["digit_code"] == "01"
        assert res["decision_gate"] in ("ACCEPTED", "ACCEPTED_WITH_WARNINGS")

        # Stage 6: Neural Classification
        cls = res["classification"]
        assert cls["primary_class"] == "W"
        assert cls["anatomical_label"] == "Whorl"
        assert cls["confidence"] > 0.50
        assert "probabilities" in cls
        assert len(cls["probabilities"]) == 5

        # Stage 7: Subtype Refinement
        sub = res["subtype"]
        assert "Whorl" in sub["subtype"] or "whorl" in sub["subtype"].lower()
        assert sub["status"] == "Resolved"
        assert "confidence" in sub
        assert "forensic_verifiability" in sub

        # Stage 8: Quality Diagnostics
        q = res["quality"]
        assert q["overall_score"] > 40.0
        assert q["quality_tier"] in ("Excellent", "Good", "Fair")
        assert q["usable_area_percentage"] > 40.0

        # Step-by-Step Forensic Explanation (5 steps)
        exp = res["explanation"]
        assert isinstance(exp, list)
        assert len(exp) == 5
        assert "Quality" in exp[0]["step"]
        assert "Deep Learning" in exp[1]["step"]
        assert "Landmark" in exp[2]["step"]
        assert "Subtype" in exp[3]["step"]
        assert "Operational Gate" in exp[4]["step"]

        # Visualizations (Base64 overlays for dynamic canvas)
        vis = res["visualizations"]
        assert "image_base64" in vis and vis["image_base64"].startswith("data:image/png;base64,")
        assert "composite_overlay_base64" in vis and vis["composite_overlay_base64"].startswith("data:image/png;base64,")
        assert "cores" in vis
        assert "deltas" in vis
        assert "streamlines" in vis
        assert len(vis["cores"]) >= 1

        # Landmark metrics & Dermatoglyphic ridge count
        feat = res["features"]
        assert "num_cores" in feat
        assert "num_deltas" in feat
        assert "ridge_count" in feat
        assert isinstance(feat["ridge_count"], (int, float))
        assert feat["ridge_count"] >= 0.0

    def test_analyze_via_sample_id(self, client):
        """Verify /api/analyze works seamlessly using sample_id parameter without re-uploading file."""
        response = client.post("/api/analyze", data={"sample_id": "nist_ulnar_loop"})
        assert response.status_code == 200

        res = response.json()
        assert res["classification"]["primary_class"] in ("L", "R")
        assert "Loop" in res["classification"]["anatomical_label"]
        assert res["features"]["num_cores"] >= 1
        assert "ridge_count" in res["features"]
        assert len(res["explanation"]) == 5

    def test_analyze_non_square_image(self, client):
        """Verify /api/analyze successfully handles non-square input dimensions (e.g. 400x512)."""
        import cv2
        sample_rel_path = SAMPLE_REGISTRY["nist_whorl"]["rel_path"]
        abs_path = os.path.join(BASE_DIR, sample_rel_path)
        img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        crop = img[50:450, 0:512]  # 400 x 512
        _, encoded = cv2.imencode(".png", crop)

        files = {"file": ("non_square.png", encoded.tobytes(), "image/png")}
        response = client.post("/api/analyze", files=files, data={"hand": "Right"})
        assert response.status_code == 200
        res = response.json()
        assert res["input_shape"] == [400, 512]
        assert "classification" in res
        assert "visualizations" in res

    def test_analyze_altered_fingerprint_detects_damage(self, client):
        """Verify /api/analyze detects physical obliteration in SOCOFing altered fingerprint."""
        response = client.post("/api/analyze", data={"sample_id": "socofing_altered_obl"})
        assert response.status_code == 200

        res = response.json()
        q = res["quality"]
        assert q["damage_detected"] is True
        assert any("Obliteration" in dt for dt in q["damage_types"])
        assert q["damage_percentage"] > 0.0

        # Decision gate reflects defect warnings
        assert res["decision_gate"] in ("ACCEPTED_WITH_WARNINGS", "REJECTED_LOW_QUALITY")

        # Damage mask visual asset generated
        assert res["visualizations"]["damage_mask_base64"].startswith("data:image/png;base64,")

    def test_analyze_missing_input_returns_400(self, client):
        """Verify /api/analyze returns 400 when neither file nor sample_id is provided."""
        response = client.post("/api/analyze", data={})
        assert response.status_code == 400
        assert "No input provided" in response.json()["detail"]

    def test_analyze_empty_file_returns_400(self, client):
        """Verify /api/analyze returns 400 when uploaded file is 0 bytes."""
        files = {"file": ("empty.png", b"", "image/png")}
        response = client.post("/api/analyze", files=files)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_analyze_corrupt_file_returns_400(self, client):
        """Verify /api/analyze returns 400 when uploaded file cannot be decoded as an image."""
        files = {"file": ("corrupt.png", b"NOT_AN_IMAGE_PAYLOAD", "image/png")}
        response = client.post("/api/analyze", files=files)
        assert response.status_code == 400
        assert "decoded" in response.json()["detail"].lower()

    def test_analyze_invalid_sample_returns_404(self, client):
        """Verify /api/analyze returns 404 for invalid sample_id."""
        response = client.post("/api/analyze", data={"sample_id": "nonexistent_sample_xyz"})
        assert response.status_code == 404

    def test_static_frontend_serving(self, client):
        """Verify root / serves index.html and static assets are accessible with all required Stage 10 controls."""
        res_index = client.get("/")
        assert res_index.status_code == 200
        assert "text/html" in res_index.headers["content-type"]
        html = res_index.text

        # Canvas & Layout
        assert "fingerprintCanvas" in html
        assert "AI Biometric Fingerprint" in html

        # Requirements: Dropzone, Ridge Count, Action Bar & Vault
        assert "Upload New Image" not in html
        assert "dropZone" in html
        assert "valRidgeCount" in html
        assert "Export Forensic Report (JSON/PDF)" not in html
        assert "Download JSON" not in html
        assert "Print / Save PDF" in html
        assert "btnFalsePrediction" in html or "False Prediction" in html
        assert "Correction Vault" in html
        assert "MPS GPU ACTIVE" not in html

        # Diagnostic Layer Toggles
        assert "Cores (Red)" in html
        assert "Deltas (Blue)" in html
        assert "Ridge Flow" in html
        assert "Damage Mask" in html

        # Forensic Inspection Lightbox Modal (Pan & Zoom)
        assert "imageModal" in html
        assert "modalCanvas" in html
        assert "modalViewport" in html
        assert "modalBtnClose" in html
        assert "modalBtnZoomIn" in html
        assert "modalBtnZoomOut" in html
        assert "modalBtnFit" in html
        assert "modalBtn100" in html
        assert "modalZoomLevel" in html
        assert "modalToggleCores" in html
        assert "modalToggleDeltas" in html
        assert "modalToggleStreamlines" in html
        assert "modalToggleDamage" in html
        assert "modalCanvasContainer" in html
        assert "modalImageMeta" in html
        assert "modal-backdrop" in html
        assert "modal-window" in html
        assert "modal-header" in html
        assert "modal-viewport" in html
        assert "Scroll to Zoom • Drag to Move • Esc to Close" in html
        assert 'title="Click to open full-screen forensic inspection view"' in html

        res_css = client.get("/static/style.css")
        assert res_css.status_code == 200
        assert "text/css" in res_css.headers.get("content-type", "")
        css = res_css.text
        assert ".modal-backdrop" in css
        assert ".modal-viewport" in css
        assert "cursor: grab" in css
        assert "cursor: grabbing" in css
        assert "#modalCanvas" in css
        assert "image-rendering: crisp-edges" in css

        res_js = client.get("/static/app.js")
        assert res_js.status_code == 200

    def test_analyze_large_image_scaled_to_max_1024(self, client):
        """Verify uploaded large images (e.g. smartphone photo 4032x3024) are scaled down to <= 1024 px."""
        import cv2
        import numpy as np

        large_img = np.random.randint(40, 220, (3024, 4032), dtype=np.uint8)
        _, encoded = cv2.imencode(".png", large_img)

        files = {"file": ("large_phone_camera.png", encoded.tobytes(), "image/png")}
        response = client.post("/api/analyze", files=files)
        assert response.status_code == 200
        res = response.json()
        assert max(res["input_shape"]) <= 1024
        assert res["input_shape"] == [768, 1024]
        assert "classification" in res
        assert "visualizations" in res

    def test_analyze_tall_large_image_scaled(self, client):
        """Verify uploaded tall images (e.g. 3000x1200) are scaled preserving aspect ratio to max dimension 1024."""
        import cv2
        import numpy as np

        tall_img = np.random.randint(40, 220, (3000, 1200), dtype=np.uint8)
        _, encoded = cv2.imencode(".png", tall_img)

        files = {"file": ("tall_camera_crop.png", encoded.tobytes(), "image/png")}
        response = client.post("/api/analyze", files=files)
        assert response.status_code == 200
        res = response.json()
        assert max(res["input_shape"]) <= 1024
        assert res["input_shape"] == [1024, 410]
        assert "classification" in res
        assert "visualizations" in res

    def test_direct_analyzer_large_image_downscaling(self):
        """Verify FingerprintSystemAnalyzer.analyze() directly scales down images > 1024."""
        import numpy as np
        from src.pipeline.analyzer import FingerprintSystemAnalyzer

        analyzer = FingerprintSystemAnalyzer()
        large_arr = np.random.randint(40, 220, (2000, 4000), dtype=np.uint8)
        result = analyzer.analyze(large_arr)
        assert max(result.input_shape) <= 1024
        assert result.input_shape == (512, 1024)

    def test_frontend_redesign_forensic_theme_and_removed_elements(self, client):
        """Verify removal of Load Sample and Rationale, plus presence of clean workstation elements."""
        res_index = client.get("/")
        assert res_index.status_code == 200
        html = res_index.text

        # Removed elements
        assert "sample-selector-wrap" not in html
        assert "btnLoadSample" not in html
        assert "sampleSelect" not in html
        assert "card-rationale" not in html
        assert "How Was This Determined?" not in html
        assert "btnUploadImage" not in html
        assert "upload-actions-bar" not in html

        # Footer text removed & empty footer container preserved
        assert "macOS Apple Silicon MPS Engine" not in html
        assert "Stage 10 Production Dashboard" not in html
        assert '<footer class="app-footer"></footer>' in html

        # Redesigned workstation title & elements
        assert "Fingerprint Analysis & Classification Workstation" in html
        assert "Biometric pattern analysis, landmark extraction & quality assessment" in html
        assert "MPS GPU ACTIVE" not in html

    def test_css_anti_blowout_rules(self, client):
        """Verify stylesheet eliminates horizontal blowout rules and neon glows."""
        res_css = client.get("/static/style.css")
        assert res_css.status_code == 200
        css = res_css.text

        # Anti-blowout rules
        assert "overflow-x: hidden" in css
        assert "grid-template-columns: 460px minmax(0, 1fr)" in css
        assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in css
        assert "word-break: break-word" in css
        assert "white-space: nowrap" not in css.split(".metric-sub")[1].split("}")[0]

        # No sci-fi neon glow variables
        assert "--cyan-glow" not in css

    def test_feedback_submission_and_retrieval(self, client):
        """Submit correction via POST /api/feedback and verify retrieval and count."""
        # 1x1 transparent PNG Base64 data URL
        dummy_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        payload = {
            "image_base64": dummy_b64,
            "filename": "f0001_01.png",
            "ai_prediction": {
                "primary_class": "W",
                "anatomical_label": "Whorl",
                "subtype": "Double Loop Whorl",
                "confidence": 0.984,
                "quality_score": 79.2,
            },
            "user_correction": "Plain Arch",
            "category": "Arch",
            "notes": "Examiner verified ridge flow forms unbroken transverse arch.",
            "metrics": {
                "cores_count": 0,
                "deltas_count": 0,
                "quality_score": 79.2,
                "usable_area": 88.4,
                "ridge_count": 0,
            },
        }

        # Submit correction
        post_res = client.post("/api/feedback", json=payload)
        assert post_res.status_code == 200
        post_data = post_res.json()
        assert post_data["status"] == "success"
        assert "id" in post_data
        corr_id = post_data["id"]
        assert corr_id.startswith("corr_")
        assert post_data["total_corrections"] >= 1

        # Query GET /api/feedback
        get_res = client.get("/api/feedback")
        assert get_res.status_code == 200
        corrections = get_res.json().get("corrections", [])
        assert any(c["id"] == corr_id for c in corrections)

        # Find the specific record
        rec = next(c for c in corrections if c["id"] == corr_id)
        assert rec["user_correction"] == "Plain Arch"
        assert rec["category"] == "Arch"
        assert rec["filename"] == "f0001_01.png"
        assert rec["ai_prediction"]["anatomical_label"] == "Whorl"
        assert rec["image_filename"] == f"{corr_id}.png"

        # Query GET /api/feedback/count
        count_res = client.get("/api/feedback/count")
        assert count_res.status_code == 200
        assert count_res.json()["count"] >= 1

        # Query GET /api/feedback/{id}
        single_res = client.get(f"/api/feedback/{corr_id}")
        assert single_res.status_code == 200
        assert single_res.json()["id"] == corr_id

        # Query nonexistent feedback ID returns 404
        missing_res = client.get("/api/feedback/nonexistent_corr_id_99999")
        assert missing_res.status_code == 404

    def test_feedback_custom_other_submission(self, client):
        """Submit correction with category='Other' and custom text."""
        custom_text = "Accidental Composite Whorl with Central Pocket"
        payload = {
            "image_base64": None,
            "filename": "f0010_02.png",
            "ai_prediction": {
                "primary_class": "W",
                "anatomical_label": "Whorl",
                "subtype": "Spiral Whorl",
                "confidence": 0.81,
            },
            "user_correction": custom_text,
            "category": "Other",
            "notes": "Rare accidental pattern with dual loop pockets.",
            "metrics": {"cores_count": 2, "deltas_count": 2},
        }

        response = client.post("/api/feedback", json=payload)
        assert response.status_code == 200
        res_data = response.json()
        assert res_data["status"] == "success"
        corr_id = res_data["id"]

        # Verify record retrieval
        get_res = client.get(f"/api/feedback/{corr_id}")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["user_correction"] == custom_text
        assert data["category"] == "Other"

    def test_feedback_filtering_by_category(self, client):
        """Verify GET /api/feedback?category=Whorl vs GET /api/feedback?category=Arch."""
        # Ensure at least one Whorl and one Arch correction exist
        whorl_payload = {
            "image_base64": None,
            "filename": "whorl_sample.png",
            "ai_prediction": {"anatomical_label": "Ulnar Loop"},
            "user_correction": "Double Loop Whorl",
            "category": "Whorl",
            "notes": "Clear dual delta triradii visible.",
        }
        arch_payload = {
            "image_base64": None,
            "filename": "arch_sample.png",
            "ai_prediction": {"anatomical_label": "Radial Loop"},
            "user_correction": "Tented Arch",
            "category": "Arch",
            "notes": "Steep vertical ridge elevation > 45 deg.",
        }

        res_w = client.post("/api/feedback", json=whorl_payload)
        res_a = client.post("/api/feedback", json=arch_payload)
        assert res_w.status_code == 200
        assert res_a.status_code == 200
        whorl_id = res_w.json()["id"]
        arch_id = res_a.json()["id"]

        # Query Whorl category
        whorl_get = client.get("/api/feedback?category=Whorl")
        assert whorl_get.status_code == 200
        whorl_records = whorl_get.json().get("corrections", [])
        whorl_ids = [c["id"] for c in whorl_records]
        assert whorl_id in whorl_ids
        assert arch_id not in whorl_ids

        # Query Arch category
        arch_get = client.get("/api/feedback?category=Arch")
        assert arch_get.status_code == 200
        arch_records = arch_get.json().get("corrections", [])
        arch_ids = [c["id"] for c in arch_records]
        assert arch_id in arch_ids
        assert whorl_id not in arch_ids

        # Query All returns both
        all_get = client.get("/api/feedback?category=All")
        assert all_get.status_code == 200
        all_ids = [c["id"] for c in all_get.json().get("corrections", [])]
        assert whorl_id in all_ids
        assert arch_id in all_ids

    def test_feedback_image_serving(self, client):
        """Query GET /api/feedback/image/{filename} and verify 200 OK image response."""
        # Red 2x2 PNG Base64 data URL
        red_png_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        payload = {
            "image_base64": red_png_b64,
            "filename": "image_test.png",
            "ai_prediction": {"anatomical_label": "Whorl"},
            "user_correction": "Radial Loop",
            "category": "Radial Loop",
        }

        res = client.post("/api/feedback", json=payload)
        assert res.status_code == 200
        corr_id = res.json()["id"]
        filename = f"{corr_id}.png"

        # Serve image
        img_res = client.get(f"/api/feedback/image/{filename}")
        assert img_res.status_code == 200
        assert img_res.headers["content-type"] == "image/png"
        assert len(img_res.content) > 10

        # Nonexistent image returns 404
        img_404 = client.get("/api/feedback/image/nonexistent_image_99999.png")
        assert img_404.status_code == 404

    def test_feedback_filter_empty_category_no_false_matches(self, client):
        """Verify records with empty category do not leak into category-specific filters."""
        payload = {
            "image_base64": None,
            "filename": "empty_cat.png",
            "ai_prediction": {"anatomical_label": "Whorl"},
            "user_correction": "Special Case",
            "category": "",
            "notes": "No category specified.",
        }
        res = client.post("/api/feedback", json=payload)
        assert res.status_code == 200
        empty_id = res.json()["id"]

        # Should NOT appear in Whorl or Arch queries
        whorl_res = client.get("/api/feedback?category=Whorl")
        whorl_ids = [c["id"] for c in whorl_res.json().get("corrections", [])]
        assert empty_id not in whorl_ids

        arch_res = client.get("/api/feedback?category=Arch")
        arch_ids = [c["id"] for c in arch_res.json().get("corrections", [])]
        assert empty_id not in arch_ids

        # SHOULD appear in All query
        all_res = client.get("/api/feedback?category=All")
        all_ids = [c["id"] for c in all_res.json().get("corrections", [])]
        assert empty_id in all_ids

    def test_feedback_image_traversal_protection(self, client):
        """Verify directory traversal attempts in image endpoint return 404."""
        traversal_res = client.get("/api/feedback/image/../../etc/passwd")
        assert traversal_res.status_code == 404

    def test_feedback_invalid_base64_gracefully_handled(self, client):
        """Verify malformed base64 image strings do not crash endpoint and set image_filename=None."""
        payload = {
            "image_base64": "NOT_VALID_BASE64_GARBAGE!@#$%",
            "filename": "garbage.png",
            "ai_prediction": {"anatomical_label": "Whorl"},
            "user_correction": "Tented Arch",
            "category": "Arch",
        }
        res = client.post("/api/feedback", json=payload)
        assert res.status_code == 200
        corr_id = res.json()["id"]

        rec = client.get(f"/api/feedback/{corr_id}").json()
        assert rec["image_filename"] is None
        assert rec["image_url"] is None

    def test_modal_hierarchy_and_clean_markup(self, client):
        """Verify modals are siblings at root and not nested inside imageModal."""
        import html.parser
        res = client.get("/")
        assert res.status_code == 200
        html_text = res.text

        void_elements = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
        stack = []
        nesting = {}

        class StrictParser(html.parser.HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag.lower() not in void_elements:
                    attrs_dict = dict(attrs)
                    elem_id = attrs_dict.get('id', '')
                    desc = f'{tag}#{elem_id}' if elem_id else tag
                    stack.append(desc)
                    if elem_id in ('imageModal', 'feedbackModal', 'vaultModal'):
                        nesting[elem_id] = list(stack)

            def handle_endtag(self, tag):
                if tag.lower() not in void_elements:
                    for i in range(len(stack)-1, -1, -1):
                        if stack[i].split('#')[0] == tag:
                            stack.pop(i)
                            break

        parser = StrictParser()
        parser.feed(html_text)

        assert 'div#imageModal' not in nesting.get('feedbackModal', [])
        assert 'div#imageModal' not in nesting.get('vaultModal', [])

    def test_feedback_export_all_zip(self, client):
        """POST 2 feedback items (one with image), call GET /api/feedback/export?format=zip, verify contents."""
        tiny_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        p1 = {
            "image_base64": tiny_png,
            "filename": "export_test1.png",
            "ai_prediction": {"anatomical_label": "Whorl"},
            "user_correction": "Arch",
            "category": "Arch",
        }
        p2 = {
            "image_base64": None,
            "filename": "export_test2.png",
            "ai_prediction": {"anatomical_label": "Loop"},
            "user_correction": "Whorl",
            "category": "Whorl",
        }
        res1 = client.post("/api/feedback", json=p1)
        assert res1.status_code == 200
        id1 = res1.json()["id"]

        res2 = client.post("/api/feedback", json=p2)
        assert res2.status_code == 200
        id2 = res2.json()["id"]

        export_res = client.get("/api/feedback/export?format=zip")
        assert export_res.status_code == 200
        assert export_res.headers["content-type"] == "application/zip"
        assert "attachment; filename=" in export_res.headers.get("content-disposition", "")
        assert export_res.headers["content-disposition"].endswith('.zip"')

        with zipfile.ZipFile(io.BytesIO(export_res.content)) as zf:
            namelist = zf.namelist()
            assert "corrections.json" in namelist
            assert "METADATA.txt" in namelist
            assert f"images/{id1}.png" in namelist
            assert f"reports/forensic_report_{id1}.pdf" in namelist
            assert f"reports/forensic_report_{id2}.pdf" in namelist

            corrections_in_zip = json.loads(zf.read("corrections.json").decode("utf-8"))
            zip_ids = {c["id"] for c in corrections_in_zip}
            assert id1 in zip_ids
            assert id2 in zip_ids

            meta = zf.read("METADATA.txt").decode("utf-8")
            assert "AI Fingerprint Forensic Classification System" in meta

    def test_feedback_export_and_clear(self, client):
        """POST 1 feedback item with image, call GET /api/feedback/export?clear=true&format=zip, verify queue cleared and image deleted."""
        tiny_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        p = {
            "image_base64": tiny_png,
            "filename": "clear_test.png",
            "ai_prediction": {"anatomical_label": "Whorl"},
            "user_correction": "Arch",
            "category": "Arch",
        }
        res = client.post("/api/feedback", json=p)
        assert res.status_code == 200
        corr_id = res.json()["id"]
        img_path = os.path.join(FEEDBACK_IMAGES_DIR, f"{corr_id}.png")
        assert os.path.isfile(img_path)

        export_res = client.get("/api/feedback/export?clear=true&format=zip")
        assert export_res.status_code == 200
        assert export_res.headers["content-type"] == "application/zip"

        count_res = client.get("/api/feedback/count")
        assert count_res.status_code == 200
        assert count_res.json()["count"] == 0

        assert not os.path.isfile(img_path)

    def test_feedback_delete_single_record(self, client):
        """POST 1 feedback item with image, call DELETE /api/feedback/{id}, verify image removed and record deleted."""
        tiny_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        p = {
            "image_base64": tiny_png,
            "filename": "delete_single_test.png",
            "ai_prediction": {"anatomical_label": "Whorl"},
            "user_correction": "Ulnar Loop",
            "category": "Ulnar Loop",
        }
        res = client.post("/api/feedback", json=p)
        assert res.status_code == 200
        corr_id = res.json()["id"]
        img_path = os.path.join(FEEDBACK_IMAGES_DIR, f"{corr_id}.png")
        assert os.path.isfile(img_path)

        del_res = client.delete(f"/api/feedback/{corr_id}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "deleted"
        assert del_res.json()["id"] == corr_id

        assert not os.path.isfile(img_path)

        get_res = client.get(f"/api/feedback/{corr_id}")
        assert get_res.status_code == 404

    def test_feedback_delete_nonexistent_returns_404(self, client):
        """DELETE /api/feedback/nonexistent_xyz returns 404."""
        del_res = client.delete("/api/feedback/nonexistent_xyz")
        assert del_res.status_code == 404

    def test_feedback_download_single_record(self, client):
        """POST 1 feedback item with image, call GET /api/feedback/{id}/download, verify 200 OK, application/zip bundle."""
        tiny_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        p = {
            "image_base64": tiny_png,
            "filename": "download_test.png",
            "ai_prediction": {"anatomical_label": "Arch"},
            "user_correction": "Tented Arch",
            "category": "Arch",
            "visualizations": {"streamlines": [[[10.0, 12.0], [15.0, 18.0]]], "cores": [{"x": 50, "y": 60}]},
            "analysis_result": {"filename": "download_test.png", "classification": {"primary_class": "A"}},
        }
        res = client.post("/api/feedback", json=p)
        assert res.status_code == 200
        corr_id = res.json()["id"]

        dl_res = client.get(f"/api/feedback/{corr_id}/download")
        assert dl_res.status_code == 200
        assert dl_res.headers["content-type"] == "application/zip"
        assert f'filename="correction_{corr_id}_bundle.zip"' in dl_res.headers.get("content-disposition", "")
        with zipfile.ZipFile(io.BytesIO(dl_res.content)) as zf:
            namelist = zf.namelist()
            assert f"{corr_id}_analysis.json" in namelist
            assert "forensic_report.pdf" in namelist
            assert "fingerprint.png" in namelist

            data = json.loads(zf.read(f"{corr_id}_analysis.json").decode("utf-8"))
            assert data["id"] == corr_id
            assert data["user_correction"] == "Tented Arch"
            assert data["visualizations"] is not None
            assert data["analysis_result"] is not None

            pdf_bytes = zf.read("forensic_report.pdf")
            assert len(pdf_bytes) > 500
            assert pdf_bytes.startswith(b"%PDF")

    def test_feedback_clear_all(self, client):
        """POST 2 items, call DELETE /api/feedback, verify 200 OK, count 0."""
        p1 = {"image_base64": None, "user_correction": "A", "category": "Arch"}
        p2 = {"image_base64": None, "user_correction": "B", "category": "Whorl"}
        res1 = client.post("/api/feedback", json=p1)
        res2 = client.post("/api/feedback", json=p2)
        assert res1.status_code == 200
        assert res2.status_code == 200

        del_res = client.delete("/api/feedback")
        assert del_res.status_code == 200
        data = del_res.json()
        assert data["status"] == "cleared"
        assert data["deleted_count"] >= 2

        count_res = client.get("/api/feedback/count")
        assert count_res.status_code == 200
        assert count_res.json()["count"] == 0

    def test_frontend_vault_export_and_delete_controls_present(self, client):
        """Verify index.html contains btnDownloadAllVault, does NOT contain btnClearAllVault, and has circular gauge."""
        res = client.get("/")
        assert res.status_code == 200
        html_text = res.text
        assert 'id="btnDownloadAllVault"' in html_text
        assert 'id="btnClearAllVault"' not in html_text
        assert 'id="btnInspectVaultPrint"' in html_text
        assert 'id="vaultConfirmModal"' in html_text
        assert 'id="vaultConfirmTitle"' in html_text
        assert 'id="vaultConfirmMessage"' in html_text
        assert 'id="btnCancelVaultConfirm"' in html_text
        assert 'id="btnAcceptVaultConfirm"' in html_text

        # Verify confidence gauge / bar
        assert 'id="confidenceBarFill"' in html_text
        assert 'id="primaryConfBadge"' in html_text
        assert 'id="confidencePercentageText"' in html_text

        # Verify removed badges are NOT in HTML
        assert 'id="inputStatusBadge"' not in html_text
        assert 'id="nistSymbolBadge"' not in html_text
        assert 'id="subtypeStatusBadge"' not in html_text
        assert 'id="pipelineLatency"' not in html_text

    def test_feedback_persists_visualizations_and_analysis_result(self, client):
        """Verify POST /api/feedback stores visualizations and analysis_result and GET /api/feedback returns them."""
        vis = {
            "cores": [{"x": 100, "y": 150}],
            "deltas": [{"x": 200, "y": 250}],
            "streamlines": [[[10, 20], [30, 40]]],
            "composite_overlay_base64": "data:image/png;base64,fake",
            "damage_mask_base64": "data:image/png;base64,fake2",
        }
        res_data = {
            "filename": "vis_test.png",
            "classification": {"primary_class": "W", "confidence": 0.99},
            "subtype": {"subtype": "Plain Whorl"},
            "visualizations": vis,
        }
        payload = {
            "filename": "vis_test.png",
            "ai_prediction": {"anatomical_label": "Whorl", "confidence": 0.99},
            "user_correction": "Plain Whorl",
            "category": "Whorl",
            "visualizations": vis,
            "analysis_result": res_data,
        }
        post_res = client.post("/api/feedback", json=payload)
        assert post_res.status_code == 200
        corr_id = post_res.json()["id"]

        get_res = client.get(f"/api/feedback/{corr_id}")
        assert get_res.status_code == 200
        record = get_res.json()
        assert record["id"] == corr_id
        assert record["visualizations"] is not None
        assert record["visualizations"]["cores"] == [{"x": 100, "y": 150}]
        assert record["analysis_result"] is not None
        assert record["analysis_result"]["classification"]["primary_class"] == "W"

    def test_feedback_accepts_and_stores_visualizations_none(self, client):
        """Verify POST /api/feedback accepts payload where visualizations and analysis_result are None."""
        payload = {
            "user_correction": "Arch",
            "category": "Arch",
            "visualizations": None,
            "analysis_result": None,
        }
        res = client.post("/api/feedback", json=payload)
        assert res.status_code == 200
        corr_id = res.json()["id"]

        get_res = client.get(f"/api/feedback/{corr_id}")
        assert get_res.status_code == 200
        rec = get_res.json()
        assert rec["visualizations"] is None
        assert rec["analysis_result"] is None

    def test_feedback_persists_arch_empty_singular_points(self, client):
        """Verify POST /api/feedback handles Arch prints with 0 cores and 0 deltas."""
        vis = {
            "cores": [],
            "deltas": [],
            "streamlines": [],
            "composite_overlay_base64": "data:image/png;base64,archfake",
            "damage_mask_base64": None,
        }
        payload = {
            "user_correction": "Plain Arch",
            "category": "Arch",
            "visualizations": vis,
            "analysis_result": {"filename": "arch.png", "features": {"cores": [], "deltas": []}},
        }
        res = client.post("/api/feedback", json=payload)
        assert res.status_code == 200
        corr_id = res.json()["id"]

        get_res = client.get(f"/api/feedback/{corr_id}")
        assert get_res.status_code == 200
        rec = get_res.json()
        assert rec["visualizations"]["cores"] == []
        assert rec["visualizations"]["deltas"] == []
        assert rec["visualizations"]["composite_overlay_base64"] == "data:image/png;base64,archfake"

    def test_frontend_vault_inspection_and_clear_button_contract(self, client):
        """Verify frontend contract: btnClearAllVault removed, btnInspectVaultPrint and overlays present."""
        # 1. index.html checks
        res = client.get("/")
        assert res.status_code == 200
        html = res.text
        assert 'id="btnClearAllVault"' not in html
        assert 'id="btnDownloadAllVault"' in html
        assert 'id="btnInspectVaultPrint"' in html

        # 2. app.js checks
        res_js = client.get("/static/app.js")
        assert res_js.status_code == 200
        js = res_js.text
        assert "btnClearAllVault" not in js
        assert "handleClearAllVault" not in js
        assert "openVaultInspectionModal" in js
        assert "submitFeedbackRecord" in js
        assert "vaultDetailImgWrap" in js
        assert "vault-img-hover-badge" in js

        # 3. style.css checks
        res_css = client.get("/static/style.css")
        assert res_css.status_code == 200
        css = res_css.text
        assert ".vault-detail-img-wrap" in css
        assert ".vault-img-hover-badge" in css
        assert "#btnInspectVaultPrint" in css

    def test_app_js_dom_resilience_node_mock(self):
        """Execute Node.js DOM mock resilience tests ensuring app.js never throws null property errors."""
        import subprocess
        test_script = os.path.join(BASE_DIR, "tests", "test_dom_resilience.js")
        proc = subprocess.run(["node", test_script], capture_output=True, text=True)
        assert proc.returncode == 0, f"Node DOM test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        assert "ALL NODE.JS DOM RESILIENCE TESTS PASSED CLEANLY!" in proc.stdout



