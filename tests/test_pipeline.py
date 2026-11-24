"""
AgriVision End-to-End System Integration Tests
Validates:
1. Backend health & GPU CUDA accessibility
2. Image upload, validation & inference pipeline
3. Grad-CAM visual saliency generation
4. Shannon entropy uncertainty guardrails
5. Context-aware agricultural chatbot (DistilBERT + FAISS)
6. ReportLab PDF report generation
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


client = TestClient(app)
TEST_DATA_DIR = Path("datasets/processed/test").resolve()


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["cuda_available"] is True
    assert "RTX 3060" in data["gpu_name"] or "NVIDIA" in data["gpu_name"]


def test_disease_detection_real_leaf():
    # Pick first available test image
    test_crop_dir = TEST_DATA_DIR / "Potato___Early_Blight"
    if not test_crop_dir.exists():
        pytest.skip("Test dataset not ready yet.")

    sample_imgs = list(test_crop_dir.glob("*.jpg"))
    if not sample_imgs:
        pytest.skip("No sample image found.")

    with open(sample_imgs[0], "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/detect",
        files={"file": ("potato_leaf.jpg", file_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "crop" in data
    assert "disease" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "gradcam_base64" in data
    assert data["advisory"] is not None
    assert len(data["top_alternatives"]) > 0


def test_invalid_small_image_rejection():
    # Create 16x16 tiny image
    img = Image.new("RGB", (16, 16), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/detect",
        files={"file": ("tiny.jpg", buf.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 422
    assert "too small" in response.json()["detail"].lower()


def test_context_aware_chat():
    payload = {
        "question": "How do I prevent this blight from spreading to other potato plants?",
        "context": {
            "crop": "Potato",
            "disease": "Early Blight",
            "confidence": 0.94
        }
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] in ["disease_prevention", "disease_management", "general_farming_inquiry"]
    assert "Potato" in data["answer"] or "blight" in data["answer"].lower()
    assert len(data["sources"]) > 0


def test_pdf_report_generation():
    payload = {
        "crop": "Potato",
        "disease": "Early Blight",
        "confidence": 0.94,
        "is_healthy": False,
        "pathogen": "Alternaria solani",
        "advisory": {
            "symptoms": "Dark brown concentric rings on lower leaves.",
            "prevention_practices": ["Use certified disease-free tubers.", "Rotate crops for 3 years."],
            "management_practices": ["Spray Mancozeb 75 WP.", "Remove infected foliage."],
            "safety_warning": "Wear protective gloves and mask."
        }
    }
    response = client.post("/api/report", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_model_evaluation_endpoints():
    res = client.get("/api/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert "cv_evaluation" in data
    assert data["cv_evaluation"]["overall_metrics"]["accuracy"] >= 0.90
    assert data["cv_evaluation"]["overall_metrics"]["macro_f1"] >= 0.85
    assert "nlp_evaluation" in data
    assert data["nlp_evaluation"]["mrr"] == 1.0

    res_cm = client.get("/api/evaluation/confusion-matrix")
    assert res_cm.status_code == 200
    assert res_cm.headers["content-type"] == "image/png"

    res_tc = client.get("/api/evaluation/training-curves")
    assert res_tc.status_code == 200
    assert res_tc.headers["content-type"] == "image/png"
