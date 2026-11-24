"""
AgriVision REST API Endpoints
Endpoints:
- POST /api/detect (Multipart image disease classification & Grad-CAM)
- POST /api/chat   (Context-aware agricultural query answering)
- POST /api/report (Diagnostic PDF report generation via ReportLab)
- GET  /api/health (System and ML engine diagnostics)
"""

import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse, FileResponse

from backend.app.schemas.diagnostic import (
    DiagnosticResponse,
    ChatRequest,
    ChatResponse,
    ReportRequest,
    AdvisoryRecord
)
from backend.app.services.cv_service import cv_service
from backend.app.services.nlp_service import nlp_service
from backend.app.services.pdf_service import pdf_service
from backend.app.core.config import settings
import torch


router = APIRouter()


@router.post("/detect", response_model=DiagnosticResponse, summary="Detect Crop Disease from Leaf Image")
async def detect_crop_disease(file: UploadFile = File(...)):
    """
    Accepts an uploaded crop leaf photograph.
    Executes validation, EfficientNet-B0 inference, Shannon entropy uncertainty scoring,
    and Grad-CAM visual attention mapping.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image format (JPEG, PNG, WEBP)."
        )

    file_bytes = await file.read()
    try:
        image = cv_service.validate_image(file_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    try:
        prediction = cv_service.predict(image, generate_gradcam=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline error: {str(e)}"
        )

    # Attach verified agronomic advisory record from knowledge base
    advisory_data = nlp_service.get_advisory_for_class(prediction["crop"], prediction["disease"])
    advisory_obj = None
    if advisory_data:
        advisory_obj = AdvisoryRecord(
            crop=advisory_data.get("crop", prediction["crop"]),
            disease=advisory_data.get("disease", prediction["disease"]),
            pathogen=advisory_data.get("pathogen"),
            symptoms=advisory_data.get("symptoms"),
            favorable_conditions=advisory_data.get("favorable_conditions"),
            prevention_practices=advisory_data.get("prevention_practices", []),
            management_practices=advisory_data.get("management_practices", []),
            safety_warning=advisory_data.get("safety_warning"),
            sources=advisory_data.get("sources", [])
        )
    elif prediction["crop"] == "Non-Target / Unknown" or "Non_Target" in prediction.get("raw_class", ""):
        advisory_obj = AdvisoryRecord(
            crop="Non-Target / Unknown",
            disease="Unsupported Crop or Non-Leaf Object",
            pathogen="N/A",
            symptoms="The uploaded photograph does not match the leaf patterns of the 4 supported crops (Wheat, Rice, Potato, Tomato). It appears to be an unsupported plant, a non-leaf plant part (e.g. tuber, fruit, panicle), or a background surface.",
            favorable_conditions="N/A",
            prevention_practices=[
                "Ensure you photograph a single, clearly visible leaf of Wheat, Rice, Potato, or Tomato.",
                "Capture the photo under natural daylight avoiding heavy shadow, intense flash, or extreme blur.",
                "Position the leaf against a neutral or field background."
            ],
            management_practices=[
                "AgriVision currently supports: Wheat (Leaf Rust, Healthy), Rice (Bacterial Leaf Blight, Brown Spot, Leaf Smut), Potato (Early Blight, Late Blight, Healthy), and Tomato (Early Blight, Late Blight, Bacterial Spot, Septoria, Leaf Mold, Mosaic Virus, Healthy).",
                "For unlisted crops or non-leaf plant parts, please consult your local Krishi Vigyan Kendra (KVK) or university extension specialist."
            ],
            safety_warning="No chemical intervention should be undertaken for unverified plants or non-crop objects.",
            sources=[{"institution": "AgriVision Diagnostic Protocol", "publication": "Out-of-Distribution Rejection Guide"}]
        )

    return DiagnosticResponse(
        crop=prediction["crop"],
        disease=prediction["disease"],
        raw_class=prediction["raw_class"],
        confidence=prediction["confidence"],
        entropy=prediction["entropy"],
        is_healthy=prediction["is_healthy"],
        is_ambiguous=prediction["is_ambiguous"],
        pathogen=advisory_obj.pathogen if advisory_obj else None,
        top_alternatives=prediction["top_alternatives"],
        gradcam_base64=prediction["gradcam_base64"],
        advisory=advisory_obj
    )


@router.post("/chat", response_model=ChatResponse, summary="Context-Aware Agricultural Chatbot")
async def agricultural_chat(req: ChatRequest):
    """
    Answers farmer and student agricultural questions.
    Uses DistilBERT for intent classification and FAISS for dense semantic retrieval
    over verified PAU, CCS HAU, and ICAR advisories.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty."
        )

    response = nlp_service.answer_question(req.question.strip(), req.context)
    return ChatResponse(
        question=response["question"],
        intent=response["intent"],
        answer=response["answer"],
        sources=response["sources"],
        context_used=response.get("context_used")
    )


@router.post("/report", summary="Download PDF Diagnostic Report")
async def generate_pdf_report(req: ReportRequest):
    """
    Generates a formal, printable PDF diagnostic report containing scan metadata,
    Grad-CAM heatmap, agronomic recommendations, and institutional citations.
    """
    try:
        pdf_buffer = pdf_service.generate_report(req.model_dump())
        filename = f"AgriVision_Report_{req.crop}_{req.disease.replace(' ', '_')}.pdf"
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF compilation error: {str(e)}"
        )


@router.get("/health", summary="Engine Health Check")
async def health_check():
    """Returns runtime status of hardware, computer vision model, and NLP services."""
    return {
        "status": "online",
        "device": settings.DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cv_model_loaded": cv_service.model is not None,
        "num_classes": len(cv_service.classes),
        "nlp_model_loaded": nlp_service.intent_model is not None,
        "faiss_index_vectors": nlp_service.faiss_index.ntotal if nlp_service.faiss_index else 0,
        "version": settings.VERSION
    }


@router.get("/evaluation", summary="Comprehensive AI Model Evaluation Metrics")
async def get_model_evaluation():
    """
    Returns authentic, genuinely evaluated test benchmarks across both AI models:
    1. EfficientNet-B0 Computer Vision metrics (Accuracy, Macro-F1, Per-crop, Per-class, ECE, Latency)
    2. DistilBERT Intent Classifier & FAISS Dense Semantic Retrieval metrics (Recall@K, MRR)
    3. Dataset partitioning and audit statistics.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    cv_eval_path = base_dir / "reports" / "evaluation" / "cv_evaluation_report.json"
    nlp_eval_path = base_dir / "reports" / "training" / "nlp_retrieval_metrics.json"
    dataset_audit_path = base_dir / "reports" / "evaluation" / "dataset_audit_report.json"

    data = {
        "timestamp": "2026-11-18",
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "device": settings.DEVICE,
            "cuda_available": torch.cuda.is_available(),
            "architecture": "EfficientNet-B0 + DistilBERT + FAISS IndexFlatIP"
        }
    }

    if cv_eval_path.exists():
        with open(cv_eval_path) as f:
            data["cv_evaluation"] = json.load(f)
    else:
        data["cv_evaluation"] = None

    if nlp_eval_path.exists():
        with open(nlp_eval_path) as f:
            data["nlp_evaluation"] = json.load(f)
    else:
        data["nlp_evaluation"] = None

    if dataset_audit_path.exists():
        with open(dataset_audit_path) as f:
            data["dataset_audit"] = json.load(f)
    else:
        data["dataset_audit"] = None

    return data


@router.get("/evaluation/confusion-matrix", summary="Confusion Matrix Heatmap")
async def get_confusion_matrix_image():
    """Serves the generated high-resolution Confusion Matrix PNG for model evaluation."""
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    cm_path = base_dir / "reports" / "evaluation" / "cv_confusion_matrix.png"
    if not cm_path.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix image not found.")
    return FileResponse(cm_path, media_type="image/png")


@router.get("/evaluation/training-curves", summary="Training Loss & Convergence Curves")
async def get_training_curves_image():
    """Serves the loss and accuracy/Macro-F1 training progression curve PNG."""
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    curves_path = base_dir / "reports" / "training" / "cv_training_curves.png"
    if not curves_path.exists():
        raise HTTPException(status_code=404, detail="Training curves image not found.")
    return FileResponse(curves_path, media_type="image/png")
