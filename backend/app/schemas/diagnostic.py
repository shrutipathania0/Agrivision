"""
Pydantic Schemas for AgriVision API
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AlternativePrediction(BaseModel):
    class_name: str
    confidence: float


class AdvisoryRecord(BaseModel):
    crop: str
    disease: str
    pathogen: Optional[str] = None
    symptoms: Optional[str] = None
    favorable_conditions: Optional[str] = None
    prevention_practices: List[str] = []
    management_practices: List[str] = []
    safety_warning: Optional[str] = None
    sources: List[Dict[str, Any]] = []


class DiagnosticResponse(BaseModel):
    crop: str
    disease: str
    raw_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    entropy: float
    is_healthy: bool
    is_ambiguous: bool
    pathogen: Optional[str] = None
    top_alternatives: List[AlternativePrediction] = []
    gradcam_base64: Optional[str] = None
    advisory: Optional[AdvisoryRecord] = None


class ChatRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    question: str
    intent: str
    answer: str
    sources: List[Dict[str, Any]] = []
    context_used: Optional[Dict[str, Any]] = None


class ReportRequest(BaseModel):
    crop: str
    disease: str
    confidence: float
    is_healthy: bool
    is_ambiguous: Optional[bool] = False
    entropy: Optional[float] = None
    pathogen: Optional[str] = None
    advisory: Optional[Dict[str, Any]] = None
    gradcam_base64: Optional[str] = None
    original_image_base64: Optional[str] = None
    top_alternatives: Optional[List[AlternativePrediction]] = []
    client_timestamp: Optional[str] = None
