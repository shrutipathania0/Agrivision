"""
AgriVision Backend Configuration
Settings for device auto-selection, model paths, confidence thresholds, and API routes.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class Settings(BaseSettings):
    PROJECT_NAME: str = "AgriVision"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATASETS_DIR: Path = BASE_DIR / "datasets"
    KNOWLEDGE_BASE_DIR: Path = BASE_DIR / "knowledge_base"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    
    # Model artifact paths
    CV_MODEL_PATH: Path = MODELS_DIR / "computer_vision" / "efficientnet_b0_agrivision_best.pth"
    CV_CLASS_MAPPING_PATH: Path = MODELS_DIR / "computer_vision" / "class_mapping.json"
    NLP_MODEL_DIR: Path = MODELS_DIR / "nlp" / "distilbert_intent_classifier"
    NLP_EMBEDDINGS_DIR: Path = MODELS_DIR / "embeddings"
    
    # Inference guardrails & thresholds
    CONFIDENCE_THRESHOLD: float = 0.65  # Minimum softmax confidence
    ENTROPY_THRESHOLD: float = 1.80      # Maximum Shannon entropy for ambiguous images
    IMAGE_SIZE: int = 224
    
    # Device management
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # PDF generation branding
    ORGANIZATION_NAME: str = "AgriVision Diagnostic Labs"
    REGIONAL_CONTEXT: str = "Punjab & Haryana Agricultural Extension"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()
