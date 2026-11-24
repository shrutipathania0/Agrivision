"""
AgriVision Backend Entrypoint
FastAPI application with CORS, request logging, and modular routing.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.endpoints import router as api_router
from backend.app.core.config import settings


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AgriVision — Smart Crop Disease Detection & Agricultural Advisory System (Punjab & Haryana)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/", summary="Root Health Landing")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "documentation": "/docs",
        "focus_crops": ["Potato", "Tomato", "Rice", "Wheat"],
        "region": settings.REGIONAL_CONTEXT
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
