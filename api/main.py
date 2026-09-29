"""
FastAPI Application Entry Point:
Provides REST API endpoints for demand forecasting, inventory optimization,
SHAP explainability, and scenario simulation.
"""
from contextlib import asynccontextmanager
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.database.session import init_db
from api.routes import analytics, forecast, inventory, explain, simulate
from api.schemas.models import HealthResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist and check model artifacts
    init_db()
    model_exists = settings.MODEL_ARTIFACT_PATH.exists()
    print(f"Startup check: Model artifact present: {model_exists}")
    yield
    print("Application shutting down.")

app = FastAPI(
    title="Retail Demand Forecasting & Inventory Intelligence API",
    description="Production-grade API for multi-horizon retail demand forecasts, inventory risk scoring, replenishment calculations, and explainable AI.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit and external frontend consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoint routers
app.include_router(analytics.router)
app.include_router(forecast.router)
app.include_router(inventory.router)
app.include_router(explain.router)
app.include_router(simulate.router)

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check verifying model readiness and environment."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.ENV,
        model_loaded=settings.MODEL_ARTIFACT_PATH.exists()
    )

@app.get("/metrics", tags=["System"])
def get_model_metrics():
    """Returns the latest training and evaluation metrics report."""
    if settings.METRICS_PATH.exists():
        with open(settings.METRICS_PATH, "r") as f:
            return json.load(f)
    return {"message": "Metrics report not generated yet. Run training pipeline."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
