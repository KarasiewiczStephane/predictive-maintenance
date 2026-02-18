"""FastAPI REST API for serving predictive maintenance predictions.

Provides health check, single prediction, and batch prediction endpoints
with request/response validation using Pydantic models.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.models.trainer import MaintenancePredictor

logger = logging.getLogger(__name__)

model: Optional[MaintenancePredictor] = None


class SensorReading(BaseModel):
    """Input schema for a single equipment sensor reading.

    Attributes:
        equipment_id: Unique identifier for the equipment.
        temperature: Temperature sensor reading in degrees.
        vibration: Vibration sensor reading.
        pressure: Pressure sensor reading in PSI.
        rpm: Rotations per minute reading.
    """

    equipment_id: str
    temperature: float = Field(..., ge=-50, le=200)
    vibration: float = Field(..., ge=0, le=100)
    pressure: float = Field(..., ge=0, le=500)
    rpm: float = Field(..., ge=0, le=10000)


class PredictionResponse(BaseModel):
    """Response schema for a prediction result.

    Attributes:
        equipment_id: Equipment identifier from the input.
        failure_probability: Predicted probability of failure.
        prediction: Human-readable prediction label.
        confidence: Confidence score for the prediction.
    """

    equipment_id: str
    failure_probability: float
    prediction: str
    confidence: float


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint.

    Attributes:
        status: Service health status.
        model_loaded: Whether the prediction model is loaded.
    """

    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model
    model_path = settings.model_dir / "model.joblib"
    if model_path.exists():
        model = MaintenancePredictor()
        model.load(model_path)
        logger.info("Model loaded successfully from %s", model_path)
    else:
        logger.warning("No model found at %s, predictions disabled", model_path)
    yield
    model = None


app = FastAPI(
    title="Predictive Maintenance API",
    description="Predict equipment failures based on sensor data",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health and model availability."""
    return HealthResponse(status="healthy", model_loaded=model is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict(reading: SensorReading) -> PredictionResponse:
    """Predict equipment failure from a single sensor reading.

    Args:
        reading: Sensor reading with equipment measurements.

    Returns:
        Prediction result with failure probability and label.

    Raises:
        HTTPException: 503 if model is not loaded.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([reading.model_dump()])
    features = df[model.feature_names] if model.feature_names else df
    proba = float(model.predict_proba(features)[0])

    return PredictionResponse(
        equipment_id=reading.equipment_id,
        failure_probability=round(proba, 4),
        prediction="failure_risk" if proba >= 0.5 else "normal",
        confidence=round(abs(proba - 0.5) * 2, 4),
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(
    readings: list[SensorReading],
) -> list[PredictionResponse]:
    """Predict equipment failure for multiple sensor readings.

    Args:
        readings: List of sensor readings.

    Returns:
        List of prediction results.

    Raises:
        HTTPException: 503 if model is not loaded.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for reading in readings:
        result = await predict(reading)
        results.append(result)
    return results


def start_server() -> None:
    """Start the FastAPI server with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
