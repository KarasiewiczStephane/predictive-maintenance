"""Tests for the FastAPI REST API."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.models.trainer import MaintenancePredictor

RAW_FEATURE_COLS = ["temperature", "vibration", "pressure", "rpm"]


@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    """Create and save a trained model for API testing.

    Trains on raw sensor columns so the predict endpoint can use them directly.
    """
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "temperature": np.random.uniform(60, 90, n),
            "vibration": np.random.uniform(3, 8, n),
            "pressure": np.random.uniform(90, 110, n),
            "rpm": np.random.uniform(2800, 3200, n),
            "failure": [0] * 180 + [1] * 20,
        }
    )

    model = MaintenancePredictor()
    model.train(df[RAW_FEATURE_COLS], df["failure"])

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model.save(model_dir / "model.joblib")
    return tmp_path


@pytest.fixture
def client_no_model() -> TestClient:
    """Create a test client without a loaded model."""
    import src.api.server as server_module
    from src.api.server import app

    original_model = server_module.model
    server_module.model = None
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    server_module.model = original_model


@pytest.fixture
def client_with_model(trained_model_path: Path) -> TestClient:
    """Create a test client with a loaded model."""
    import src.api.server as server_module
    from src.api.server import app

    model = MaintenancePredictor()
    model.load(trained_model_path / "models" / "model.joblib")

    original_model = server_module.model
    server_module.model = model
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    server_module.model = original_model


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_no_model(self, client_no_model: TestClient) -> None:
        """Health endpoint reports model not loaded."""
        response = client_no_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False

    def test_health_with_model(self, client_with_model: TestClient) -> None:
        """Health endpoint reports model loaded."""
        response = client_with_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_no_model(self, client_no_model: TestClient) -> None:
        """Returns 503 when model is not loaded."""
        response = client_no_model.post(
            "/predict",
            json={
                "equipment_id": "EQ-001",
                "temperature": 75.0,
                "vibration": 5.0,
                "pressure": 100.0,
                "rpm": 3000.0,
            },
        )
        assert response.status_code == 503

    def test_predict_valid_input(self, client_with_model: TestClient) -> None:
        """Valid prediction returns expected response format."""
        response = client_with_model.post(
            "/predict",
            json={
                "equipment_id": "EQ-001",
                "temperature": 75.0,
                "vibration": 5.0,
                "pressure": 100.0,
                "rpm": 3000.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["equipment_id"] == "EQ-001"
        assert 0 <= data["failure_probability"] <= 1
        assert data["prediction"] in ("normal", "failure_risk")
        assert 0 <= data["confidence"] <= 1

    def test_predict_invalid_temperature(self, client_with_model: TestClient) -> None:
        """Temperature out of range returns 422."""
        response = client_with_model.post(
            "/predict",
            json={
                "equipment_id": "EQ-001",
                "temperature": 999.0,
                "vibration": 5.0,
                "pressure": 100.0,
                "rpm": 3000.0,
            },
        )
        assert response.status_code == 422

    def test_predict_negative_vibration(self, client_with_model: TestClient) -> None:
        """Negative vibration returns 422."""
        response = client_with_model.post(
            "/predict",
            json={
                "equipment_id": "EQ-001",
                "temperature": 75.0,
                "vibration": -1.0,
                "pressure": 100.0,
                "rpm": 3000.0,
            },
        )
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict_no_model(self, client_no_model: TestClient) -> None:
        """Returns 503 when model is not loaded."""
        response = client_no_model.post(
            "/predict/batch",
            json=[
                {
                    "equipment_id": "EQ-001",
                    "temperature": 75.0,
                    "vibration": 5.0,
                    "pressure": 100.0,
                    "rpm": 3000.0,
                }
            ],
        )
        assert response.status_code == 503

    def test_batch_predict_multiple(self, client_with_model: TestClient) -> None:
        """Batch prediction handles multiple readings."""
        readings = [
            {
                "equipment_id": f"EQ-{i:03d}",
                "temperature": 75.0 + i,
                "vibration": 5.0,
                "pressure": 100.0,
                "rpm": 3000.0,
            }
            for i in range(3)
        ]
        response = client_with_model.post("/predict/batch", json=readings)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
