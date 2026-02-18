"""Tests for the model training module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureEngineer
from src.models.trainer import MaintenancePredictor


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic training data with engineered features."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "equipment_id": ["EQ-001"] * n,
            "temperature": np.random.uniform(60, 90, n),
            "vibration": np.random.uniform(3, 8, n),
            "pressure": np.random.uniform(90, 110, n),
            "rpm": np.random.uniform(2800, 3200, n),
            "failure": [0] * 90 + [1] * 10,
        }
    )
    fe = FeatureEngineer(window_sizes=[3, 5])
    df = fe.create_features(df)
    X = df[fe.get_feature_names()]
    y = df["failure"]
    return X, y


class TestMaintenancePredictor:
    """Tests for MaintenancePredictor class."""

    def test_train_random_forest(self, training_data: tuple) -> None:
        """Random forest trains successfully and returns metrics."""
        X, y = training_data
        model = MaintenancePredictor(model_type="random_forest")
        metrics = model.train(X, y)
        assert "cv_roc_auc_mean" in metrics
        assert "cv_roc_auc_std" in metrics
        assert 0 <= metrics["cv_roc_auc_mean"] <= 1

    def test_train_xgboost(self, training_data: tuple) -> None:
        """XGBoost trains successfully and returns metrics."""
        X, y = training_data
        model = MaintenancePredictor(model_type="xgboost")
        metrics = model.train(X, y)
        assert "cv_roc_auc_mean" in metrics
        assert 0 <= metrics["cv_roc_auc_mean"] <= 1

    def test_predict_shape(self, training_data: tuple) -> None:
        """Predictions return correct shape."""
        X, y = training_data
        model = MaintenancePredictor()
        model.train(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, training_data: tuple) -> None:
        """Probability predictions return values in [0, 1]."""
        X, y = training_data
        model = MaintenancePredictor()
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X),)
        assert proba.min() >= 0
        assert proba.max() <= 1

    def test_save_load(self, training_data: tuple, tmp_path: Path) -> None:
        """Save and load preserves model state."""
        X, y = training_data
        model = MaintenancePredictor()
        model.train(X, y)
        original_preds = model.predict(X)

        model_path = tmp_path / "model.joblib"
        model.save(model_path)
        assert model_path.exists()

        loaded_model = MaintenancePredictor()
        loaded_model.load(model_path)
        loaded_preds = loaded_model.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_missing_file(self) -> None:
        """Loading from missing path raises FileNotFoundError."""
        model = MaintenancePredictor()
        with pytest.raises(FileNotFoundError):
            model.load(Path("/nonexistent/model.joblib"))

    def test_unknown_model_type(self) -> None:
        """Unknown model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            MaintenancePredictor(model_type="neural_net")

    def test_feature_names_stored(self, training_data: tuple) -> None:
        """Feature names are stored after training."""
        X, y = training_data
        model = MaintenancePredictor()
        model.train(X, y)
        assert model.feature_names == list(X.columns)

    def test_custom_params(self, training_data: tuple) -> None:
        """Custom model parameters are applied."""
        X, y = training_data
        model = MaintenancePredictor(model_type="random_forest", n_estimators=50, max_depth=5)
        metrics = model.train(X, y)
        assert "cv_roc_auc_mean" in metrics
