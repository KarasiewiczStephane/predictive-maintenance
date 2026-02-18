"""Tests for the main application entry point."""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.main import build_parser, setup_logging, train_pipeline


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample sensor data."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "equipment_id": ["EQ-001"] * n,
            "temperature": np.random.uniform(60, 90, n),
            "vibration": np.random.uniform(3, 8, n),
            "pressure": np.random.uniform(90, 110, n),
            "rpm": np.random.uniform(2800, 3200, n),
            "failure": [0] * 180 + [1] * 20,
        }
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_csv(data_dir / "test_data.csv", index=False)
    return tmp_path


class TestBuildParser:
    """Tests for CLI argument parsing."""

    def test_train_argument(self) -> None:
        """--train argument parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["--train", "data.csv"])
        assert args.train == "data.csv"

    def test_serve_argument(self) -> None:
        """--serve flag parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["--serve"])
        assert args.serve is True

    def test_no_arguments(self) -> None:
        """No arguments results in defaults."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.train is None
        assert args.serve is False


class TestSetupLogging:
    """Tests for logging configuration."""

    def test_logging_configured(self) -> None:
        """setup_logging configures the root logger."""
        setup_logging()
        root = logging.getLogger()
        assert root.level is not None
        assert len(root.handlers) > 0


class TestTrainPipeline:
    """Tests for the training pipeline."""

    def test_train_pipeline_end_to_end(self, sample_data_dir: Path) -> None:
        """Full training pipeline runs and saves artifacts."""
        model_dir = sample_data_dir / "models"
        with patch("src.main.settings") as mock_settings:
            mock_settings.data_dir = sample_data_dir / "data"
            mock_settings.model_dir = model_dir
            mock_settings.model_type = "random_forest"
            mock_settings.test_size = 0.2
            mock_settings.random_state = 42
            mock_settings.missing_value_strategy = "interpolate"
            mock_settings.log_level = "WARNING"
            mock_settings.n_estimators = 50
            mock_settings.max_depth = 5
            mock_settings.get_window_sizes.return_value = [3, 5]

            metrics = train_pipeline("test_data.csv")

        assert "cv_roc_auc_mean" in metrics
        assert "roc_auc" in metrics
        assert (model_dir / "model.joblib").exists()
        assert (model_dir / "metrics.json").exists()
