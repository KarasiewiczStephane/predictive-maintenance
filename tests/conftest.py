"""Shared test fixtures for the predictive maintenance test suite.

Provides reusable fixtures for sample DataFrames, temporary directories,
and trained model instances used across multiple test modules.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureEngineer
from src.models.trainer import MaintenancePredictor


@pytest.fixture
def sample_sensor_df() -> pd.DataFrame:
    """Create a sample sensor DataFrame with 100 hourly readings.

    Returns:
        DataFrame with sensor columns and 5% failure rate.
    """
    np.random.seed(42)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "equipment_id": ["EQ-001"] * 100,
            "temperature": np.random.uniform(60, 90, 100),
            "vibration": np.random.uniform(3, 8, 100),
            "pressure": np.random.uniform(90, 110, 100),
            "rpm": np.random.uniform(2800, 3200, 100),
            "failure": [0] * 95 + [1] * 5,
        }
    )


@pytest.fixture
def temp_data_dir() -> Path:
    """Create a temporary directory for test data artifacts.

    Yields:
        Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def trained_model(
    sample_sensor_df: pd.DataFrame,
) -> tuple[MaintenancePredictor, list[str]]:
    """Train a model on sample data for integration testing.

    Args:
        sample_sensor_df: Sample sensor DataFrame fixture.

    Returns:
        Tuple of (trained MaintenancePredictor, list of feature names).
    """
    fe = FeatureEngineer(window_sizes=[3, 5])
    df = fe.create_features(sample_sensor_df)
    X = df[fe.get_feature_names()]
    y = df["failure"]

    model = MaintenancePredictor()
    model.train(X, y)
    return model, fe.get_feature_names()
