"""Tests for the data loading and validation module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a valid sample CSV file for testing."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
            "equipment_id": ["EQ-001"] * 20,
            "temperature": np.random.uniform(60, 90, 20),
            "vibration": np.random.uniform(3, 8, 20),
            "pressure": np.random.uniform(90, 110, 20),
            "rpm": np.random.uniform(2800, 3200, 20),
            "failure": [0] * 18 + [1] * 2,
        }
    )
    csv_path = tmp_path / "test_sensors.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


@pytest.fixture
def csv_with_missing(tmp_path: Path) -> Path:
    """Create a CSV file with missing values."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "equipment_id": ["EQ-001"] * 10,
            "temperature": [70.0, np.nan, 72.0, np.nan, 74.0, 75.0, 76.0, np.nan, 78.0, 79.0],
            "vibration": [5.0] * 10,
            "pressure": [100.0] * 10,
            "rpm": [3000.0] * 10,
            "failure": [0] * 10,
        }
    )
    csv_path = tmp_path / "missing.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_valid_csv(self, sample_csv: Path) -> None:
        """Valid CSV loads successfully with correct shape."""
        loader = DataLoader(sample_csv)
        df = loader.load_csv("test_sensors.csv")
        assert len(df) == 20
        assert "timestamp" in df.columns
        assert "equipment_id" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        loader = DataLoader(tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent.csv")

    def test_missing_columns(self, tmp_path: Path) -> None:
        """CSV with missing required columns raises ValueError."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
                "temperature": [70.0] * 5,
            }
        )
        df.to_csv(tmp_path / "bad.csv", index=False)
        loader = DataLoader(tmp_path)
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load_csv("bad.csv")

    def test_handle_missing_interpolate(self, csv_with_missing: Path) -> None:
        """Interpolation fills missing numeric values."""
        loader = DataLoader(csv_with_missing)
        df = loader.load_csv("missing.csv")
        assert df["temperature"].isnull().sum() == 3

        df_clean = loader.handle_missing(df, strategy="interpolate")
        assert df_clean["temperature"].isnull().sum() == 0

    def test_handle_missing_drop(self, csv_with_missing: Path) -> None:
        """Drop strategy removes rows with missing values."""
        loader = DataLoader(csv_with_missing)
        df = loader.load_csv("missing.csv")
        df_clean = loader.handle_missing(df, strategy="drop")
        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) < len(df)

    def test_handle_missing_ffill(self, csv_with_missing: Path) -> None:
        """Forward fill strategy fills missing values."""
        loader = DataLoader(csv_with_missing)
        df = loader.load_csv("missing.csv")
        df_clean = loader.handle_missing(df, strategy="ffill")
        assert df_clean.isnull().sum().sum() == 0

    def test_handle_missing_unknown_strategy(self, sample_csv: Path) -> None:
        """Unknown strategy raises ValueError."""
        loader = DataLoader(sample_csv)
        df = loader.load_csv("test_sensors.csv")
        df.loc[0, "temperature"] = np.nan
        with pytest.raises(ValueError, match="Unknown missing value strategy"):
            loader.handle_missing(df, strategy="magic")

    def test_handle_no_missing_values(self, sample_csv: Path) -> None:
        """No missing values returns DataFrame unchanged."""
        loader = DataLoader(sample_csv)
        df = loader.load_csv("test_sensors.csv")
        df_clean = loader.handle_missing(df)
        assert len(df_clean) == len(df)
