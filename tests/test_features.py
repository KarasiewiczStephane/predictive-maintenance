"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureEngineer


@pytest.fixture
def sensor_df() -> pd.DataFrame:
    """Create a synthetic sensor DataFrame for testing."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "equipment_id": ["EQ-001"] * n,
            "temperature": np.random.uniform(60, 90, n),
            "vibration": np.random.uniform(3, 8, n),
            "pressure": np.random.uniform(90, 110, n),
            "rpm": np.random.uniform(2800, 3200, n),
            "failure": [0] * 45 + [1] * 5,
        }
    )


@pytest.fixture
def multi_equipment_df() -> pd.DataFrame:
    """Create a DataFrame with multiple equipment IDs."""
    np.random.seed(42)
    n = 30
    eq1 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "equipment_id": ["EQ-001"] * n,
            "temperature": np.random.uniform(60, 90, n),
            "vibration": np.random.uniform(3, 8, n),
            "pressure": np.random.uniform(90, 110, n),
            "rpm": np.random.uniform(2800, 3200, n),
            "failure": [0] * n,
        }
    )
    eq2 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "equipment_id": ["EQ-002"] * n,
            "temperature": np.random.uniform(70, 100, n),
            "vibration": np.random.uniform(4, 10, n),
            "pressure": np.random.uniform(85, 115, n),
            "rpm": np.random.uniform(2700, 3100, n),
            "failure": [0] * n,
        }
    )
    return pd.concat([eq1, eq2], ignore_index=True)


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_create_features_shape(self, sensor_df: pd.DataFrame) -> None:
        """Feature creation produces expected output shape."""
        fe = FeatureEngineer(window_sizes=[3, 5])
        df = fe.create_features(sensor_df)
        assert len(df) > 0
        assert len(df) < len(sensor_df)  # NaN rows dropped

    def test_rolling_features_exist(self, sensor_df: pd.DataFrame) -> None:
        """Rolling mean and std features are created."""
        fe = FeatureEngineer(window_sizes=[3])
        df = fe.create_features(sensor_df)
        assert "temperature_rolling_mean_3" in df.columns
        assert "temperature_rolling_std_3" in df.columns
        assert "vibration_rolling_mean_3" in df.columns

    def test_lag_features_exist(self, sensor_df: pd.DataFrame) -> None:
        """Lag features are created for each sensor column."""
        fe = FeatureEngineer(window_sizes=[3])
        df = fe.create_features(sensor_df)
        assert "temperature_lag_1" in df.columns
        assert "temperature_lag_3" in df.columns
        assert "temperature_lag_5" in df.columns

    def test_lag_values_correct(self, sensor_df: pd.DataFrame) -> None:
        """Lag features shift values correctly."""
        fe = FeatureEngineer(window_sizes=[3])
        df_sorted = sensor_df.sort_values(["equipment_id", "timestamp"]).reset_index(drop=True)
        df = fe.create_features(sensor_df)
        # After dropping NaN, verify lag_1 matches the previous value
        if len(df) > 1:
            idx = df.index[0]
            original_idx = 5  # After dropping lag/rolling NaN rows
            assert df.loc[idx, "temperature_lag_1"] == pytest.approx(
                df_sorted.loc[original_idx - 1, "temperature"], rel=1e-5
            )

    def test_rate_of_change_features(self, sensor_df: pd.DataFrame) -> None:
        """Rate of change (diff) features are created."""
        fe = FeatureEngineer(window_sizes=[3])
        df = fe.create_features(sensor_df)
        assert "temperature_diff" in df.columns
        assert "vibration_diff" in df.columns

    def test_time_features(self, sensor_df: pd.DataFrame) -> None:
        """Hour and day_of_week features are created."""
        fe = FeatureEngineer(window_sizes=[3])
        df = fe.create_features(sensor_df)
        assert "hour" in df.columns
        assert "day_of_week" in df.columns
        assert df["hour"].min() >= 0
        assert df["hour"].max() <= 23
        assert df["day_of_week"].min() >= 0
        assert df["day_of_week"].max() <= 6

    def test_feature_names_populated(self, sensor_df: pd.DataFrame) -> None:
        """get_feature_names returns non-empty list after create_features."""
        fe = FeatureEngineer(window_sizes=[3])
        assert fe.get_feature_names() == []
        fe.create_features(sensor_df)
        names = fe.get_feature_names()
        assert len(names) > 0
        assert "timestamp" not in names
        assert "equipment_id" not in names
        assert "failure" not in names

    def test_no_nan_in_output(self, sensor_df: pd.DataFrame) -> None:
        """Output DataFrame has no NaN values."""
        fe = FeatureEngineer(window_sizes=[3, 5])
        df = fe.create_features(sensor_df)
        assert df.isnull().sum().sum() == 0

    def test_multi_equipment_grouping(self, multi_equipment_df: pd.DataFrame) -> None:
        """Features are computed per equipment_id group."""
        fe = FeatureEngineer(window_sizes=[3])
        df = fe.create_features(multi_equipment_df)
        assert "EQ-001" in df["equipment_id"].values
        assert "EQ-002" in df["equipment_id"].values

    def test_custom_window_sizes(self, sensor_df: pd.DataFrame) -> None:
        """Custom window sizes produce corresponding features."""
        fe = FeatureEngineer(window_sizes=[7])
        df = fe.create_features(sensor_df)
        assert "temperature_rolling_mean_7" in df.columns
        assert "temperature_rolling_std_7" in df.columns
