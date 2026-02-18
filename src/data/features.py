"""Feature engineering pipeline for predictive maintenance.

Transforms raw sensor data into ML-ready features including rolling
statistics, lag features, rate of change, and time-based features.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SENSOR_COLUMNS = ["temperature", "vibration", "pressure", "rpm"]
DEFAULT_WINDOW_SIZES = [5, 10, 30]
DEFAULT_LAG_PERIODS = [1, 3, 5]
NON_FEATURE_COLUMNS = {"timestamp", "equipment_id", "failure"}


class FeatureEngineer:
    """Creates ML features from raw equipment sensor data.

    Generates rolling statistics, lag features, rate-of-change indicators,
    and time-based features grouped by equipment_id.

    Args:
        window_sizes: Window sizes for rolling statistics calculations.
    """

    def __init__(self, window_sizes: Optional[list[int]] = None) -> None:
        self.window_sizes = window_sizes or DEFAULT_WINDOW_SIZES
        self.feature_columns: list[str] = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for predictive maintenance.

        Generates rolling means/stds, lag features, rate of change,
        and time-based features for each sensor column.

        Args:
            df: DataFrame with sensor data including timestamp,
                equipment_id, and sensor columns.

        Returns:
            DataFrame with all original and engineered features,
            rows with NaN from lag/rolling operations dropped.
        """
        df = df.copy()
        df = df.sort_values(["equipment_id", "timestamp"]).reset_index(drop=True)

        logger.info(
            "Creating features with window sizes %s for %d records",
            self.window_sizes,
            len(df),
        )

        for col in SENSOR_COLUMNS:
            if col not in df.columns:
                continue
            self._add_rolling_features(df, col)
            self._add_lag_features(df, col)
            self._add_rate_of_change(df, col)

        self._add_time_features(df)

        df = df.dropna().reset_index(drop=True)

        self.feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
        logger.info("Created %d features, %d records remaining", len(self.feature_columns), len(df))
        return df

    def _add_rolling_features(self, df: pd.DataFrame, col: str) -> None:
        """Add rolling mean and std features for a sensor column.

        Args:
            df: DataFrame to modify in place.
            col: Sensor column name to compute rolling stats for.
        """
        for window in self.window_sizes:
            df[f"{col}_rolling_mean_{window}"] = df.groupby("equipment_id")[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"{col}_rolling_std_{window}"] = df.groupby("equipment_id")[col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

    def _add_lag_features(self, df: pd.DataFrame, col: str) -> None:
        """Add lag features for a sensor column.

        Args:
            df: DataFrame to modify in place.
            col: Sensor column name to create lags for.
        """
        for lag in DEFAULT_LAG_PERIODS:
            df[f"{col}_lag_{lag}"] = df.groupby("equipment_id")[col].shift(lag)

    def _add_rate_of_change(self, df: pd.DataFrame, col: str) -> None:
        """Add rate of change (first difference) for a sensor column.

        Args:
            df: DataFrame to modify in place.
            col: Sensor column name to differentiate.
        """
        df[f"{col}_diff"] = df.groupby("equipment_id")[col].diff()

    def _add_time_features(self, df: pd.DataFrame) -> None:
        """Add time-based features from the timestamp column.

        Args:
            df: DataFrame to modify in place. Must have a datetime timestamp column.
        """
        if "timestamp" in df.columns:
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek

    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names after feature creation.

        Returns:
            List of feature column names (excludes timestamp, equipment_id, failure).
        """
        return self.feature_columns
