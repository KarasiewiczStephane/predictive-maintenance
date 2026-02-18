"""Data loading and validation for equipment sensor data.

Provides CSV loading with schema validation, missing value handling,
and data quality checks for the predictive maintenance pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SensorDataSchema(BaseModel):
    """Schema definition for equipment sensor data records.

    Attributes:
        timestamp: ISO format timestamp of the reading.
        equipment_id: Unique identifier for the equipment.
        temperature: Temperature sensor reading in degrees.
        vibration: Vibration sensor reading.
        pressure: Pressure sensor reading in PSI.
        rpm: Rotations per minute reading.
        failure: Target label (1=failure, 0=normal).
    """

    timestamp: str
    equipment_id: str
    temperature: float
    vibration: float
    pressure: float
    rpm: float
    failure: Optional[int] = 0


REQUIRED_COLUMNS = [
    "timestamp",
    "equipment_id",
    "temperature",
    "vibration",
    "pressure",
    "rpm",
]

SENSOR_COLUMNS = ["temperature", "vibration", "pressure", "rpm"]


class DataLoader:
    """Loads and validates equipment sensor data from CSV files.

    Args:
        data_path: Base directory containing data files.
    """

    def __init__(self, data_path: Path) -> None:
        self.data_path = Path(data_path)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load and validate a CSV file containing sensor data.

        Args:
            filename: Name of the CSV file relative to data_path.

        Returns:
            DataFrame with parsed timestamps and validated schema.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing.
        """
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info("Loading data from %s", filepath)
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        self._validate_schema(df)
        logger.info("Loaded %d records with %d columns", len(df), len(df.columns))
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate that the DataFrame has all required columns.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If required columns are missing.
        """
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def handle_missing(self, df: pd.DataFrame, strategy: str = "interpolate") -> pd.DataFrame:
        """Handle missing values in the DataFrame.

        Args:
            df: DataFrame with potential missing values.
            strategy: Method to handle missing values.
                'interpolate' uses linear interpolation for numeric columns.
                'drop' removes rows with any missing values.
                'ffill' forward-fills missing values.

        Returns:
            DataFrame with missing values handled.

        Raises:
            ValueError: If an unknown strategy is provided.
        """
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            logger.info("No missing values found")
            return df

        logger.info("Found %d missing values, applying '%s' strategy", missing_count, strategy)

        if strategy == "interpolate":
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
            df = df.ffill().bfill()
        elif strategy == "drop":
            df = df.dropna()
        elif strategy == "ffill":
            df = df.ffill().bfill()
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

        return df
