"""Generate realistic sample sensor data for predictive maintenance demos.

Creates a CSV dataset with simulated equipment sensor readings including
temperature, vibration, pressure, and RPM with degradation patterns
leading to equipment failures.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(n_equipment: int = 5, n_days: int = 30) -> pd.DataFrame:
    """Generate synthetic equipment sensor data with failure patterns.

    Simulates sensor readings for multiple pieces of equipment over a
    specified number of days. Equipment 1 degrades after day 20 and
    fails on day 28+. Equipment 3 shows intermittent anomalies.

    Args:
        n_equipment: Number of equipment units to simulate.
        n_days: Number of days of data to generate.

    Returns:
        DataFrame with timestamped sensor readings and failure labels.
    """
    np.random.seed(42)
    records = []

    for eq_id in range(1, n_equipment + 1):
        base_temp = np.random.uniform(60, 80)
        base_vibration = np.random.uniform(4, 6)
        base_pressure = np.random.uniform(95, 105)
        base_rpm = np.random.uniform(2900, 3100)

        for day in range(n_days):
            for hour in range(24):
                timestamp = datetime(2024, 1, 1) + timedelta(days=day, hours=hour)

                # Equipment 1: gradual degradation leading to failure
                degradation = 0
                if day > 20 and eq_id == 1:
                    degradation = (day - 20) * 2

                # Equipment 3: intermittent spikes
                spike = 0
                if eq_id == 3 and day % 7 == 0 and 10 <= hour <= 14:
                    spike = np.random.uniform(5, 15)

                temp = base_temp + degradation + spike + np.random.normal(0, 2)
                vib = base_vibration + degradation * 0.5 + spike * 0.3 + np.random.normal(0, 0.5)
                pres = base_pressure + np.random.normal(0, 5) - degradation * 0.3
                rpm = base_rpm + np.random.normal(0, 50) - degradation * 10

                failure = 1 if (day >= 28 and eq_id == 1) else 0

                records.append(
                    {
                        "timestamp": timestamp,
                        "equipment_id": f"EQ-{eq_id:03d}",
                        "temperature": round(temp, 2),
                        "vibration": round(vib, 2),
                        "pressure": round(pres, 2),
                        "rpm": round(rpm, 2),
                        "failure": failure,
                    }
                )

    return pd.DataFrame(records)


if __name__ == "__main__":
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_sample_data()
    output_path = output_dir / "equipment_sensors.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records -> {output_path}")
    print(f"Equipment IDs: {df['equipment_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Failure rate: {df['failure'].mean():.2%}")
