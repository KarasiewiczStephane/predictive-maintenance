"""Tests for the predictive maintenance dashboard data generators."""

import pandas as pd

from src.dashboard.app import (
    SENSOR_NAMES,
    generate_model_comparison,
    generate_prediction_distribution,
    generate_sensor_data,
    generate_threshold_analysis,
)


class TestModelComparison:
    def test_returns_dataframe(self) -> None:
        df = generate_model_comparison()
        assert isinstance(df, pd.DataFrame)

    def test_has_two_models(self) -> None:
        df = generate_model_comparison()
        assert len(df) == 2

    def test_has_required_columns(self) -> None:
        df = generate_model_comparison()
        for col in ["model", "precision", "recall", "f1", "roc_auc"]:
            assert col in df.columns

    def test_scores_bounded(self) -> None:
        df = generate_model_comparison()
        for col in ["precision", "recall", "f1", "roc_auc"]:
            assert (df[col] >= 0).all()
            assert (df[col] <= 1).all()

    def test_reproducible(self) -> None:
        df1 = generate_model_comparison(seed=99)
        df2 = generate_model_comparison(seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestSensorData:
    def test_returns_dataframe(self) -> None:
        df = generate_sensor_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_200_readings(self) -> None:
        df = generate_sensor_data()
        assert len(df) == 200

    def test_has_all_sensors(self) -> None:
        df = generate_sensor_data()
        for sensor in SENSOR_NAMES:
            assert sensor in df.columns

    def test_has_failure_column(self) -> None:
        df = generate_sensor_data()
        assert "failure" in df.columns
        assert df["failure"].dtype == bool

    def test_sensor_values_positive(self) -> None:
        df = generate_sensor_data()
        for sensor in SENSOR_NAMES:
            assert (df[sensor] > 0).all()


class TestPredictionDistribution:
    def test_returns_dataframe(self) -> None:
        df = generate_prediction_distribution()
        assert isinstance(df, pd.DataFrame)

    def test_has_400_samples(self) -> None:
        df = generate_prediction_distribution()
        assert len(df) == 400

    def test_has_both_classes(self) -> None:
        df = generate_prediction_distribution()
        assert set(df["actual"].unique()) == {"Healthy", "Failure"}

    def test_probabilities_bounded(self) -> None:
        df = generate_prediction_distribution()
        assert (df["probability"] >= 0).all()
        assert (df["probability"] <= 1).all()


class TestThresholdAnalysis:
    def test_returns_dataframe(self) -> None:
        df = generate_threshold_analysis()
        assert isinstance(df, pd.DataFrame)

    def test_has_entries(self) -> None:
        df = generate_threshold_analysis()
        assert len(df) > 10

    def test_threshold_bounded(self) -> None:
        df = generate_threshold_analysis()
        assert (df["threshold"] >= 0.1).all()
        assert (df["threshold"] <= 0.9).all()

    def test_metrics_bounded(self) -> None:
        df = generate_threshold_analysis()
        for col in ["precision", "recall", "f1"]:
            assert (df[col] >= 0).all()
            assert (df[col] <= 1).all()
