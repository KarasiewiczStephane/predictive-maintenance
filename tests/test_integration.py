"""Integration tests for the predictive maintenance pipeline.

Tests full data-to-prediction flows and component interactions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import FeatureEngineer
from src.data.loader import DataLoader
from src.models.evaluation import ModelEvaluator
from src.models.trainer import MaintenancePredictor


class TestDataToPredictionFlow:
    """Integration tests for the full pipeline."""

    def test_load_engineer_train_predict(self, tmp_path: Path) -> None:
        """Full flow: load CSV -> engineer features -> train -> predict."""
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
        csv_path = tmp_path / "data"
        csv_path.mkdir()
        df.to_csv(csv_path / "sensors.csv", index=False)

        # Load
        loader = DataLoader(csv_path)
        loaded = loader.load_csv("sensors.csv")
        assert len(loaded) == n

        # Handle missing
        clean = loader.handle_missing(loaded)
        assert clean.isnull().sum().sum() == 0

        # Feature engineering
        fe = FeatureEngineer(window_sizes=[3, 5])
        featured = fe.create_features(clean)
        assert len(fe.get_feature_names()) > 0
        assert featured.isnull().sum().sum() == 0

        # Train
        X = featured[fe.get_feature_names()]
        y = featured["failure"]
        model = MaintenancePredictor()
        metrics = model.train(X, y)
        assert metrics["cv_roc_auc_mean"] >= 0

        # Predict
        preds = model.predict(X)
        assert len(preds) == len(X)
        probas = model.predict_proba(X)
        assert all(0 <= p <= 1 for p in probas)

    def test_train_evaluate_save_load(self, tmp_path: Path) -> None:
        """Train -> evaluate -> save -> load -> predictions match."""
        np.random.seed(42)
        n = 150
        X = pd.DataFrame(
            {
                "temperature": np.random.uniform(60, 90, n),
                "vibration": np.random.uniform(3, 8, n),
                "pressure": np.random.uniform(90, 110, n),
                "rpm": np.random.uniform(2800, 3200, n),
            }
        )
        y = pd.Series([0] * 130 + [1] * 20)

        # Train
        model = MaintenancePredictor()
        model.train(X, y)
        original_probas = model.predict_proba(X)

        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y.values, original_probas)
        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

        # Save metrics
        evaluator.save_metrics(tmp_path / "metrics.json")
        assert (tmp_path / "metrics.json").exists()

        # Save and reload model
        model.save(tmp_path / "model.joblib")
        loaded_model = MaintenancePredictor()
        loaded_model.load(tmp_path / "model.joblib")
        loaded_probas = loaded_model.predict_proba(X)
        np.testing.assert_array_almost_equal(original_probas, loaded_probas)

    def test_multi_equipment_pipeline(self) -> None:
        """Pipeline handles multiple equipment IDs correctly."""
        np.random.seed(42)
        dfs = []
        for eq_id in ["EQ-001", "EQ-002", "EQ-003"]:
            n = 50
            dfs.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
                        "equipment_id": [eq_id] * n,
                        "temperature": np.random.uniform(60, 90, n),
                        "vibration": np.random.uniform(3, 8, n),
                        "pressure": np.random.uniform(90, 110, n),
                        "rpm": np.random.uniform(2800, 3200, n),
                        "failure": [0] * 45 + [1] * 5,
                    }
                )
            )
        combined = pd.concat(dfs, ignore_index=True)

        fe = FeatureEngineer(window_sizes=[3])
        featured = fe.create_features(combined)
        assert featured["equipment_id"].nunique() == 3

        X = featured[fe.get_feature_names()]
        y = featured["failure"]
        model = MaintenancePredictor()
        metrics = model.train(X, y)
        assert metrics["cv_roc_auc_mean"] >= 0

    def test_xgboost_pipeline(self, tmp_path: Path) -> None:
        """XGBoost model works through the full pipeline."""
        np.random.seed(42)
        n = 150
        X = pd.DataFrame(
            {
                "temperature": np.random.uniform(60, 90, n),
                "vibration": np.random.uniform(3, 8, n),
                "pressure": np.random.uniform(90, 110, n),
                "rpm": np.random.uniform(2800, 3200, n),
            }
        )
        y = pd.Series([0] * 130 + [1] * 20)

        model = MaintenancePredictor(model_type="xgboost")
        metrics = model.train(X, y)
        assert "cv_roc_auc_mean" in metrics

        model.save(tmp_path / "xgb_model.joblib")
        loaded = MaintenancePredictor(model_type="xgboost")
        loaded.load(tmp_path / "xgb_model.joblib")
        preds = loaded.predict(X)
        assert len(preds) == n

    def test_optimal_threshold_integration(self) -> None:
        """Optimal threshold finder works with real model outputs."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame(
            {
                "temperature": np.random.uniform(60, 90, n),
                "vibration": np.random.uniform(3, 8, n),
                "pressure": np.random.uniform(90, 110, n),
                "rpm": np.random.uniform(2800, 3200, n),
            }
        )
        y = pd.Series([0] * 85 + [1] * 15)

        model = MaintenancePredictor()
        model.train(X, y)
        probas = model.predict_proba(X)

        evaluator = ModelEvaluator()
        threshold = evaluator.find_optimal_threshold(y.values, probas, metric="f1")
        assert 0.1 <= threshold <= 0.85

        # Re-evaluate with optimal threshold
        evaluator_opt = ModelEvaluator(threshold=threshold)
        metrics = evaluator_opt.evaluate(y.values, probas)
        assert metrics["threshold"] == threshold
