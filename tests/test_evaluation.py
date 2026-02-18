"""Tests for the model evaluation module."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.evaluation import ModelEvaluator


@pytest.fixture
def binary_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Create known binary classification results."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.5, 0.7, 0.8, 0.9, 0.95])
    return y_true, y_pred_proba


@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Create perfect predictions for edge case testing."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred_proba = np.array([0.0, 0.1, 0.1, 0.2, 0.2, 0.8, 0.9, 0.9, 1.0, 1.0])
    return y_true, y_pred_proba


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_evaluate_returns_all_metrics(self, binary_predictions: tuple) -> None:
        """Evaluate returns all expected metric keys."""
        y_true, y_pred_proba = binary_predictions
        evaluator = ModelEvaluator(threshold=0.5)
        metrics = evaluator.evaluate(y_true, y_pred_proba)

        expected_keys = {
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "f1_score",
            "confusion_matrix",
            "threshold",
        }
        assert set(metrics.keys()) == expected_keys

    def test_evaluate_metric_ranges(self, binary_predictions: tuple) -> None:
        """All probability-based metrics are in [0, 1]."""
        y_true, y_pred_proba = binary_predictions
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred_proba)

        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_perfect_predictions(self, perfect_predictions: tuple) -> None:
        """Perfect predictions yield high metrics."""
        y_true, y_pred_proba = perfect_predictions
        evaluator = ModelEvaluator(threshold=0.5)
        metrics = evaluator.evaluate(y_true, y_pred_proba)

        assert metrics["roc_auc"] >= 0.9
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_confusion_matrix_shape(self, binary_predictions: tuple) -> None:
        """Confusion matrix has correct 2x2 shape."""
        y_true, y_pred_proba = binary_predictions
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred_proba)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_threshold_stored(self) -> None:
        """Custom threshold is stored in metrics."""
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.3, 0.7, 0.4, 0.8])
        evaluator = ModelEvaluator(threshold=0.6)
        metrics = evaluator.evaluate(y_true, y_pred_proba)
        assert metrics["threshold"] == 0.6

    def test_find_optimal_threshold(self, binary_predictions: tuple) -> None:
        """Optimal threshold is valid and in range."""
        y_true, y_pred_proba = binary_predictions
        evaluator = ModelEvaluator()
        threshold = evaluator.find_optimal_threshold(y_true, y_pred_proba, metric="f1")
        assert 0.1 <= threshold <= 0.85

    def test_find_optimal_threshold_unsupported_metric(self, binary_predictions: tuple) -> None:
        """Unsupported metric raises ValueError."""
        y_true, y_pred_proba = binary_predictions
        evaluator = ModelEvaluator()
        with pytest.raises(ValueError, match="Unsupported metric"):
            evaluator.find_optimal_threshold(y_true, y_pred_proba, metric="accuracy")

    def test_save_metrics(self, binary_predictions: tuple, tmp_path: Path) -> None:
        """Metrics save to JSON correctly."""
        y_true, y_pred_proba = binary_predictions
        evaluator = ModelEvaluator()
        evaluator.evaluate(y_true, y_pred_proba)

        metrics_path = tmp_path / "metrics.json"
        evaluator.save_metrics(metrics_path)

        assert metrics_path.exists()
        with open(metrics_path) as f:
            loaded = json.load(f)
        assert "roc_auc" in loaded
        assert "f1_score" in loaded

    def test_all_same_class(self) -> None:
        """Handles edge case where all predictions are the same class."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.1, 0.2])
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred_proba)
        assert metrics["roc_auc"] == 0.0
