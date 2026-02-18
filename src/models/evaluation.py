"""Model evaluation module with comprehensive metrics for imbalanced classification.

Provides precision, recall, F1, ROC-AUC, average precision, confusion matrix,
and optimal threshold finding for predictive maintenance models.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates classification models with metrics suited for imbalanced data.

    Args:
        threshold: Decision threshold for converting probabilities to labels.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.metrics: dict[str, Any] = {}

    def evaluate(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict[str, Any]:
        """Compute comprehensive evaluation metrics.

        Args:
            y_true: True binary labels (0 or 1).
            y_pred_proba: Predicted probabilities for the positive class.

        Returns:
            Dictionary with roc_auc, average_precision, precision, recall,
            f1_score, confusion_matrix, and threshold values.
        """
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # Handle edge case where only one class is present
        if len(np.unique(y_true)) < 2:
            roc_auc = 0.0
            avg_precision = 0.0
            logger.warning("ROC-AUC undefined (only one class present)")
        else:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)

        positive_key = "1" if "1" in report else "1.0"
        positive_metrics = report.get(positive_key, {})

        self.metrics = {
            "roc_auc": float(roc_auc),
            "average_precision": float(avg_precision),
            "precision": float(positive_metrics.get("precision", 0.0)),
            "recall": float(positive_metrics.get("recall", 0.0)),
            "f1_score": float(positive_metrics.get("f1-score", 0.0)),
            "confusion_matrix": cm.tolist(),
            "threshold": self.threshold,
        }

        logger.info("ROC-AUC: %.4f", self.metrics["roc_auc"])
        logger.info("Precision: %.4f", self.metrics["precision"])
        logger.info("Recall: %.4f", self.metrics["recall"])
        logger.info("F1-Score: %.4f", self.metrics["f1_score"])

        return self.metrics

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """Find the optimal decision threshold based on a target metric.

        Args:
            y_true: True binary labels.
            y_pred_proba: Predicted probabilities for the positive class.
            metric: Metric to optimize ('f1' supported).

        Returns:
            Optimal threshold value that maximizes the target metric.

        Raises:
            ValueError: If the metric is not supported.
        """
        if metric != "f1":
            raise ValueError(f"Unsupported metric: {metric}. Only 'f1' is supported.")

        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = 0.0

        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(thresh)

        logger.info(
            "Optimal threshold for %s: %.2f (score: %.4f)",
            metric,
            best_threshold,
            best_score,
        )
        return best_threshold

    def save_metrics(self, path: Path) -> None:
        """Save evaluation metrics to a JSON file.

        Args:
            path: File path for the metrics JSON output.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info("Metrics saved to %s", path)
