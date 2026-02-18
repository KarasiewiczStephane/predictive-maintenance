"""Model training module for predictive maintenance.

Supports multiple classification algorithms (Random Forest, XGBoost)
with cross-validation, model persistence, and prediction capabilities.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MaintenancePredictor:
    """Predictive maintenance model with training, prediction, and persistence.

    Supports Random Forest and XGBoost classifiers with feature scaling
    and cross-validation scoring.

    Args:
        model_type: Algorithm to use ('random_forest' or 'xgboost').
        **model_params: Additional parameters passed to the underlying model.
    """

    def __init__(self, model_type: str = "random_forest", **model_params: Any) -> None:
        self.model_type = model_type
        self.model_params = model_params
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

    def _create_model(self) -> Any:
        """Create the underlying classification model.

        Returns:
            Configured classifier instance.

        Raises:
            ValueError: If model_type is not supported.
        """
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.model_params.get("n_estimators", 100),
                max_depth=self.model_params.get("max_depth", 10),
                class_weight="balanced",
                random_state=self.model_params.get("random_state", 42),
            )
        elif self.model_type == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=self.model_params.get("n_estimators", 100),
                max_depth=self.model_params.get("max_depth", 6),
                scale_pos_weight=self.model_params.get("scale_pos_weight", 10),
                random_state=self.model_params.get("random_state", 42),
                eval_metric="logloss",
            )
        raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Train the model and return cross-validation metrics.

        Args:
            X: Feature DataFrame.
            y: Target series with binary labels (0/1).

        Returns:
            Dictionary with cv_roc_auc_mean and cv_roc_auc_std.
        """
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring="roc_auc")
        metrics = {
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
        }
        logger.info(
            "Training complete - CV ROC-AUC: %.4f (+/- %.4f)",
            metrics["cv_roc_auc_mean"],
            metrics["cv_roc_auc_std"],
        )
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict failure labels for input features.

        Args:
            X: Feature DataFrame matching training schema.

        Returns:
            Array of predicted labels (0 or 1).
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict failure probabilities for input features.

        Args:
            X: Feature DataFrame matching training schema.

        Returns:
            Array of failure probabilities (class 1).
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path: Path) -> None:
        """Save model, scaler, and feature names to disk.

        Args:
            path: File path for the saved model artifact.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "model_type": self.model_type,
            },
            path,
        )
        logger.info("Model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load model, scaler, and feature names from disk.

        Args:
            path: File path to load the model artifact from.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.model_type = data.get("model_type", "random_forest")
        logger.info("Model loaded from %s", path)
