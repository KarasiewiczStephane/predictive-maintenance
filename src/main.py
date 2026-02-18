"""Main application entry point for the predictive maintenance pipeline.

Orchestrates data loading, feature engineering, model training,
evaluation, and API serving through CLI commands.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import settings
from src.data.features import FeatureEngineer
from src.data.loader import DataLoader
from src.models.evaluation import ModelEvaluator
from src.models.trainer import MaintenancePredictor

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure structured logging based on application settings."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def train_pipeline(data_file: str) -> dict:
    """Run the full training pipeline from data to saved model.

    Args:
        data_file: Filename of the CSV data file relative to the data directory.

    Returns:
        Dictionary with training and evaluation metrics.
    """
    # Load data
    logger.info("Loading data from %s...", data_file)
    loader = DataLoader(settings.data_dir)
    df = loader.load_csv(data_file)
    df = loader.handle_missing(df, strategy=settings.missing_value_strategy)

    # Feature engineering
    logger.info("Creating features...")
    fe = FeatureEngineer(window_sizes=settings.get_window_sizes())
    df = fe.create_features(df)

    # Split data
    feature_cols = fe.get_feature_names()
    X = df[feature_cols]
    y = df["failure"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y,
    )

    # Train model
    logger.info("Training %s model...", settings.model_type)
    model = MaintenancePredictor(
        model_type=settings.model_type,
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        random_state=settings.random_state,
    )
    train_metrics = model.train(X_train, y_train)
    logger.info(
        "CV ROC-AUC: %.4f (+/- %.4f)",
        train_metrics["cv_roc_auc_mean"],
        train_metrics["cv_roc_auc_std"],
    )

    # Evaluate
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator()
    y_pred_proba = model.predict_proba(X_test)
    eval_metrics = evaluator.evaluate(y_test.values, y_pred_proba)

    # Save artifacts
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model.joblib")
    evaluator.save_metrics(model_dir / "metrics.json")
    logger.info("Model and metrics saved to %s", model_dir)

    return {**train_metrics, **eval_metrics}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance - Equipment Failure Prediction"
    )
    parser.add_argument(
        "--train",
        type=str,
        metavar="FILE",
        help="Train model with the specified data file",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the prediction API server",
    )
    return parser


def main() -> None:
    """Main entry point for CLI execution."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.train:
        train_pipeline(args.train)
    elif args.serve:
        from src.api.server import start_server

        start_server()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
