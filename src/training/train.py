"""
Training orchestrator for recommendation models.

This module provides the main entry point for training models:
- Loads data and builds interaction matrices
- Trains configured models (Item-Item, ALS)
- Evaluates on validation set
- Logs everything to MLflow
- Returns the best model based on primary metric

Usage:
    # As a module (CLI)
    python -m src.training.train

    # Programmatically
    from src.training.train import train_all_models, train_single_model
    results = train_all_models()
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from src.config import get_training_config
from src.data.mappings import IDMapper
from src.data.split import load_splits
from src.models.als import ALSRecommender
from src.models.base import BaseRecommender
from src.models.item_item import ItemItemRecommender
from src.training.evaluate import evaluate_model, print_metrics
from src.training.mlflow_utils import (
    log_dict_as_artifact,
    log_metrics,
    log_model_params,
    log_params,
    setup_mlflow,
    start_run,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result from training a single model."""

    model_name: str
    model: BaseRecommender
    metrics: dict[str, float]
    run_id: str


def build_interaction_matrix(
    df,
    mapper: IDMapper,
) -> csr_matrix:
    """Build sparse interaction matrix from DataFrame.

    Args:
        df: DataFrame with user_id and item_id columns.
        mapper: ID mapper for converting IDs to indices.

    Returns:
        Sparse CSR matrix of shape (n_users, n_items).
    """
    rows = df["user_id"].map(mapper.user_to_idx).values
    cols = df["item_id"].map(mapper.item_to_idx).values
    data = np.ones(len(df))

    return csr_matrix(
        (data, (rows, cols)),
        shape=(mapper.n_users, mapper.n_items),
    )


def train_item_item(
    train_matrix: csr_matrix,
    val_matrix: csr_matrix,
    k_neighbors: int = 50,
    min_similarity: float = 0.0,
    k_values: Optional[list[int]] = None,
) -> TrainingResult:
    """Train and evaluate Item-Item model.

    Args:
        train_matrix: Training interaction matrix.
        val_matrix: Validation interaction matrix.
        k_neighbors: Number of neighbors for similarity.
        min_similarity: Minimum similarity threshold.
        k_values: K values for evaluation metrics.

    Returns:
        TrainingResult with model and metrics.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    # config = get_training_config()

    with start_run(run_name="item-item", tags={"model_type": "item_item"}) as run:
        # Create and train model
        model = ItemItemRecommender(
            k_neighbors=k_neighbors,
            min_similarity=min_similarity,
        )

        logger.info("Training Item-Item model...")
        model.fit(train_matrix, show_progress=True)

        # Log parameters
        log_model_params(model)
        log_params(
            {
                "n_users": train_matrix.shape[0],
                "n_items": train_matrix.shape[1],
                "n_train_interactions": train_matrix.nnz,
            }
        )

        # Evaluate
        logger.info("Evaluating Item-Item model...")
        metrics = evaluate_model(
            model,
            train_matrix,
            val_matrix,
            k_values=k_values,
            show_progress=True,
        )

        # Log metrics
        log_metrics(metrics)

        # Log config as artifact
        log_dict_as_artifact(model.get_params(), "model_params.json")

        print_metrics(metrics, title="Item-Item Model - Validation")

        return TrainingResult(
            model_name="item_item",
            model=model,
            metrics=metrics,
            run_id=run.info.run_id,
        )


def train_als(
    train_matrix: csr_matrix,
    val_matrix: csr_matrix,
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 15,
    alpha: float = 1.0,
    random_state: int = 42,
    k_values: Optional[list[int]] = None,
) -> TrainingResult:
    """Train and evaluate ALS model.

    Args:
        train_matrix: Training interaction matrix.
        val_matrix: Validation interaction matrix.
        factors: Number of latent factors.
        regularization: L2 regularization.
        iterations: Number of ALS iterations.
        alpha: Confidence scaling factor.
        random_state: Random seed.
        k_values: K values for evaluation metrics.

    Returns:
        TrainingResult with model and metrics.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    with start_run(run_name="als", tags={"model_type": "als"}) as run:
        # Create and train model
        model = ALSRecommender(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
            random_state=random_state,
        )

        logger.info("Training ALS model...")
        model.fit(train_matrix, show_progress=True)

        # Log parameters
        log_model_params(model)
        log_params(
            {
                "n_users": train_matrix.shape[0],
                "n_items": train_matrix.shape[1],
                "n_train_interactions": train_matrix.nnz,
            }
        )

        # Evaluate
        logger.info("Evaluating ALS model...")
        metrics = evaluate_model(
            model,
            train_matrix,
            val_matrix,
            k_values=k_values,
            show_progress=True,
        )

        # Log metrics
        log_metrics(metrics)

        # Log config as artifact
        log_dict_as_artifact(model.get_params(), "model_params.json")

        print_metrics(metrics, title="ALS Model - Validation")

        return TrainingResult(
            model_name="als",
            model=model,
            metrics=metrics,
            run_id=run.info.run_id,
        )


def train_all_models() -> dict[str, TrainingResult]:
    """Train all enabled models and return results.

    Reads configuration from training.yaml to determine which models
    to train and their hyperparameters.

    Returns:
        Dictionary mapping model names to TrainingResult objects.
    """
    config = get_training_config()

    # Setup MLflow
    setup_mlflow(
        experiment_name=config.mlflow.experiment_name,
        tracking_uri=config.mlflow.tracking_uri,
    )

    # Load data
    logger.info("Loading data...")
    train_df, val_df, test_df = load_splits()
    mapper = IDMapper.from_disk()

    logger.info(f"Train: {len(train_df):,} interactions")
    logger.info(f"Val: {len(val_df):,} interactions")
    logger.info(f"Users: {mapper.n_users:,}, Items: {mapper.n_items:,}")

    # Build matrices
    logger.info("Building interaction matrices...")
    train_matrix = build_interaction_matrix(train_df, mapper)
    val_matrix = build_interaction_matrix(val_df, mapper)

    results: dict[str, TrainingResult] = {}
    k_values = config.evaluation.k_values

    # Train Item-Item if enabled
    if config.models.item_item.enabled:
        logger.info("\n" + "=" * 50)
        logger.info("Training Item-Item Model")
        logger.info("=" * 50)

        results["item_item"] = train_item_item(
            train_matrix=train_matrix,
            val_matrix=val_matrix,
            k_neighbors=config.models.item_item.k_neighbors,
            min_similarity=config.models.item_item.min_similarity,
            k_values=k_values,
        )

    # Train ALS if enabled
    if config.models.als.enabled:
        logger.info("\n" + "=" * 50)
        logger.info("Training ALS Model")
        logger.info("=" * 50)

        results["als"] = train_als(
            train_matrix=train_matrix,
            val_matrix=val_matrix,
            factors=config.models.als.factors,
            regularization=config.models.als.regularization,
            iterations=config.models.als.iterations,
            alpha=config.models.als.alpha,
            random_state=config.random_seed,
            k_values=k_values,
        )

    # Determine best model
    if results:
        primary_metric = config.evaluation.primary_metric

        best_model_name = max(
            results.keys(),
            key=lambda k: results[k].metrics.get(primary_metric, 0),
        )

        logger.info("\n" + "=" * 50)
        logger.info("Training Summary")
        logger.info("=" * 50)

        for name, result in results.items():
            metric_value = result.metrics.get(primary_metric, 0)
            marker = " ← BEST" if name == best_model_name else ""
            logger.info(f"{name}: {primary_metric}={metric_value:.4f}{marker}")

    return results


def get_best_model(results: dict[str, TrainingResult]) -> Optional[TrainingResult]:
    """Get the best model from training results.

    Args:
        results: Dictionary of training results.

    Returns:
        Best TrainingResult based on primary metric, or None if empty.
    """
    if not results:
        return None

    config = get_training_config()
    primary_metric = config.evaluation.primary_metric

    best_name = max(
        results.keys(),
        key=lambda k: results[k].metrics.get(primary_metric, 0),
    )

    return results[best_name]


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument(
        "--model",
        choices=["item_item", "als", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    args = parser.parse_args()

    if args.model == "all":
        results = train_all_models()
        best = get_best_model(results)
        if best:
            print(f"\n✓ Best model: {best.model_name}")
    else:
        # Load data
        config = get_training_config()
        setup_mlflow(config.mlflow.experiment_name, config.mlflow.tracking_uri)

        train_df, val_df, _ = load_splits()
        mapper = IDMapper.from_disk()
        train_matrix = build_interaction_matrix(train_df, mapper)
        val_matrix = build_interaction_matrix(val_df, mapper)

        if args.model == "item_item":
            result = train_item_item(
                train_matrix,
                val_matrix,
                k_neighbors=config.models.item_item.k_neighbors,
                k_values=config.evaluation.k_values,
            )
        else:
            result = train_als(
                train_matrix,
                val_matrix,
                factors=config.models.als.factors,
                regularization=config.models.als.regularization,
                iterations=config.models.als.iterations,
                k_values=config.evaluation.k_values,
            )

        print(f"\n✓ Training complete: {result.model_name}")
