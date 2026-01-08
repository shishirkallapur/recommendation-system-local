"""
MLflow utilities for experiment tracking and model registry.

This module provides helper functions for:
- Setting up MLflow experiments
- Logging parameters, metrics, and artifacts
- Managing model registry (staging, production)
- Comparing runs and promoting models

Usage:
    from src.training.mlflow_utils import (
        setup_mlflow,
        start_run,
        log_model_params,
        log_metrics,
        log_artifacts,
    )

    setup_mlflow("movie-recommender")

    with start_run(run_name="als-v1"):
        log_model_params(model)
        log_metrics(metrics)
        log_artifacts(artifacts_dir)
"""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import mlflow
from mlflow.tracking import MlflowClient

from src.config import get_project_root

logger = logging.getLogger(__name__)

# Default MLflow tracking URI (local file storage)
DEFAULT_TRACKING_URI = f"file://{get_project_root() / 'mlruns'}"


def setup_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
) -> Any:
    """Set up MLflow tracking.

    Creates the experiment if it doesn't exist.

    Args:
        experiment_name: Name of the experiment.
        tracking_uri: MLflow tracking URI. Defaults to local ./mlruns.

    Returns:
        Experiment ID.
    """
    if tracking_uri is None:
        tracking_uri = DEFAULT_TRACKING_URI

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created experiment '{experiment_name}' with ID {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(
            f"Using existing experiment '{experiment_name}' (ID: {experiment_id})"
        )

    mlflow.set_experiment(experiment_name)

    return experiment_id


@contextmanager
def start_run(
    run_name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[dict[str, str]] = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager for MLflow run.

    Args:
        run_name: Name for the run.
        nested: Whether this is a nested run.
        tags: Optional tags to add to the run.

    Yields:
        Active MLflow run.

    Example:
        with start_run(run_name="als-experiment") as run:
            # Training code here
            log_metrics({"ndcg@10": 0.18})
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags(tags)
        logger.info(f"Started MLflow run: {run.info.run_id} ({run_name})")
        yield run
        logger.info(f"Finished MLflow run: {run.info.run_id}")


def log_model_params(model: Any) -> None:
    """Log model hyperparameters to MLflow.

    Expects the model to have a get_params() method.

    Args:
        model: Model with get_params() method.
    """
    if hasattr(model, "get_params"):
        params = model.get_params()
        mlflow.log_params(params)
        logger.debug(f"Logged params: {params}")
    else:
        logger.warning(f"Model {type(model)} has no get_params() method")


def log_params(params: dict[str, Any]) -> None:
    """Log arbitrary parameters to MLflow.

    Args:
        params: Dictionary of parameter names to values.
    """
    # MLflow params must be strings, numbers, or booleans
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)):
            clean_params[k] = v
        else:
            clean_params[k] = str(v)

    mlflow.log_params(clean_params)
    logger.debug(f"Logged params: {clean_params}")


def sanitize_metric_name(name: str) -> str:
    """Sanitize metric name for MLflow compatibility.

    MLflow only allows alphanumerics, underscores, dashes, periods,
    spaces, colons, and slashes.

    Args:
        name: Original metric name (e.g., "ndcg@10")

    Returns:
        Sanitized name (e.g., "ndcg_at_10")
    """
    # Replace @ with _at_
    return name.replace("@", "_at_")


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to MLflow.

    Args:
        metrics: Dictionary of metric names to values.
        step: Optional step number for tracking metrics over time.
    """
    # Sanitize metric names for MLflow compatibility
    sanitized = {sanitize_metric_name(k): v for k, v in metrics.items()}
    mlflow.log_metrics(sanitized, step=step)
    logger.debug(f"Logged metrics: {sanitized}")


def log_artifact(
    local_path: Union[str, Path], artifact_path: Optional[str] = None
) -> None:
    """Log a single artifact file to MLflow.

    Args:
        local_path: Path to the local file.
        artifact_path: Optional subdirectory in the artifact store.
    """
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
    logger.debug(f"Logged artifact: {local_path}")


def log_artifacts(
    local_dir: Union[str, Path], artifact_path: Optional[str] = None
) -> None:
    """Log all files in a directory as artifacts.

    Args:
        local_dir: Path to local directory.
        artifact_path: Optional subdirectory in the artifact store.
    """
    mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
    logger.debug(f"Logged artifacts from: {local_dir}")


def log_dict_as_artifact(data: dict, filename: str) -> None:
    """Log a dictionary as a JSON artifact.

    Args:
        data: Dictionary to save.
        filename: Name of the JSON file (e.g., "config.json").
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        mlflow.log_artifact(str(filepath))

    logger.debug(f"Logged dict as artifact: {filename}")


def get_best_run(
    experiment_name: str,
    metric: str = "ndcg@10",
    higher_is_better: bool = True,
) -> Optional[dict[str, Any]]:
    """Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment.
        metric: Metric to compare runs by (will be sanitized for MLflow).
        higher_is_better: Whether higher metric values are better.

    Returns:
        Dictionary with run info, or None if no runs found.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return None

    # Sanitize metric name for MLflow query
    sanitized_metric = sanitize_metric_name(metric)

    # Search for runs with the metric
    order = "DESC" if higher_is_better else "ASC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.`{sanitized_metric}` > 0",
        order_by=[f"metrics.`{sanitized_metric}` {order}"],
        max_results=1,
    )

    if not runs:
        logger.warning(f"No runs found with metric '{sanitized_metric}'")
        return None

    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "run_name": best_run.info.run_name,
        "metrics": best_run.data.metrics,
        "params": best_run.data.params,
        "artifact_uri": best_run.info.artifact_uri,
    }


def compare_runs(
    experiment_name: str,
    metric: str = "ndcg@10",
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Compare top runs in an experiment.

    Args:
        experiment_name: Name of the experiment.
        metric: Metric to compare by (will be sanitized for MLflow).
        top_n: Number of top runs to return.

    Returns:
        List of run info dictionaries, sorted by metric (best first).
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return []

    # Sanitize metric name for MLflow query
    sanitized_metric = sanitize_metric_name(metric)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.`{sanitized_metric}` DESC"],
        max_results=top_n,
    )

    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            metric: run.data.metrics.get(sanitized_metric),
            "params": run.data.params,
        }
        for run in runs
    ]


def get_run_artifacts_path(run_id: str) -> Path:
    """Get the local path to a run's artifacts.

    Args:
        run_id: MLflow run ID.

    Returns:
        Path to artifacts directory.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri

    # Convert file:// URI to path
    if artifact_uri.startswith("file://"):
        return Path(artifact_uri[7:])

    return Path(artifact_uri)


def should_promote_model(
    new_metrics: dict[str, float],
    current_metrics: Optional[dict[str, float]],
    primary_metric: str = "ndcg@10",
    min_improvement: float = 0.01,
    max_regression: float = 0.05,
) -> tuple[bool, str]:
    """Determine if a new model should be promoted to production.

    Args:
        new_metrics: Metrics from the new model.
        current_metrics: Metrics from current production model (None if no current).
        primary_metric: Primary metric to compare.
        min_improvement: Minimum relative improvement required on primary metric.
        max_regression: Maximum allowed regression on any metric.

    Returns:
        Tuple of (should_promote, reason).
    """
    # No current model - always promote
    if current_metrics is None:
        return True, "No existing production model"

    # Check primary metric improvement
    new_value = new_metrics.get(primary_metric, 0)
    current_value = current_metrics.get(primary_metric, 0)

    if current_value > 0:
        improvement = (new_value - current_value) / current_value
    else:
        improvement = 1.0 if new_value > 0 else 0.0

    if improvement < min_improvement:
        return False, (
            f"Insufficient improvement on {primary_metric}: "
            f"{improvement:.2%} < {min_improvement:.2%} required"
        )

    # Check for regression on other metrics
    for metric, new_val in new_metrics.items():
        if metric == primary_metric or metric == "n_users_evaluated":
            continue

        current_val = current_metrics.get(metric, 0)
        if current_val > 0:
            change = (new_val - current_val) / current_val
            if change < -max_regression:
                return False, (
                    f"Regression on {metric}: {change:.2%} "
                    f"exceeds max allowed {-max_regression:.2%}"
                )

    return True, f"Improved {primary_metric} by {improvement:.2%}"


def get_current_run_id() -> Optional[str]:
    """Get the current active run ID.

    Returns:
        Run ID or None if no active run.
    """
    run = mlflow.active_run()
    return run.info.run_id if run else None


def set_run_tag(key: str, value: str) -> None:
    """Set a tag on the current run.

    Args:
        key: Tag key.
        value: Tag value.
    """
    mlflow.set_tag(key, value)


def end_run_if_active() -> None:
    """End the current run if one is active.

    Useful for cleanup in error handling.
    """
    if mlflow.active_run():
        mlflow.end_run()
