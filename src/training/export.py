"""
Export trained models to production artifact format.

This module takes trained models and saves all artifacts needed by the
serving layer in a standardized format. It bridges the offline training
pipeline and the online serving layer.

Production artifacts include:
- model_metadata.json: Version, metrics, hyperparameters, timestamps
- user_embeddings.npy: User latent factors
- item_embeddings.npy: Item latent factors
- item_index.faiss: FAISS index for similarity search
- user_mapping.json: user_id → matrix_index
- item_mapping.json: item_id → matrix_index
- item_features.json: item_id → {title, year, genres}

Usage:
    from src.training.export import export_model
    from src.models.als import ALSRecommender

    model = ALSRecommender(...)
    model.fit(train_matrix)

    export_model(
        model=model,
        metrics={"ndcg@10": 0.18, "recall@10": 0.25},
        version="v1.0.0",
    )
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from src.config import get_training_config
from src.data.mappings import load_mappings
from src.features.build import get_genre_names, load_item_features
from src.models.als import ALSRecommender
from src.models.index import FAISSIndex

logger = logging.getLogger(__name__)


def get_production_path() -> Path:
    """Get the path to production artifacts directory.

    Returns:
        Path to models/production/
    """
    config = get_training_config()
    return Path(config.output.models_dir) / config.output.production_dir


def generate_version() -> str:
    """Generate a version string based on current timestamp.

    Returns:
        Version string like "v20240115_143022"
    """
    now = datetime.now(timezone.utc)
    return f"v{now.strftime('%Y%m%d_%H%M%S')}"


def build_item_features_dict() -> dict[str, dict[str, Any]]:
    """Build item features dictionary for serving layer.

    Loads item features from CSV and converts to a dictionary keyed
    by item_id (as string for JSON compatibility).

    Returns:
        Dictionary mapping item_id -> {title, year, genres}
    """
    logger.info("Building item features dictionary...")

    # Load item features DataFrame
    item_features_df = load_item_features()
    genre_names = get_genre_names()

    features_dict: dict[str, dict[str, Any]] = {}

    for _, row in item_features_df.iterrows():
        item_id = str(int(row["item_id"]))

        # Extract genres where the binary flag is 1
        genres = [genre for genre in genre_names if row.get(genre, 0) == 1]

        features_dict[item_id] = {
            "title": row["title"],
            "year": int(row["year"]) if not np.isnan(row["year"]) else None,
            "genres": genres,
        }

    logger.info(f"Built features for {len(features_dict):,} items")
    return features_dict


def export_model(
    model: ALSRecommender,
    metrics: dict[str, float],
    version: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    backup_existing: bool = True,
) -> Path:
    """Export a trained model to production artifact format.

    This function saves all artifacts needed by the serving layer:
    - Embeddings (NumPy arrays)
    - FAISS index for similarity search
    - ID mappings (JSON)
    - Item features (JSON)
    - Model metadata (JSON)

    Args:
        model: Trained ALSRecommender model.
        metrics: Dictionary of evaluation metrics.
        version: Version string. Auto-generated if not provided.
        output_dir: Directory to save artifacts. Defaults to models/production/.
        run_id: MLflow run ID for traceability.
        backup_existing: If True, backup existing production artifacts before overwriting.

    Returns:
        Path to the export directory.

    Raises:
        ValueError: If model is not fitted or not an ALS model.
    """
    # Validate model
    if not isinstance(model, ALSRecommender):
        raise ValueError(
            f"Export currently only supports ALSRecommender, got {type(model).__name__}"
        )

    if not model.is_fitted:
        raise ValueError("Model must be fitted before export")

    # Setup paths
    if output_dir is None:
        output_dir = get_production_path()
    output_dir = Path(output_dir)

    # Generate version if not provided
    if version is None:
        version = generate_version()

    logger.info(f"Exporting model version {version} to {output_dir}")

    # Backup existing artifacts if requested
    if backup_existing and output_dir.exists():
        backup_dir = (
            output_dir.parent
            / f"production_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        logger.info(f"Backing up existing artifacts to {backup_dir}")
        shutil.copytree(output_dir, backup_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save embeddings
    logger.info("Saving embeddings...")
    user_embeddings = model.get_all_user_embeddings()
    item_embeddings = model.get_all_item_embeddings()

    np.save(output_dir / "user_embeddings.npy", user_embeddings)
    np.save(output_dir / "item_embeddings.npy", item_embeddings)
    logger.info(
        f"  User embeddings: {user_embeddings.shape}, "
        f"Item embeddings: {item_embeddings.shape}"
    )

    # 2. Build and save FAISS index
    logger.info("Building FAISS index...")
    faiss_index = FAISSIndex(
        embedding_dim=model.factors,
        index_type="flat",  # Exact search for accuracy
    )
    faiss_index.build(item_embeddings, normalize=True, show_progress=True)
    faiss_index.save(output_dir / "item_index.faiss")
    logger.info(f"  FAISS index: {faiss_index.n_items:,} items")

    # 3. Copy ID mappings
    logger.info("Saving ID mappings...")
    user_mapping, item_mapping = load_mappings()

    # Save with string keys for JSON compatibility
    with open(output_dir / "user_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in user_mapping.items()}, f, indent=2)

    with open(output_dir / "item_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in item_mapping.items()}, f, indent=2)

    logger.info(f"  Users: {len(user_mapping):,}, Items: {len(item_mapping):,}")

    # 4. Build and save item features
    logger.info("Saving item features...")
    item_features = build_item_features_dict()
    with open(output_dir / "item_features.json", "w") as f:
        json.dump(item_features, f, indent=2)

    # 5. Save metadata
    logger.info("Saving model metadata...")
    metadata = {
        "version": version,
        "model_type": model.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": run_id,
        "hyperparameters": model.get_params(),
        "metrics": metrics,
        "dimensions": {
            "n_users": int(user_embeddings.shape[0]),
            "n_items": int(item_embeddings.shape[0]),
            "embedding_dim": int(user_embeddings.shape[1]),
        },
        "artifacts": [
            "user_embeddings.npy",
            "item_embeddings.npy",
            "item_index.faiss",
            "user_mapping.json",
            "item_mapping.json",
            "item_features.json",
            "model_metadata.json",
        ],
    }

    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 50)
    logger.info(f"✓ Model exported successfully to {output_dir}")
    logger.info(f"  Version: {version}")
    logger.info(f"  Primary metric (ndcg@10): {metrics.get('ndcg@10', 'N/A')}")
    logger.info("=" * 50)

    return output_dir


def load_production_metadata(
    production_dir: Optional[Union[str, Path]] = None
) -> Optional[dict[str, Any]]:
    """Load metadata for the current production model.

    Args:
        production_dir: Path to production directory. Uses default if not provided.

    Returns:
        Metadata dictionary, or None if no production model exists.
    """
    if production_dir is None:
        production_dir = get_production_path()
    production_dir = Path(production_dir)

    metadata_path = production_dir / "model_metadata.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        data: dict[str, Any] = json.load(f)
        return data


def is_production_model_exists(
    production_dir: Optional[Union[str, Path]] = None
) -> bool:
    """Check if a production model exists.

    Args:
        production_dir: Path to production directory. Uses default if not provided.

    Returns:
        True if production model artifacts exist.
    """
    if production_dir is None:
        production_dir = get_production_path()
    production_dir = Path(production_dir)

    required_files = [
        "model_metadata.json",
        "user_embeddings.npy",
        "item_embeddings.npy",
        "item_index.faiss",
        "user_mapping.json",
        "item_mapping.json",
        "item_features.json",
    ]

    return all((production_dir / f).exists() for f in required_files)


def get_production_version(
    production_dir: Optional[Union[str, Path]] = None
) -> Optional[str]:
    """Get the version of the current production model.

    Args:
        production_dir: Path to production directory. Uses default if not provided.

    Returns:
        Version string, or None if no production model exists.
    """
    metadata = load_production_metadata(production_dir)
    return metadata.get("version") if metadata else None


def get_production_metrics(
    production_dir: Optional[Union[str, Path]] = None
) -> Optional[dict[str, float]]:
    """Get the metrics of the current production model.

    Args:
        production_dir: Path to production directory. Uses default if not provided.

    Returns:
        Metrics dictionary, or None if no production model exists.
    """
    metadata = load_production_metadata(production_dir)
    return metadata.get("metrics") if metadata else None


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Export trained model to production format"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if production model exists and show metadata",
    )
    args = parser.parse_args()

    if args.check:
        if is_production_model_exists():
            metadata = load_production_metadata()
            if metadata is not None:
                print("\n" + "=" * 50)
                print("Production Model Found")
                print("=" * 50)
                print(f"Version: {metadata.get('version')}")
                print(f"Model type: {metadata.get('model_type')}")
                print(f"Created at: {metadata.get('created_at')}")
                print("\nDimensions:")
                dims = metadata.get("dimensions", {})
                print(f"  Users: {dims.get('n_users'):,}")
                print(f"  Items: {dims.get('n_items'):,}")
                print(f"  Embedding dim: {dims.get('embedding_dim')}")
                print("\nMetrics:")
                for metric, value in metadata.get("metrics", {}).items():
                    print(f"  {metric}: {value:.4f}")
            else:
                print("Production model exists but metadata could not be loaded.")
        else:
            print("No production model found.")
            print(f"Expected location: {get_production_path()}")
    else:
        print("To export a model, use the training pipeline:")
        print("  python -m src.training.train")
        print("\nThen export the best model:")
        print("  from src.training.export import export_model")
        print("  export_model(model, metrics)")
        print("\nOr check existing production model:")
        print("  python -m src.training.export --check")
