"""
Model loader for the recommendation API.

This module loads production artifacts into memory at startup:
- User and item embeddings (NumPy arrays)
- FAISS index for similarity search
- ID mappings (user_id <-> index, item_id <-> index)
- Item features (title, genres, year)

The loaded model is stored in a singleton that can be accessed
throughout the application.

Usage:
    from src.api.model_loader import model_store, load_model

    # Load at startup
    load_model()

    # Access in request handlers
    embeddings = model_store.get_user_embedding(user_idx)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.models.index import FAISSIndex
from src.training.export import get_production_path

logger = logging.getLogger(__name__)


@dataclass
class ItemFeatures:
    """Features for a single item."""

    item_id: int
    title: str
    year: Optional[int]
    genres: list[str]


@dataclass
class ModelStore:
    """Container for all loaded model artifacts.

    This class holds all the data needed to serve recommendations:
    - Embeddings for scoring
    - FAISS index for similarity
    - Mappings for ID conversion
    - Item features for response enrichment

    Attributes:
        is_loaded: Whether the model has been loaded
        version: Model version string
        loaded_at: Timestamp when model was loaded
        user_embeddings: NumPy array of shape (n_users, embedding_dim)
        item_embeddings: NumPy array of shape (n_items, embedding_dim)
        faiss_index: FAISS index for item similarity search
        user_to_idx: Mapping from user_id to matrix index
        idx_to_user: Mapping from matrix index to user_id
        item_to_idx: Mapping from item_id to matrix index
        idx_to_item: Mapping from matrix index to item_id
        item_features: Mapping from item_id to ItemFeatures
        metadata: Full model metadata dict
    """

    is_loaded: bool = False
    version: Optional[str] = None
    loaded_at: Optional[datetime] = None

    # Embeddings
    user_embeddings: Optional[np.ndarray] = None
    item_embeddings: Optional[np.ndarray] = None

    # FAISS index
    faiss_index: Optional[FAISSIndex] = None

    # ID mappings
    user_to_idx: dict[int, int] = field(default_factory=dict)
    idx_to_user: dict[int, int] = field(default_factory=dict)
    item_to_idx: dict[int, int] = field(default_factory=dict)
    idx_to_item: dict[int, int] = field(default_factory=dict)

    # Item features
    item_features: dict[int, ItemFeatures] = field(default_factory=dict)

    # Full metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        """Number of users in the model."""
        if self.user_embeddings is None:
            return 0
        return self.user_embeddings.shape[0]

    @property
    def n_items(self) -> int:
        """Number of items in the model."""
        if self.item_embeddings is None:
            return 0
        return self.item_embeddings.shape[0]

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        if self.user_embeddings is None:
            return 0
        return self.user_embeddings.shape[1]

    def has_user(self, user_id: int) -> bool:
        """Check if user exists in the model."""
        return user_id in self.user_to_idx

    def has_item(self, item_id: int) -> bool:
        """Check if item exists in the model."""
        return item_id in self.item_to_idx

    def get_user_idx(self, user_id: int) -> Optional[int]:
        """Convert user_id to matrix index."""
        return self.user_to_idx.get(user_id)

    def get_item_idx(self, item_id: int) -> Optional[int]:
        """Convert item_id to matrix index."""
        return self.item_to_idx.get(item_id)

    def get_user_id(self, idx: int) -> Optional[int]:
        """Convert matrix index to user_id."""
        return self.idx_to_user.get(idx)

    def get_item_id(self, idx: int) -> Optional[int]:
        """Convert matrix index to item_id."""
        return self.idx_to_item.get(idx)

    def get_user_embedding(self, user_idx: int) -> Optional[np.ndarray]:
        """Get embedding for a user by matrix index."""
        if self.user_embeddings is None:
            return None
        if user_idx < 0 or user_idx >= self.n_users:
            return None
        embedding: np.ndarray = self.user_embeddings[user_idx]
        return embedding

    def get_item_embedding(self, item_idx: int) -> Optional[np.ndarray]:
        """Get embedding for an item by matrix index."""
        if self.item_embeddings is None:
            return None
        if item_idx < 0 or item_idx >= self.n_items:
            return None
        embedding: np.ndarray = self.item_embeddings[item_idx]
        return embedding

    def get_item_features(self, item_id: int) -> Optional[ItemFeatures]:
        """Get features for an item by item_id."""
        return self.item_features.get(item_id)

    def get_all_item_ids(self) -> list[int]:
        """Get all item IDs in the model."""
        return list(self.item_to_idx.keys())

    def clear(self) -> None:
        """Clear all loaded data."""
        self.is_loaded = False
        self.version = None
        self.loaded_at = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.faiss_index = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.item_features = {}
        self.metadata = {}


# Global singleton instance
model_store = ModelStore()


def load_model(production_dir: Optional[Path] = None) -> ModelStore:
    """Load production model artifacts into the model store.

    This function loads all artifacts needed for serving:
    - User/item embeddings
    - FAISS index
    - ID mappings
    - Item features
    - Model metadata

    Args:
        production_dir: Path to production artifacts. Uses default if not provided.

    Returns:
        The loaded ModelStore instance.

    Raises:
        FileNotFoundError: If required artifacts are missing.
        ValueError: If artifacts are invalid or corrupted.
    """
    global model_store

    if production_dir is None:
        production_dir = get_production_path()
    production_dir = Path(production_dir)

    logger.info(f"Loading model from {production_dir}")

    # Clear any existing data
    model_store.clear()

    # 1. Load metadata
    metadata_path = production_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        model_store.metadata = json.load(f)
    model_store.version = model_store.metadata.get("version", "unknown")
    logger.info(f"  Model version: {model_store.version}")

    # 2. Load user embeddings
    user_emb_path = production_dir / "user_embeddings.npy"
    if not user_emb_path.exists():
        raise FileNotFoundError(f"User embeddings not found: {user_emb_path}")

    model_store.user_embeddings = np.load(user_emb_path)
    if model_store.user_embeddings is not None:
        logger.info(f"  User embeddings: {model_store.user_embeddings.shape}")

    # 3. Load item embeddings
    item_emb_path = production_dir / "item_embeddings.npy"
    if not item_emb_path.exists():
        raise FileNotFoundError(f"Item embeddings not found: {item_emb_path}")

    model_store.item_embeddings = np.load(item_emb_path)
    if model_store.item_embeddings is not None:
        logger.info(f"  Item embeddings: {model_store.item_embeddings.shape}")

    # 4. Load FAISS index
    faiss_path = production_dir / "item_index.faiss"
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    model_store.faiss_index = FAISSIndex.load(faiss_path)
    logger.info(f"  FAISS index: {model_store.faiss_index.n_items} items")

    # 5. Load user mapping
    user_map_path = production_dir / "user_mapping.json"
    if not user_map_path.exists():
        raise FileNotFoundError(f"User mapping not found: {user_map_path}")

    with open(user_map_path) as f:
        user_map_raw = json.load(f)
    model_store.user_to_idx = {int(k): v for k, v in user_map_raw.items()}
    model_store.idx_to_user = {v: int(k) for k, v in user_map_raw.items()}
    logger.info(f"  User mapping: {len(model_store.user_to_idx)} users")

    # 6. Load item mapping
    item_map_path = production_dir / "item_mapping.json"
    if not item_map_path.exists():
        raise FileNotFoundError(f"Item mapping not found: {item_map_path}")

    with open(item_map_path) as f:
        item_map_raw = json.load(f)
    model_store.item_to_idx = {int(k): v for k, v in item_map_raw.items()}
    model_store.idx_to_item = {v: int(k) for k, v in item_map_raw.items()}
    logger.info(f"  Item mapping: {len(model_store.item_to_idx)} items")

    # 7. Load item features
    features_path = production_dir / "item_features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Item features not found: {features_path}")

    with open(features_path) as f:
        features_raw = json.load(f)

    for item_id_str, features in features_raw.items():
        item_id = int(item_id_str)
        model_store.item_features[item_id] = ItemFeatures(
            item_id=item_id,
            title=features.get("title", f"Movie {item_id}"),
            year=features.get("year"),
            genres=features.get("genres", []),
        )
    logger.info(f"  Item features: {len(model_store.item_features)} items")

    # Mark as loaded
    model_store.is_loaded = True
    model_store.loaded_at = datetime.now(timezone.utc)

    logger.info(
        f"✓ Model loaded successfully: {model_store.n_users} users, {model_store.n_items} items"
    )

    return model_store


def get_model_store() -> ModelStore:
    """Get the global model store instance.

    Returns:
        The ModelStore singleton.

    Raises:
        RuntimeError: If model has not been loaded.
    """
    if not model_store.is_loaded:
        raise RuntimeError(
            "Model not loaded. Call load_model() at startup or check /health endpoint."
        )
    return model_store


def is_model_loaded() -> bool:
    """Check if the model has been loaded."""
    return model_store.is_loaded


def reload_model(production_dir: Optional[Path] = None) -> ModelStore:
    """Reload the model (for hot-reload scenarios).

    This clears the current model and loads fresh artifacts.

    Args:
        production_dir: Path to production artifacts.

    Returns:
        The reloaded ModelStore instance.
    """
    logger.info("Reloading model...")
    return load_model(production_dir)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        store = load_model()

        print("\n" + "=" * 50)
        print("Model Store Summary")
        print("=" * 50)
        print(f"Version: {store.version}")
        print(f"Loaded at: {store.loaded_at}")
        print(f"Users: {store.n_users:,}")
        print(f"Items: {store.n_items:,}")
        print(f"Embedding dim: {store.embedding_dim}")

        # Test lookups
        print("\nSample lookups:")
        sample_user_id = list(store.user_to_idx.keys())[0]
        sample_item_id = list(store.item_to_idx.keys())[0]

        print(f"  User {sample_user_id} -> idx {store.get_user_idx(sample_user_id)}")
        print(f"  Item {sample_item_id} -> idx {store.get_item_idx(sample_item_id)}")

        features = store.get_item_features(sample_item_id)
        if features:
            print(f"  Item {sample_item_id} title: {features.title}")

        print("\n✓ Model loader working correctly!")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Run training and export first.")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise
