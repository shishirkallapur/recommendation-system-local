"""
ID mapping utilities for the recommendation system.

This module creates and manages mappings between original IDs (user_id, item_id)
and contiguous matrix indices (0, 1, 2, ...) needed for matrix factorization.

Mappings are built from training data only - users/items appearing only in
val/test are considered cold-start and won't have mappings.

Usage:
    # As a module (CLI)
    python -m src.data.mappings

    # Programmatically
    from src.data.mappings import build_mappings, load_mappings
    user_map, item_map = build_mappings()
    user_map, item_map = load_mappings()
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import get_data_config
from src.data.split import get_split_paths

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Type alias for ID mappings
IDMapping = dict[int, int]


def create_id_mapping(ids: pd.Series) -> IDMapping:
    """Create a mapping from original IDs to contiguous indices.

    Args:
        ids: Series of original IDs (may have duplicates)

    Returns:
        Dictionary mapping original_id -> matrix_index

    Example:
        >>> ids = pd.Series([196, 22, 196, 244, 22])
        >>> create_id_mapping(ids)
        {22: 0, 196: 1, 244: 2}  # Sorted for consistency
    """
    unique_ids = sorted(ids.unique())
    return {original_id: idx for idx, original_id in enumerate(unique_ids)}


def build_mappings(save: bool = True) -> tuple[IDMapping, IDMapping]:
    """Build user and item ID mappings from training data.

    Creates mappings from original IDs to contiguous matrix indices.
    Only includes users/items that appear in the training set.

    Args:
        save: Whether to save mappings to disk

    Returns:
        Tuple of (user_mapping, item_mapping)
    """
    logger.info("Building ID mappings from training data...")

    # Load training data only
    train_path, _, _ = get_split_paths()
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df):,} training interactions")

    # Create mappings
    user_mapping = create_id_mapping(train_df["user_id"])
    item_mapping = create_id_mapping(train_df["item_id"])

    logger.info(f"Created user mapping: {len(user_mapping):,} users")
    logger.info(f"Created item mapping: {len(item_mapping):,} items")

    # Log some statistics
    logger.info(
        f"User ID range: {min(user_mapping.keys())} - {max(user_mapping.keys())}"
    )
    logger.info(
        f"Item ID range: {min(item_mapping.keys())} - {max(item_mapping.keys())}"
    )

    if save:
        config = get_data_config()
        processed_path = config.get_processed_path()

        user_map_path = processed_path / "user_mapping.json"
        item_map_path = processed_path / "item_mapping.json"

        # Save as JSON (convert int keys to strings for JSON compatibility)
        with open(user_map_path, "w") as f:
            json.dump({str(k): v for k, v in user_mapping.items()}, f)

        with open(item_map_path, "w") as f:
            json.dump({str(k): v for k, v in item_mapping.items()}, f)

        logger.info(f"Saved user mapping to {user_map_path}")
        logger.info(f"Saved item mapping to {item_map_path}")

    return user_mapping, item_mapping


def get_mapping_paths() -> tuple[Path, Path]:
    """Get paths to the mapping files.

    Returns:
        Tuple of (user_mapping_path, item_mapping_path)

    Raises:
        FileNotFoundError: If mappings haven't been created yet
    """
    config = get_data_config()
    processed_path = config.get_processed_path()

    user_map_path = processed_path / "user_mapping.json"
    item_map_path = processed_path / "item_mapping.json"

    if not user_map_path.exists():
        raise FileNotFoundError(
            f"User mapping not found at {user_map_path}. "
            "Run 'python -m src.data.mappings' first."
        )

    if not item_map_path.exists():
        raise FileNotFoundError(
            f"Item mapping not found at {item_map_path}. "
            "Run 'python -m src.data.mappings' first."
        )

    return user_map_path, item_map_path


def load_mappings() -> tuple[IDMapping, IDMapping]:
    """Load user and item mappings from disk.

    Returns:
        Tuple of (user_mapping, item_mapping)

    Raises:
        FileNotFoundError: If mappings haven't been created yet
    """
    user_map_path, item_map_path = get_mapping_paths()

    with open(user_map_path) as f:
        user_mapping = {int(k): v for k, v in json.load(f).items()}

    with open(item_map_path) as f:
        item_mapping = {int(k): v for k, v in json.load(f).items()}

    return user_mapping, item_mapping


def create_reverse_mapping(mapping: IDMapping) -> dict[int, int]:
    """Create reverse mapping (matrix_index -> original_id).

    Useful for converting model outputs back to original IDs.

    Args:
        mapping: Original ID -> matrix index mapping

    Returns:
        Matrix index -> original ID mapping
    """
    return {v: k for k, v in mapping.items()}


class IDMapper:
    """Utility class for converting between ID spaces.

    Provides convenient methods for mapping IDs in both directions
    and handling unknown IDs gracefully.

    Usage:
        mapper = IDMapper.from_disk()

        # Original ID to matrix index
        idx = mapper.user_to_index(196)

        # Matrix index to original ID
        user_id = mapper.index_to_user(0)

        # Check if user exists
        if mapper.has_user(999):
            ...
    """

    def __init__(self, user_mapping: IDMapping, item_mapping: IDMapping):
        """Initialize with mappings.

        Args:
            user_mapping: user_id -> matrix_index
            item_mapping: item_id -> matrix_index
        """
        self.user_to_idx = user_mapping
        self.item_to_idx = item_mapping
        self.idx_to_user = create_reverse_mapping(user_mapping)
        self.idx_to_item = create_reverse_mapping(item_mapping)

    @classmethod
    def from_disk(cls) -> "IDMapper":
        """Load mappings from disk and create IDMapper.

        Returns:
            IDMapper instance with loaded mappings
        """
        user_mapping, item_mapping = load_mappings()
        return cls(user_mapping, item_mapping)

    @property
    def n_users(self) -> int:
        """Number of users in the mapping."""
        return len(self.user_to_idx)

    @property
    def n_items(self) -> int:
        """Number of items in the mapping."""
        return len(self.item_to_idx)

    def has_user(self, user_id: int) -> bool:
        """Check if user exists in the mapping."""
        return user_id in self.user_to_idx

    def has_item(self, item_id: int) -> bool:
        """Check if item exists in the mapping."""
        return item_id in self.item_to_idx

    def user_to_index(self, user_id: int) -> Optional[int]:
        """Convert user ID to matrix index.

        Args:
            user_id: Original user ID

        Returns:
            Matrix index, or None if user not found
        """
        return self.user_to_idx.get(user_id)

    def item_to_index(self, item_id: int) -> Optional[int]:
        """Convert item ID to matrix index.

        Args:
            item_id: Original item ID

        Returns:
            Matrix index, or None if item not found
        """
        return self.item_to_idx.get(item_id)

    def index_to_user(self, idx: int) -> Optional[int]:
        """Convert matrix index to user ID.

        Args:
            idx: Matrix index

        Returns:
            Original user ID, or None if index not found
        """
        return self.idx_to_user.get(idx)

    def index_to_item(self, idx: int) -> Optional[int]:
        """Convert matrix index to item ID.

        Args:
            idx: Matrix index

        Returns:
            Original item ID, or None if index not found
        """
        return self.idx_to_item.get(idx)

    def get_all_user_ids(self) -> list:
        """Get all original user IDs."""
        return list(self.user_to_idx.keys())

    def get_all_item_ids(self) -> list:
        """Get all original item IDs."""
        return list(self.item_to_idx.keys())


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build ID mappings for recommendation system"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save mappings to disk (for testing)",
    )
    args = parser.parse_args()

    try:
        user_mapping, item_mapping = build_mappings(save=not args.no_save)

        print("\n" + "=" * 50)
        print("ID Mappings Complete")
        print("=" * 50)
        print(f"Users: {len(user_mapping):,}")
        print(f"Items: {len(item_mapping):,}")

        # Show sample mappings
        print("\nSample user mappings (first 5):")
        for user_id, idx in list(user_mapping.items())[:5]:
            print(f"  user_id {user_id} -> index {idx}")

        print("\nSample item mappings (first 5):")
        for item_id, idx in list(item_mapping.items())[:5]:
            print(f"  item_id {item_id} -> index {idx}")

        print("\n✓ Mappings complete!")

    except FileNotFoundError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Mapping creation failed: {e}")
        raise
