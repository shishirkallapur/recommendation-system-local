"""
Time-based splitting of interaction data.

This module splits processed interactions into train/validation/test sets
using global timestamp cutoffs. This prevents data leakage by ensuring
the model only trains on past data to predict future interactions.

Usage:
    # As a module (CLI)
    python -m src.data.split

    # Programmatically
    from src.data.split import split_interactions
    train_df, val_df, test_df = split_interactions()
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import get_data_config
from src.data.preprocess import load_interactions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def timestamp_to_datetime(ts: int) -> str:
    """Convert Unix timestamp to readable datetime string.

    Args:
        ts: Unix timestamp (seconds since epoch)

    Returns:
        Formatted datetime string
    """
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def compute_split_timestamps(
    timestamps: pd.Series,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int]:
    """Compute the timestamp cutoffs for train/val/test splits.

    Args:
        timestamps: Series of Unix timestamps
        train_ratio: Fraction of data for training (e.g., 0.70)
        val_ratio: Fraction of data for validation (e.g., 0.15)

    Returns:
        Tuple of (train_cutoff, val_cutoff) timestamps
    """
    # Sort timestamps to find percentile cutoffs
    sorted_ts = timestamps.sort_values()
    n = len(sorted_ts)

    # Find indices for cutoffs
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    # Get timestamp values at those indices
    train_cutoff = sorted_ts.iloc[train_idx]
    val_cutoff = sorted_ts.iloc[val_idx]

    return train_cutoff, val_cutoff


def split_by_timestamp(
    df: pd.DataFrame,
    train_cutoff: int,
    val_cutoff: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/val/test based on timestamp cutoffs.

    Args:
        df: DataFrame with 'timestamp' column
        train_cutoff: Timestamp separating train from val
        val_cutoff: Timestamp separating val from test

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = df[df["timestamp"] < train_cutoff].copy()
    val_df = df[
        (df["timestamp"] >= train_cutoff) & (df["timestamp"] < val_cutoff)
    ].copy()
    test_df = df[df["timestamp"] >= val_cutoff].copy()

    return train_df, val_df, test_df


def log_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_cutoff: int,
    val_cutoff: int,
) -> None:
    """Log detailed statistics about the split."""
    total = len(train_df) + len(val_df) + len(test_df)

    logger.info("=" * 50)
    logger.info("Split Statistics")
    logger.info("=" * 50)

    # Size statistics
    logger.info(
        f"Train: {len(train_df):,} interactions "
        f"({len(train_df) / total * 100:.1f}%)"
    )
    logger.info(
        f"Val:   {len(val_df):,} interactions " f"({len(val_df) / total * 100:.1f}%)"
    )
    logger.info(
        f"Test:  {len(test_df):,} interactions " f"({len(test_df) / total * 100:.1f}%)"
    )

    # Time range statistics
    logger.info("-" * 50)
    logger.info("Time Ranges:")
    logger.info(
        f"Train: {timestamp_to_datetime(train_df['timestamp'].min())} → "
        f"{timestamp_to_datetime(train_df['timestamp'].max())}"
    )
    logger.info(
        f"Val:   {timestamp_to_datetime(val_df['timestamp'].min())} → "
        f"{timestamp_to_datetime(val_df['timestamp'].max())}"
    )
    logger.info(
        f"Test:  {timestamp_to_datetime(test_df['timestamp'].min())} → "
        f"{timestamp_to_datetime(test_df['timestamp'].max())}"
    )

    # Cutoff timestamps
    logger.info("-" * 50)
    logger.info(f"Train/Val cutoff: {timestamp_to_datetime(train_cutoff)}")
    logger.info(f"Val/Test cutoff:  {timestamp_to_datetime(val_cutoff)}")

    # User/item coverage
    logger.info("-" * 50)
    logger.info("User Coverage:")
    train_users = set(train_df["user_id"])
    val_users = set(val_df["user_id"])
    test_users = set(test_df["user_id"])

    val_new_users = val_users - train_users
    test_new_users = test_users - train_users - val_users

    logger.info(f"  Users in train: {len(train_users):,}")
    logger.info(
        f"  Users in val: {len(val_users):,} " f"({len(val_new_users)} not in train)"
    )
    logger.info(
        f"  Users in test: {len(test_users):,} "
        f"({len(test_new_users)} not in train/val)"
    )

    # Item coverage
    logger.info("Item Coverage:")
    train_items = set(train_df["item_id"])
    val_items = set(val_df["item_id"])
    test_items = set(test_df["item_id"])

    val_new_items = val_items - train_items
    test_new_items = test_items - train_items - val_items

    logger.info(f"  Items in train: {len(train_items):,}")
    logger.info(
        f"  Items in val: {len(val_items):,} " f"({len(val_new_items)} not in train)"
    )
    logger.info(
        f"  Items in test: {len(test_items):,} "
        f"({len(test_new_items)} not in train/val)"
    )


def split_interactions(
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions into train/validation/test sets.

    Uses global time-based splitting with ratios from config.

    Args:
        save: Whether to save splits to disk

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    config = get_data_config()

    # Load processed interactions
    logger.info("Loading processed interactions...")
    interactions_df = load_interactions()
    logger.info(f"Loaded {len(interactions_df):,} interactions")

    # Compute cutoff timestamps
    train_cutoff, val_cutoff = compute_split_timestamps(
        interactions_df["timestamp"],
        train_ratio=config.splitting.train_ratio,
        val_ratio=config.splitting.val_ratio,
    )

    # Perform the split
    train_df, val_df, test_df = split_by_timestamp(
        interactions_df, train_cutoff, val_cutoff
    )

    # Log statistics
    log_split_statistics(train_df, val_df, test_df, train_cutoff, val_cutoff)

    if save:
        processed_path = config.get_processed_path()

        train_path = processed_path / "train.csv"
        val_path = processed_path / "val.csv"
        test_path = processed_path / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("-" * 50)
        logger.info(f"Saved train.csv: {train_path}")
        logger.info(f"Saved val.csv: {val_path}")
        logger.info(f"Saved test.csv: {test_path}")

    return train_df, val_df, test_df


# =============================================================================
# Loading Functions (for use by later phases)
# =============================================================================


def get_split_paths() -> tuple[Path, Path, Path]:
    """Get paths to the train/val/test split files.

    Returns:
        Tuple of (train_path, val_path, test_path)

    Raises:
        FileNotFoundError: If splits haven't been created yet
    """
    config = get_data_config()
    processed_path = config.get_processed_path()

    train_path = processed_path / "train.csv"
    val_path = processed_path / "val.csv"
    test_path = processed_path / "test.csv"

    for path, name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name}.csv not found at {path}. "
                "Run 'python -m src.data.split' first."
            )

    return train_path, val_path, test_path


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train/val/test split DataFrames.

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        FileNotFoundError: If splits haven't been created yet
    """
    train_path, val_path, test_path = get_split_paths()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split interactions into train/val/test sets"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save splits to disk (for testing)",
    )
    args = parser.parse_args()

    try:
        train_df, val_df, test_df = split_interactions(save=not args.no_save)
        print("\n✓ Splitting complete!")

    except FileNotFoundError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Splitting failed: {e}")
        raise
