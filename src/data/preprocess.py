"""
Preprocess MovieLens data for the recommendation system.
This module handles:
- Loading raw MovieLens data files
- Converting explicit ratings to implicit feedback
- Filtering users/items with insufficient interactions
- Saving processed interactions
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import get_data_config
from src.data.download import get_expected_data_path, is_data_downloaded

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================


def load_ratings() -> pd.DataFrame:
    """Load the raw ratings data from u.data.

    Returns:
        DataFrame with columns: user_id, item_id, rating, timestamp

    Raises:
        FileNotFoundError: If data hasn't been downloaded yet
    """
    if not is_data_downloaded():
        raise FileNotFoundError(
            "MovieLens data not found. Run 'python -m src.data.download' first."
        )

    data_path = get_expected_data_path()
    ratings_path = data_path / "u.data"

    logger.info(f"Loading ratings from {ratings_path}")

    # u.data is tab-separated with no header
    df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "rating": "int8",
            "timestamp": "int64",
        },
    )

    logger.info(f"Loaded {len(df):,} ratings")
    return df


def load_movies() -> pd.DataFrame:
    """Load the movie metadata from u.item.

    Returns:
        DataFrame with columns: item_id, title, release_date, and genre columns

    Raises:
        FileNotFoundError: If data hasn't been downloaded yet
    """
    if not is_data_downloaded():
        raise FileNotFoundError(
            "MovieLens data not found. Run 'python -m src.data.download' first."
        )

    data_path = get_expected_data_path()
    movies_path = data_path / "u.item"

    logger.info(f"Loading movies from {movies_path}")

    # Genre names from u.genre
    genres = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    # Column names for u.item
    columns = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
    ] + genres

    # u.item is pipe-separated
    df = pd.read_csv(
        movies_path,
        sep="|",
        names=columns,
        encoding="latin-1",
        dtype={"item_id": "int32"},
    )

    # Drop columns we don't need
    df = df.drop(columns=["video_release_date", "imdb_url"])

    logger.info(f"Loaded {len(df):,} movies")
    return df


# =============================================================================
# Preprocessing
# =============================================================================


def convert_to_implicit(
    ratings_df: pd.DataFrame,
    threshold: int,
) -> pd.DataFrame:
    """Convert explicit ratings to implicit feedback.

    Ratings >= threshold become positive interactions (kept).
    Ratings < threshold are discarded.

    Args:
        ratings_df: DataFrame with rating column
        threshold: Minimum rating to consider as positive

    Returns:
        DataFrame with only positive interactions (rating column removed)
    """
    logger.info(f"Converting to implicit feedback (threshold >= {threshold})")

    n_before = len(ratings_df)

    # Keep only ratings >= threshold
    implicit_df = ratings_df[ratings_df["rating"] >= threshold].copy()

    # Drop the rating column - it's now implicit (all positive)
    implicit_df = implicit_df.drop(columns=["rating"])

    n_after = len(implicit_df)
    pct_kept = (n_after / n_before) * 100

    logger.info(f"Kept {n_after:,} / {n_before:,} interactions ({pct_kept:.1f}%)")

    return implicit_df


def filter_by_interactions(
    df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    """Filter users and items with too few interactions.

    Iteratively removes users and items until counts stabilize.
    This is necessary because removing items affects user counts
    and vice versa.

    Args:
        df: DataFrame with user_id and item_id columns
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item

    Returns:
        Filtered DataFrame
    """
    logger.info(
        f"Filtering: min_user={min_user_interactions}, "
        f"min_item={min_item_interactions}"
    )

    n_start = len(df)
    iteration = 0

    while True:
        iteration += 1
        n_before = len(df)

        # Count interactions per user and item
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()

        # Find users and items that meet the threshold
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index

        # Filter
        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]

        n_after = len(df)

        logger.debug(f"Iteration {iteration}: {n_before:,} -> {n_after:,} interactions")

        # Stop when no more changes
        if n_after == n_before:
            break

        # Safety check to prevent infinite loops
        if iteration > 100:
            logger.warning("Filtering did not converge after 100 iterations")
            break

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()

    logger.info(
        f"After filtering: {len(df):,} interactions "
        f"({n_users:,} users, {n_items:,} items)"
    )
    logger.info(
        f"Removed {n_start - len(df):,} interactions "
        f"({(n_start - len(df)) / n_start * 100:.1f}%)"
    )

    return df


def preprocess_movielens(save: bool = True) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Steps:
    1. Load raw ratings
    2. Convert to implicit feedback
    3. Filter sparse users/items
    4. Save to processed directory

    Args:
        save: Whether to save the processed data to disk

    Returns:
        Processed interactions DataFrame
    """
    config = get_data_config()

    # Load raw data
    ratings_df = load_ratings()

    # Convert to implicit
    interactions_df = convert_to_implicit(
        ratings_df,
        threshold=config.preprocessing.implicit_threshold,
    )

    # Filter sparse users/items
    interactions_df = filter_by_interactions(
        interactions_df,
        min_user_interactions=config.preprocessing.min_user_interactions,
        min_item_interactions=config.preprocessing.min_item_interactions,
    )

    # Sort by timestamp (important for time-based splitting)
    interactions_df = interactions_df.sort_values("timestamp").reset_index(drop=True)

    if save:
        # Save processed data
        processed_path = config.get_processed_path()
        processed_path.mkdir(parents=True, exist_ok=True)

        output_file = processed_path / "interactions.csv"
        interactions_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")

    return interactions_df


def get_interactions_path() -> Path:
    """Get the path to the processed interactions file.

    Returns:
        Path to interactions.csv

    Raises:
        FileNotFoundError: If preprocessing hasn't been run yet
    """
    config = get_data_config()
    path = config.get_processed_path() / "interactions.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Interactions file not found at {path}. "
            "Run 'python -m src.data.preprocess' first."
        )

    return path


def load_interactions() -> pd.DataFrame:
    """Load the processed interactions file.

    Returns:
        DataFrame with columns: user_id, item_id, timestamp

    Raises:
        FileNotFoundError: If preprocessing hasn't been run yet
    """
    path = get_interactions_path()
    return pd.read_csv(path)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens data for recommendation system"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save processed data to disk (for testing)",
    )
    args = parser.parse_args()

    try:
        interactions_df = preprocess_movielens(save=not args.no_save)

        print("\n" + "=" * 50)
        print("Preprocessing Complete")
        print("=" * 50)
        print(f"Total interactions: {len(interactions_df):,}")
        print(f"Unique users: {interactions_df['user_id'].nunique():,}")
        print(f"Unique items: {interactions_df['item_id'].nunique():,}")
        print(
            f"Density: {len(interactions_df) / (interactions_df['user_id'].nunique() * interactions_df['item_id'].nunique()) * 100:.2f}%"
        )
        print("\nSample data:")
        print(interactions_df.head(10).to_string(index=False))

    except FileNotFoundError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
