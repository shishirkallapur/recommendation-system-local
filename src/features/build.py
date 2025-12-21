"""
Build features for the recommendation system.

This module creates:
- Item features: genre one-hot encoding, title, year
- Popularity features: interaction counts and ranks from training data

Usage:
    # As a module (CLI)
    python -m src.features.build

    # Programmatically
    from src.features.build import build_all_features
    item_features, popularity = build_all_features()
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import get_data_config
from src.data.download import get_expected_data_path, is_data_downloaded
from src.data.split import get_split_paths

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Genre names in MovieLens 100K (in order they appear in u.item)
GENRE_NAMES = [
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


# =============================================================================
# Item Features
# =============================================================================


def extract_year_from_title(title: str) -> Optional[int]:
    """Extract release year from movie title.

    MovieLens titles are formatted as "Movie Name (YYYY)".

    Args:
        title: Movie title string

    Returns:
        Year as integer, or None if not found

    Examples:
        >>> extract_year_from_title("Toy Story (1995)")
        1995
        >>> extract_year_from_title("Some Movie")
        None
    """
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        return int(match.group(1))
    return None


def load_raw_movies() -> pd.DataFrame:
    """Load raw movie data from u.item.

    Returns:
        DataFrame with item_id, title, and genre binary columns
    """
    if not is_data_downloaded():
        raise FileNotFoundError(
            "MovieLens data not found. Run 'python -m src.data.download' first."
        )

    data_path = get_expected_data_path()
    movies_path = data_path / "u.item"

    # Column names: id, title, release_date, video_date, url, then 19 genre flags
    columns = ["item_id", "title", "release_date", "video_date", "url"] + GENRE_NAMES

    df = pd.read_csv(
        movies_path,
        sep="|",
        names=columns,
        encoding="latin-1",
        dtype={"item_id": "int32"},
    )

    return df


def build_item_features(save: bool = True) -> pd.DataFrame:
    """Build item feature matrix with genres and metadata.

    Creates a DataFrame with:
    - item_id: Movie ID
    - title: Movie title (including year)
    - year: Extracted release year
    - Genre columns: Binary flags for each genre

    Args:
        save: Whether to save to disk

    Returns:
        Item features DataFrame
    """
    logger.info("Building item features...")

    # Load raw movie data
    movies_df = load_raw_movies()
    logger.info(f"Loaded {len(movies_df):,} movies from raw data")

    # Select and rename columns we need
    feature_columns = ["item_id", "title"] + GENRE_NAMES
    item_features = movies_df[feature_columns].copy()

    # Extract year from title
    item_features["year"] = item_features["title"].apply(extract_year_from_title)

    # Reorder columns: item_id, title, year, then genres
    column_order = ["item_id", "title", "year"] + GENRE_NAMES
    item_features = item_features[column_order]

    # Log statistics
    n_with_year = item_features["year"].notna().sum()
    logger.info(f"Extracted year for {n_with_year:,} / {len(item_features):,} movies")

    # Genre statistics
    genre_counts = item_features[GENRE_NAMES].sum().sort_values(ascending=False)
    logger.info("Top 5 genres by movie count:")
    for genre, count in genre_counts.head().items():
        logger.info(f"  {genre}: {count:,} movies")

    if save:
        config = get_data_config()
        features_path = config.get_features_path()
        features_path.mkdir(parents=True, exist_ok=True)

        output_file = features_path / "item_features.csv"
        item_features.to_csv(output_file, index=False)
        logger.info(f"Saved item features to {output_file}")

    return item_features


# =============================================================================
# Popularity Features
# =============================================================================


def build_popularity_features(save: bool = True) -> pd.DataFrame:
    """Build popularity features from training data.

    Computes interaction counts and popularity ranks for each item.
    Uses only training data to avoid data leakage.

    Args:
        save: Whether to save to disk

    Returns:
        Popularity features DataFrame with columns:
        - item_id: Movie ID
        - interaction_count: Number of interactions in training set
        - popularity_rank: Rank (1 = most popular)
        - popularity_percentile: Percentile (100 = most popular)
    """
    logger.info("Building popularity features from training data...")

    # Load training data only (to avoid leakage)
    train_path, _, _ = get_split_paths()
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df):,} training interactions")

    # Count interactions per item
    interaction_counts = train_df.groupby("item_id").size()
    interaction_counts.name = "interaction_count"
    item_counts = interaction_counts.reset_index()

    # Compute rank (1 = most popular)
    item_counts["popularity_rank"] = (
        item_counts["interaction_count"].rank(method="min", ascending=False).astype(int)
    )

    # Compute percentile (100 = most popular)
    item_counts["popularity_percentile"] = (
        item_counts["interaction_count"].rank(pct=True) * 100
    ).round(1)

    # Sort by popularity
    popularity_df: pd.DataFrame = item_counts.sort_values(
        "popularity_rank"
    ).reset_index(drop=True)

    # Log statistics
    logger.info(f"Computed popularity for {len(popularity_df):,} items")
    logger.info(
        f"Interaction counts: "
        f"min={popularity_df['interaction_count'].min()}, "
        f"median={popularity_df['interaction_count'].median():.0f}, "
        f"max={popularity_df['interaction_count'].max()}"
    )

    logger.info("Top 10 most popular items:")
    top_10 = popularity_df.head(10)
    for _, row in top_10.iterrows():
        logger.info(
            f"  Rank {row['popularity_rank']}: "
            f"item_id={row['item_id']}, "
            f"count={row['interaction_count']}"
        )

    if save:
        config = get_data_config()
        features_path = config.get_features_path()
        features_path.mkdir(parents=True, exist_ok=True)

        output_file = features_path / "popularity.csv"
        popularity_df.to_csv(output_file, index=False)
        logger.info(f"Saved popularity features to {output_file}")

    return popularity_df


# =============================================================================
# Combined Feature Building
# =============================================================================


def build_all_features(
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all features (item features and popularity).

    Args:
        save: Whether to save features to disk

    Returns:
        Tuple of (item_features_df, popularity_df)
    """
    logger.info("=" * 50)
    logger.info("Building all features")
    logger.info("=" * 50)

    item_features = build_item_features(save=save)

    logger.info("-" * 50)

    popularity = build_popularity_features(save=save)

    logger.info("=" * 50)
    logger.info("Feature building complete!")

    return item_features, popularity


# =============================================================================
# Loading Functions (for use by later phases)
# =============================================================================


def get_item_features_path() -> Path:
    """Get path to item features file.

    Returns:
        Path to item_features.csv

    Raises:
        FileNotFoundError: If features haven't been built yet
    """
    config = get_data_config()
    path = config.get_features_path() / "item_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Item features not found at {path}. "
            "Run 'python -m src.features.build' first."
        )

    return path


def get_popularity_path() -> Path:
    """Get path to popularity features file.

    Returns:
        Path to popularity.csv

    Raises:
        FileNotFoundError: If features haven't been built yet
    """
    config = get_data_config()
    path = config.get_features_path() / "popularity.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Popularity features not found at {path}. "
            "Run 'python -m src.features.build' first."
        )

    return path


def load_item_features() -> pd.DataFrame:
    """Load item features DataFrame.

    Returns:
        DataFrame with item_id, title, year, and genre columns
    """
    path = get_item_features_path()
    return pd.read_csv(path)


def load_popularity() -> pd.DataFrame:
    """Load popularity features DataFrame.

    Returns:
        DataFrame with item_id, interaction_count, popularity_rank, popularity_percentile
    """
    path = get_popularity_path()
    return pd.read_csv(path)


def get_genre_names() -> list[str]:
    """Get list of genre column names.

    Returns:
        List of genre names in order
    """
    return GENRE_NAMES.copy()


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build features for recommendation system"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save features to disk (for testing)",
    )
    parser.add_argument(
        "--item-features-only",
        action="store_true",
        help="Only build item features (skip popularity)",
    )
    parser.add_argument(
        "--popularity-only",
        action="store_true",
        help="Only build popularity features (skip item features)",
    )
    args = parser.parse_args()

    try:
        if args.item_features_only:
            build_item_features(save=not args.no_save)
        elif args.popularity_only:
            build_popularity_features(save=not args.no_save)
        else:
            build_all_features(save=not args.no_save)

        print("\n✓ Feature building complete!")

    except FileNotFoundError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Feature building failed: {e}")
        raise
