"""
Fallback handlers for cold-start scenarios.

This module provides fallback recommendations when the main
recommendation engine cannot help:
- Unknown users: Return popular items
- User with seed item: Return similar items to seed
- Empty results: Return popular items as backup

The fallback system ensures every user gets some recommendations,
even if they're not personalized.

Usage:
    from src.api.fallback import FallbackHandler

    handler = FallbackHandler()

    # For unknown users
    popular = handler.get_popular(k=10)

    # For users who provide a seed movie
    similar = handler.get_similar_to_seed(movie_id=1, k=10)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.api.model_loader import get_model_store, is_model_loaded
from src.api.recommender import RecommendationEngine, SimilarityResult
from src.config import get_data_config

logger = logging.getLogger(__name__)


@dataclass
class PopularItem:
    """A popular item result."""

    movie_id: int
    title: str
    popularity_score: float
    genres: list[str]
    year: Optional[int]
    rank: int


class FallbackHandler:
    """Handler for cold-start and fallback scenarios.

    This class provides non-personalized recommendations for cases
    where the main recommendation engine cannot help:

    1. Unknown users: Users not in training data
    2. Seed-based: Users who provide a movie they like
    3. Empty results: When filters are too restrictive

    The handler loads popularity data from the features directory
    and uses the model_store for item metadata.

    Example:
        handler = FallbackHandler()

        # Get popular movies
        popular = handler.get_popular(k=10)

        # Get popular movies filtered by genre
        action_popular = handler.get_popular(k=10, genre="Action")

        # Get similar movies for a seed
        similar = handler.get_similar_to_seed(movie_id=1, k=10)
    """

    def __init__(self, popularity_path: Optional[Path] = None) -> None:
        """Initialize the fallback handler.

        Args:
            popularity_path: Path to popularity.csv. Uses default if not provided.
        """
        self._popularity_df: Optional[pd.DataFrame] = None
        self._popularity_path = popularity_path
        self._engine: Optional[RecommendationEngine] = None

    def _load_popularity(self) -> pd.DataFrame:
        """Load popularity data from disk.

        Returns:
            DataFrame with columns: item_id, interaction_count,
            popularity_rank, popularity_percentile

        Raises:
            FileNotFoundError: If popularity file doesn't exist.
        """
        if self._popularity_df is not None:
            return self._popularity_df

        if self._popularity_path is None:
            config = get_data_config()
            self._popularity_path = config.get_features_path() / "popularity.csv"

        if not self._popularity_path.exists():
            raise FileNotFoundError(
                f"Popularity data not found at {self._popularity_path}. "
                "Run feature building first."
            )

        self._popularity_df = pd.read_csv(self._popularity_path)
        logger.info(f"Loaded popularity data: {len(self._popularity_df)} items")

        return self._popularity_df

    def _get_engine(self) -> RecommendationEngine:
        """Get or create the recommendation engine.

        Returns:
            RecommendationEngine instance.
        """
        if self._engine is None:
            self._engine = RecommendationEngine()
        return self._engine

    def get_popular(
        self,
        k: int = 10,
        genre: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> list[PopularItem]:
        """Get popular items as fallback recommendations.

        Returns items sorted by popularity (interaction count in training data).

        Args:
            k: Number of items to return.
            genre: Filter to items with this genre.
            year_min: Minimum release year.
            year_max: Maximum release year.

        Returns:
            List of PopularItem objects sorted by popularity.
        """
        popularity_df = self._load_popularity()

        # Start with all items sorted by rank
        df = popularity_df.sort_values("popularity_rank").copy()

        # Get item features for filtering and enrichment
        store = get_model_store() if is_model_loaded() else None

        results: list[PopularItem] = []

        for _, row in df.iterrows():
            item_id = int(row["item_id"])

            # Get features from model store
            features = store.get_item_features(item_id) if store else None

            # Apply genre filter
            if genre and features:
                if genre not in features.genres:
                    continue
            elif genre and not features:
                # Skip if we need to filter but have no features
                continue

            # Apply year filter
            if features and features.year is not None:
                if year_min is not None and features.year < year_min:
                    continue
                if year_max is not None and features.year > year_max:
                    continue

            # Build result
            results.append(
                PopularItem(
                    movie_id=item_id,
                    title=features.title if features else f"Movie {item_id}",
                    popularity_score=float(row["interaction_count"]),
                    genres=features.genres if features else [],
                    year=features.year if features else None,
                    rank=int(row["popularity_rank"]),
                )
            )

            if len(results) >= k:
                break

        return results

    def get_similar_to_seed(
        self,
        movie_id: int,
        k: int = 10,
    ) -> tuple[list[SimilarityResult], bool]:
        """Get items similar to a seed movie.

        This is useful for new users who provide a movie they like.
        Uses the recommendation engine's similar_items method.

        Args:
            movie_id: Seed movie ID.
            k: Number of similar items to return.

        Returns:
            Tuple of:
            - List of SimilarityResult objects
            - success: True if seed was found, False otherwise
        """
        engine = self._get_engine()

        # Check if movie exists
        if not engine.check_item_exists(movie_id):
            logger.warning(f"Seed movie {movie_id} not found in model")
            return [], False

        try:
            similar = engine.similar_items(movie_id=movie_id, k=k)
            return similar, True
        except ValueError as e:
            logger.warning(f"Could not get similar items for {movie_id}: {e}")
            return [], False

    def get_fallback_for_user(
        self,
        user_id: int,
        k: int = 10,
        seed_movie_id: Optional[int] = None,
        genre: Optional[str] = None,
    ) -> tuple[list[PopularItem], str]:
        """Get fallback recommendations for an unknown user.

        Tries strategies in order:
        1. If seed_movie_id provided: Return similar items
        2. Otherwise: Return popular items (optionally filtered by genre)

        Args:
            user_id: The unknown user ID (for logging).
            k: Number of recommendations.
            seed_movie_id: Optional seed movie for similarity-based fallback.
            genre: Optional genre filter for popularity fallback.

        Returns:
            Tuple of:
            - List of PopularItem or converted SimilarityResult
            - reason: String explaining which fallback was used
        """
        logger.info(f"Fallback triggered for user {user_id}")

        # Strategy 1: Seed-based similarity
        if seed_movie_id is not None:
            similar, success = self.get_similar_to_seed(seed_movie_id, k=k)
            if success and similar:
                # Convert SimilarityResult to PopularItem format
                results = [
                    PopularItem(
                        movie_id=item.movie_id,
                        title=item.title,
                        popularity_score=item.similarity_score,
                        genres=item.genres,
                        year=item.year,
                        rank=i + 1,
                    )
                    for i, item in enumerate(similar)
                ]
                return results, f"similar_to_seed:{seed_movie_id}"

        # Strategy 2: Popularity-based
        popular = self.get_popular(k=k, genre=genre)
        reason = "popularity"
        if genre:
            reason = f"popularity:genre={genre}"

        return popular, reason

    def is_user_known(self, user_id: int) -> bool:
        """Check if a user is known to the model.

        Args:
            user_id: User ID to check.

        Returns:
            True if user exists in model, False otherwise.
        """
        if not is_model_loaded():
            return False

        store = get_model_store()
        return store.has_user(user_id)


# =============================================================================
# Module-level convenience
# =============================================================================

_fallback_handler: Optional[FallbackHandler] = None


def get_fallback_handler() -> FallbackHandler:
    """Get the global fallback handler instance.

    Returns:
        FallbackHandler instance.
    """
    global _fallback_handler
    if _fallback_handler is None:
        _fallback_handler = FallbackHandler()
    return _fallback_handler


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.api.model_loader import load_model

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load model first
    print("Loading model...")
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Some features may not work without model.")

    # Create handler
    handler = FallbackHandler()

    print("\n" + "=" * 50)
    print("Testing Fallback Handler")
    print("=" * 50)

    # Test popular items
    print("\n1. Top 5 popular movies:")
    try:
        popular = handler.get_popular(k=5)
        for item in popular:
            print(
                f"   #{item.rank}: {item.title} ({item.popularity_score:.0f} interactions)"
            )
    except Exception as e:
        print(f"   Error: {e}")

    # Test popular with genre filter
    print("\n2. Top 5 popular Action movies:")
    try:
        popular = handler.get_popular(k=5, genre="Action")
        for item in popular:
            print(f"   #{item.rank}: {item.title} - {item.genres}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test similar to seed
    print("\n3. Movies similar to movie 1 (seed-based fallback):")
    try:
        store = get_model_store()
        sample_movie = list(store.item_to_idx.keys())[0]
        similar, success = handler.get_similar_to_seed(movie_id=sample_movie, k=5)
        if success:
            seed_info = store.get_item_features(sample_movie)
            print(f"   Seed: {seed_info.title if seed_info else sample_movie}")
            for sim_item in similar:
                print(
                    f"   - {sim_item.title} (similarity: {sim_item.similarity_score:.4f})"
                )
        else:
            print("   Could not find similar items")
    except Exception as e:
        print(f"   Error: {e}")

    # Test full fallback flow
    print("\n4. Fallback for unknown user 999999:")
    try:
        results, reason = handler.get_fallback_for_user(user_id=999999, k=5)
        print(f"   Reason: {reason}")
        for item in results:
            print(f"   - {item.title}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test is_user_known
    print("\n5. User existence check:")
    try:
        store = get_model_store()
        known_user = list(store.user_to_idx.keys())[0]
        print(f"   User {known_user} known: {handler.is_user_known(known_user)}")
        print(f"   User 999999 known: {handler.is_user_known(999999)}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n✓ Fallback handler tests complete!")
