"""
Recommendation engine for the API.

This module contains the core recommendation logic:
- Personalized recommendations via embedding dot product
- Item similarity via FAISS index
- Filtering by genres, year, seen items
- Response enrichment with movie metadata

The engine is stateless - all data comes from the model_store.

Usage:
    from src.api.recommender import RecommendationEngine

    engine = RecommendationEngine()
    recommendations = engine.recommend(user_id=196, k=10)
    similar = engine.similar_items(movie_id=1, k=10)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.api.model_loader import ItemFeatures, get_model_store

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Result of a recommendation request."""

    movie_id: int
    title: str
    score: float
    genres: list[str]
    year: Optional[int]


@dataclass
class SimilarityResult:
    """Result of a similarity request."""

    movie_id: int
    title: str
    similarity_score: float
    genres: list[str]
    year: Optional[int]


class RecommendationEngine:
    """Core recommendation engine.

    This class provides methods for generating recommendations:
    - recommend(): Personalized recommendations for a user
    - similar_items(): Items similar to a given item

    The engine is stateless and thread-safe. All data is read from
    the global model_store which is loaded at startup.

    Example:
        engine = RecommendationEngine()

        # Get recommendations for user 196
        recs = engine.recommend(user_id=196, k=10)

        # Get movies similar to movie 1
        similar = engine.similar_items(movie_id=1, k=10)
    """

    def __init__(self) -> None:
        """Initialize the recommendation engine."""
        # Engine is stateless - we access model_store when needed
        pass

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
        filter_genres: Optional[list[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> tuple[list[RecommendationResult], bool, Optional[str]]:
        """Generate personalized recommendations for a user.

        This method:
        1. Looks up the user's embedding
        2. Computes scores for all items via dot product
        3. Applies filters (seen items, genres, year)
        4. Returns top-K items with metadata

        Args:
            user_id: Original user ID (not matrix index).
            k: Number of recommendations to return.
            exclude_seen: Whether to exclude items the user has seen.
            seen_items: Set of item IDs to exclude (if exclude_seen is True).
                       If None and exclude_seen is True, we can't filter seen items
                       (would need interaction history which we don't store).
            filter_genres: Only include items with at least one of these genres.
            year_min: Minimum release year (inclusive).
            year_max: Maximum release year (inclusive).

        Returns:
            Tuple of:
            - List of RecommendationResult objects
            - is_fallback: True if fallback was used (always False here)
            - fallback_reason: None (fallback handled by separate module)

        Raises:
            ValueError: If user_id is not found in the model.
        """
        store = get_model_store()

        # Check if user exists
        user_idx = store.get_user_idx(user_id)
        if user_idx is None:
            raise ValueError(f"User {user_id} not found in model")

        # Get user embedding
        user_embedding = store.get_user_embedding(user_idx)
        if user_embedding is None:
            raise ValueError(f"Could not get embedding for user {user_id}")

        # Score all items via dot product
        # scores[i] = user_embedding · item_embeddings[i]
        # Higher score = better match
        if store.item_embeddings is None:
            raise ValueError("Item embeddings not loaded")

        scores: np.ndarray = store.item_embeddings @ user_embedding

        # Build candidate list with scores
        candidates: list[tuple[int, int, float]] = []  # (item_id, item_idx, score)

        for item_idx in range(store.n_items):
            item_id = store.get_item_id(item_idx)
            if item_id is None:
                continue

            score = float(scores[item_idx])
            candidates.append((item_id, item_idx, score))

        # Apply filters
        filtered = self._apply_filters(
            candidates=candidates,
            exclude_item_ids=seen_items if exclude_seen else None,
            filter_genres=filter_genres,
            year_min=year_min,
            year_max=year_max,
        )

        # Sort by score descending and take top-K
        filtered.sort(key=lambda x: x[2], reverse=True)
        top_k = filtered[:k]

        # Enrich with metadata
        results = []
        for item_id, _, score in top_k:
            features = store.get_item_features(item_id)
            results.append(
                RecommendationResult(
                    movie_id=item_id,
                    title=features.title if features else f"Movie {item_id}",
                    score=score,
                    genres=features.genres if features else [],
                    year=features.year if features else None,
                )
            )

        return results, False, None

    def similar_items(
        self,
        movie_id: int,
        k: int = 10,
    ) -> list[SimilarityResult]:
        """Find items similar to a given movie.

        Uses the FAISS index for efficient similarity search.
        Similarity is based on cosine similarity of item embeddings.

        Args:
            movie_id: Original movie ID (not matrix index).
            k: Number of similar items to return.

        Returns:
            List of SimilarityResult objects.

        Raises:
            ValueError: If movie_id is not found in the model.
        """
        store = get_model_store()

        # Check if item exists
        item_idx = store.get_item_idx(movie_id)
        if item_idx is None:
            raise ValueError(f"Movie {movie_id} not found in model")

        # Get item embedding
        item_embedding = store.get_item_embedding(item_idx)
        if item_embedding is None:
            raise ValueError(f"Could not get embedding for movie {movie_id}")

        # Search FAISS index
        if store.faiss_index is None:
            raise ValueError("FAISS index not loaded")

        # Request k+1 because the item itself might be in results
        search_results = store.faiss_index.search(
            item_embedding, k=k + 1, normalize=True
        )

        # Convert to results, excluding the query item
        results = []
        for similar_idx, similarity in search_results:
            similar_item_id = store.get_item_id(similar_idx)
            if similar_item_id is None:
                continue

            # Skip the query item itself
            if similar_item_id == movie_id:
                continue

            features = store.get_item_features(similar_item_id)
            results.append(
                SimilarityResult(
                    movie_id=similar_item_id,
                    title=features.title if features else f"Movie {similar_item_id}",
                    similarity_score=similarity,
                    genres=features.genres if features else [],
                    year=features.year if features else None,
                )
            )

            # Stop once we have k results
            if len(results) >= k:
                break

        return results

    def get_item_info(self, movie_id: int) -> Optional[ItemFeatures]:
        """Get information about a specific movie.

        Args:
            movie_id: Original movie ID.

        Returns:
            ItemFeatures if found, None otherwise.
        """
        store = get_model_store()
        return store.get_item_features(movie_id)

    def _apply_filters(
        self,
        candidates: list[tuple[int, int, float]],
        exclude_item_ids: Optional[set[int]] = None,
        filter_genres: Optional[list[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> list[tuple[int, int, float]]:
        """Apply filters to candidate items.

        Args:
            candidates: List of (item_id, item_idx, score) tuples.
            exclude_item_ids: Item IDs to exclude.
            filter_genres: Only include items with at least one of these genres.
            year_min: Minimum release year.
            year_max: Maximum release year.

        Returns:
            Filtered list of candidates.
        """
        store = get_model_store()
        filtered = []

        for item_id, item_idx, score in candidates:
            # Exclude specific items
            if exclude_item_ids and item_id in exclude_item_ids:
                continue

            # Get features for genre/year filtering
            features = store.get_item_features(item_id)

            # Filter by genres
            if filter_genres and features:
                # Check if item has at least one of the requested genres
                item_genres = set(features.genres)
                requested_genres = set(filter_genres)
                if not item_genres & requested_genres:
                    continue

            # Filter by year
            if features and features.year is not None:
                if year_min is not None and features.year < year_min:
                    continue
                if year_max is not None and features.year > year_max:
                    continue

            filtered.append((item_id, item_idx, score))

        return filtered

    def check_user_exists(self, user_id: int) -> bool:
        """Check if a user exists in the model.

        Args:
            user_id: Original user ID.

        Returns:
            True if user exists, False otherwise.
        """
        store = get_model_store()
        return store.has_user(user_id)

    def check_item_exists(self, movie_id: int) -> bool:
        """Check if an item exists in the model.

        Args:
            movie_id: Original movie ID.

        Returns:
            True if item exists, False otherwise.
        """
        store = get_model_store()
        return store.has_item(movie_id)


# =============================================================================
# Module-level convenience functions
# =============================================================================

# Global engine instance (stateless, so safe to share)
_engine: Optional[RecommendationEngine] = None


def get_engine() -> RecommendationEngine:
    """Get the global recommendation engine instance.

    Returns:
        RecommendationEngine instance.
    """
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

    from src.api.model_loader import load_model

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load model first
    print("Loading model...")
    load_model()

    # Create engine
    engine = RecommendationEngine()

    print("\n" + "=" * 50)
    print("Testing Recommendation Engine")
    print("=" * 50)

    # Get a sample user ID
    store = get_model_store()
    sample_user_id = list(store.user_to_idx.keys())[0]
    sample_movie_id = list(store.item_to_idx.keys())[0]

    # Test recommendations
    print(f"\n1. Recommendations for user {sample_user_id}:")
    try:
        recs, is_fallback, reason = engine.recommend(user_id=sample_user_id, k=5)
        for i, rec in enumerate(recs, 1):
            print(f"   {i}. {rec.title} (score: {rec.score:.4f})")
    except Exception as e:
        print(f"   Error: {e}")

    # Test with genre filter
    print(f"\n2. Action movie recommendations for user {sample_user_id}:")
    try:
        recs, _is_fallback, _reason = engine.recommend(
            user_id=sample_user_id, k=5, filter_genres=["Action"]
        )
        for i, rec in enumerate(recs, 1):
            print(f"   {i}. {rec.title} - {rec.genres}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test similar items
    print(f"\n3. Movies similar to movie {sample_movie_id}:")
    try:
        similar = engine.similar_items(movie_id=sample_movie_id, k=5)
        query_info = engine.get_item_info(sample_movie_id)
        if query_info:
            print(f"   Query: {query_info.title}")
        for i, item in enumerate(similar, 1):
            print(f"   {i}. {item.title} (similarity: {item.similarity_score:.4f})")
    except Exception as e:
        print(f"   Error: {e}")

    # Test unknown user
    print("\n4. Testing unknown user (999999):")
    try:
        engine.recommend(user_id=999999, k=5)
        print("   Unexpected: Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")

    print("\n✓ Recommendation engine tests complete!")
