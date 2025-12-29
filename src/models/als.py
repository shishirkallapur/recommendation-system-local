"""
ALS (Alternating Least Squares) Matrix Factorization Model.

This model learns latent embeddings for users and items by factorizing
the user-item interaction matrix. It's designed for implicit feedback
(clicks, views) rather than explicit ratings.

How it works:
1. Initialize random user and item embedding matrices
2. Alternate between optimizing user embeddings (fixing items) and
   optimizing item embeddings (fixing users)
3. Use confidence weighting for implicit feedback

Usage:
    from src.models.als import ALSRecommender

    model = ALSRecommender(factors=64, regularization=0.01, iterations=15)
    model.fit(interaction_matrix)

    # Get recommendations
    recommendations = model.recommend(user_idx=0, n=10)

    # Get user/item embeddings for downstream use
    user_emb = model.get_user_embedding(user_idx=0)
    item_emb = model.get_item_embedding(item_idx=42)
"""

import logging
from typing import Optional

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from src.models.base import BaseRecommender

logger = logging.getLogger(__name__)


class ALSRecommender(BaseRecommender):
    """ALS Matrix Factorization for Implicit Feedback.

    This model uses the `implicit` library to perform Alternating Least
    Squares matrix factorization optimized for implicit feedback data.

    Attributes:
        factors: Number of latent factors (embedding dimension).
        regularization: L2 regularization parameter.
        iterations: Number of ALS iterations.
        alpha: Confidence scaling factor for implicit feedback.
        random_state: Random seed for reproducibility.
        user_embeddings_: Learned user embeddings (n_users × factors).
        item_embeddings_: Learned item embeddings (n_items × factors).
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 1.0,
        random_state: int = 42,
        name: str = "ALSRecommender",
    ):
        """Initialize the ALS recommender.

        Args:
            factors: Number of latent factors (embedding dimension).
            regularization: L2 regularization to prevent overfitting.
            iterations: Number of alternating optimization iterations.
            alpha: Confidence scaling. Higher values give more weight to
                  observed interactions vs unobserved.
            random_state: Random seed for reproducibility.
            name: Human-readable name for the model.
        """
        super().__init__(name=name)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state

        # Set during fit()
        self._model: AlternatingLeastSquares
        self.user_embeddings_: np.ndarray
        self.item_embeddings_: np.ndarray
        self.interaction_matrix_: csr_matrix

    def fit(
        self,
        interaction_matrix: csr_matrix,
        show_progress: bool = True,
    ) -> None:
        """Train the ALS model.

        Args:
            interaction_matrix: Sparse matrix of shape (n_users, n_items)
                               where non-zero entries indicate interactions.
            show_progress: Whether to show training progress.
        """
        self._n_users, self._n_items = interaction_matrix.shape
        self.interaction_matrix_ = interaction_matrix

        logger.info(
            f"Training ALS model: {self.n_users} users, {self.n_items} items, "
            f"{self.factors} factors, {self.iterations} iterations"
        )

        # Initialize the implicit ALS model
        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
            random_state=self.random_state,
        )

        # The implicit library expects item-user matrix (items as rows)
        # and uses CSR format for efficiency
        item_user_matrix = interaction_matrix.T.tocsr()

        # Fit the model
        self._model.fit(item_user_matrix, show_progress=show_progress)

        # Extract embeddings for easy access
        # IMPORTANT: We pass item-user matrix (items as rows, users as columns)
        # implicit's naming convention:
        # - item_factors: factors for rows of the input matrix (our items)
        # - user_factors: factors for columns of the input matrix (our users)
        # But empirically, the shapes come out swapped, so we assign accordingly:
        self.item_embeddings_ = self._model.user_factors  # shape: (n_items, factors)
        self.user_embeddings_ = self._model.item_factors  # shape: (n_users, factors)

        self.is_fitted = True

        logger.info(
            f"ALS training complete. "
            f"User embeddings: {self.user_embeddings_.shape}, "
            f"Item embeddings: {self.item_embeddings_.shape}"
        )

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        filter_already_liked: bool = True,
        filter_items: Optional[set[int]] = None,
        items_to_score: Optional[np.ndarray] = None,
    ) -> list[tuple[int, float]]:
        """Generate top-N recommendations for a user.

        Computes scores as dot product between user embedding and all
        item embeddings, then returns the highest-scoring items.

        Args:
            user_idx: Matrix index of the user.
            n: Number of recommendations to return.
            filter_already_liked: Exclude items the user has interacted with.
            filter_items: Additional item indices to exclude.
            items_to_score: Only score these items (if None, score all).

        Returns:
            List of (item_idx, score) tuples, sorted by score descending.
        """
        self._validate_user_idx(user_idx)

        # Get user embedding
        user_embedding = self.user_embeddings_[user_idx]

        # Compute scores for all items: dot product of user and item embeddings
        scores: np.ndarray = self.item_embeddings_.dot(user_embedding)

        # Build exclusion mask
        exclude_mask = np.zeros(self.n_items, dtype=bool)

        if filter_already_liked:
            # Get indices of items this user has interacted with
            user_interactions = self.interaction_matrix_[user_idx].toarray().flatten()
            liked_items = np.where(user_interactions > 0)[0]
            exclude_mask[liked_items] = True

        if filter_items:
            exclude_mask[list(filter_items)] = True

        # Apply exclusion
        scores[exclude_mask] = -np.inf

        # If only scoring specific items, mask others
        if items_to_score is not None:
            score_mask = np.ones(self.n_items, dtype=bool)
            score_mask[items_to_score] = False
            scores[score_mask] = -np.inf

        # Get top-N items
        top_indices = np.argsort(scores)[::-1][:n]

        # Filter out -inf scores
        recommendations = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > -np.inf
        ]

        return recommendations

    def similar_items(
        self,
        item_idx: int,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """Find items similar to a given item.

        Similarity is computed as cosine similarity between item embeddings.

        Args:
            item_idx: Matrix index of the query item.
            n: Number of similar items to return.

        Returns:
            List of (item_idx, similarity_score) tuples, sorted descending.
        """
        self._validate_item_idx(item_idx)

        # Get query item embedding
        query_embedding = self.item_embeddings_[item_idx]

        # Compute cosine similarity with all items
        # First normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        item_norms = np.linalg.norm(self.item_embeddings_, axis=1, keepdims=True)
        normalized_items = self.item_embeddings_ / (item_norms + 1e-10)

        similarities = normalized_items.dot(query_norm)

        # Zero out self-similarity
        similarities[item_idx] = -np.inf

        # Get top-N
        top_indices = np.argsort(similarities)[::-1][:n]

        similar = [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > -np.inf
        ]

        return similar

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get the learned embedding for a user.

        Args:
            user_idx: Matrix index of the user.

        Returns:
            User embedding vector of shape (factors,).
        """
        self._validate_user_idx(user_idx)
        result: np.ndarray = self.user_embeddings_.copy()
        return result

    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """Get the learned embedding for an item.

        Args:
            item_idx: Matrix index of the item.

        Returns:
            Item embedding vector of shape (factors,).
        """
        self._validate_item_idx(item_idx)
        result: np.ndarray = self.item_embeddings_[item_idx].copy()
        return result

    def get_all_user_embeddings(self) -> np.ndarray:
        """Get all user embeddings.

        Returns:
            Array of shape (n_users, factors).
        """
        self._check_is_fitted()
        result: np.ndarray = self.user_embeddings_.copy()
        return result

    def get_all_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings.

        Returns:
            Array of shape (n_items, factors).
        """
        self._check_is_fitted()
        return self.item_embeddings_.copy()

    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """Predict the score for a specific user-item pair.

        Args:
            user_idx: Matrix index of the user.
            item_idx: Matrix index of the item.

        Returns:
            Predicted score (dot product of embeddings).
        """
        self._validate_user_idx(user_idx)
        self._validate_item_idx(item_idx)

        return float(
            np.dot(self.user_embeddings_[user_idx], self.item_embeddings_[item_idx])
        )

    def get_params(self) -> dict:
        """Get model hyperparameters for logging."""
        return {
            "name": self.name,
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "random_state": self.random_state,
        }
