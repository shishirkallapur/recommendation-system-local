"""
Abstract base class for recommendation models.

All recommender models should inherit from BaseRecommender and implement
the required methods. This ensures a consistent interface across different
model types (collaborative filtering, matrix factorization, etc.).

Usage:
    class MyModel(BaseRecommender):
        def fit(self, interactions, ...) -> None:
            # Train the model
            ...

        def recommend(self, user_id, ...) -> list[tuple[int, float]]:
            # Generate recommendations
            ...
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix


class BaseRecommender(ABC):
    """Abstract base class for all recommendation models.

    This class defines the interface that all recommender models must implement.
    It provides a consistent API for training, generating recommendations,
    and finding similar items.

    Attributes:
        name: Human-readable name for the model
        is_fitted: Whether the model has been trained
    """

    def __init__(self, name: str = "BaseRecommender"):
        """Initialize the recommender.

        Args:
            name: Human-readable name for the model
        """
        self.name = name
        self.is_fitted = False
        self._n_users: Optional[int] = None
        self._n_items: Optional[int] = None

    @property
    def n_users(self) -> int:
        """Number of users the model was trained on."""
        if self._n_users is None:
            raise ValueError("Model has not been fitted yet")
        return self._n_users

    @property
    def n_items(self) -> int:
        """Number of items the model was trained on."""
        if self._n_items is None:
            raise ValueError("Model has not been fitted yet")
        return self._n_items

    @abstractmethod
    def fit(
        self,
        interaction_matrix: csr_matrix,
        show_progress: bool = True,
    ) -> None:
        """Train the model on user-item interactions.

        Args:
            interaction_matrix: Sparse matrix of shape (n_users, n_items)
                               where non-zero entries indicate interactions
            show_progress: Whether to show a progress bar during training

        Note:
            After calling fit(), is_fitted should be True and n_users/n_items
            should be set.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        filter_already_liked: bool = True,
        filter_items: Optional[set[int]] = None,
        items_to_score: Optional[np.ndarray] = None,
    ) -> list[tuple[int, float]]:
        """Generate top-N recommendations for a user.

        Args:
            user_idx: Matrix index of the user (not original user_id)
            n: Number of recommendations to return
            filter_already_liked: If True, exclude items the user has interacted with
            filter_items: Optional set of item indices to exclude
            items_to_score: Optional array of item indices to score (if None, score all)

        Returns:
            List of (item_idx, score) tuples, sorted by score descending.
            Item indices are matrix indices (not original item_ids).

        Raises:
            ValueError: If model hasn't been fitted or user_idx is invalid
        """
        pass

    @abstractmethod
    def similar_items(
        self,
        item_idx: int,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """Find items similar to a given item.

        Args:
            item_idx: Matrix index of the item (not original item_id)
            n: Number of similar items to return

        Returns:
            List of (item_idx, score) tuples, sorted by similarity descending.
            Does not include the query item itself.

        Raises:
            ValueError: If model hasn't been fitted or item_idx is invalid
        """
        pass

    def _check_is_fitted(self) -> None:
        """Raise an error if the model hasn't been fitted.

        Raises:
            ValueError: If is_fitted is False
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} has not been fitted. Call fit() first.")

    def _validate_user_idx(self, user_idx: int) -> None:
        """Validate that a user index is valid.

        Args:
            user_idx: User index to validate

        Raises:
            ValueError: If user_idx is out of bounds
        """
        self._check_is_fitted()
        if user_idx < 0 or user_idx >= self.n_users:
            raise ValueError(
                f"user_idx {user_idx} out of bounds. "
                f"Valid range: 0 to {self.n_users - 1}"
            )

    def _validate_item_idx(self, item_idx: int) -> None:
        """Validate that an item index is valid.

        Args:
            item_idx: Item index to validate

        Raises:
            ValueError: If item_idx is out of bounds
        """
        self._check_is_fitted()
        if item_idx < 0 or item_idx >= self.n_items:
            raise ValueError(
                f"item_idx {item_idx} out of bounds. "
                f"Valid range: 0 to {self.n_items - 1}"
            )

    def get_params(self) -> dict:
        """Get model hyperparameters.

        Override this in subclasses to return model-specific parameters.
        Used for logging to MLflow.

        Returns:
            Dictionary of parameter names to values
        """
        return {"name": self.name}

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name}({status})"
