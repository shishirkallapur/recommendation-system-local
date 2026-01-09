"""
Unit tests for recommendation models.

Tests cover:
- ALSRecommender: training, recommendations, embeddings, similar items
- ItemItemRecommender: training, recommendations, similarity computation
- Edge cases: cold-start users, invalid indices, empty data

Run with: pytest tests/test_models.py -v
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.als import ALSRecommender
from src.models.item_item import ItemItemRecommender

# =============================================================================
# Fixtures: Shared test data
# =============================================================================


@pytest.fixture
def simple_interaction_matrix() -> csr_matrix:
    """Create a simple interaction matrix for testing.

    Matrix (5 users x 6 items):
        Items:  0  1  2  3  4  5
    User 0:    [1, 1, 0, 0, 0, 0]  -> likes items 0, 1
    User 1:    [1, 0, 1, 0, 0, 0]  -> likes items 0, 2
    User 2:    [0, 1, 1, 1, 0, 0]  -> likes items 1, 2, 3
    User 3:    [0, 0, 0, 1, 1, 0]  -> likes items 3, 4
    User 4:    [0, 0, 0, 0, 1, 1]  -> likes items 4, 5
    """
    data = [
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1],
    ]
    return csr_matrix(data, dtype=np.float32)


@pytest.fixture
def larger_interaction_matrix() -> csr_matrix:
    """Create a larger random interaction matrix for more robust testing."""
    np.random.seed(42)
    n_users, n_items = 50, 30
    # Sparse interactions (~10% density)
    data = (np.random.rand(n_users, n_items) > 0.9).astype(np.float32)
    return csr_matrix(data)


# =============================================================================
# ALSRecommender Tests
# =============================================================================


class TestALSRecommender:
    """Tests for ALSRecommender model."""

    def test_init_default_params(self):
        """Test model initializes with default parameters."""
        model = ALSRecommender()
        assert model.factors == 64
        assert model.regularization == 0.01
        assert model.iterations == 15
        assert model.alpha == 1.0
        assert model.is_fitted is False

    def test_init_custom_params(self):
        """Test model initializes with custom parameters."""
        model = ALSRecommender(
            factors=32,
            regularization=0.1,
            iterations=10,
            alpha=2.0,
            random_state=123,
        )
        assert model.factors == 32
        assert model.regularization == 0.1
        assert model.iterations == 10
        assert model.alpha == 2.0
        assert model.random_state == 123

    def test_fit_creates_embeddings(self, simple_interaction_matrix):
        """Test that fit() creates user and item embeddings."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        assert model.is_fitted is True
        assert model.n_users == 5
        assert model.n_items == 6

        # Check embedding shapes
        assert model.user_embeddings_.shape == (5, 8)
        assert model.item_embeddings_.shape == (6, 8)

    def test_fit_embeddings_not_all_zeros(self, simple_interaction_matrix):
        """Test that embeddings have non-zero values after training."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        # Embeddings should not be all zeros
        assert np.any(model.user_embeddings_ != 0)
        assert np.any(model.item_embeddings_ != 0)

    def test_recommend_returns_correct_format(self, simple_interaction_matrix):
        """Test recommend() returns list of (item_idx, score) tuples."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        recs = model.recommend(user_idx=0, n=3)

        assert isinstance(recs, list)
        assert len(recs) <= 3
        for item_idx, score in recs:
            assert isinstance(item_idx, int)
            assert isinstance(score, float)

    def test_recommend_excludes_seen_items(self, simple_interaction_matrix):
        """Test that recommendations exclude items user has interacted with."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        # User 0 has interacted with items 0 and 1
        recs = model.recommend(user_idx=0, n=10, filter_already_liked=True)
        recommended_items = {item_idx for item_idx, _ in recs}

        assert 0 not in recommended_items
        assert 1 not in recommended_items

    def test_recommend_includes_seen_when_disabled(self, simple_interaction_matrix):
        """Test that seen items are included when filter is disabled."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        recs = model.recommend(user_idx=0, n=10, filter_already_liked=False)
        recommended_items = {item_idx for item_idx, _ in recs}

        # Should include all items when not filtering
        assert len(recommended_items) == 6

    def test_recommend_respects_filter_items(self, simple_interaction_matrix):
        """Test that filter_items parameter excludes specified items."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        # Exclude items 2 and 3 in addition to seen items
        recs = model.recommend(user_idx=0, n=10, filter_items={2, 3})
        recommended_items = {item_idx for item_idx, _ in recs}

        assert 2 not in recommended_items
        assert 3 not in recommended_items

    def test_recommend_sorted_by_score(self, simple_interaction_matrix):
        """Test that recommendations are sorted by score descending."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        recs = model.recommend(user_idx=0, n=5)
        scores = [score for _, score in recs]

        assert scores == sorted(scores, reverse=True)

    def test_similar_items_returns_correct_format(self, simple_interaction_matrix):
        """Test similar_items() returns correctly formatted results."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        similar = model.similar_items(item_idx=0, n=3)

        assert isinstance(similar, list)
        assert len(similar) <= 3
        for item_idx, score in similar:
            assert isinstance(item_idx, int)
            assert isinstance(score, float)

    def test_similar_items_excludes_query_item(self, simple_interaction_matrix):
        """Test that similar_items excludes the query item itself."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        similar = model.similar_items(item_idx=0, n=10)
        similar_items = {item_idx for item_idx, _ in similar}

        assert 0 not in similar_items

    def test_get_user_embedding(self, simple_interaction_matrix):
        """Test get_user_embedding returns correct shape."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        emb = model.get_user_embedding(user_idx=0)

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (5, 8)  # Returns all user embeddings

    def test_get_item_embedding(self, simple_interaction_matrix):
        """Test get_item_embedding returns correct shape."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        emb = model.get_item_embedding(item_idx=0)

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (8,)

    def test_get_all_embeddings(self, simple_interaction_matrix):
        """Test get_all_user/item_embeddings return correct shapes."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        user_embs = model.get_all_user_embeddings()
        item_embs = model.get_all_item_embeddings()

        assert user_embs.shape == (5, 8)
        assert item_embs.shape == (6, 8)

    def test_predict_score(self, simple_interaction_matrix):
        """Test predict_score returns a float."""
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(simple_interaction_matrix, show_progress=False)

        score = model.predict_score(user_idx=0, item_idx=0)

        assert isinstance(score, float)

    def test_get_params(self):
        """Test get_params returns hyperparameters."""
        model = ALSRecommender(factors=32, regularization=0.05)
        params = model.get_params()

        assert params["factors"] == 32
        assert params["regularization"] == 0.05
        assert "name" in params

    def test_invalid_user_idx_raises(self, simple_interaction_matrix):
        """Test that invalid user index raises ValueError."""
        model = ALSRecommender(factors=8, iterations=5)
        model.fit(simple_interaction_matrix, show_progress=False)

        with pytest.raises(ValueError, match="user_idx.*out of bounds"):
            model.recommend(user_idx=100, n=5)

    def test_invalid_item_idx_raises(self, simple_interaction_matrix):
        """Test that invalid item index raises ValueError."""
        model = ALSRecommender(factors=8, iterations=5)
        model.fit(simple_interaction_matrix, show_progress=False)

        with pytest.raises(ValueError, match="item_idx.*out of bounds"):
            model.similar_items(item_idx=100, n=5)

    def test_not_fitted_raises(self):
        """Test that calling methods before fit raises error."""
        model = ALSRecommender()

        with pytest.raises(ValueError, match="not been fitted"):
            model.recommend(user_idx=0, n=5)


# =============================================================================
# ItemItemRecommender Tests
# =============================================================================


class TestItemItemRecommender:
    """Tests for ItemItemRecommender model."""

    def test_init_default_params(self):
        """Test model initializes with default parameters."""
        model = ItemItemRecommender()
        assert model.k_neighbors == 100
        assert model.min_similarity == 0.0
        assert model.is_fitted is False

    def test_init_custom_params(self):
        """Test model initializes with custom parameters."""
        model = ItemItemRecommender(k_neighbors=50, min_similarity=0.1)
        assert model.k_neighbors == 50
        assert model.min_similarity == 0.1

    def test_fit_creates_similarity_matrix(self, simple_interaction_matrix):
        """Test that fit() creates similarity matrix."""
        model = ItemItemRecommender(k_neighbors=None)
        model.fit(simple_interaction_matrix, show_progress=False)

        assert model.is_fitted is True
        assert model.n_users == 5
        assert model.n_items == 6
        assert model.similarity_matrix_.shape == (6, 6)

    def test_similarity_matrix_symmetric(self, simple_interaction_matrix):
        """Test that similarity matrix is symmetric."""
        model = ItemItemRecommender(k_neighbors=None, min_similarity=0.0)
        model.fit(simple_interaction_matrix, show_progress=False)

        # Similarity should be symmetric
        np.testing.assert_array_almost_equal(
            model.similarity_matrix_,
            model.similarity_matrix_.T,
        )

    def test_similarity_diagonal_zero(self, simple_interaction_matrix):
        """Test that self-similarity (diagonal) is zero."""
        model = ItemItemRecommender(k_neighbors=None)
        model.fit(simple_interaction_matrix, show_progress=False)

        diagonal = np.diag(model.similarity_matrix_)
        np.testing.assert_array_equal(diagonal, np.zeros(6))

    def test_similarity_bounded(self, simple_interaction_matrix):
        """Test that similarities are bounded [0, 1]."""
        model = ItemItemRecommender(k_neighbors=None)
        model.fit(simple_interaction_matrix, show_progress=False)

        assert np.all(model.similarity_matrix_ >= 0)
        assert np.all(model.similarity_matrix_ <= 1)

    def test_similar_items_expected_results(self, simple_interaction_matrix):
        """Test that similar items make intuitive sense.

        Items 0 and 1 are both liked by user 0, so they should be similar.
        Items 0 and 5 have no users in common, so similarity should be 0.
        """
        model = ItemItemRecommender(k_neighbors=None)
        model.fit(simple_interaction_matrix, show_progress=False)

        # Items 0 and 1 are co-interacted by user 0
        sim_0_1 = model.get_similarity(0, 1)
        # Items 0 and 5 have no common users
        sim_0_5 = model.get_similarity(0, 5)

        assert sim_0_1 > sim_0_5

    def test_recommend_returns_correct_format(self, simple_interaction_matrix):
        """Test recommend() returns list of (item_idx, score) tuples."""
        model = ItemItemRecommender(k_neighbors=None)
        model.fit(simple_interaction_matrix, show_progress=False)

        recs = model.recommend(user_idx=0, n=3)

        assert isinstance(recs, list)
        for item_idx, score in recs:
            assert isinstance(item_idx, int)
            assert isinstance(score, float)

    def test_recommend_excludes_seen_items(self, simple_interaction_matrix):
        """Test that recommendations exclude items user has interacted with."""
        model = ItemItemRecommender(k_neighbors=None)
        model.fit(simple_interaction_matrix, show_progress=False)

        # User 0 has interacted with items 0 and 1
        recs = model.recommend(user_idx=0, n=10)
        recommended_items = {item_idx for item_idx, _ in recs}

        assert 0 not in recommended_items
        assert 1 not in recommended_items

    def test_recommend_cold_start_user(self, simple_interaction_matrix):
        """Test that user with no interactions gets empty recommendations."""
        # Create matrix with one user having no interactions
        data = simple_interaction_matrix.toarray()
        data = np.vstack([data, np.zeros(6)])  # Add user with no interactions
        matrix = csr_matrix(data)

        model = ItemItemRecommender(k_neighbors=None)
        model.fit(matrix, show_progress=False)

        # User 5 has no interactions
        recs = model.recommend(user_idx=5, n=10)
        assert recs == []

    def test_similar_items_format(self, simple_interaction_matrix):
        """Test similar_items returns correctly formatted results."""
        model = ItemItemRecommender(k_neighbors=None)
        model.fit(simple_interaction_matrix, show_progress=False)

        similar = model.similar_items(item_idx=0, n=3)

        assert isinstance(similar, list)
        for item_idx, score in similar:
            assert isinstance(item_idx, int)
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_similar_items_excludes_self(self, simple_interaction_matrix):
        """Test that similar_items excludes the query item."""
        model = ItemItemRecommender(k_neighbors=None)
        model.fit(simple_interaction_matrix, show_progress=False)

        similar = model.similar_items(item_idx=0, n=10)
        similar_items = {item_idx for item_idx, _ in similar}

        assert 0 not in similar_items

    def test_k_neighbors_limits_similarity(self, larger_interaction_matrix):
        """Test that k_neighbors limits non-zero similarities per item."""
        k = 5
        model = ItemItemRecommender(k_neighbors=k)
        model.fit(larger_interaction_matrix, show_progress=False)

        # Each row should have at most k non-zero entries
        for i in range(model.n_items):
            n_nonzero = np.count_nonzero(model.similarity_matrix_[i])
            assert n_nonzero <= k

    def test_min_similarity_filters(self, simple_interaction_matrix):
        """Test that min_similarity filters low similarities."""
        model_no_filter = ItemItemRecommender(k_neighbors=None, min_similarity=0.0)
        model_no_filter.fit(simple_interaction_matrix, show_progress=False)

        model_filtered = ItemItemRecommender(k_neighbors=None, min_similarity=0.5)
        model_filtered.fit(simple_interaction_matrix, show_progress=False)

        # Filtered model should have fewer non-zero entries
        n_nonzero_no_filter = np.count_nonzero(model_no_filter.similarity_matrix_)
        n_nonzero_filtered = np.count_nonzero(model_filtered.similarity_matrix_)

        assert n_nonzero_filtered <= n_nonzero_no_filter

    def test_get_similarity(self, simple_interaction_matrix):
        """Test get_similarity returns correct value."""
        model = ItemItemRecommender(k_neighbors=None)
        model.fit(simple_interaction_matrix, show_progress=False)

        sim = model.get_similarity(0, 1)

        assert isinstance(sim, float)
        assert 0 <= sim <= 1

    def test_get_params(self):
        """Test get_params returns hyperparameters."""
        model = ItemItemRecommender(k_neighbors=50, min_similarity=0.1)
        params = model.get_params()

        assert params["k_neighbors"] == 50
        assert params["min_similarity"] == 0.1
        assert "name" in params

    def test_invalid_user_idx_raises(self, simple_interaction_matrix):
        """Test that invalid user index raises ValueError."""
        model = ItemItemRecommender()
        model.fit(simple_interaction_matrix, show_progress=False)

        with pytest.raises(ValueError, match="user_idx.*out of bounds"):
            model.recommend(user_idx=100, n=5)

    def test_invalid_item_idx_raises(self, simple_interaction_matrix):
        """Test that invalid item index raises ValueError."""
        model = ItemItemRecommender()
        model.fit(simple_interaction_matrix, show_progress=False)

        with pytest.raises(ValueError, match="item_idx.*out of bounds"):
            model.similar_items(item_idx=100, n=5)


# =============================================================================
# Integration Tests
# =============================================================================


class TestModelComparison:
    """Tests comparing behavior between models."""

    def test_both_models_fit_same_data(self, simple_interaction_matrix):
        """Test that both models can fit the same data."""
        als = ALSRecommender(factors=8, iterations=5, random_state=42)
        item_item = ItemItemRecommender(k_neighbors=None)

        als.fit(simple_interaction_matrix, show_progress=False)
        item_item.fit(simple_interaction_matrix, show_progress=False)

        assert als.is_fitted
        assert item_item.is_fitted
        assert als.n_users == item_item.n_users
        assert als.n_items == item_item.n_items

    def test_both_models_recommend_same_user(self, simple_interaction_matrix):
        """Test that both models can recommend for the same user."""
        als = ALSRecommender(factors=8, iterations=5, random_state=42)
        item_item = ItemItemRecommender(k_neighbors=None)

        als.fit(simple_interaction_matrix, show_progress=False)
        item_item.fit(simple_interaction_matrix, show_progress=False)

        als_recs = als.recommend(user_idx=0, n=3)
        ii_recs = item_item.recommend(user_idx=0, n=3)

        # Both should return recommendations (may differ in items)
        assert len(als_recs) > 0
        assert len(ii_recs) > 0

    def test_both_models_similar_items(self, simple_interaction_matrix):
        """Test that both models can find similar items."""
        als = ALSRecommender(factors=8, iterations=5, random_state=42)
        item_item = ItemItemRecommender(k_neighbors=None)

        als.fit(simple_interaction_matrix, show_progress=False)
        item_item.fit(simple_interaction_matrix, show_progress=False)

        als_similar = als.similar_items(item_idx=0, n=3)
        ii_similar = item_item.similar_items(item_idx=0, n=3)

        # Both should return similar items
        assert len(als_similar) > 0
        assert len(ii_similar) > 0
