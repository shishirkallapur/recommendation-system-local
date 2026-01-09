"""
Unit tests for evaluation metrics.

Tests cover:
- Individual metrics: precision, recall, NDCG, MRR, hit rate
- Edge cases: empty recommendations, empty relevant items
- Full evaluation pipeline

Run with: pytest tests/test_evaluation.py -v
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.als import ALSRecommender
from src.training.evaluate import (
    compute_metrics,
    evaluate_model,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# =============================================================================
# Precision@K Tests
# =============================================================================


class TestPrecisionAtK:
    """Tests for precision_at_k metric."""

    def test_perfect_precision(self):
        """Test precision = 1.0 when all recommendations are relevant."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(recommended, relevant, k=5) == 1.0

    def test_zero_precision(self):
        """Test precision = 0.0 when no recommendations are relevant."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {10, 11, 12}
        assert precision_at_k(recommended, relevant, k=5) == 0.0

    def test_partial_precision(self):
        """Test precision with some relevant items."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 10}  # 2 relevant in top 5
        assert precision_at_k(recommended, relevant, k=5) == 2 / 5

    def test_precision_with_k_less_than_recommendations(self):
        """Test precision when k < len(recommendations)."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2}  # Both in top 2
        assert precision_at_k(recommended, relevant, k=2) == 1.0
        assert precision_at_k(recommended, relevant, k=5) == 2 / 5

    def test_precision_empty_recommendations(self):
        """Test precision with empty recommendations."""
        recommended: list[int] = []
        relevant = {1, 2, 3}
        # k=0 case
        assert precision_at_k(recommended, relevant, k=0) == 0.0

    def test_precision_k_zero(self):
        """Test precision when k=0."""
        recommended = [1, 2, 3]
        relevant = {1, 2}
        assert precision_at_k(recommended, relevant, k=0) == 0.0


# =============================================================================
# Recall@K Tests
# =============================================================================


class TestRecallAtK:
    """Tests for recall_at_k metric."""

    def test_perfect_recall(self):
        """Test recall = 1.0 when all relevant items are recommended."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, k=5) == 1.0

    def test_zero_recall(self):
        """Test recall = 0.0 when no relevant items are recommended."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {10, 11, 12}
        assert recall_at_k(recommended, relevant, k=5) == 0.0

    def test_partial_recall(self):
        """Test recall with some relevant items recommended."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 10, 11}  # 2 out of 4 relevant
        assert recall_at_k(recommended, relevant, k=5) == 2 / 4

    def test_recall_empty_relevant(self):
        """Test recall when no items are relevant."""
        recommended = [1, 2, 3]
        relevant: set[int] = set()
        assert recall_at_k(recommended, relevant, k=3) == 0.0

    def test_recall_k_affects_result(self):
        """Test that k parameter affects recall."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {3, 5}  # Item 3 at position 3, item 5 at position 5
        assert recall_at_k(recommended, relevant, k=2) == 0.0  # Neither in top 2
        assert recall_at_k(recommended, relevant, k=3) == 0.5  # Only 3 in top 3
        assert recall_at_k(recommended, relevant, k=5) == 1.0  # Both in top 5


# =============================================================================
# NDCG@K Tests
# =============================================================================


class TestNDCGAtK:
    """Tests for ndcg_at_k metric."""

    def test_perfect_ndcg_single_item(self):
        """Test NDCG = 1.0 when single relevant item is at position 1."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1}
        assert ndcg_at_k(recommended, relevant, k=5) == 1.0

    def test_perfect_ndcg_multiple_items(self):
        """Test NDCG = 1.0 when relevant items are at optimal positions."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2}  # At positions 1 and 2 (optimal)
        assert ndcg_at_k(recommended, relevant, k=5) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        """Test NDCG = 0.0 when no relevant items are recommended."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {10, 11, 12}
        assert ndcg_at_k(recommended, relevant, k=5) == 0.0

    def test_ndcg_position_matters(self):
        """Test that NDCG is higher when relevant items are ranked higher."""
        relevant = {1}

        # Item 1 at position 1
        recs_good = [1, 2, 3, 4, 5]
        # Item 1 at position 5
        recs_bad = [2, 3, 4, 5, 1]

        ndcg_good = ndcg_at_k(recs_good, relevant, k=5)
        ndcg_bad = ndcg_at_k(recs_bad, relevant, k=5)

        assert ndcg_good > ndcg_bad

    def test_ndcg_empty_relevant(self):
        """Test NDCG when no items are relevant."""
        recommended = [1, 2, 3]
        relevant: set[int] = set()
        assert ndcg_at_k(recommended, relevant, k=3) == 0.0

    def test_ndcg_bounded(self):
        """Test that NDCG is bounded between 0 and 1."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4}

        ndcg = ndcg_at_k(recommended, relevant, k=5)
        assert 0.0 <= ndcg <= 1.0

    def test_ndcg_k_zero(self):
        """Test NDCG when k=0."""
        recommended = [1, 2, 3]
        relevant = {1, 2}
        assert ndcg_at_k(recommended, relevant, k=0) == 0.0

    def test_ndcg_calculation_example(self):
        """Test NDCG calculation with known expected value.

        Example:
        - Recommended: [1, 2, 3, 4, 5]
        - Relevant: {2, 4}
        - DCG = 1/log2(3) + 1/log2(5) = 0.6309 + 0.4307 = 1.0616
        - IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        - NDCG = 1.0616 / 1.6309 = 0.651
        """
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4}

        # DCG: rel at pos 2 and 4 (0-indexed: 1 and 3)
        dcg = 1 / np.log2(3) + 1 / np.log2(5)  # positions 2 and 4 (1-indexed)
        idcg = 1 / np.log2(2) + 1 / np.log2(3)  # ideal: positions 1 and 2
        expected_ndcg = dcg / idcg

        assert ndcg_at_k(recommended, relevant, k=5) == pytest.approx(
            expected_ndcg, rel=1e-3
        )


# =============================================================================
# Reciprocal Rank (MRR) Tests
# =============================================================================


class TestReciprocalRank:
    """Tests for reciprocal_rank metric."""

    def test_first_position(self):
        """Test RR = 1.0 when relevant item is at position 1."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1}
        assert reciprocal_rank(recommended, relevant) == 1.0

    def test_second_position(self):
        """Test RR = 0.5 when relevant item is at position 2."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {2}
        assert reciprocal_rank(recommended, relevant) == 0.5

    def test_third_position(self):
        """Test RR = 1/3 when relevant item is at position 3."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {3}
        assert reciprocal_rank(recommended, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_found(self):
        """Test RR = 0.0 when no relevant item is found."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {10, 11, 12}
        assert reciprocal_rank(recommended, relevant) == 0.0

    def test_multiple_relevant_uses_first(self):
        """Test that RR uses the first relevant item found."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4}  # First relevant at position 2
        assert reciprocal_rank(recommended, relevant) == 0.5

    def test_empty_recommendations(self):
        """Test RR with empty recommendations."""
        recommended: list[int] = []
        relevant = {1, 2, 3}
        assert reciprocal_rank(recommended, relevant) == 0.0


# =============================================================================
# Hit@K Tests
# =============================================================================


class TestHitAtK:
    """Tests for hit_at_k metric."""

    def test_hit_when_relevant_in_top_k(self):
        """Test hit = 1.0 when relevant item is in top K."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {3}
        assert hit_at_k(recommended, relevant, k=5) == 1.0

    def test_no_hit_when_relevant_not_in_top_k(self):
        """Test hit = 0.0 when relevant item is not in top K."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {10}
        assert hit_at_k(recommended, relevant, k=5) == 0.0

    def test_hit_k_cutoff(self):
        """Test that K cutoff is respected."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {5}  # At position 5 (0-indexed: 4)
        assert hit_at_k(recommended, relevant, k=4) == 0.0  # Not in top 4
        assert hit_at_k(recommended, relevant, k=5) == 1.0  # In top 5

    def test_hit_empty_relevant(self):
        """Test hit when no items are relevant."""
        recommended = [1, 2, 3]
        relevant: set[int] = set()
        assert hit_at_k(recommended, relevant, k=3) == 0.0


# =============================================================================
# compute_metrics Tests
# =============================================================================


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_returns_all_metrics(self):
        """Test that compute_metrics returns all expected metrics."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4}
        k = 5

        metrics = compute_metrics(recommended, relevant, k)

        assert f"precision@{k}" in metrics
        assert f"recall@{k}" in metrics
        assert f"ndcg@{k}" in metrics
        assert f"hit@{k}" in metrics
        assert "mrr" in metrics

    def test_metrics_values_correct(self):
        """Test that metric values are consistent with individual functions."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3}
        k = 5

        metrics = compute_metrics(recommended, relevant, k)

        assert metrics[f"precision@{k}"] == precision_at_k(recommended, relevant, k)
        assert metrics[f"recall@{k}"] == recall_at_k(recommended, relevant, k)
        assert metrics[f"ndcg@{k}"] == ndcg_at_k(recommended, relevant, k)
        assert metrics[f"hit@{k}"] == hit_at_k(recommended, relevant, k)
        assert metrics["mrr"] == reciprocal_rank(recommended, relevant)


# =============================================================================
# evaluate_model Tests
# =============================================================================


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    @pytest.fixture
    def fitted_model_and_matrices(self):
        """Create a fitted model with train/test matrices."""
        # Create train matrix (5 users x 10 items)
        train_data = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # User 0
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # User 1
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # User 2
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # User 3
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # User 4
            ]
        )
        train_matrix = csr_matrix(train_data, dtype=np.float32)

        # Create test matrix (same users, different items)
        test_data = np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # User 0 interacted with item 2
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # User 1 interacted with item 3
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # User 2 interacted with item 4
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # User 3 interacted with item 5
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # User 4 interacted with item 6
            ]
        )
        test_matrix = csr_matrix(test_data, dtype=np.float32)

        # Train model
        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(train_matrix, show_progress=False)

        return model, train_matrix, test_matrix

    def test_evaluate_returns_metrics(self, fitted_model_and_matrices):
        """Test that evaluate_model returns expected metrics."""
        model, train_matrix, test_matrix = fitted_model_and_matrices

        metrics = evaluate_model(
            model,
            train_matrix,
            test_matrix,
            k_values=[5, 10],
            show_progress=False,
        )

        # Check all expected metrics are present
        assert "precision@5" in metrics
        assert "precision@10" in metrics
        assert "recall@5" in metrics
        assert "recall@10" in metrics
        assert "ndcg@5" in metrics
        assert "ndcg@10" in metrics
        assert "hit@5" in metrics
        assert "hit@10" in metrics
        assert "mrr" in metrics
        assert "n_users_evaluated" in metrics

    def test_evaluate_metrics_bounded(self, fitted_model_and_matrices):
        """Test that all metrics are bounded between 0 and 1."""
        model, train_matrix, test_matrix = fitted_model_and_matrices

        metrics = evaluate_model(
            model,
            train_matrix,
            test_matrix,
            k_values=[5],
            show_progress=False,
        )

        for name, value in metrics.items():
            if name != "n_users_evaluated":
                assert 0.0 <= value <= 1.0, f"{name} = {value} is out of bounds"

    def test_evaluate_respects_max_users(self, fitted_model_and_matrices):
        """Test that max_users parameter limits evaluation."""
        model, train_matrix, test_matrix = fitted_model_and_matrices

        metrics = evaluate_model(
            model,
            train_matrix,
            test_matrix,
            k_values=[5],
            max_users=2,
            show_progress=False,
        )

        assert metrics["n_users_evaluated"] <= 2

    def test_evaluate_no_test_users(self):
        """Test evaluation when no users have test interactions."""
        train_matrix = csr_matrix(np.eye(5, 10), dtype=np.float32)
        test_matrix = csr_matrix(np.zeros((5, 10)), dtype=np.float32)

        model = ALSRecommender(factors=8, iterations=5, random_state=42)
        model.fit(train_matrix, show_progress=False)

        metrics = evaluate_model(
            model, train_matrix, test_matrix, k_values=[5], show_progress=False
        )

        assert metrics["n_users_evaluated"] == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_item_recommendation(self):
        """Test metrics with single item recommended."""
        recommended = [1]
        relevant = {1}

        assert precision_at_k(recommended, relevant, k=1) == 1.0
        assert recall_at_k(recommended, relevant, k=1) == 1.0
        assert ndcg_at_k(recommended, relevant, k=1) == 1.0
        assert hit_at_k(recommended, relevant, k=1) == 1.0
        assert reciprocal_rank(recommended, relevant) == 1.0

    def test_large_k_with_few_recommendations(self):
        """Test when k is larger than number of recommendations."""
        recommended = [1, 2, 3]
        relevant = {1, 2}

        # Should work correctly even if k > len(recommended)
        p = precision_at_k(recommended, relevant, k=10)
        r = recall_at_k(recommended, relevant, k=10)

        # With 3 recs and 2 relevant, precision@10 should still be 2/10
        # (padding with non-relevant)
        # Actually, implementation uses min(k, len(recommended))
        assert p == pytest.approx(2 / 10)  # 2 relevant / k=10
        assert r == 1.0  # All relevant found

    def test_duplicate_items_in_recommendations(self):
        """Test behavior with duplicate items (shouldn't happen but test anyway)."""
        recommended = [1, 1, 2, 2, 3]
        relevant = {1, 2}

        # First occurrence matters for RR
        assert reciprocal_rank(recommended, relevant) == 1.0

    def test_all_metrics_with_empty_inputs(self):
        """Test all metrics handle empty inputs gracefully."""
        empty_recs: list[int] = []
        empty_relevant: set[int] = set()
        normal_recs = [1, 2, 3]
        normal_relevant = {1, 2}

        # Empty recommendations
        assert precision_at_k(empty_recs, normal_relevant, k=5) == 0.0
        assert recall_at_k(empty_recs, normal_relevant, k=5) == 0.0
        assert ndcg_at_k(empty_recs, normal_relevant, k=5) == 0.0
        assert hit_at_k(empty_recs, normal_relevant, k=5) == 0.0
        assert reciprocal_rank(empty_recs, normal_relevant) == 0.0

        # Empty relevant
        assert recall_at_k(normal_recs, empty_relevant, k=5) == 0.0
        assert ndcg_at_k(normal_recs, empty_relevant, k=5) == 0.0
        assert hit_at_k(normal_recs, empty_relevant, k=5) == 0.0
        assert reciprocal_rank(normal_recs, empty_relevant) == 0.0
