"""
Evaluation metrics for recommendation models.

This module provides ranking metrics to evaluate recommender systems:
- Precision@K: Fraction of recommended items that are relevant
- Recall@K: Fraction of relevant items that were recommended
- NDCG@K: Normalized Discounted Cumulative Gain (position-aware)
- MRR: Mean Reciprocal Rank (position of first relevant item)
- Hit Rate@K: Fraction of users with at least one relevant recommendation

Usage:
    from src.training.evaluate import evaluate_model, compute_metrics

    # Evaluate a model on test data
    metrics = evaluate_model(model, train_matrix, test_matrix, k_values=[5, 10, 20])

    # Compute metrics for a single user
    user_metrics = compute_metrics(recommended_items, relevant_items, k=10)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from src.models.base import BaseRecommender

logger = logging.getLogger(__name__)


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Compute Precision@K.

    Precision@K = (# relevant items in top K) / K

    Args:
        recommended: List of recommended item indices (ordered by score).
        relevant: Set of relevant (ground truth) item indices.
        k: Number of top recommendations to consider.

    Returns:
        Precision@K score between 0 and 1.
    """
    if k <= 0:
        return 0.0

    top_k = recommended[:k]
    n_relevant_in_top_k = len(set(top_k) & relevant)

    return n_relevant_in_top_k / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Compute Recall@K.

    Recall@K = (# relevant items in top K) / (# total relevant items)

    Args:
        recommended: List of recommended item indices (ordered by score).
        relevant: Set of relevant (ground truth) item indices.
        k: Number of top recommendations to consider.

    Returns:
        Recall@K score between 0 and 1.
    """
    if len(relevant) == 0:
        return 0.0

    top_k = recommended[:k]
    n_relevant_in_top_k = len(set(top_k) & relevant)

    return n_relevant_in_top_k / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Compute NDCG@K (Normalized Discounted Cumulative Gain).

    NDCG accounts for the position of relevant items - items ranked higher
    contribute more to the score.

    DCG@K = Σ (rel_i / log2(i + 1)) for i = 1 to K
    NDCG@K = DCG@K / IDCG@K (ideal DCG with perfect ranking)

    Args:
        recommended: List of recommended item indices (ordered by score).
        relevant: Set of relevant (ground truth) item indices.
        k: Number of top recommendations to consider.

    Returns:
        NDCG@K score between 0 and 1.
    """
    if len(relevant) == 0 or k <= 0:
        return 0.0

    top_k = recommended[:k]

    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            # Position is 1-indexed for the formula
            dcg += 1.0 / float(np.log2(i + 2))  # +2 because i is 0-indexed

    # Compute IDCG (ideal DCG - all relevant items at top)
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / float(np.log2(i + 2)) for i in range(n_relevant))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def reciprocal_rank(recommended: list[int], relevant: set[int]) -> float:
    """Compute Reciprocal Rank.

    RR = 1 / (position of first relevant item)

    Args:
        recommended: List of recommended item indices (ordered by score).
        relevant: Set of relevant (ground truth) item indices.

    Returns:
        Reciprocal Rank between 0 and 1 (0 if no relevant item found).
    """
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)

    return 0.0


def hit_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Compute Hit@K.

    Hit@K = 1 if any relevant item in top K, else 0

    Args:
        recommended: List of recommended item indices (ordered by score).
        relevant: Set of relevant (ground truth) item indices.
        k: Number of top recommendations to consider.

    Returns:
        1.0 if hit, 0.0 otherwise.
    """
    top_k = set(recommended[:k])
    return 1.0 if len(top_k & relevant) > 0 else 0.0


def compute_metrics(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> dict[str, float]:
    """Compute all ranking metrics for a single user.

    Args:
        recommended: List of recommended item indices (ordered by score).
        relevant: Set of relevant (ground truth) item indices.
        k: Number of top recommendations to consider.

    Returns:
        Dictionary with precision, recall, ndcg, mrr, and hit metrics.
    """
    return {
        f"precision@{k}": precision_at_k(recommended, relevant, k),
        f"recall@{k}": recall_at_k(recommended, relevant, k),
        f"ndcg@{k}": ndcg_at_k(recommended, relevant, k),
        "mrr": reciprocal_rank(recommended, relevant),
        f"hit@{k}": hit_at_k(recommended, relevant, k),
    }


def evaluate_model(
    model: BaseRecommender,
    train_matrix: csr_matrix,
    test_matrix: csr_matrix,
    k_values: Optional[list[int]] = None,
    max_users: Optional[int] = None,
    show_progress: bool = True,
) -> dict[str, float]:
    """Evaluate a recommendation model on test data.

    For each user in the test set:
    1. Get recommendations (excluding items in training set)
    2. Compare with items the user actually interacted with in test set
    3. Compute ranking metrics

    Args:
        model: Trained recommender model.
        train_matrix: Training interaction matrix (n_users, n_items).
        test_matrix: Test interaction matrix (n_users, n_items).
        k_values: List of K values to compute metrics for. Default: [5, 10, 20].
        max_users: Maximum number of users to evaluate (for speed). None = all.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary of metric names to average values.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    max_k = max(k_values)

    # Find users with test interactions
    test_users = np.where(np.diff(test_matrix.indptr) > 0)[0]

    if max_users is not None:
        test_users = test_users[:max_users]

    logger.info(f"Evaluating {len(test_users)} users with test interactions")

    # Collect metrics for all users
    all_metrics: dict[str, list[float]] = {f"precision@{k}": [] for k in k_values}
    all_metrics.update({f"recall@{k}": [] for k in k_values})
    all_metrics.update({f"ndcg@{k}": [] for k in k_values})
    all_metrics.update({f"hit@{k}": [] for k in k_values})
    all_metrics["mrr"] = []

    # Evaluate each user
    iterator = tqdm(test_users, desc="Evaluating") if show_progress else test_users

    for user_idx in iterator:
        # Get relevant items (items user interacted with in test set)
        relevant_items = set(test_matrix[user_idx].indices)

        if len(relevant_items) == 0:
            continue

        # Get recommendations
        try:
            recs = model.recommend(
                user_idx=user_idx,
                n=max_k,
                filter_already_liked=True,  # Exclude training items
            )
            recommended = [item_idx for item_idx, _ in recs]
        except (ValueError, IndexError):
            # User might not exist in training data
            continue

        if len(recommended) == 0:
            continue

        # Compute metrics for each k
        for k in k_values:
            all_metrics[f"precision@{k}"].append(
                precision_at_k(recommended, relevant_items, k)
            )
            all_metrics[f"recall@{k}"].append(
                recall_at_k(recommended, relevant_items, k)
            )
            all_metrics[f"ndcg@{k}"].append(ndcg_at_k(recommended, relevant_items, k))
            all_metrics[f"hit@{k}"].append(hit_at_k(recommended, relevant_items, k))

        all_metrics["mrr"].append(reciprocal_rank(recommended, relevant_items))

    # Compute averages
    avg_metrics = {}
    for name, values in all_metrics.items():
        if len(values) > 0:
            avg_metrics[name] = float(np.mean(values))
        else:
            avg_metrics[name] = 0.0

    # Add count of evaluated users
    avg_metrics["n_users_evaluated"] = len(all_metrics["mrr"])

    return avg_metrics


def evaluate_model_df(
    model: BaseRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_mapping: dict[int, int],
    item_mapping: dict[int, int],
    k_values: Optional[list[int]] = None,
    show_progress: bool = True,
) -> dict[str, float]:
    """Evaluate a model using DataFrames instead of sparse matrices.

    Convenience wrapper that builds sparse matrices from DataFrames.

    Args:
        model: Trained recommender model.
        train_df: Training interactions DataFrame with user_id, item_id columns.
        test_df: Test interactions DataFrame with user_id, item_id columns.
        user_mapping: Mapping from user_id to matrix index.
        item_mapping: Mapping from item_id to matrix index.
        k_values: List of K values for metrics. Default: [5, 10, 20].
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary of metric names to average values.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    n_users = len(user_mapping)
    n_items = len(item_mapping)

    # Build train matrix
    train_rows = train_df["user_id"].map(user_mapping).dropna().astype(int)
    train_cols = train_df["item_id"].map(item_mapping).dropna().astype(int)
    valid_train = train_rows.index.intersection(train_cols.index)
    train_matrix = csr_matrix(
        ([1] * len(valid_train), (train_rows[valid_train], train_cols[valid_train])),
        shape=(n_users, n_items),
    )

    # Build test matrix
    test_rows = test_df["user_id"].map(user_mapping).dropna().astype(int)
    test_cols = test_df["item_id"].map(item_mapping).dropna().astype(int)
    valid_test = test_rows.index.intersection(test_cols.index)
    test_matrix = csr_matrix(
        ([1] * len(valid_test), (test_rows[valid_test], test_cols[valid_test])),
        shape=(n_users, n_items),
    )

    return evaluate_model(
        model=model,
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        k_values=k_values,
        show_progress=show_progress,
    )


def print_metrics(metrics: dict[str, float], title: str = "Evaluation Results") -> None:
    """Pretty print evaluation metrics.

    Args:
        metrics: Dictionary of metric names to values.
        title: Title to display.
    """
    print(f"\n{'=' * 50}")
    print(title)
    print("=" * 50)

    # Group by metric type
    precision_metrics = {k: v for k, v in metrics.items() if k.startswith("precision")}
    recall_metrics = {k: v for k, v in metrics.items() if k.startswith("recall")}
    ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith("ndcg")}
    hit_metrics = {k: v for k, v in metrics.items() if k.startswith("hit")}

    if precision_metrics:
        print("\nPrecision:")
        for k, v in sorted(precision_metrics.items()):
            print(f"  {k}: {v:.4f}")

    if recall_metrics:
        print("\nRecall:")
        for k, v in sorted(recall_metrics.items()):
            print(f"  {k}: {v:.4f}")

    if ndcg_metrics:
        print("\nNDCG:")
        for k, v in sorted(ndcg_metrics.items()):
            print(f"  {k}: {v:.4f}")

    if hit_metrics:
        print("\nHit Rate:")
        for k, v in sorted(hit_metrics.items()):
            print(f"  {k}: {v:.4f}")

    if "mrr" in metrics:
        print(f"\nMRR: {metrics['mrr']:.4f}")

    if "n_users_evaluated" in metrics:
        print(f"\nUsers evaluated: {int(metrics['n_users_evaluated'])}")

    print("=" * 50)
