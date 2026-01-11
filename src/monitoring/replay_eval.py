"""
Replay evaluation for measuring recommendation quality.

This module evaluates logged recommendations against actual user
interactions to measure real-world performance:
- Hit Rate: % of users who interacted with a recommendation
- Precision: # of relevant recommendations / # shown
- CTR Proxy: Estimated click-through rate

Usage:
    from src.monitoring.replay_eval import ReplayEvaluator

    evaluator = ReplayEvaluator()
    results = evaluator.evaluate(
        interactions_path="data/processed/test.csv",
        hours=24,
    )
    print(results)
"""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import get_data_config

logger = logging.getLogger(__name__)


@dataclass
class ReplayMetrics:
    """Metrics from replay evaluation."""

    # Core metrics
    hit_rate: float  # % of requests with at least one hit
    precision: float  # Average precision across requests
    ctr_proxy: float  # Total hits / total recommendations shown

    # Counts
    total_requests: int
    requests_with_hits: int
    total_recommendations: int
    total_hits: int

    # By model version (if multiple)
    metrics_by_version: dict[str, dict[str, float]]

    # Evaluation metadata
    evaluation_window_hours: float
    requests_start: Optional[datetime]
    requests_end: Optional[datetime]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Convert datetimes to strings
        if self.requests_start:
            result["requests_start"] = self.requests_start.isoformat()
        if self.requests_end:
            result["requests_end"] = self.requests_end.isoformat()
        return result


@dataclass
class UserReplayResult:
    """Replay result for a single user request."""

    request_id: str
    user_id: int
    recommendations: list[int]
    interactions: set[int]
    hits: list[int]
    precision: float
    is_hit: bool
    model_version: Optional[str]


class ReplayEvaluator:
    """Evaluator for replay-based recommendation quality.

    This class joins logged recommendations with actual user interactions
    to measure how well recommendations performed in practice.

    The evaluation flow:
    1. Load recommendation logs from SQLite
    2. Load actual interactions from CSV (or another source)
    3. For each logged recommendation, check if user interacted with
       any of the recommended items within a time window
    4. Compute aggregate metrics

    Example:
        evaluator = ReplayEvaluator()

        # Evaluate using test set as "future" interactions
        results = evaluator.evaluate(
            interactions_path="data/processed/test.csv",
            hours=24,
        )

        print(f"Hit Rate: {results.hit_rate:.1%}")
        print(f"Precision: {results.precision:.3f}")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize the replay evaluator.

        Args:
            db_path: Path to request logs database. Uses default if not provided.
        """
        if db_path is None:
            config = get_data_config()
            db_path = config.get_logs_path() / "requests.db"

        self.db_path = Path(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Request logs not found at {self.db_path}. "
                "Run some API requests first."
            )

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def load_request_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        endpoint: str = "/recommend",
    ) -> pd.DataFrame:
        """Load recommendation logs from database.

        Args:
            start_time: Start of time window (optional).
            end_time: End of time window (optional).
            endpoint: Endpoint to filter by.

        Returns:
            DataFrame with columns: request_id, user_id, recommendations,
            model_version, timestamp, is_fallback
        """
        conn = self._get_connection()

        try:
            query = """
                SELECT
                    request_id,
                    user_id,
                    recommendations,
                    model_version,
                    timestamp,
                    is_fallback
                FROM request_logs
                WHERE endpoint = ?
                AND user_id IS NOT NULL
                AND recommendations IS NOT NULL
            """
            params: list[Any] = [endpoint]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp < ?"
                params.append(end_time.isoformat())

            query += " ORDER BY timestamp"

            df = pd.read_sql_query(query, conn, params=params)

            # Parse recommendations JSON
            df["recommendations"] = df["recommendations"].apply(
                lambda x: json.loads(x) if x else []
            )

            # Parse timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            return df

        finally:
            conn.close()

    def load_interactions(
        self,
        interactions_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Load user interactions for evaluation.

        Args:
            interactions_path: Path to interactions CSV. Uses test.csv if not provided.

        Returns:
            DataFrame with columns: user_id, item_id
        """
        if interactions_path is None:
            config = get_data_config()
            interactions_path = config.get_processed_path() / "test.csv"

        interactions_path = Path(interactions_path)

        if not interactions_path.exists():
            raise FileNotFoundError(
                f"Interactions file not found at {interactions_path}"
            )

        df = pd.read_csv(interactions_path)

        # Ensure required columns exist
        if "user_id" not in df.columns or "item_id" not in df.columns:
            raise ValueError("Interactions file must have user_id and item_id columns")

        return df[["user_id", "item_id"]]

    def build_user_interactions_map(
        self,
        interactions_df: pd.DataFrame,
    ) -> dict[int, set[int]]:
        """Build a map of user_id -> set of item_ids they interacted with.

        Args:
            interactions_df: DataFrame with user_id and item_id columns.

        Returns:
            Dict mapping user_id to set of item_ids.
        """
        user_items: dict[int, set[int]] = {}

        for _, row in interactions_df.iterrows():
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])

            if user_id not in user_items:
                user_items[user_id] = set()

            user_items[user_id].add(item_id)

        return user_items

    def evaluate_single_request(
        self,
        request_id: str,
        user_id: int,
        recommendations: list[int],
        user_interactions: set[int],
        model_version: Optional[str] = None,
    ) -> UserReplayResult:
        """Evaluate a single recommendation request.

        Args:
            request_id: Unique request identifier.
            user_id: User who received recommendations.
            recommendations: List of recommended item IDs.
            user_interactions: Set of items user actually interacted with.
            model_version: Model version used.

        Returns:
            UserReplayResult with metrics for this request.
        """
        # Find hits (recommended items that user interacted with)
        rec_set = set(recommendations)
        hits = list(rec_set & user_interactions)

        # Compute precision
        precision = len(hits) / len(recommendations) if recommendations else 0.0

        return UserReplayResult(
            request_id=request_id,
            user_id=user_id,
            recommendations=recommendations,
            interactions=user_interactions,
            hits=hits,
            precision=precision,
            is_hit=len(hits) > 0,
            model_version=model_version,
        )

    def evaluate(
        self,
        interactions_path: Optional[Path] = None,
        hours: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ReplayMetrics:
        """Run replay evaluation on logged recommendations.

        Args:
            interactions_path: Path to interactions CSV for ground truth.
            hours: Number of hours of logs to evaluate (alternative to start/end).
            start_time: Start of evaluation window.
            end_time: End of evaluation window.

        Returns:
            ReplayMetrics with aggregate results.
        """
        # Determine time window
        if hours is not None:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

        logger.info(f"Running replay evaluation from {start_time} to {end_time}")

        # Load data
        logs_df = self.load_request_logs(start_time, end_time)
        if logs_df.empty:
            logger.warning("No recommendation logs found in time window")
            return ReplayMetrics(
                hit_rate=0.0,
                precision=0.0,
                ctr_proxy=0.0,
                total_requests=0,
                requests_with_hits=0,
                total_recommendations=0,
                total_hits=0,
                metrics_by_version={},
                evaluation_window_hours=hours or 0,
                requests_start=start_time,
                requests_end=end_time,
            )

        interactions_df = self.load_interactions(interactions_path)
        user_interactions = self.build_user_interactions_map(interactions_df)

        logger.info(f"Loaded {len(logs_df)} recommendation logs")
        logger.info(f"Loaded interactions for {len(user_interactions)} users")

        # Evaluate each request
        results: list[UserReplayResult] = []
        version_results: dict[str, list[UserReplayResult]] = {}

        for _, row in logs_df.iterrows():
            user_id = int(row["user_id"])

            # Get user's actual interactions
            user_items = user_interactions.get(user_id, set())

            result = self.evaluate_single_request(
                request_id=row["request_id"],
                user_id=user_id,
                recommendations=row["recommendations"],
                user_interactions=user_items,
                model_version=row["model_version"],
            )
            results.append(result)

            # Group by version
            version = row["model_version"] or "unknown"
            if version not in version_results:
                version_results[version] = []
            version_results[version].append(result)

        # Compute aggregate metrics
        total_requests = len(results)
        requests_with_hits = sum(1 for r in results if r.is_hit)
        total_recommendations = sum(len(r.recommendations) for r in results)
        total_hits = sum(len(r.hits) for r in results)

        hit_rate = requests_with_hits / total_requests if total_requests > 0 else 0
        avg_precision = (
            sum(r.precision for r in results) / total_requests
            if total_requests > 0
            else 0
        )
        ctr_proxy = (
            total_hits / total_recommendations if total_recommendations > 0 else 0
        )

        # Compute metrics by version
        metrics_by_version: dict[str, dict[str, float]] = {}
        for version, v_results in version_results.items():
            v_total = len(v_results)
            v_hits = sum(1 for r in v_results if r.is_hit)
            v_total_recs = sum(len(r.recommendations) for r in v_results)
            v_total_hits = sum(len(r.hits) for r in v_results)

            metrics_by_version[version] = {
                "hit_rate": v_hits / v_total if v_total > 0 else 0,
                "precision": (
                    sum(r.precision for r in v_results) / v_total if v_total > 0 else 0
                ),
                "ctr_proxy": v_total_hits / v_total_recs if v_total_recs > 0 else 0,
                "request_count": v_total,
            }

        # Get actual time range from logs
        requests_start = logs_df["timestamp"].min()
        requests_end = logs_df["timestamp"].max()

        # Convert pandas Timestamp to datetime if needed
        if hasattr(requests_start, "to_pydatetime"):
            requests_start = requests_start.to_pydatetime()
        if hasattr(requests_end, "to_pydatetime"):
            requests_end = requests_end.to_pydatetime()

        return ReplayMetrics(
            hit_rate=round(hit_rate, 4),
            precision=round(avg_precision, 4),
            ctr_proxy=round(ctr_proxy, 4),
            total_requests=total_requests,
            requests_with_hits=requests_with_hits,
            total_recommendations=total_recommendations,
            total_hits=total_hits,
            metrics_by_version=metrics_by_version,
            evaluation_window_hours=hours or 0,
            requests_start=requests_start,
            requests_end=requests_end,
        )

    def print_results(self, metrics: ReplayMetrics) -> None:
        """Print formatted replay evaluation results.

        Args:
            metrics: ReplayMetrics to print.
        """
        print("\n" + "=" * 60)
        print(" REPLAY EVALUATION RESULTS")
        print("=" * 60)

        print("\nEvaluation Window:")
        print(f"  Start: {metrics.requests_start}")
        print(f"  End:   {metrics.requests_end}")

        print("\n--- Overall Metrics ---")
        print(f"  Hit Rate:  {metrics.hit_rate:.1%}")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  CTR Proxy: {metrics.ctr_proxy:.3f}")

        print("\n--- Counts ---")
        print(f"  Total requests:      {metrics.total_requests:,}")
        print(f"  Requests with hits:  {metrics.requests_with_hits:,}")
        print(f"  Total recommendations: {metrics.total_recommendations:,}")
        print(f"  Total hits:          {metrics.total_hits:,}")

        if metrics.metrics_by_version:
            print("\n--- By Model Version ---")
            for version, v_metrics in metrics.metrics_by_version.items():
                print(f"\n  {version}:")
                print(f"    Requests:  {v_metrics['request_count']:,}")
                print(f"    Hit Rate:  {v_metrics['hit_rate']:.1%}")
                print(f"    Precision: {v_metrics['precision']:.3f}")

        print("\n" + "=" * 60)

    def save_results(
        self,
        metrics: ReplayMetrics,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save replay evaluation results to JSON.

        Args:
            metrics: ReplayMetrics to save.
            output_path: Output path. Auto-generated if not provided.

        Returns:
            Path to saved file.
        """
        if output_path is None:
            config = get_data_config()
            reports_dir = config.get_logs_path() / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = reports_dir / f"replay_eval_{timestamp}.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"Replay evaluation saved to {output_path}")
        return output_path


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run replay evaluation")
    parser.add_argument(
        "--hours",
        type=float,
        default=24,
        help="Hours of logs to evaluate (default: 24)",
    )
    parser.add_argument(
        "--interactions",
        type=str,
        help="Path to interactions CSV (default: test.csv)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    try:
        evaluator = ReplayEvaluator()

        interactions_path = Path(args.interactions) if args.interactions else None
        metrics = evaluator.evaluate(
            interactions_path=interactions_path,
            hours=args.hours,
        )

        evaluator.print_results(metrics)

        if args.save:
            saved_path = evaluator.save_results(metrics)
            print(f"\n✓ Results saved to {saved_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise
