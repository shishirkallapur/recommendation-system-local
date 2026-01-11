"""
KPI computation for the recommendation service.

This module computes key performance indicators from request logs:
- Traffic metrics: request count, requests per minute, unique users
- Latency metrics: mean, p50, p95, p99
- Quality metrics: fallback rate, catalog coverage
- Error tracking: error count and rate

Usage:
    from src.monitoring.kpis import KPICalculator

    calculator = KPICalculator()
    kpis = calculator.compute_all(hours=24)
    print(kpis)
"""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import get_data_config

logger = logging.getLogger(__name__)


@dataclass
class TrafficKPIs:
    """Traffic-related KPIs."""

    total_requests: int
    requests_per_minute: float
    unique_users: int
    requests_by_endpoint: dict[str, int]


@dataclass
class LatencyKPIs:
    """Latency-related KPIs."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    latency_by_endpoint: dict[str, float]


@dataclass
class QualityKPIs:
    """Recommendation quality KPIs."""

    fallback_rate: float
    fallback_count: int
    catalog_coverage: float
    unique_items_recommended: int
    total_catalog_size: int
    avg_recommendations_per_request: float


@dataclass
class KPIReport:
    """Complete KPI report."""

    start_time: datetime
    end_time: datetime
    duration_hours: float
    traffic: TrafficKPIs
    latency: LatencyKPIs
    quality: QualityKPIs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_hours": self.duration_hours,
            "traffic": asdict(self.traffic),
            "latency": asdict(self.latency),
            "quality": asdict(self.quality),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class KPICalculator:
    """Calculator for recommendation service KPIs.

    Reads from the request logs SQLite database and computes
    various performance and quality metrics.

    Example:
        calculator = KPICalculator()

        # Compute KPIs for last 24 hours
        report = calculator.compute_all(hours=24)
        print(f"Total requests: {report.traffic.total_requests}")
        print(f"p95 latency: {report.latency.p95_ms:.1f}ms")
        print(f"Fallback rate: {report.quality.fallback_rate:.1%}")

        # Save report
        calculator.save_report(report, "reports/daily_kpis.json")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize the KPI calculator.

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

    def compute_traffic_kpis(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> TrafficKPIs:
        """Compute traffic-related KPIs.

        Args:
            start_time: Start of time window.
            end_time: End of time window.

        Returns:
            TrafficKPIs dataclass.
        """
        conn = self._get_connection()

        try:
            # Total requests
            cursor = conn.execute(
                """
                SELECT COUNT(*) as total
                FROM request_logs
                WHERE timestamp >= ? AND timestamp < ?
                """,
                (start_time.isoformat(), end_time.isoformat()),
            )
            total_requests = cursor.fetchone()["total"]

            # Unique users (excluding NULL for /popular)
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT user_id) as unique_users
                FROM request_logs
                WHERE timestamp >= ? AND timestamp < ?
                AND user_id IS NOT NULL
                """,
                (start_time.isoformat(), end_time.isoformat()),
            )
            unique_users = cursor.fetchone()["unique_users"]

            # Requests by endpoint
            cursor = conn.execute(
                """
                SELECT endpoint, COUNT(*) as count
                FROM request_logs
                WHERE timestamp >= ? AND timestamp < ?
                GROUP BY endpoint
                """,
                (start_time.isoformat(), end_time.isoformat()),
            )
            requests_by_endpoint = {row["endpoint"]: row["count"] for row in cursor}

            # Calculate requests per minute
            duration_minutes = (end_time - start_time).total_seconds() / 60
            requests_per_minute = (
                total_requests / duration_minutes if duration_minutes > 0 else 0
            )

            return TrafficKPIs(
                total_requests=total_requests,
                requests_per_minute=round(requests_per_minute, 2),
                unique_users=unique_users,
                requests_by_endpoint=requests_by_endpoint,
            )

        finally:
            conn.close()

    def compute_latency_kpis(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> LatencyKPIs:
        """Compute latency-related KPIs.

        Args:
            start_time: Start of time window.
            end_time: End of time window.

        Returns:
            LatencyKPIs dataclass.
        """
        conn = self._get_connection()

        try:
            # Get all latencies
            cursor = conn.execute(
                """
                SELECT latency_ms
                FROM request_logs
                WHERE timestamp >= ? AND timestamp < ?
                AND latency_ms IS NOT NULL
                """,
                (start_time.isoformat(), end_time.isoformat()),
            )
            latencies = [row["latency_ms"] for row in cursor]

            if not latencies:
                return LatencyKPIs(
                    mean_ms=0,
                    p50_ms=0,
                    p95_ms=0,
                    p99_ms=0,
                    max_ms=0,
                    latency_by_endpoint={},
                )

            latencies_array = np.array(latencies)

            # Latency by endpoint
            cursor = conn.execute(
                """
                SELECT endpoint, AVG(latency_ms) as avg_latency
                FROM request_logs
                WHERE timestamp >= ? AND timestamp < ?
                AND latency_ms IS NOT NULL
                GROUP BY endpoint
                """,
                (start_time.isoformat(), end_time.isoformat()),
            )
            latency_by_endpoint = {
                row["endpoint"]: round(row["avg_latency"], 2) for row in cursor
            }

            return LatencyKPIs(
                mean_ms=round(float(np.mean(latencies_array)), 2),
                p50_ms=round(float(np.percentile(latencies_array, 50)), 2),
                p95_ms=round(float(np.percentile(latencies_array, 95)), 2),
                p99_ms=round(float(np.percentile(latencies_array, 99)), 2),
                max_ms=round(float(np.max(latencies_array)), 2),
                latency_by_endpoint=latency_by_endpoint,
            )

        finally:
            conn.close()

    def compute_quality_kpis(
        self,
        start_time: datetime,
        end_time: datetime,
        total_catalog_size: Optional[int] = None,
    ) -> QualityKPIs:
        """Compute recommendation quality KPIs.

        Args:
            start_time: Start of time window.
            end_time: End of time window.
            total_catalog_size: Total items in catalog (for coverage calculation).

        Returns:
            QualityKPIs dataclass.
        """
        conn = self._get_connection()

        try:
            # Fallback rate (only for /recommend endpoint)
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(is_fallback) as fallback_count
                FROM request_logs
                WHERE timestamp >= ? AND timestamp < ?
                AND endpoint = '/recommend'
                """,
                (start_time.isoformat(), end_time.isoformat()),
            )
            row = cursor.fetchone()
            total_recommend = row["total"]
            fallback_count = row["fallback_count"] or 0
            fallback_rate = (
                fallback_count / total_recommend if total_recommend > 0 else 0
            )

            # Unique items recommended
            cursor = conn.execute(
                """
                SELECT recommendations
                FROM request_logs
                WHERE timestamp >= ? AND timestamp < ?
                AND recommendations IS NOT NULL
                """,
                (start_time.isoformat(), end_time.isoformat()),
            )

            all_items: set[int] = set()
            total_recs = 0
            rec_count = 0

            for row in cursor:
                try:
                    items = json.loads(row["recommendations"])
                    all_items.update(items)
                    total_recs += len(items)
                    rec_count += 1
                except (json.JSONDecodeError, TypeError):
                    continue

            unique_items = len(all_items)
            avg_recs = total_recs / rec_count if rec_count > 0 else 0

            # Catalog coverage
            if total_catalog_size is None:
                # Try to get from model store
                try:
                    from src.api.model_loader import model_store

                    total_catalog_size = model_store.n_items
                except Exception:
                    total_catalog_size = unique_items  # Fallback

            coverage = (
                unique_items / total_catalog_size if total_catalog_size > 0 else 0
            )

            return QualityKPIs(
                fallback_rate=round(fallback_rate, 4),
                fallback_count=fallback_count,
                catalog_coverage=round(coverage, 4),
                unique_items_recommended=unique_items,
                total_catalog_size=total_catalog_size,
                avg_recommendations_per_request=round(avg_recs, 2),
            )

        finally:
            conn.close()

    def compute_all(
        self,
        hours: float = 24,
        end_time: Optional[datetime] = None,
        total_catalog_size: Optional[int] = None,
    ) -> KPIReport:
        """Compute all KPIs for a time window.

        Args:
            hours: Number of hours to look back.
            end_time: End of time window. Defaults to now.
            total_catalog_size: Total items in catalog.

        Returns:
            Complete KPIReport.
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        start_time = end_time - timedelta(hours=hours)

        logger.info(f"Computing KPIs from {start_time} to {end_time}")

        traffic = self.compute_traffic_kpis(start_time, end_time)
        latency = self.compute_latency_kpis(start_time, end_time)
        quality = self.compute_quality_kpis(start_time, end_time, total_catalog_size)

        return KPIReport(
            start_time=start_time,
            end_time=end_time,
            duration_hours=hours,
            traffic=traffic,
            latency=latency,
            quality=quality,
        )

    def save_report(
        self,
        report: KPIReport,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save KPI report to JSON file.

        Args:
            report: KPI report to save.
            output_path: Output file path. Auto-generated if not provided.

        Returns:
            Path to saved report.
        """
        if output_path is None:
            config = get_data_config()
            reports_dir = config.get_logs_path() / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            timestamp = report.end_time.strftime("%Y%m%d_%H%M%S")
            output_path = reports_dir / f"kpi_report_{timestamp}.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(report.to_json())

        logger.info(f"KPI report saved to {output_path}")
        return output_path

    def print_report(self, report: KPIReport) -> None:
        """Print a formatted KPI report to console.

        Args:
            report: KPI report to print.
        """
        print("\n" + "=" * 60)
        print(" KPI REPORT")
        print("=" * 60)
        print(f"\nTime window: {report.start_time} to {report.end_time}")
        print(f"Duration: {report.duration_hours} hours")

        print("\n--- Traffic ---")
        print(f"  Total requests: {report.traffic.total_requests:,}")
        print(f"  Requests/minute: {report.traffic.requests_per_minute:.2f}")
        print(f"  Unique users: {report.traffic.unique_users:,}")
        print("  By endpoint:")
        for endpoint, count in report.traffic.requests_by_endpoint.items():
            print(f"    {endpoint}: {count:,}")

        print("\n--- Latency ---")
        print(f"  Mean: {report.latency.mean_ms:.1f}ms")
        print(f"  p50:  {report.latency.p50_ms:.1f}ms")
        print(f"  p95:  {report.latency.p95_ms:.1f}ms")
        print(f"  p99:  {report.latency.p99_ms:.1f}ms")
        print(f"  Max:  {report.latency.max_ms:.1f}ms")

        print("\n--- Quality ---")
        print(f"  Fallback rate: {report.quality.fallback_rate:.1%}")
        print(f"  Fallback count: {report.quality.fallback_count:,}")
        print(f"  Catalog coverage: {report.quality.catalog_coverage:.1%}")
        print(
            f"  Unique items recommended: {report.quality.unique_items_recommended:,}"
        )
        print(
            f"  Avg recommendations/request: {report.quality.avg_recommendations_per_request:.1f}"
        )

        print("\n" + "=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Compute KPIs from request logs")
    parser.add_argument(
        "--hours",
        type=float,
        default=24,
        help="Number of hours to analyze (default: 24)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional)",
    )
    args = parser.parse_args()

    try:
        calculator = KPICalculator()
        report = calculator.compute_all(hours=args.hours)
        calculator.print_report(report)

        if args.save:
            output_path = Path(args.output) if args.output else None
            saved_path = calculator.save_report(report, output_path)
            print(f"\n✓ Report saved to {saved_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run some API requests first.")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise
