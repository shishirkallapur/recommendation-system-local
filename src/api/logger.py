"""
Request logger for the recommendation API.

This module logs all API requests to SQLite for:
- Monitoring (traffic, latency, error rates)
- Debugging (investigate specific requests)
- Offline evaluation (replay recommendations)
- Retraining (use logged data to improve model)

Logging is done asynchronously to avoid blocking responses.

Usage:
    from src.api.logger import RequestLogger, get_logger

    logger = get_logger()

    # Log a recommendation request
    logger.log_recommendation(
        request_id="...",
        user_id=196,
        recommendations=[1, 2, 3],
        scores=[0.9, 0.8, 0.7],
        latency_ms=15.5,
        model_version="v1.0",
    )

    # Query logs
    recent = logger.get_recent_requests(limit=100)
"""

import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Optional

from src.config import get_data_config

logger = logging.getLogger(__name__)


@dataclass
class RequestLogEntry:
    """A single request log entry."""

    request_id: str
    timestamp: datetime
    user_id: Optional[int]
    endpoint: str
    model_version: Optional[str]
    recommendations: list[int]
    scores: list[float]
    latency_ms: float
    is_fallback: bool
    fallback_reason: Optional[str]


class RequestLogger:
    """Async request logger with SQLite backend.

    This class provides thread-safe, non-blocking logging of API requests.
    Writes are queued and processed by a background thread to avoid
    blocking the main request thread.

    The logger creates and manages a SQLite database with the schema:
    - request_id: UUID for the request
    - timestamp: When request was received
    - user_id: Requested user ID
    - endpoint: API endpoint called
    - model_version: Model version used
    - recommendations: JSON array of movie IDs
    - scores: JSON array of scores
    - latency_ms: Response time in milliseconds
    - is_fallback: Whether fallback was used
    - fallback_reason: Why fallback was used

    Example:
        logger = RequestLogger()
        logger.start()

        logger.log_recommendation(
            request_id=str(uuid.uuid4()),
            user_id=196,
            recommendations=[1, 2, 3],
            scores=[0.9, 0.8, 0.7],
            latency_ms=15.5,
            model_version="v1.0",
        )

        logger.stop()
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize the request logger.

        Args:
            db_path: Path to SQLite database. Uses default if not provided.
        """
        if db_path is None:
            config = get_data_config()
            db_path = config.get_logs_path() / "requests.db"

        self.db_path = Path(db_path)
        self._queue: Queue[Optional[RequestLogEntry]] = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Create the database table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS request_logs (
                    request_id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    user_id INTEGER,
                    endpoint TEXT NOT NULL,
                    model_version TEXT,
                    recommendations TEXT,
                    scores TEXT,
                    latency_ms REAL,
                    is_fallback INTEGER DEFAULT 0,
                    fallback_reason TEXT
                )
            """
            )

            # Create indexes for common queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON request_logs(timestamp)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_id
                ON request_logs(user_id)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_endpoint
                ON request_logs(endpoint)
            """
            )

            conn.commit()

        logger.info(f"Request logger initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection.

        Yields:
            SQLite connection.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="RequestLoggerWorker",
        )
        self._worker_thread.start()
        logger.info("Request logger worker started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background worker thread.

        Args:
            timeout: Maximum time to wait for pending writes.
        """
        if not self._running:
            return

        self._running = False

        # Send poison pill to stop worker
        self._queue.put(None)

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)

        logger.info("Request logger worker stopped")

    def _worker_loop(self) -> None:
        """Background worker loop that processes the write queue."""
        while self._running or not self._queue.empty():
            try:
                entry = self._queue.get(timeout=1.0)

                if entry is None:
                    # Poison pill received
                    break

                self._write_entry(entry)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error writing log entry: {e}")

    def _write_entry(self, entry: RequestLogEntry) -> None:
        """Write a single log entry to the database.

        Args:
            entry: Log entry to write.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO request_logs (
                    request_id, timestamp, user_id, endpoint, model_version,
                    recommendations, scores, latency_ms, is_fallback, fallback_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.request_id,
                    entry.timestamp.isoformat(),
                    entry.user_id,
                    entry.endpoint,
                    entry.model_version,
                    json.dumps(entry.recommendations),
                    json.dumps(entry.scores),
                    entry.latency_ms,
                    1 if entry.is_fallback else 0,
                    entry.fallback_reason,
                ),
            )
            conn.commit()

    def log_recommendation(
        self,
        request_id: str,
        user_id: int,
        recommendations: list[int],
        scores: list[float],
        latency_ms: float,
        model_version: Optional[str] = None,
        is_fallback: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> None:
        """Log a recommendation request.

        Args:
            request_id: Unique request ID (UUID).
            user_id: User ID from request.
            recommendations: List of recommended movie IDs.
            scores: List of recommendation scores.
            latency_ms: Response time in milliseconds.
            model_version: Model version used.
            is_fallback: Whether fallback was used.
            fallback_reason: Reason for fallback.
        """
        entry = RequestLogEntry(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            endpoint="/recommend",
            model_version=model_version,
            recommendations=recommendations,
            scores=scores,
            latency_ms=latency_ms,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
        )
        self._queue.put(entry)

    def log_similar(
        self,
        request_id: str,
        movie_id: int,
        similar_items: list[int],
        scores: list[float],
        latency_ms: float,
        model_version: Optional[str] = None,
    ) -> None:
        """Log a similar items request.

        Args:
            request_id: Unique request ID.
            movie_id: Query movie ID.
            similar_items: List of similar movie IDs.
            scores: List of similarity scores.
            latency_ms: Response time in milliseconds.
            model_version: Model version used.
        """
        entry = RequestLogEntry(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            user_id=movie_id,  # Store movie_id in user_id field for similar
            endpoint="/similar",
            model_version=model_version,
            recommendations=similar_items,
            scores=scores,
            latency_ms=latency_ms,
            is_fallback=False,
            fallback_reason=None,
        )
        self._queue.put(entry)

    def log_popular(
        self,
        request_id: str,
        recommendations: list[int],
        scores: list[float],
        latency_ms: float,
        genre_filter: Optional[str] = None,
    ) -> None:
        """Log a popular items request.

        Args:
            request_id: Unique request ID.
            recommendations: List of popular movie IDs.
            scores: List of popularity scores.
            latency_ms: Response time in milliseconds.
            genre_filter: Genre filter applied (if any).
        """
        entry = RequestLogEntry(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            user_id=None,
            endpoint="/popular",
            model_version=None,
            recommendations=recommendations,
            scores=scores,
            latency_ms=latency_ms,
            is_fallback=False,
            fallback_reason=f"genre={genre_filter}" if genre_filter else None,
        )
        self._queue.put(entry)

    def get_recent_requests(
        self,
        limit: int = 100,
        endpoint: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get recent request logs.

        Args:
            limit: Maximum number of entries to return.
            endpoint: Filter by endpoint (optional).

        Returns:
            List of log entries as dictionaries.
        """
        with self._get_connection() as conn:
            if endpoint:
                cursor = conn.execute(
                    """
                    SELECT * FROM request_logs
                    WHERE endpoint = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (endpoint, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM request_logs
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_request_count(
        self,
        since: Optional[datetime] = None,
        endpoint: Optional[str] = None,
    ) -> int:
        """Get count of requests.

        Args:
            since: Only count requests after this time.
            endpoint: Filter by endpoint.

        Returns:
            Number of matching requests.
        """
        with self._get_connection() as conn:
            query = "SELECT COUNT(*) FROM request_logs WHERE 1=1"
            params: list[Any] = []

            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())

            if endpoint:
                query += " AND endpoint = ?"
                params.append(endpoint)

            cursor = conn.execute(query, params)
            result = cursor.fetchone()

            return int(result[0]) if result else 0

    def get_average_latency(
        self,
        since: Optional[datetime] = None,
        endpoint: Optional[str] = None,
    ) -> Optional[float]:
        """Get average latency.

        Args:
            since: Only include requests after this time.
            endpoint: Filter by endpoint.

        Returns:
            Average latency in ms, or None if no requests.
        """
        with self._get_connection() as conn:
            query = "SELECT AVG(latency_ms) FROM request_logs WHERE 1=1"
            params: list[Any] = []

            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())

            if endpoint:
                query += " AND endpoint = ?"
                params.append(endpoint)

            cursor = conn.execute(query, params)
            result = cursor.fetchone()

            return float(result[0]) if result and result[0] else None

    def get_fallback_rate(
        self,
        since: Optional[datetime] = None,
    ) -> Optional[float]:
        """Get fallback rate for /recommend endpoint.

        Args:
            since: Only include requests after this time.

        Returns:
            Fallback rate (0-1), or None if no requests.
        """
        with self._get_connection() as conn:
            query = """
                SELECT
                    AVG(is_fallback) as fallback_rate
                FROM request_logs
                WHERE endpoint = '/recommend'
            """
            params: list[Any] = []

            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())

            cursor = conn.execute(query, params)
            result = cursor.fetchone()

            return float(result[0]) if result and result[0] is not None else None


# =============================================================================
# Module-level convenience
# =============================================================================

_request_logger: Optional[RequestLogger] = None


def get_logger() -> RequestLogger:
    """Get the global request logger instance.

    Returns:
        RequestLogger instance.
    """
    global _request_logger
    if _request_logger is None:
        _request_logger = RequestLogger()
    return _request_logger


def generate_request_id() -> str:
    """Generate a unique request ID.

    Returns:
        UUID string.
    """
    return str(uuid.uuid4())


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 50)
    print("Testing Request Logger")
    print("=" * 50)

    # Create logger with test database
    test_db = Path("data/logs/test_requests.db")
    req_logger = RequestLogger(db_path=test_db)
    req_logger.start()

    try:
        # Log some test requests
        print("\n1. Logging test requests...")

        for i in range(5):
            req_logger.log_recommendation(
                request_id=generate_request_id(),
                user_id=100 + i,
                recommendations=[1, 2, 3, 4, 5],
                scores=[0.9, 0.8, 0.7, 0.6, 0.5],
                latency_ms=10.0 + i * 2,
                model_version="v_test",
                is_fallback=i % 2 == 0,
                fallback_reason="test" if i % 2 == 0 else None,
            )

        req_logger.log_similar(
            request_id=generate_request_id(),
            movie_id=1,
            similar_items=[2, 3, 4],
            scores=[0.95, 0.90, 0.85],
            latency_ms=5.0,
            model_version="v_test",
        )

        req_logger.log_popular(
            request_id=generate_request_id(),
            recommendations=[10, 20, 30],
            scores=[100, 90, 80],
            latency_ms=3.0,
            genre_filter="Action",
        )

        # Wait for async writes
        time.sleep(1)

        # Query logs
        print("\n2. Recent requests:")
        recent = req_logger.get_recent_requests(limit=3)
        for req in recent:
            print(
                f"   {req['endpoint']}: user={req['user_id']}, latency={req['latency_ms']}ms"
            )

        # Get stats
        print("\n3. Statistics:")
        print(f"   Total requests: {req_logger.get_request_count()}")
        print(
            f"   /recommend requests: {req_logger.get_request_count(endpoint='/recommend')}"
        )
        print(f"   Average latency: {req_logger.get_average_latency():.2f}ms")
        fallback_rate = req_logger.get_fallback_rate()
        if fallback_rate is not None:
            print(f"   Fallback rate: {fallback_rate:.1%}")

        print("\n✓ Request logger tests complete!")

    finally:
        req_logger.stop()

        # Cleanup test database
        if test_db.exists():
            test_db.unlink()
            print(f"\n   Cleaned up test database: {test_db}")
