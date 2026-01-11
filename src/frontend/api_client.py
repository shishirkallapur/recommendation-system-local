"""
API client for communicating with the recommendation service.

This module provides a clean interface for the Streamlit frontend to
interact with the FastAPI backend. It handles:
- HTTP requests with proper error handling
- Response parsing and validation
- Timeout management
- Connection error recovery

Usage:
    from src.frontend.api_client import APIClient

    client = APIClient(base_url="http://localhost:8000")

    # Check API health
    health = client.health_check()

    # Get personalized recommendations
    recs = client.get_recommendations(user_id=42, k=10)

    # Get similar movies
    similar = client.get_similar_movies(movie_id=1, k=10)

    # Get popular movies
    popular = client.get_popular_movies(k=20, genre="Action")
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

logger = logging.getLogger(__name__)


# =============================================================================
# Response Data Classes
# =============================================================================
# These mirror the API schemas but as simple dataclasses for frontend use


@dataclass
class Movie:
    """Represents a movie with recommendation/similarity score."""

    movie_id: int
    title: str
    score: float
    genres: Optional[list[str]] = None
    year: Optional[int] = None


@dataclass
class HealthStatus:
    """API health check response."""

    status: str
    model_version: Optional[str] = None
    model_loaded_at: Optional[str] = None
    uptime_seconds: float = 0.0
    n_users: Optional[int] = None
    n_items: Optional[int] = None


@dataclass
class RecommendationResult:
    """Result from /recommend endpoint."""

    user_id: int
    recommendations: list[Movie]
    model_version: str
    is_fallback: bool = False
    fallback_reason: Optional[str] = None


@dataclass
class SimilarityResult:
    """Result from /similar endpoint."""

    movie_id: int
    title: str
    similar_items: list[Movie]
    model_version: str


@dataclass
class PopularResult:
    """Result from /popular endpoint."""

    recommendations: list[Movie]
    source: str = "popularity"


@dataclass
class APIError:
    """Represents an API error."""

    error: str
    message: str
    details: Optional[dict[str, Any]] = None


# =============================================================================
# API Client
# =============================================================================


class APIClient:
    """
    Client for the movie recommendation API.

    Handles all communication with the FastAPI backend, including
    error handling and response parsing.

    Attributes:
        base_url: Base URL of the API (default: http://localhost:8000)
        timeout: Request timeout in seconds (default: 10)
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 10.0):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the recommendation API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def _handle_response(
        self, response: requests.Response
    ) -> tuple[Optional[dict[str, Any]], Optional[APIError]]:
        """
        Handle API response and extract JSON or error.

        Args:
            response: The requests Response object

        Returns:
            Tuple of (data, error) - one will be None
        """
        try:
            data = response.json()
        except ValueError:
            return None, APIError(
                error="parse_error", message="Failed to parse API response as JSON"
            )

        if response.status_code >= 400:
            return None, APIError(
                error=data.get("error", "unknown_error"),
                message=data.get("message", f"HTTP {response.status_code}"),
                details=data.get("details"),
            )

        return data, None

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[dict[str, Any]], Optional[APIError]]:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/health")
            json: JSON body for POST requests
            params: Query parameters for GET requests

        Returns:
            Tuple of (data, error) - one will be None
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self._session.request(
                method=method, url=url, json=json, params=params, timeout=self.timeout
            )
            return self._handle_response(response)

        except ConnectionError:
            logger.error(f"Connection error: Could not connect to {self.base_url}")
            return None, APIError(
                error="connection_error",
                message=f"Could not connect to API at {self.base_url}. Is the server running?",
            )
        except Timeout:
            logger.error(f"Timeout: Request to {url} timed out after {self.timeout}s")
            return None, APIError(
                error="timeout",
                message=f"Request timed out after {self.timeout} seconds",
            )
        except RequestException as e:
            logger.error(f"Request error: {e}")
            return None, APIError(error="request_error", message=str(e))

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def health_check(self) -> tuple[Optional[HealthStatus], Optional[APIError]]:
        """
        Check API health status.

        Returns:
            Tuple of (HealthStatus, None) on success, or (None, APIError) on failure

        Example:
            health, error = client.health_check()
            if error:
                print(f"API is down: {error.message}")
            else:
                print(f"API status: {health.status}")
        """
        data, error = self._make_request("GET", "/health")

        if error or data is None:
            return None, error

        return (
            HealthStatus(
                status=data.get("status", "unknown"),
                model_version=data.get("model_version"),
                model_loaded_at=data.get("model_loaded_at"),
                uptime_seconds=data.get("uptime_seconds", 0.0),
                n_users=data.get("n_users"),
                n_items=data.get("n_items"),
            ),
            None,
        )

    def get_recommendations(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
        genres: Optional[list[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> tuple[Optional[RecommendationResult], Optional[APIError]]:
        """
        Get personalized recommendations for a user.

        Args:
            user_id: The user ID to get recommendations for
            k: Number of recommendations to return (1-100)
            exclude_seen: Whether to exclude already-seen items
            genres: Optional list of genres to filter by
            year_min: Optional minimum release year
            year_max: Optional maximum release year

        Returns:
            Tuple of (RecommendationResult, None) on success,
            or (None, APIError) on failure

        Example:
            recs, error = client.get_recommendations(user_id=42, k=10)
            if error:
                print(f"Error: {error.message}")
            else:
                for movie in recs.recommendations:
                    print(f"{movie.title}: {movie.score:.3f}")
        """
        # Build request body
        body: dict[str, Any] = {
            "user_id": user_id,
            "k": k,
            "exclude_seen": exclude_seen,
        }

        # Add filters if any are specified
        filters: dict[str, Any] = {}
        if genres:
            filters["genres"] = genres
        if year_min is not None:
            filters["year_min"] = year_min
        if year_max is not None:
            filters["year_max"] = year_max
        if filters:
            body["filters"] = filters

        data, error = self._make_request("POST", "/recommend", json=body)

        if error or data is None:
            return None, error

        # Parse recommendations
        recommendations = [
            Movie(
                movie_id=rec["movie_id"],
                title=rec["title"],
                score=rec["score"],
                genres=rec.get("genres"),
                year=rec.get("year"),
            )
            for rec in data.get("recommendations", [])
        ]

        return (
            RecommendationResult(
                user_id=data["user_id"],
                recommendations=recommendations,
                model_version=data["model_version"],
                is_fallback=data.get("is_fallback", False),
                fallback_reason=data.get("fallback_reason"),
            ),
            None,
        )

    def get_similar_movies(
        self, movie_id: int, k: int = 10
    ) -> tuple[Optional[SimilarityResult], Optional[APIError]]:
        """
        Get movies similar to a given movie.

        Args:
            movie_id: The seed movie ID
            k: Number of similar movies to return (1-100)

        Returns:
            Tuple of (SimilarityResult, None) on success,
            or (None, APIError) on failure

        Example:
            similar, error = client.get_similar_movies(movie_id=1, k=10)
            if error:
                print(f"Error: {error.message}")
            else:
                print(f"Movies similar to {similar.title}:")
                for movie in similar.similar_items:
                    print(f"  {movie.title}: {movie.score:.3f}")
        """
        body: dict[str, Any] = {"movie_id": movie_id, "k": k}

        data, error = self._make_request("POST", "/similar", json=body)

        if error or data is None:
            return None, error

        # Parse similar items
        similar_items = [
            Movie(
                movie_id=item["movie_id"],
                title=item["title"],
                score=item["similarity_score"],
                genres=item.get("genres"),
                year=item.get("year"),
            )
            for item in data.get("similar_items", [])
        ]

        return (
            SimilarityResult(
                movie_id=data["movie_id"],
                title=data["title"],
                similar_items=similar_items,
                model_version=data["model_version"],
            ),
            None,
        )

    def get_popular_movies(
        self, k: int = 10, genre: Optional[str] = None
    ) -> tuple[Optional[PopularResult], Optional[APIError]]:
        """
        Get popular movies (non-personalized).

        Args:
            k: Number of popular movies to return (1-100)
            genre: Optional genre to filter by

        Returns:
            Tuple of (PopularResult, None) on success,
            or (None, APIError) on failure

        Example:
            popular, error = client.get_popular_movies(k=20, genre="Action")
            if error:
                print(f"Error: {error.message}")
            else:
                for movie in popular.recommendations:
                    print(f"{movie.title}: {movie.score:.1f}")
        """
        params: dict[str, Any] = {"k": k}
        if genre:
            params["genre"] = genre

        data, error = self._make_request("GET", "/popular", params=params)

        if error or data is None:
            return None, error

        # Parse popular movies
        recommendations = [
            Movie(
                movie_id=rec["movie_id"],
                title=rec["title"],
                score=rec["popularity_score"],
                genres=rec.get("genres"),
                year=rec.get("year"),
            )
            for rec in data.get("recommendations", [])
        ]

        return (
            PopularResult(
                recommendations=recommendations, source=data.get("source", "popularity")
            ),
            None,
        )

    def is_available(self) -> bool:
        """
        Quick check if the API is reachable.

        Returns:
            True if API responds to health check, False otherwise

        Example:
            if client.is_available():
                print("API is up!")
            else:
                print("API is down")
        """
        health, error = self.health_check()
        return error is None and health is not None

    def get_all_movies(self) -> tuple[Optional[list[Movie]], Optional[APIError]]:
        """
        Get a list of all available movies (via popular endpoint with high k).

        This is useful for populating dropdowns in the UI.
        Note: Returns up to 100 movies sorted by popularity.

        Returns:
            Tuple of (list[Movie], None) on success,
            or (None, APIError) on failure
        """
        result, error = self.get_popular_movies(k=100)
        if error or result is None:
            return None, error
        return result.recommendations, None
