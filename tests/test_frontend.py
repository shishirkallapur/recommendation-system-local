"""
Tests for the frontend components.

Tests the API client, data classes, and helper functions.
Uses mocking to avoid requiring a running API server.
"""

from unittest.mock import Mock, patch

import requests

from src.frontend.api_client import (
    APIClient,
    APIError,
    HealthStatus,
    Movie,
)

# =============================================================================
# Data Class Tests
# =============================================================================


class TestMovieDataclass:
    """Tests for the Movie dataclass."""

    def test_movie_creation(self):
        """Test creating a Movie with all fields."""
        movie = Movie(
            movie_id=1,
            title="Toy Story",
            score=0.95,
            genres=["Animation", "Children", "Comedy"],
            year=1995,
        )
        assert movie.movie_id == 1
        assert movie.title == "Toy Story"
        assert movie.score == 0.95
        assert movie.genres == ["Animation", "Children", "Comedy"]
        assert movie.year == 1995

    def test_movie_optional_fields(self):
        """Test creating a Movie with only required fields."""
        movie = Movie(movie_id=1, title="Test Movie", score=0.5)
        assert movie.movie_id == 1
        assert movie.title == "Test Movie"
        assert movie.score == 0.5
        assert movie.genres is None
        assert movie.year is None


class TestHealthStatusDataclass:
    """Tests for the HealthStatus dataclass."""

    def test_health_status_creation(self):
        """Test creating a HealthStatus with all fields."""
        health = HealthStatus(
            status="healthy",
            model_version="v20240115",
            model_loaded_at="2024-01-15T10:00:00Z",
            uptime_seconds=3600.0,
            n_users=943,
            n_items=1682,
        )
        assert health.status == "healthy"
        assert health.model_version == "v20240115"
        assert health.n_users == 943
        assert health.n_items == 1682

    def test_health_status_defaults(self):
        """Test HealthStatus with default values."""
        health = HealthStatus(status="healthy")
        assert health.status == "healthy"
        assert health.model_version is None
        assert health.uptime_seconds == 0.0


class TestAPIErrorDataclass:
    """Tests for the APIError dataclass."""

    def test_api_error_creation(self):
        """Test creating an APIError."""
        error = APIError(
            error="not_found",
            message="User not found",
            details={"user_id": 999},
        )
        assert error.error == "not_found"
        assert error.message == "User not found"
        assert error.details == {"user_id": 999}


# =============================================================================
# API Client Tests
# =============================================================================


class TestAPIClientInit:
    """Tests for APIClient initialization."""

    def test_default_url(self):
        """Test default base URL."""
        client = APIClient()
        assert client.base_url == "http://localhost:8000"

    def test_custom_url(self):
        """Test custom base URL."""
        client = APIClient(base_url="http://api.example.com:9000")
        assert client.base_url == "http://api.example.com:9000"

    def test_url_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from URL."""
        client = APIClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_custom_timeout(self):
        """Test custom timeout setting."""
        client = APIClient(timeout=30.0)
        assert client.timeout == 30.0


class TestAPIClientHealthCheck:
    """Tests for the health_check method."""

    @patch.object(requests.Session, "request")
    def test_health_check_success(self, mock_request):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "model_version": "v20240115",
            "model_loaded_at": "2024-01-15T10:00:00Z",
            "uptime_seconds": 3600.0,
            "n_users": 943,
            "n_items": 1682,
        }
        mock_request.return_value = mock_response

        client = APIClient()
        health, error = client.health_check()

        assert error is None
        assert health is not None
        assert health.status == "healthy"
        assert health.model_version == "v20240115"
        assert health.n_users == 943

    @patch.object(requests.Session, "request")
    def test_health_check_connection_error(self, mock_request):
        """Test health check when API is unreachable."""
        mock_request.side_effect = requests.exceptions.ConnectionError()

        client = APIClient()
        health, error = client.health_check()

        assert health is None
        assert error is not None
        assert error.error == "connection_error"

    @patch.object(requests.Session, "request")
    def test_health_check_timeout(self, mock_request):
        """Test health check timeout."""
        mock_request.side_effect = requests.exceptions.Timeout()

        client = APIClient()
        health, error = client.health_check()

        assert health is None
        assert error is not None
        assert error.error == "timeout"


class TestAPIClientRecommendations:
    """Tests for the get_recommendations method."""

    @patch.object(requests.Session, "request")
    def test_get_recommendations_success(self, mock_request):
        """Test successful recommendations request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": 42,
            "recommendations": [
                {
                    "movie_id": 1,
                    "title": "Toy Story",
                    "score": 0.95,
                    "year": 1995,
                    "genres": ["Animation"],
                },
                {
                    "movie_id": 2,
                    "title": "GoldenEye",
                    "score": 0.88,
                    "year": 1995,
                    "genres": ["Action"],
                },
            ],
            "model_version": "v20240115",
            "is_fallback": False,
        }
        mock_request.return_value = mock_response

        client = APIClient()
        result, error = client.get_recommendations(user_id=42, k=10)

        assert error is None
        assert result is not None
        assert result.user_id == 42
        assert len(result.recommendations) == 2
        assert result.recommendations[0].title == "Toy Story"
        assert result.is_fallback is False

    @patch.object(requests.Session, "request")
    def test_get_recommendations_fallback(self, mock_request):
        """Test recommendations with fallback for unknown user."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": 99999,
            "recommendations": [
                {
                    "movie_id": 50,
                    "title": "Star Wars",
                    "score": 583,
                    "year": 1977,
                    "genres": ["Sci-Fi"],
                },
            ],
            "model_version": "v20240115",
            "is_fallback": True,
            "fallback_reason": "popularity",
        }
        mock_request.return_value = mock_response

        client = APIClient()
        result, error = client.get_recommendations(user_id=99999, k=10)

        assert error is None
        assert result is not None
        assert result.is_fallback is True
        assert result.fallback_reason == "popularity"

    @patch.object(requests.Session, "request")
    def test_get_recommendations_with_filters(self, mock_request):
        """Test recommendations with genre and year filters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": 42,
            "recommendations": [],
            "model_version": "v20240115",
            "is_fallback": False,
        }
        mock_request.return_value = mock_response

        client = APIClient()
        result, error = client.get_recommendations(
            user_id=42,
            k=10,
            genres=["Action"],
            year_min=1990,
            year_max=2000,
        )

        # Verify the request was made with correct body
        call_args = mock_request.call_args
        request_body = call_args.kwargs.get("json", {})
        assert request_body["user_id"] == 42
        assert request_body["filters"]["genres"] == ["Action"]
        assert request_body["filters"]["year_min"] == 1990


class TestAPIClientSimilar:
    """Tests for the get_similar_movies method."""

    @patch.object(requests.Session, "request")
    def test_get_similar_success(self, mock_request):
        """Test successful similar movies request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "movie_id": 1,
            "title": "Toy Story",
            "similar_items": [
                {
                    "movie_id": 2,
                    "title": "A Bug's Life",
                    "similarity_score": 0.89,
                    "year": 1998,
                },
                {
                    "movie_id": 3,
                    "title": "Antz",
                    "similarity_score": 0.75,
                    "year": 1998,
                },
            ],
            "model_version": "v20240115",
        }
        mock_request.return_value = mock_response

        client = APIClient()
        result, error = client.get_similar_movies(movie_id=1, k=10)

        assert error is None
        assert result is not None
        assert result.movie_id == 1
        assert result.title == "Toy Story"
        assert len(result.similar_items) == 2
        assert result.similar_items[0].score == 0.89

    @patch.object(requests.Session, "request")
    def test_get_similar_not_found(self, mock_request):
        """Test similar movies for unknown movie."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": "not_found",
            "message": "Movie not found",
        }
        mock_request.return_value = mock_response

        client = APIClient()
        result, error = client.get_similar_movies(movie_id=99999, k=10)

        assert result is None
        assert error is not None
        assert error.error == "not_found"


class TestAPIClientPopular:
    """Tests for the get_popular_movies method."""

    @patch.object(requests.Session, "request")
    def test_get_popular_success(self, mock_request):
        """Test successful popular movies request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "recommendations": [
                {"movie_id": 50, "title": "Star Wars", "popularity_score": 583},
                {"movie_id": 100, "title": "Fargo", "popularity_score": 508},
            ],
            "source": "popularity",
        }
        mock_request.return_value = mock_response

        client = APIClient()
        result, error = client.get_popular_movies(k=10)

        assert error is None
        assert result is not None
        assert len(result.recommendations) == 2
        assert result.recommendations[0].title == "Star Wars"
        assert result.source == "popularity"

    @patch.object(requests.Session, "request")
    def test_get_popular_with_genre(self, mock_request):
        """Test popular movies with genre filter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "recommendations": [],
            "source": "popularity:genre=Horror",
        }
        mock_request.return_value = mock_response

        client = APIClient()
        result, error = client.get_popular_movies(k=10, genre="Horror")

        # Verify the request was made with correct params
        call_args = mock_request.call_args
        request_params = call_args.kwargs.get("params", {})
        assert request_params["genre"] == "Horror"


class TestAPIClientIsAvailable:
    """Tests for the is_available method."""

    @patch.object(requests.Session, "request")
    def test_is_available_true(self, mock_request):
        """Test is_available returns True when API is up."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_request.return_value = mock_response

        client = APIClient()
        assert client.is_available() is True

    @patch.object(requests.Session, "request")
    def test_is_available_false(self, mock_request):
        """Test is_available returns False when API is down."""
        mock_request.side_effect = requests.exceptions.ConnectionError()

        client = APIClient()
        assert client.is_available() is False
