"""
API tests for the movie recommendation service.

These tests verify all API endpoints work correctly:
- GET /health
- POST /recommend
- POST /similar
- GET /popular

Tests use FastAPI's TestClient for synchronous testing
without needing to run a real server.

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Import after setting up test fixtures to ensure model is loaded
from src.api.main import app
from src.api.model_loader import is_model_loaded, load_model, model_store

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure model is loaded before running tests."""
    if not is_model_loaded():
        try:
            load_model()
        except FileNotFoundError:
            pytest.skip("Production model not found. Run training first.")


@pytest.fixture(scope="module")
def client():
    """Create a test client for the API."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def known_user_id() -> int:
    """Get a user ID that exists in the model."""
    if not model_store.user_to_idx:
        pytest.skip("No users in model")
    return list(model_store.user_to_idx.keys())[0]


@pytest.fixture
def known_movie_id() -> int:
    """Get a movie ID that exists in the model."""
    if not model_store.item_to_idx:
        pytest.skip("No items in model")
    return list(model_store.item_to_idx.keys())[0]


@pytest.fixture
def unknown_user_id() -> int:
    """Get a user ID that doesn't exist in the model."""
    return 9999999


@pytest.fixture
def unknown_movie_id() -> int:
    """Get a movie ID that doesn't exist in the model."""
    return 9999999


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, client: TestClient):
        """Test that /health returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient):
        """Test that /health returns expected fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_version" in data
        assert "uptime_seconds" in data
        assert "n_users" in data
        assert "n_items" in data

    def test_health_status_healthy(self, client: TestClient):
        """Test that status is 'healthy' when model is loaded."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"

    def test_health_has_model_info(self, client: TestClient):
        """Test that health includes model information."""
        response = client.get("/health")
        data = response.json()

        assert data["n_users"] is not None
        assert data["n_users"] > 0
        assert data["n_items"] is not None
        assert data["n_items"] > 0


# =============================================================================
# Recommend Endpoint Tests
# =============================================================================


class TestRecommendEndpoint:
    """Tests for POST /recommend endpoint."""

    def test_recommend_known_user_returns_200(
        self, client: TestClient, known_user_id: int
    ):
        """Test that /recommend returns 200 for known user."""
        response = client.post("/recommend", json={"user_id": known_user_id, "k": 5})
        assert response.status_code == 200

    def test_recommend_returns_recommendations(
        self, client: TestClient, known_user_id: int
    ):
        """Test that /recommend returns a list of recommendations."""
        response = client.post("/recommend", json={"user_id": known_user_id, "k": 5})
        data = response.json()

        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) > 0

    def test_recommend_respects_k_parameter(
        self, client: TestClient, known_user_id: int
    ):
        """Test that /recommend returns at most k items."""
        k = 3
        response = client.post("/recommend", json={"user_id": known_user_id, "k": k})
        data = response.json()

        assert len(data["recommendations"]) <= k

    def test_recommend_response_structure(self, client: TestClient, known_user_id: int):
        """Test that recommendation items have expected fields."""
        response = client.post("/recommend", json={"user_id": known_user_id, "k": 5})
        data = response.json()

        assert "user_id" in data
        assert "recommendations" in data
        assert "model_version" in data
        assert "is_fallback" in data

        # Check recommendation item structure
        rec = data["recommendations"][0]
        assert "movie_id" in rec
        assert "title" in rec
        assert "score" in rec

    def test_recommend_known_user_not_fallback(
        self, client: TestClient, known_user_id: int
    ):
        """Test that known user doesn't trigger fallback."""
        response = client.post("/recommend", json={"user_id": known_user_id, "k": 5})
        data = response.json()

        assert data["is_fallback"] is False

    def test_recommend_unknown_user_triggers_fallback(
        self, client: TestClient, unknown_user_id: int
    ):
        """Test that unknown user triggers fallback."""
        response = client.post("/recommend", json={"user_id": unknown_user_id, "k": 5})
        data = response.json()

        assert response.status_code == 200
        assert data["is_fallback"] is True
        assert data["fallback_reason"] is not None

    def test_recommend_unknown_user_returns_results(
        self, client: TestClient, unknown_user_id: int
    ):
        """Test that unknown user still gets recommendations (via fallback)."""
        response = client.post("/recommend", json={"user_id": unknown_user_id, "k": 5})
        data = response.json()

        assert len(data["recommendations"]) > 0

    def test_recommend_with_genre_filter(self, client: TestClient, known_user_id: int):
        """Test that genre filter is applied."""
        response = client.post(
            "/recommend",
            json={
                "user_id": known_user_id,
                "k": 10,
                "filters": {"genres": ["Action"]},
            },
        )
        data = response.json()

        assert response.status_code == 200

        # All returned movies should have Action genre
        for rec in data["recommendations"]:
            if rec.get("genres"):  # Some might not have genre info
                assert "Action" in rec["genres"]

    def test_recommend_with_year_filter(self, client: TestClient, known_user_id: int):
        """Test that year filter is applied."""
        response = client.post(
            "/recommend",
            json={
                "user_id": known_user_id,
                "k": 10,
                "filters": {"year_min": 1990, "year_max": 1995},
            },
        )
        data = response.json()

        assert response.status_code == 200

        # All returned movies should be in year range
        for rec in data["recommendations"]:
            if rec.get("year"):  # Some might not have year info
                assert 1990 <= rec["year"] <= 1995

    def test_recommend_exclude_seen_default(
        self, client: TestClient, known_user_id: int
    ):
        """Test that exclude_seen defaults to True."""
        response = client.post("/recommend", json={"user_id": known_user_id, "k": 5})

        assert response.status_code == 200

    def test_recommend_invalid_k_too_low(self, client: TestClient, known_user_id: int):
        """Test that k < 1 returns validation error."""
        response = client.post("/recommend", json={"user_id": known_user_id, "k": 0})

        assert response.status_code == 422  # Validation error

    def test_recommend_invalid_k_too_high(self, client: TestClient, known_user_id: int):
        """Test that k > 100 returns validation error."""
        response = client.post("/recommend", json={"user_id": known_user_id, "k": 101})

        assert response.status_code == 422  # Validation error


# =============================================================================
# Similar Endpoint Tests
# =============================================================================


class TestSimilarEndpoint:
    """Tests for POST /similar endpoint."""

    def test_similar_valid_movie_returns_200(
        self, client: TestClient, known_movie_id: int
    ):
        """Test that /similar returns 200 for valid movie."""
        response = client.post("/similar", json={"movie_id": known_movie_id, "k": 5})
        assert response.status_code == 200

    def test_similar_returns_similar_items(
        self, client: TestClient, known_movie_id: int
    ):
        """Test that /similar returns a list of similar items."""
        response = client.post("/similar", json={"movie_id": known_movie_id, "k": 5})
        data = response.json()

        assert "similar_items" in data
        assert isinstance(data["similar_items"], list)
        assert len(data["similar_items"]) > 0

    def test_similar_respects_k_parameter(
        self, client: TestClient, known_movie_id: int
    ):
        """Test that /similar returns at most k items."""
        k = 3
        response = client.post("/similar", json={"movie_id": known_movie_id, "k": k})
        data = response.json()

        assert len(data["similar_items"]) <= k

    def test_similar_response_structure(self, client: TestClient, known_movie_id: int):
        """Test that /similar response has expected fields."""
        response = client.post("/similar", json={"movie_id": known_movie_id, "k": 5})
        data = response.json()

        assert "movie_id" in data
        assert "title" in data
        assert "similar_items" in data
        assert "model_version" in data

        # Check similar item structure
        item = data["similar_items"][0]
        assert "movie_id" in item
        assert "title" in item
        assert "similarity_score" in item

    def test_similar_scores_bounded(self, client: TestClient, known_movie_id: int):
        """Test that similarity scores are between 0 and 1."""
        response = client.post("/similar", json={"movie_id": known_movie_id, "k": 5})
        data = response.json()

        for item in data["similar_items"]:
            assert 0 <= item["similarity_score"] <= 1

    def test_similar_excludes_query_movie(
        self, client: TestClient, known_movie_id: int
    ):
        """Test that the query movie is not in results."""
        response = client.post("/similar", json={"movie_id": known_movie_id, "k": 10})
        data = response.json()

        result_ids = [item["movie_id"] for item in data["similar_items"]]
        assert known_movie_id not in result_ids

    def test_similar_invalid_movie_returns_404(
        self, client: TestClient, unknown_movie_id: int
    ):
        """Test that /similar returns 404 for unknown movie."""
        response = client.post("/similar", json={"movie_id": unknown_movie_id, "k": 5})

        assert response.status_code == 404

    def test_similar_invalid_movie_error_message(
        self, client: TestClient, unknown_movie_id: int
    ):
        """Test that 404 response has error details."""
        response = client.post("/similar", json={"movie_id": unknown_movie_id, "k": 5})
        data = response.json()

        assert "detail" in data


# =============================================================================
# Popular Endpoint Tests
# =============================================================================


class TestPopularEndpoint:
    """Tests for GET /popular endpoint."""

    def test_popular_returns_200(self, client: TestClient):
        """Test that /popular returns 200 OK."""
        response = client.get("/popular")
        assert response.status_code == 200

    def test_popular_returns_recommendations(self, client: TestClient):
        """Test that /popular returns a list of recommendations."""
        response = client.get("/popular")
        data = response.json()

        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) > 0

    def test_popular_respects_k_parameter(self, client: TestClient):
        """Test that /popular returns at most k items."""
        k = 3
        response = client.get("/popular", params={"k": k})
        data = response.json()

        assert len(data["recommendations"]) <= k

    def test_popular_response_structure(self, client: TestClient):
        """Test that /popular response has expected fields."""
        response = client.get("/popular")
        data = response.json()

        assert "recommendations" in data
        assert "source" in data

        # Check recommendation item structure
        rec = data["recommendations"][0]
        assert "movie_id" in rec
        assert "title" in rec
        assert "popularity_score" in rec

    def test_popular_sorted_by_popularity(self, client: TestClient):
        """Test that results are sorted by popularity descending."""
        response = client.get("/popular", params={"k": 10})
        data = response.json()

        scores = [rec["popularity_score"] for rec in data["recommendations"]]
        assert scores == sorted(scores, reverse=True)

    def test_popular_with_genre_filter(self, client: TestClient):
        """Test that genre filter is applied."""
        response = client.get("/popular", params={"k": 10, "genre": "Comedy"})
        data = response.json()

        assert response.status_code == 200

        # All returned movies should have Comedy genre
        for rec in data["recommendations"]:
            if rec.get("genres"):
                assert "Comedy" in rec["genres"]

    def test_popular_source_field(self, client: TestClient):
        """Test that source field indicates popularity-based."""
        response = client.get("/popular")
        data = response.json()

        assert "popularity" in data["source"]

    def test_popular_genre_filter_in_source(self, client: TestClient):
        """Test that genre filter is reflected in source field."""
        response = client.get("/popular", params={"genre": "Action"})
        data = response.json()

        assert "Action" in data["source"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_endpoint_returns_404(self, client: TestClient):
        """Test that unknown endpoints return 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_recommend_missing_user_id(self, client: TestClient):
        """Test that missing user_id returns validation error."""
        response = client.post("/recommend", json={"k": 5})
        assert response.status_code == 422

    def test_similar_missing_movie_id(self, client: TestClient):
        """Test that missing movie_id returns validation error."""
        response = client.post("/similar", json={"k": 5})
        assert response.status_code == 422

    def test_recommend_invalid_json(self, client: TestClient):
        """Test that invalid JSON returns error."""
        response = client.post(
            "/recommend",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# =============================================================================
# API Documentation Tests
# =============================================================================


class TestAPIDocs:
    """Tests for API documentation endpoints."""

    def test_openapi_json_accessible(self, client: TestClient):
        """Test that OpenAPI JSON is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_swagger_ui_accessible(self, client: TestClient):
        """Test that Swagger UI is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_accessible(self, client: TestClient):
        """Test that ReDoc is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
