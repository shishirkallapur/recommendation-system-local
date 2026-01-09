"""
Pydantic schemas for API request/response validation.

These schemas define the contract between the API and its clients:
- Request validation with automatic error messages
- Response serialization with consistent structure
- OpenAPI documentation generation

Usage:
    from src.api.schemas import RecommendRequest, RecommendResponse

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(request: RecommendRequest):
        ...
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# =============================================================================
# Shared Models
# =============================================================================


class MovieRecommendation(BaseModel):
    """A single movie recommendation."""

    movie_id: int = Field(..., description="Unique movie identifier")
    title: str = Field(..., description="Movie title")
    score: float = Field(..., description="Recommendation score (higher is better)")
    genres: Optional[list[str]] = Field(default=None, description="Movie genres")
    year: Optional[int] = Field(default=None, description="Release year")


class SimilarMovie(BaseModel):
    """A similar movie result."""

    movie_id: int = Field(..., description="Unique movie identifier")
    title: str = Field(..., description="Movie title")
    similarity_score: float = Field(
        ..., ge=0, le=1, description="Similarity score (0-1)"
    )
    genres: Optional[list[str]] = Field(default=None, description="Movie genres")
    year: Optional[int] = Field(default=None, description="Release year")


class PopularMovie(BaseModel):
    """A popular movie result."""

    movie_id: int = Field(..., description="Unique movie identifier")
    title: str = Field(..., description="Movie title")
    popularity_score: float = Field(..., description="Popularity score")
    genres: Optional[list[str]] = Field(default=None, description="Movie genres")
    year: Optional[int] = Field(default=None, description="Release year")


# =============================================================================
# Request Schemas
# =============================================================================


class RecommendFilters(BaseModel):
    """Optional filters for recommendations."""

    genres: Optional[list[str]] = Field(
        default=None, description="Filter by genres (include only these)"
    )
    year_min: Optional[int] = Field(
        default=None, ge=1900, le=2030, description="Minimum release year"
    )
    year_max: Optional[int] = Field(
        default=None, ge=1900, le=2030, description="Maximum release year"
    )


class RecommendRequest(BaseModel):
    """Request body for /recommend endpoint."""

    user_id: int = Field(..., description="User identifier")
    k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    exclude_seen: bool = Field(
        default=True, description="Exclude items user has already interacted with"
    )
    filters: Optional[RecommendFilters] = Field(
        default=None, description="Optional filters"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": 196,
                    "k": 10,
                    "exclude_seen": True,
                    "filters": {"genres": ["Action", "Sci-Fi"], "year_min": 1990},
                }
            ]
        }
    }


class SimilarRequest(BaseModel):
    """Request body for /similar endpoint."""

    movie_id: int = Field(..., description="Seed movie identifier")
    k: int = Field(default=10, ge=1, le=100, description="Number of similar items")

    model_config = {"json_schema_extra": {"examples": [{"movie_id": 1, "k": 10}]}}


# =============================================================================
# Response Schemas
# =============================================================================


class RecommendResponse(BaseModel):
    """Response body for /recommend endpoint."""

    user_id: int = Field(..., description="Requested user ID")
    recommendations: list[MovieRecommendation] = Field(
        ..., description="List of recommended movies"
    )
    model_version: str = Field(..., description="Model version used")
    is_fallback: bool = Field(
        default=False, description="Whether fallback logic was used"
    )
    fallback_reason: Optional[str] = Field(
        default=None, description="Reason for fallback (if applicable)"
    )


class SimilarResponse(BaseModel):
    """Response body for /similar endpoint."""

    movie_id: int = Field(..., description="Seed movie ID")
    title: str = Field(..., description="Seed movie title")
    similar_items: list[SimilarMovie] = Field(..., description="List of similar movies")
    model_version: str = Field(..., description="Model version used")


class PopularResponse(BaseModel):
    """Response body for /popular endpoint."""

    recommendations: list[PopularMovie] = Field(
        ..., description="List of popular movies"
    )
    source: str = Field(default="popularity", description="Source of recommendations")


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str = Field(..., description="Service status")
    model_version: Optional[str] = Field(
        default=None, description="Loaded model version"
    )
    model_loaded_at: Optional[datetime] = Field(
        default=None, description="When model was loaded"
    )
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    n_users: Optional[int] = Field(default=None, description="Number of users in model")
    n_items: Optional[int] = Field(default=None, description="Number of items in model")


# =============================================================================
# Error Schemas
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(default=None, description="Additional details")


class NotFoundError(BaseModel):
    """404 Not Found error response."""

    error: str = Field(default="not_found", description="Error type")
    message: str = Field(..., description="What was not found")
