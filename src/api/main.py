"""
FastAPI application for the movie recommendation service.

This module defines the REST API endpoints:
- GET /health: Service health check
- POST /recommend: Personalized recommendations
- POST /similar: Item similarity
- GET /popular: Popular items fallback

Usage:
    # Run with uvicorn
    uvicorn src.api.main:app --reload --port 8000

    # Or run directly
    python -m src.api.main
"""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.fallback import get_fallback_handler
from src.api.logger import generate_request_id, get_logger
from src.api.model_loader import load_model, model_store
from src.api.recommender import get_engine
from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    MovieRecommendation,
    NotFoundError,
    PopularMovie,
    PopularResponse,
    RecommendRequest,
    RecommendResponse,
    SimilarMovie,
    SimilarRequest,
    SimilarResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Track startup time for uptime calculation
_startup_time: Optional[datetime] = None


# =============================================================================
# Application Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    On startup:
    - Load the production model
    - Start the request logger

    On shutdown:
    - Stop the request logger (flush pending writes)
    """
    global _startup_time

    logger.info("Starting recommendation service...")

    # Load model
    try:
        load_model()
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Service will start but recommendations won't work")

    # Start request logger
    request_logger = get_logger()
    request_logger.start()
    logger.info("Request logger started")

    # Record startup time
    _startup_time = datetime.now(timezone.utc)

    logger.info("Recommendation service ready!")

    yield

    # Shutdown
    logger.info("Shutting down recommendation service...")
    request_logger.stop()
    logger.info("Request logger stopped")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Movie Recommendation API",
    description="""
    A production-style movie recommendation service.

    ## Endpoints

    - **GET /health**: Check service health and model status
    - **POST /recommend**: Get personalized recommendations for a user
    - **POST /similar**: Find movies similar to a given movie
    - **GET /popular**: Get popular movies (non-personalized fallback)

    ## Cold Start Handling

    For unknown users, the service falls back to:
    1. Similar items (if a seed movie is provided)
    2. Popular items (default fallback)
    """,
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(ValueError)
async def value_error_handler(
    request, exc: ValueError
) -> JSONResponse:  # noqa: ANN001, ARG001
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content={"error": "bad_request", "message": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(
    request, exc: RuntimeError
) -> JSONResponse:  # noqa: ANN001, ARG001
    """Handle RuntimeError exceptions (e.g., model not loaded)."""
    return JSONResponse(
        status_code=503,
        content={"error": "service_unavailable", "message": str(exc)},
    )


# =============================================================================
# Health Endpoint
# =============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check service health and model status.",
)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns service status, model version, and uptime.
    """
    uptime = 0.0
    if _startup_time:
        uptime = (datetime.now(timezone.utc) - _startup_time).total_seconds()

    return HealthResponse(
        status="healthy" if model_store.is_loaded else "degraded",
        model_version=model_store.version,
        model_loaded_at=model_store.loaded_at,
        uptime_seconds=uptime,
        n_users=model_store.n_users if model_store.is_loaded else None,
        n_items=model_store.n_items if model_store.is_loaded else None,
    )


# =============================================================================
# Recommend Endpoint
# =============================================================================


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["Recommendations"],
    summary="Get personalized recommendations",
    description="""
    Get personalized movie recommendations for a user.

    For known users, recommendations are based on collaborative filtering.
    For unknown users, popular items are returned as fallback.

    **Filters:**
    - `genres`: Only include movies with at least one of these genres
    - `year_min`/`year_max`: Filter by release year range
    """,
)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Get personalized recommendations for a user."""
    start_time = time.perf_counter()
    request_id = generate_request_id()

    engine = get_engine()
    fallback_handler = get_fallback_handler()
    request_logger = get_logger()

    is_fallback = False
    fallback_reason: Optional[str] = None
    recommendations: list[MovieRecommendation] = []

    try:
        # Check if user is known
        if not fallback_handler.is_user_known(request.user_id):
            # Use fallback for unknown user
            is_fallback = True
            fallback_reason = "unknown_user"

            popular_items, reason = fallback_handler.get_fallback_for_user(
                user_id=request.user_id,
                k=request.k,
                genre=(
                    request.filters.genres[0]
                    if request.filters and request.filters.genres
                    else None
                ),
            )
            fallback_reason = reason

            recommendations = [
                MovieRecommendation(
                    movie_id=item.movie_id,
                    title=item.title,
                    score=item.popularity_score,
                    genres=item.genres,
                    year=item.year,
                )
                for item in popular_items
            ]
        else:
            # Use main recommendation engine
            # Extract filters
            filter_genres = None
            year_min = None
            year_max = None

            if request.filters:
                filter_genres = request.filters.genres
                year_min = request.filters.year_min
                year_max = request.filters.year_max

            recs, is_fallback, fallback_reason = engine.recommend(
                user_id=request.user_id,
                k=request.k,
                exclude_seen=request.exclude_seen,
                filter_genres=filter_genres,
                year_min=year_min,
                year_max=year_max,
            )

            recommendations = [
                MovieRecommendation(
                    movie_id=rec.movie_id,
                    title=rec.title,
                    score=rec.score,
                    genres=rec.genres,
                    year=rec.year,
                )
                for rec in recs
            ]

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Log request (async)
        request_logger.log_recommendation(
            request_id=request_id,
            user_id=request.user_id,
            recommendations=[r.movie_id for r in recommendations],
            scores=[r.score for r in recommendations],
            latency_ms=latency_ms,
            model_version=model_store.version,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
        )

        return RecommendResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_version=model_store.version or "unknown",
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
        )

    except Exception as e:
        logger.error(f"Error in /recommend: {e}")
        raise


# =============================================================================
# Similar Endpoint
# =============================================================================


@app.post(
    "/similar",
    response_model=SimilarResponse,
    responses={
        404: {"model": NotFoundError, "description": "Movie not found"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["Recommendations"],
    summary="Find similar movies",
    description="""
    Find movies similar to a given movie.

    Uses item-item similarity based on collaborative filtering embeddings.
    Useful for "Because you watched X" recommendations.
    """,
)
async def similar(request: SimilarRequest) -> SimilarResponse:
    """Find movies similar to a given movie."""
    start_time = time.perf_counter()
    request_id = generate_request_id()

    engine = get_engine()
    request_logger = get_logger()

    # Check if movie exists
    if not engine.check_item_exists(request.movie_id):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "message": f"Movie {request.movie_id} not found in model",
            },
        )

    # Get movie info for response
    movie_info = engine.get_item_info(request.movie_id)
    movie_title = movie_info.title if movie_info else f"Movie {request.movie_id}"

    # Get similar items
    similar_items = engine.similar_items(
        movie_id=request.movie_id,
        k=request.k,
    )

    # Build response
    similar_movies = [
        SimilarMovie(
            movie_id=item.movie_id,
            title=item.title,
            similarity_score=item.similarity_score,
            genres=item.genres,
            year=item.year,
        )
        for item in similar_items
    ]

    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Log request (async)
    request_logger.log_similar(
        request_id=request_id,
        movie_id=request.movie_id,
        similar_items=[m.movie_id for m in similar_movies],
        scores=[m.similarity_score for m in similar_movies],
        latency_ms=latency_ms,
        model_version=model_store.version,
    )

    return SimilarResponse(
        movie_id=request.movie_id,
        title=movie_title,
        similar_items=similar_movies,
        model_version=model_store.version or "unknown",
    )


# =============================================================================
# Popular Endpoint
# =============================================================================


@app.get(
    "/popular",
    response_model=PopularResponse,
    tags=["Recommendations"],
    summary="Get popular movies",
    description="""
    Get popular movies based on interaction count.

    This is a non-personalized endpoint useful for:
    - Homepage recommendations before user logs in
    - Fallback when personalized recommendations aren't available
    """,
)
async def popular(
    k: int = Query(default=10, ge=1, le=100, description="Number of items"),
    genre: Optional[str] = Query(default=None, description="Filter by genre"),
) -> PopularResponse:
    """Get popular movies."""
    start_time = time.perf_counter()
    request_id = generate_request_id()

    fallback_handler = get_fallback_handler()
    request_logger = get_logger()

    # Get popular items
    popular_items = fallback_handler.get_popular(k=k, genre=genre)

    # Build response
    recommendations = [
        PopularMovie(
            movie_id=item.movie_id,
            title=item.title,
            popularity_score=item.popularity_score,
            genres=item.genres,
            year=item.year,
        )
        for item in popular_items
    ]

    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Log request (async)
    request_logger.log_popular(
        request_id=request_id,
        recommendations=[r.movie_id for r in recommendations],
        scores=[r.popularity_score for r in recommendations],
        latency_ms=latency_ms,
        genre_filter=genre,
    )

    return PopularResponse(
        recommendations=recommendations,
        source=f"popularity:genre={genre}" if genre else "popularity",
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
