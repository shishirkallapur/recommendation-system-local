"""
Personalized recommendations page.

This page allows known users to get personalized movie recommendations
based on their viewing history. For unknown users, it gracefully falls
back to popular movies with a clear explanation.

Features:
- User ID input with validation
- Configurable number of recommendations
- Optional genre and year filters
- Clear fallback indication for unknown users
- Movie cards with details (title, year, genres, score)
"""

from typing import Optional, cast

import streamlit as st

from src.frontend.api_client import APIClient, Movie

# =============================================================================
# Configuration
# =============================================================================

# Default API URL - can be overridden via environment or session state
DEFAULT_API_URL = "http://localhost:8000"


def get_client() -> APIClient:
    """Get or create the API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(base_url=DEFAULT_API_URL)
    return cast(APIClient, st.session_state.api_client)


# =============================================================================
# UI Components
# =============================================================================


def render_movie_card(movie: Movie, rank: int) -> None:
    """Render a single movie as a card.

    Args:
        movie: Movie object with details
        rank: Display rank (1-indexed)
    """
    # Format genres
    genres_str = ", ".join(movie.genres) if movie.genres else "Unknown"

    # Format year
    year_str = f"({movie.year})" if movie.year else ""

    # Format score
    score_str = f"{movie.score:.4f}"

    # Create card using columns
    with st.container():
        col1, col2 = st.columns([1, 11])

        with col1:
            st.markdown(
                f"<div style='font-size: 24px; font-weight: bold; "
                f"color: #888; text-align: center;'>{rank}</div>",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(f"**{movie.title}** {year_str}")
            st.caption(f"🎬 {genres_str} · Score: {score_str}")

        st.divider()


def render_movie_list(movies: list[Movie], start_rank: int = 1) -> None:
    """Render a list of movies.

    Args:
        movies: List of Movie objects
        start_rank: Starting rank number
    """
    if not movies:
        st.info("No movies to display.")
        return

    for i, movie in enumerate(movies):
        render_movie_card(movie, rank=start_rank + i)


def render_fallback_notice(reason: Optional[str]) -> None:
    """Render a notice explaining why fallback was used.

    Args:
        reason: Fallback reason string from API
    """
    if not reason:
        return

    if reason == "popularity":
        st.info(
            "👤 **New User Detected**\n\n"
            "We don't have enough information about your preferences yet. "
            "Here are our most popular movies to get you started!"
        )
    elif reason.startswith("popularity:genre="):
        genre = reason.split("=")[1]
        st.info(
            f"👤 **New User Detected**\n\n"
            f"We don't have your preferences yet, but here are popular "
            f"**{genre}** movies you might enjoy!"
        )
    elif reason.startswith("similar_to_seed:"):
        movie_id = reason.split(":")[1]
        st.info(
            f"👤 **New User Detected**\n\n"
            f"Based on movie #{movie_id} you selected, here are similar movies!"
        )
    else:
        st.info(f"ℹ️ Showing fallback recommendations: {reason}")


# =============================================================================
# Main Page
# =============================================================================


def render_page() -> None:
    """Render the personalized recommendations page."""
    st.title("🎬 Personalized Recommendations")
    st.markdown(
        "Get movie recommendations tailored to your taste. "
        "Enter your user ID to see what we think you'll love!"
    )

    client = get_client()

    # Check API availability
    if not client.is_available():
        st.error(
            "⚠️ **API Unavailable**\n\n"
            "The recommendation service is not responding. "
            "Please ensure the API is running at "
            f"`{client.base_url}`"
        )
        st.code("# Start the API with:\npython -m src.api.main", language="bash")
        return

    st.success("✓ API Connected", icon="🟢")

    # Input form
    st.subheader("Get Your Recommendations")

    with st.form("recommend_form"):
        col1, col2 = st.columns([2, 1])

        with col1:
            user_id = st.number_input(
                "User ID",
                min_value=1,
                value=196,
                step=1,
                help="Enter your user ID to get personalized recommendations",
            )

        with col2:
            k = st.slider(
                "Number of movies",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="How many recommendations to show",
            )

        # Advanced filters in expander
        with st.expander("🔧 Advanced Filters", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)

            with filter_col1:
                # Genre filter - common genres
                genre_options = [
                    "",
                    "Action",
                    "Adventure",
                    "Animation",
                    "Children's",
                    "Comedy",
                    "Crime",
                    "Documentary",
                    "Drama",
                    "Fantasy",
                    "Film-Noir",
                    "Horror",
                    "Musical",
                    "Mystery",
                    "Romance",
                    "Sci-Fi",
                    "Thriller",
                    "War",
                    "Western",
                ]
                selected_genre = st.selectbox(
                    "Filter by Genre",
                    options=genre_options,
                    index=0,
                    help="Only show movies with this genre",
                )

            with filter_col2:
                year_min = st.number_input(
                    "Min Year",
                    min_value=1900,
                    max_value=2030,
                    value=1900,
                    help="Minimum release year",
                )

            with filter_col3:
                year_max = st.number_input(
                    "Max Year",
                    min_value=1900,
                    max_value=2030,
                    value=2030,
                    help="Maximum release year",
                )

        exclude_seen = st.checkbox(
            "Exclude movies I've already seen",
            value=True,
            help="Don't recommend movies you've previously interacted with",
        )

        submitted = st.form_submit_button(
            "🎯 Get Recommendations", use_container_width=True
        )

    # Process form submission
    if submitted:
        with st.spinner("Finding movies you'll love..."):
            # Prepare filter arguments
            genres = [selected_genre] if selected_genre else None
            year_min_val = year_min if year_min > 1900 else None
            year_max_val = year_max if year_max < 2030 else None

            # Call API
            result, error = client.get_recommendations(
                user_id=int(user_id),
                k=k,
                exclude_seen=exclude_seen,
                genres=genres,
                year_min=year_min_val,
                year_max=year_max_val,
            )

        if error:
            st.error(f"❌ **Error:** {error.message}")
            return

        if result is None:
            st.error("❌ Unexpected error: No result returned")
            return

        # Display results
        st.subheader(f"Recommendations for User {result.user_id}")

        # Show fallback notice if applicable
        if result.is_fallback:
            render_fallback_notice(result.fallback_reason)

        # Show model version
        st.caption(f"Model: {result.model_version}")

        # Render movies
        if result.recommendations:
            render_movie_list(result.recommendations)
        else:
            st.warning(
                "No recommendations found. Try adjusting your filters "
                "or check if the user ID exists."
            )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Allow running this page directly for testing
    st.set_page_config(
        page_title="Personalized Recommendations",
        page_icon="🎬",
        layout="wide",
    )
    render_page()
