"""
Popular movies page.

This page displays the most popular movies in the catalog, with optional
genre filtering. It serves multiple purposes:
- Discovery for new users (cold-start)
- Browsing trending content
- Genre-based exploration

Features:
- Popularity-ranked movie list
- Genre filter dropdown
- Configurable number of results
- Popularity scores displayed
"""

from typing import cast

import streamlit as st

from src.frontend.api_client import APIClient, Movie

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_API_URL = "http://localhost:8000"

# Available genres for filtering (matches MovieLens 100K)
GENRE_OPTIONS = [
    "All Genres",
    "Action",
    "Adventure",
    "Animation",
    "Children",
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


def get_client() -> APIClient:
    """Get or create the API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(base_url=DEFAULT_API_URL)
    return cast(APIClient, st.session_state.api_client)


# =============================================================================
# UI Components
# =============================================================================


def render_popular_movie_card(movie: Movie, rank: int) -> None:
    """Render a popular movie as a card.

    Args:
        movie: Movie object with popularity score
        rank: Display rank (1-indexed)
    """
    # Format genres
    genres_str = ", ".join(movie.genres) if movie.genres else "Unknown"

    # Format year
    year_str = f"({movie.year})" if movie.year else ""

    # Format popularity score (interaction count)
    popularity_score = int(movie.score)

    # Medal for top 3
    if rank == 1:
        medal = "🥇"
    elif rank == 2:
        medal = "🥈"
    elif rank == 3:
        medal = "🥉"
    else:
        medal = ""

    with st.container():
        col1, col2, col3 = st.columns([1, 9, 2])

        with col1:
            if medal:
                st.markdown(
                    f"<div style='font-size: 28px; text-align: center;'>{medal}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='font-size: 24px; font-weight: bold; "
                    f"color: #888; text-align: center;'>{rank}</div>",
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown(f"**{movie.title}** {year_str}")
            st.caption(f"🎬 {genres_str}")

        with col3:
            st.markdown(
                f"<div style='text-align: right;'>"
                f"<span style='font-size: 1.1rem; font-weight: bold;'>"
                f"{popularity_score}</span><br/>"
                f"<span style='font-size: 0.8rem; color: #888;'>interactions</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()


def render_popular_movies_list(movies: list[Movie], start_rank: int = 1) -> None:
    """Render a list of popular movies.

    Args:
        movies: List of Movie objects with popularity scores
        start_rank: Starting rank number (for pagination)
    """
    if not movies:
        st.info("No movies found.")
        return

    for i, movie in enumerate(movies):
        render_popular_movie_card(movie, rank=start_rank + i)


def render_stats_header(total_movies: int, genre: str) -> None:
    """Render a stats header showing current filter state.

    Args:
        total_movies: Number of movies being displayed
        genre: Current genre filter (or "All Genres")
    """
    genre_text = f"**{genre}**" if genre != "All Genres" else "**all genres**"

    st.markdown(
        f"Showing top **{total_movies}** popular movies from {genre_text}",
    )


# =============================================================================
# Main Page
# =============================================================================


def render_page() -> None:
    """Render the popular movies page."""
    st.title("🔥 Popular Movies")
    st.markdown(
        "Discover the most popular movies in our catalog. "
        "These are the films that users interact with the most!"
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

    # Filter controls
    st.subheader("Browse Popular Movies")

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_genre = st.selectbox(
            "Filter by Genre",
            options=GENRE_OPTIONS,
            index=0,
            help="Show popular movies from a specific genre",
        )

    with col2:
        k = st.slider(
            "Number of movies",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="How many popular movies to show",
        )

    # Fetch button
    if st.button("🔄 Load Popular Movies", use_container_width=True):
        st.session_state.load_popular = True

    # Auto-load on first visit or when button clicked
    if st.session_state.get("load_popular", True):
        # Determine genre filter for API
        genre_filter = None if selected_genre == "All Genres" else selected_genre

        with st.spinner("Loading popular movies..."):
            result, error = client.get_popular_movies(k=k, genre=genre_filter)

        if error:
            st.error(f"❌ **Error:** {error.message}")
            return

        if result is None:
            st.error("❌ Unexpected error: No result returned")
            return

        # Display results
        st.divider()

        # Stats header
        render_stats_header(
            total_movies=len(result.recommendations),
            genre=selected_genre,
        )

        # Show source info
        st.caption(f"Source: {result.source}")

        # Render movies
        if result.recommendations:
            render_popular_movies_list(result.recommendations)
        else:
            if genre_filter:
                st.warning(
                    f"No popular movies found in the **{genre_filter}** genre. "
                    "Try selecting a different genre."
                )
            else:
                st.warning("No popular movies found.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Popular Movies",
        page_icon="🔥",
        layout="wide",
    )
    render_page()
