"""
Similar movies page.

This page allows users to discover movies similar to one they select.
It's useful for:
- Movie discovery ("If you liked X, try these...")
- Cold-start users who can provide a seed movie
- Exploring the catalog based on a known favorite

Features:
- Searchable movie dropdown
- Configurable number of results
- Similarity scores displayed as percentages
- Seed movie displayed for context
"""

from typing import Optional, cast

import streamlit as st

from src.frontend.api_client import APIClient, Movie

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_API_URL = "http://localhost:8000"


def get_client() -> APIClient:
    """Get or create the API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(base_url=DEFAULT_API_URL)
    return cast(APIClient, st.session_state.api_client)


# =============================================================================
# UI Components
# =============================================================================


def render_seed_movie(title: str, movie_id: int) -> None:
    """Render the seed movie header.

    Args:
        title: Seed movie title
        movie_id: Seed movie ID
    """
    st.markdown(
        f"""
        <div style='background-color: #262730; padding: 1rem;
        border-radius: 0.5rem; margin-bottom: 1rem;'>
            <p style='margin: 0; color: #888;'>Finding movies similar to:</p>
            <h3 style='margin: 0.5rem 0 0 0;'>🎬 {title}</h3>
            <p style='margin: 0.25rem 0 0 0; color: #888; font-size: 0.9rem;'>
                Movie ID: {movie_id}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_similar_movie_card(movie: Movie, rank: int) -> None:
    """Render a similar movie as a card.

    Args:
        movie: Movie object with similarity score
        rank: Display rank (1-indexed)
    """
    # Format genres
    genres_str = ", ".join(movie.genres) if movie.genres else "Unknown"

    # Format year
    year_str = f"({movie.year})" if movie.year else ""

    # Format similarity as percentage
    similarity_pct = movie.score * 100

    # Color based on similarity
    if similarity_pct >= 70:
        color = "#00c853"  # Green - very similar
    elif similarity_pct >= 50:
        color = "#ffc107"  # Yellow - moderately similar
    else:
        color = "#888"  # Gray - less similar

    with st.container():
        col1, col2, col3 = st.columns([1, 9, 2])

        with col1:
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
                f"<div style='text-align: right; font-size: 1.1rem; "
                f"font-weight: bold; color: {color};'>{similarity_pct:.1f}%</div>",
                unsafe_allow_html=True,
            )

        st.divider()


def render_similar_movies_list(movies: list[Movie]) -> None:
    """Render a list of similar movies.

    Args:
        movies: List of Movie objects with similarity scores
    """
    if not movies:
        st.info("No similar movies found.")
        return

    for i, movie in enumerate(movies):
        render_similar_movie_card(movie, rank=i + 1)


def load_movie_options(client: APIClient) -> Optional[dict[str, int]]:
    """Load movie options for the dropdown.

    Returns:
        Dictionary mapping "Title (Year)" to movie_id, or None on error
    """
    # Check if already cached in session state
    if "movie_options" in st.session_state:
        return cast(dict[str, int], st.session_state.movie_options)

    # Fetch from API
    movies, error = client.get_all_movies()

    if error or movies is None:
        return None

    # Build options dict: "Title (Year)" -> movie_id
    options: dict[str, int] = {}
    for movie in movies:
        year_str = f" ({movie.year})" if movie.year else ""
        label = f"{movie.title}{year_str}"
        options[label] = movie.movie_id

    # Cache in session state
    st.session_state.movie_options = options

    return options


# =============================================================================
# Main Page
# =============================================================================


def render_page() -> None:
    """Render the similar movies page."""
    st.title("🔍 Find Similar Movies")
    st.markdown(
        "Discover movies similar to your favorites. "
        "Select a movie you like and we'll find others you might enjoy!"
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

    # Load movie options for dropdown
    movie_options = load_movie_options(client)

    if movie_options is None:
        st.error(
            "❌ Could not load movie list. "
            "Please check that the API is working correctly."
        )
        return

    if not movie_options:
        st.warning("No movies available in the system.")
        return

    # Input form
    st.subheader("Select a Movie")

    with st.form("similar_form"):
        # Movie selection dropdown
        selected_movie_label = st.selectbox(
            "Choose a movie you like",
            options=list(movie_options.keys()),
            index=0,
            help="Start typing to search for a movie",
        )

        # Number of results
        k = st.slider(
            "Number of similar movies",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            help="How many similar movies to show",
        )

        submitted = st.form_submit_button(
            "🔍 Find Similar Movies", use_container_width=True
        )

    # Process form submission
    if submitted and selected_movie_label:
        movie_id = movie_options[selected_movie_label]

        with st.spinner("Finding similar movies..."):
            result, error = client.get_similar_movies(movie_id=movie_id, k=k)

        if error:
            st.error(f"❌ **Error:** {error.message}")
            return

        if result is None:
            st.error("❌ Unexpected error: No result returned")
            return

        # Display results
        st.subheader("Similar Movies")

        # Show seed movie
        render_seed_movie(title=result.title, movie_id=result.movie_id)

        # Show model version
        st.caption(f"Model: {result.model_version}")

        # Render similar movies
        if result.similar_items:
            render_similar_movies_list(result.similar_items)
        else:
            st.warning(
                "No similar movies found. This might happen if the movie "
                "has very few interactions in our dataset."
            )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Find Similar Movies",
        page_icon="🔍",
        layout="wide",
    )
    render_page()
