"""
About/Status page.

This page provides system information, health status, and documentation
about how the recommendation system works.

Features:
- API health status with detailed metrics
- Model information (version, dimensions)
- System architecture overview
- Usage tips and help
"""

from typing import cast

import streamlit as st

from src.frontend.api_client import APIClient

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


def render_health_status(client: APIClient) -> None:
    """Render the API health status section."""
    st.subheader("🔌 System Status")

    health, error = client.health_check()

    if error:
        st.error(
            f"**API Unavailable**\n\n"
            f"Error: {error.message}\n\n"
            f"Please ensure the API is running at `{client.base_url}`"
        )
        st.code(
            "# Start the API with:\nPYTHONPATH=. python -m src.api.main",
            language="bash",
        )
        return

    if health is None:
        st.error("Could not retrieve health status.")
        return

    # Status indicator
    if health.status == "healthy":
        st.success("✓ System is healthy and ready", icon="🟢")
    else:
        st.warning(f"System status: {health.status}", icon="🟡")

    # Metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Model Version",
            value=health.model_version or "Unknown",
        )

    with col2:
        st.metric(
            label="Users in Model",
            value=f"{health.n_users:,}" if health.n_users else "Unknown",
        )

    with col3:
        st.metric(
            label="Movies in Model",
            value=f"{health.n_items:,}" if health.n_items else "Unknown",
        )

    # Additional details in expander
    with st.expander("📊 More Details"):
        if health.model_loaded_at:
            st.write(f"**Model Loaded:** {health.model_loaded_at}")
        if health.uptime_seconds:
            uptime_min = health.uptime_seconds / 60
            if uptime_min < 60:
                st.write(f"**Uptime:** {uptime_min:.1f} minutes")
            else:
                uptime_hr = uptime_min / 60
                st.write(f"**Uptime:** {uptime_hr:.1f} hours")
        st.write(f"**API URL:** {client.base_url}")


def render_how_it_works() -> None:
    """Render the 'How It Works' section."""
    st.subheader("🧠 How It Works")

    st.markdown(
        """
    This recommendation system uses **collaborative filtering** to suggest movies
    you might enjoy based on the preferences of similar users.

    **The Technology:**

    - **Algorithm:** Alternating Least Squares (ALS) matrix factorization
    - **Similarity Search:** FAISS for fast nearest-neighbor lookups
    - **Data:** Trained on the MovieLens 100K dataset

    **How Recommendations Are Made:**

    1. **For known users:** We compute a personalized score for each movie based
       on your historical interactions and return the highest-scoring unwatched films.

    2. **For new users:** We show popular movies or, if you select a seed movie,
       we find similar films based on viewing patterns.

    3. **Similar movies:** We use item embeddings to find movies that are
       "close" in the latent space—meaning users who liked one tend to like the others.
    """
    )


def render_features() -> None:
    """Render the features overview section."""
    st.subheader("✨ Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **🎬 Personalized Recommendations**
        - Enter your user ID
        - Get tailored movie suggestions
        - Filter by genre and year
        - Exclude already-seen movies
        """
        )

        st.markdown(
            """
        **🔥 Popular Movies**
        - Browse trending films
        - Filter by genre
        - Great for discovery
        """
        )

    with col2:
        st.markdown(
            """
        **🔍 Similar Movies**
        - Select a movie you like
        - Discover similar films
        - Perfect for "if you liked X..."
        """
        )

        st.markdown(
            """
        **📊 Transparency**
        - See recommendation scores
        - View similarity percentages
        - Check system health
        """
        )


def render_usage_tips() -> None:
    """Render usage tips section."""
    st.subheader("💡 Tips")

    st.markdown(
        """
    - **New here?** Start with **Popular Movies** to explore the catalog,
      or use **Find Similar** with a movie you already love.

    - **Have a user ID?** Go to **Personalized Recommendations** for
      tailored suggestions based on your history.

    - **Filters not working?** Some combinations may return few results.
      Try broadening your year range or removing genre filters.

    - **Scores explained:**
      - *Personalized:* Higher scores = stronger predicted preference
      - *Similar:* Percentage indicates how closely related movies are
      - *Popular:* Shows total user interactions (more = more popular)
    """
    )


def render_about_data() -> None:
    """Render information about the dataset."""
    st.subheader("📚 About the Data")

    st.markdown(
        """
    This system is trained on the **MovieLens 100K** dataset:

    - ~100,000 ratings from ~1,000 users on ~1,700 movies
    - Ratings converted to implicit feedback (rating ≥ 4 = positive interaction)
    - Movies span from 1922 to 1998
    - 19 genre categories

    *MovieLens data provided by GroupLens Research at the University of Minnesota.*
    """
    )


# =============================================================================
# Main Page
# =============================================================================


def render_page() -> None:
    """Render the about/status page."""
    st.title("ℹ️ About This System")
    st.markdown(
        "Learn how the Movie Recommendation System works and check its current status."
    )

    client = get_client()

    # Health status (most important, show first)
    render_health_status(client)

    st.divider()

    # How it works
    render_how_it_works()

    st.divider()

    # Features overview
    render_features()

    st.divider()

    # Usage tips
    render_usage_tips()

    st.divider()

    # About the data
    render_about_data()

    # Footer
    st.divider()
    st.caption(
        "Built as a learning project for production ML systems. "
        "Powered by FastAPI, Streamlit, and ❤️"
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="About - Movie Recommender",
        page_icon="ℹ️",
        layout="wide",
    )
    render_page()
