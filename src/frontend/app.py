"""
Main Streamlit application for the Movie Recommendation System.

This is the entry point for the frontend. It provides:
- Sidebar navigation between pages
- Shared API client state
- Consistent page configuration
- Branding and layout

Usage:
    PYTHONPATH=. streamlit run src/frontend/app.py --server.port 8502
"""

import streamlit as st

from src.frontend.pages import about, personalized, popular, similar

# =============================================================================
# Page Configuration
# =============================================================================

# Must be the first Streamlit command
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Navigation
# =============================================================================

# Define pages
PAGES = {
    "🎬 Personalized": personalized,
    "🔍 Find Similar": similar,
    "🔥 Popular": popular,
    "ℹ️ About": about,
}


def render_sidebar() -> str:
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        # Logo/Title
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 style='margin: 0;'>🎬</h1>
                <h2 style='margin: 0;'>Movie Recommender</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Navigation
        st.markdown("### Navigation")
        selected_page = st.radio(
            label="Go to",
            options=list(PAGES.keys()),
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # Quick stats (if API is available)
        render_sidebar_stats()

        # Footer
        st.divider()
        st.caption(
            "Built with Streamlit, FastAPI, and ❤️\n\n"
            "A learning project for production ML systems."
        )

    return str(selected_page)


def render_sidebar_stats() -> None:
    """Render quick stats in the sidebar."""
    from src.frontend.api_client import APIClient

    # Get or create client
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()

    client = st.session_state.api_client

    # Check health
    health, error = client.health_check()

    if error:
        st.markdown("### Status")
        st.error("API Offline", icon="🔴")
    elif health:
        st.markdown("### Status")
        st.success("API Online", icon="🟢")

        # Show compact stats
        if health.n_users and health.n_items:
            st.caption(f"👥 {health.n_users:,} users")
            st.caption(f"🎬 {health.n_items:,} movies")


# =============================================================================
# Main App
# =============================================================================


def main() -> None:
    """Main application entry point."""
    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Render the selected page
    page_module = PAGES[selected_page]
    page_module.render_page()


if __name__ == "__main__":
    main()
