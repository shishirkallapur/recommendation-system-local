"""
Streamlit monitoring dashboard for the recommendation service.

Provides real-time visibility into:
- API health status
- Traffic metrics (requests, unique users)
- Latency metrics (mean, p50, p95, p99)
- Quality metrics (fallback rate, catalog coverage)

Usage:
    PYTHONPATH=. streamlit run src/monitoring/dashboard.py --server.port 8501
"""

from typing import Optional

import streamlit as st

from src.frontend.api_client import APIClient
from src.monitoring.kpis import KPICalculator, KPIReport

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
)

# =============================================================================
# Helper Functions
# =============================================================================


def get_api_client() -> APIClient:
    """Get or create the API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()
    client: APIClient = st.session_state.api_client
    return client


def load_kpis(hours: float) -> Optional[KPIReport]:
    """Load KPIs for the specified time window.

    Args:
        hours: Number of hours to look back.

    Returns:
        KPIReport or None if no data available.
    """
    try:
        calculator = KPICalculator()
        return calculator.compute_all(hours=hours)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error computing KPIs: {e}")
        return None


# =============================================================================
# UI Components
# =============================================================================


def render_api_health() -> None:
    """Render API health status section."""
    st.subheader("🔌 API Health")

    client = get_api_client()
    health, error = client.health_check()

    if error:
        st.error(f"**API Offline:** {error.message}")
        return

    if health is None:
        st.warning("Could not retrieve health status.")
        return

    # Status indicator
    if health.status == "healthy":
        st.success("✓ API is healthy", icon="🟢")
    else:
        st.warning(f"Status: {health.status}", icon="🟡")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Version", health.model_version or "N/A")

    with col2:
        st.metric("Users", f"{health.n_users:,}" if health.n_users else "N/A")

    with col3:
        st.metric("Items", f"{health.n_items:,}" if health.n_items else "N/A")

    with col4:
        if health.uptime_seconds:
            uptime_hrs = health.uptime_seconds / 3600
            st.metric("Uptime", f"{uptime_hrs:.1f}h")
        else:
            st.metric("Uptime", "N/A")


def render_traffic_metrics(report: KPIReport) -> None:
    """Render traffic metrics section."""
    st.subheader("📈 Traffic")

    traffic = report.traffic

    # Main metrics row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Requests", f"{traffic.total_requests:,}")

    with col2:
        st.metric("Requests/Min", f"{traffic.requests_per_minute:.2f}")

    with col3:
        st.metric("Unique Users", f"{traffic.unique_users:,}")

    # Requests by endpoint
    if traffic.requests_by_endpoint:
        st.markdown("**Requests by Endpoint:**")
        cols = st.columns(len(traffic.requests_by_endpoint))
        for i, (endpoint, count) in enumerate(traffic.requests_by_endpoint.items()):
            with cols[i]:
                st.metric(endpoint, f"{count:,}")


def render_latency_metrics(report: KPIReport) -> None:
    """Render latency metrics section."""
    st.subheader("⏱️ Latency")

    latency = report.latency

    # Main metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Mean", f"{latency.mean_ms:.1f}ms")

    with col2:
        st.metric("p50", f"{latency.p50_ms:.1f}ms")

    with col3:
        # Color p95 based on target (<100ms)
        p95_val = latency.p95_ms
        delta = None
        if p95_val > 0:
            delta = "OK" if p95_val < 100 else "High"
        st.metric("p95", f"{p95_val:.1f}ms", delta=delta)

    with col4:
        st.metric("p99", f"{latency.p99_ms:.1f}ms")

    with col5:
        st.metric("Max", f"{latency.max_ms:.1f}ms")

    # Latency by endpoint
    if latency.latency_by_endpoint:
        st.markdown("**Average Latency by Endpoint:**")
        cols = st.columns(len(latency.latency_by_endpoint))
        for i, (endpoint, avg_ms) in enumerate(latency.latency_by_endpoint.items()):
            with cols[i]:
                st.metric(endpoint, f"{avg_ms:.1f}ms")


def render_quality_metrics(report: KPIReport) -> None:
    """Render quality metrics section."""
    st.subheader("🎯 Quality")

    quality = report.quality

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Color fallback rate based on target (<10%)
        fallback_pct = quality.fallback_rate * 100
        delta = None
        if quality.fallback_count > 0:
            delta = "OK" if fallback_pct < 10 else "High"
        st.metric("Fallback Rate", f"{fallback_pct:.1f}%", delta=delta)

    with col2:
        st.metric("Fallback Count", f"{quality.fallback_count:,}")

    with col3:
        # Color coverage based on target (>50%)
        coverage_pct = quality.catalog_coverage * 100
        delta = None
        if quality.unique_items_recommended > 0:
            delta = "Good" if coverage_pct > 50 else "Low"
        st.metric("Catalog Coverage", f"{coverage_pct:.1f}%", delta=delta)

    with col4:
        st.metric("Avg Recs/Request", f"{quality.avg_recommendations_per_request:.1f}")

    # Additional details
    with st.expander("📊 Coverage Details"):
        st.write(f"**Unique items recommended:** {quality.unique_items_recommended:,}")
        st.write(f"**Total catalog size:** {quality.total_catalog_size:,}")


def render_time_window_selector() -> float:
    """Render time window selector and return selected hours."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 📊 Monitoring Dashboard")

    with col2:
        hours = st.selectbox(
            "Time Window",
            options=[1, 6, 12, 24, 48, 168],
            index=3,  # Default to 24 hours
            format_func=lambda x: f"Last {x}h" if x < 168 else "Last 7 days",
        )

    return float(hours)


def render_no_data_message() -> None:
    """Render message when no data is available."""
    st.warning(
        "📭 **No Request Data Available**\n\n"
        "The monitoring dashboard requires request logs to compute metrics. "
        "Start using the recommendation API to generate data.\n\n"
        "**Quick start:**\n"
        "1. Start the API: `PYTHONPATH=. python -m src.api.main`\n"
        "2. Make some requests via the frontend or curl\n"
        "3. Refresh this dashboard"
    )


def render_report_actions(report: KPIReport) -> None:
    """Render report action buttons."""
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("💾 Save Report", use_container_width=True):
            try:
                calculator = KPICalculator()
                path = calculator.save_report(report)
                st.success(f"Report saved to {path}")
            except Exception as e:
                st.error(f"Failed to save report: {e}")

    with col3:
        st.caption(
            f"Report period: {report.start_time.strftime('%Y-%m-%d %H:%M')} "
            f"to {report.end_time.strftime('%Y-%m-%d %H:%M')} UTC"
        )


# =============================================================================
# Main Dashboard
# =============================================================================


def main() -> None:
    """Main dashboard entry point."""
    # Time window selector
    hours = render_time_window_selector()

    st.divider()

    # API Health (always show)
    render_api_health()

    st.divider()

    # Load KPIs
    report = load_kpis(hours)

    if report is None:
        render_no_data_message()
        return

    # Check if there's any data
    if report.traffic.total_requests == 0:
        render_no_data_message()
        return

    # Traffic metrics
    render_traffic_metrics(report)

    st.divider()

    # Latency metrics
    render_latency_metrics(report)

    st.divider()

    # Quality metrics
    render_quality_metrics(report)

    # Report actions
    render_report_actions(report)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
