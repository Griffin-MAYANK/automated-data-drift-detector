"""Reusable Streamlit presentation components."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


STATUS_COLORS = {
    "ALERT": "#c0392b", "INVESTIGATE": "#d68910", "MONITOR": "#2471a3",
    "NO_DRIFT": "#1e8449", "HEALTHY": "#1e8449", "INSUFFICIENT_DATA": "#6c757d",
}


def render_header(database_path: str) -> None:
    """Render the dashboard header and database status."""

    st.title("Automated Data Drift Monitor")
    st.caption("Production-oriented monitoring for statistical and practical data drift across historical detection runs.")
    st.caption(f"SQLite history: `{database_path}`")


def render_summary_metrics(latest_run: dict[str, Any], total_runs: int) -> None:
    """Render top-level latest-run and historical metrics."""

    columns = st.columns(5)
    columns[0].metric("Overall Status", latest_run.get("overall_status", "UNKNOWN"))
    columns[1].metric("Alerts", latest_run.get("alert_count", 0))
    columns[2].metric("Investigations", latest_run.get("investigate_count", 0))
    columns[3].metric("Monitor", latest_run.get("monitor_count", 0))
    columns[4].metric("Total Runs", total_runs)
    status = latest_run.get("overall_status", "UNKNOWN")
    color = STATUS_COLORS.get(status, "#495057")
    st.markdown(
        f'<div style="border-left: 5px solid {color}; padding: 0.5rem 0.75rem; margin: 0.5rem 0 1rem 0;">'
        f'<strong>Status:</strong> {status}</div>', unsafe_allow_html=True
    )


def render_run_metadata(run: dict[str, Any]) -> None:
    """Render metadata for one historical run."""

    columns = st.columns(4)
    columns[0].write(f"**Run ID**  \n`{run['run_id']}`")
    columns[1].write(f"**Created**  \n{run['created_at']}")
    columns[2].write(f"**Dataset**  \n{run['dataset']}")
    columns[3].write(f"**Split date**  \n{run['split_date']}")
    st.caption(f"Reference rows: {run['reference_rows']:,} | Current rows: {run['current_rows']:,} | Features: {run['number_of_features']}")


def _format_p_value(value: Any) -> str:
    return "-" if pd.isna(value) else f"{float(value):.3e}"


def render_feature_table(features: pd.DataFrame) -> None:
    """Render feature results with readable numeric formatting."""

    display = features.copy()
    for column in ("p_value", "adjusted_p_value"):
        if column in display:
            display[column] = display[column].map(_format_p_value)
    for column in ("magnitude", "normalized_magnitude"):
        if column in display:
            display[column] = display[column].map(lambda value: "-" if pd.isna(value) else f"{float(value):.4f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_status_trend(trend: pd.DataFrame) -> None:
    """Render a historical status-count chart."""

    if trend.empty:
        st.info("Not enough historical data to show a status trend.")
    else:
        st.bar_chart(trend, stack=True, use_container_width=True)


def render_feature_history(history: pd.DataFrame) -> None:
    """Render selected feature history and a magnitude trend."""

    if history.empty:
        st.info("No history is available for this feature.")
        return
    chart = history.copy()
    chart["created_at"] = pd.to_datetime(chart["created_at"], errors="coerce")
    chart = chart.dropna(subset=["created_at"]).set_index("created_at")
    metric = "normalized_magnitude" if not chart["normalized_magnitude"].dropna().empty else "magnitude"
    st.line_chart(chart[[metric]].rename(columns={metric: "magnitude"}), use_container_width=True)
    st.dataframe(history, use_container_width=True, hide_index=True)