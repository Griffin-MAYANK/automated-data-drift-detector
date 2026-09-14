"""Streamlit entry point for the historical drift monitoring dashboard."""

from __future__ import annotations

import sqlite3

import streamlit as st

if __package__:
    from .components import (
        render_feature_history, render_feature_table, render_header,
        render_run_metadata, render_status_trend, render_summary_metrics,
    )
    from .data import (
        build_status_trend, feature_history_to_dataframe, features_to_dataframe,
        get_database_path, get_feature_history, get_latest_run, get_recent_runs,
        get_repository, run_to_dataframe,
    )
else:
    from components import (
        render_feature_history, render_feature_table, render_header,
        render_run_metadata, render_status_trend, render_summary_metrics,
    )
    from data import (
        build_status_trend, feature_history_to_dataframe, features_to_dataframe,
        get_database_path, get_feature_history, get_latest_run, get_recent_runs,
        get_repository, run_to_dataframe,
    )
from drift_detector.logging_config import get_logger


logger = get_logger("dashboard")


def main() -> None:
    """Render the monitoring dashboard."""

    st.set_page_config(page_title="Automated Data Drift Monitor", page_icon=":bar_chart:", layout="wide")
    database_path = get_database_path()
    render_header(str(database_path))
    if st.sidebar.button("Refresh dashboard"):
        st.rerun()
    run_limit = st.sidebar.slider("Historical runs to display", 5, 100, 20, 5)
    repository = get_repository(database_path)

    try:
        recent_runs = get_recent_runs(repository, limit=run_limit)
        total_runs = repository.count_runs()
    except sqlite3.Error:
        logger.exception("dashboard database access failed")
        st.error("Historical data is temporarily unavailable.")
        return

    if not recent_runs:
        st.info("No detection runs have been recorded yet.")
        st.caption("Run the detector or API to populate historical monitoring data.")
        return
    latest_run = get_latest_run(repository)
    if latest_run is None:
        st.info("No detection runs have been recorded yet.")
        return

    render_summary_metrics(latest_run, total_runs)
    st.divider()
    st.subheader("Latest Detection Run")
    render_run_metadata(latest_run)
    render_feature_table(features_to_dataframe(latest_run.get("features", [])))

    st.divider()
    st.subheader("Historical Monitoring")
    render_status_trend(build_status_trend(recent_runs))
    st.dataframe(run_to_dataframe(recent_runs), use_container_width=True, hide_index=True)
    selected_run_id = st.selectbox("Inspect historical run", [run["run_id"] for run in recent_runs])
    try:
        selected_run = repository.get_run(selected_run_id)
    except sqlite3.Error:
        logger.exception("dashboard run lookup failed")
        st.error("The selected run could not be loaded.")
        selected_run = None
    if selected_run:
        with st.expander("Selected run details", expanded=False):
            render_run_metadata(selected_run)
            render_feature_table(features_to_dataframe(selected_run.get("features", [])))

    st.divider()
    st.subheader("Feature Analysis")
    available_features = sorted({item["feature"] for item in latest_run.get("features", []) if item.get("feature")})
    if not available_features:
        st.info("No feature results are available for analysis.")
        return
    selected_feature = st.selectbox("Feature", available_features)
    try:
        history = get_feature_history(repository, selected_feature, limit=run_limit)
    except sqlite3.Error:
        logger.exception("dashboard feature history lookup failed")
        st.error("Feature history is temporarily unavailable.")
        return
    render_feature_history(feature_history_to_dataframe(history))


if __name__ == "__main__":
    main()