"""
Reporting and export utilities for the automated
data drift detection system.

This module handles:

1. Final report preparation
2. Operational summary generation
3. CSV export
4. JSON export
5. Export verification
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


def calculate_operational_summary(
    report: pd.DataFrame,
) -> dict:
    """
    Calculate high-level operational information
    from the final drift report.

    Parameters
    ----------
    report : pandas.DataFrame
        Final feature-level drift report.

    Returns
    -------
    dict
        Operational summary containing decision counts
        and overall dataset status.
    """

    if report.empty:
        raise ValueError(
            "Cannot calculate a summary from an empty report."
        )

    decision_counts = (
        report["final_decision"]
        .value_counts()
    )

    alert_count = int(
        decision_counts.get("ALERT", 0)
    )

    investigate_count = int(
        decision_counts.get("INVESTIGATE", 0)
    )

    monitor_count = int(
        decision_counts.get("MONITOR", 0)
    )

    no_drift_count = int(
        decision_counts.get("NO_DRIFT", 0)
    )

    insufficient_data_count = int(
        decision_counts.get(
            "INSUFFICIENT_DATA",
            0,
        )
    )

    if alert_count > 0:
        overall_status = "ALERT"

    elif investigate_count > 0:
        overall_status = "INVESTIGATE"

    elif monitor_count > 0:
        overall_status = "MONITOR"

    elif insufficient_data_count > 0:
        overall_status = "INSUFFICIENT_DATA"

    else:
        overall_status = "HEALTHY"

    return {
        "overall_status": overall_status,
        "alert_count": alert_count,
        "investigate_count": investigate_count,
        "monitor_count": monitor_count,
        "no_drift_count": no_drift_count,
        "insufficient_data_count": (
            insufficient_data_count
        ),
        "number_of_features": int(
            len(report)
        ),
    }


def prepare_json_records(
    report: pd.DataFrame,
) -> list:
    """
    Convert a DataFrame into JSON-safe records.

    Handles:

    - NaN
    - NumPy numeric types
    - NumPy boolean types
    - pandas numeric types
    """

    records = []

    for record in report.to_dict(
        orient="records"
    ):

        clean_record = {}

        for key, value in record.items():

            if pd.isna(value):
                clean_record[key] = None

            elif isinstance(
                value,
                np.integer,
            ):
                clean_record[key] = int(value)

            elif isinstance(
                value,
                np.floating,
            ):
                clean_record[key] = float(value)

            elif isinstance(
                value,
                np.bool_,
            ):
                clean_record[key] = bool(value)

            else:
                clean_record[key] = value

        records.append(clean_record)

    return records


def export_drift_report(
    report: pd.DataFrame,
    output_directory: str | Path = "../reports",
    dataset_name: str = "UCI Online Retail",
    reference_rows: int | None = None,
    current_rows: int | None = None,
) -> dict:
    """
    Export a final drift report to CSV and JSON.

    Parameters
    ----------
    report : pandas.DataFrame
        Final production drift report.

    output_directory : str or pathlib.Path
        Directory where reports will be stored.

    dataset_name : str
        Name of the monitored dataset.

    reference_rows : int, optional
        Number of rows in the reference dataset.

    current_rows : int, optional
        Number of rows in the current dataset.

    Returns
    -------
    dict
        Paths and metadata describing the exported reports.
    """

    if report.empty:
        raise ValueError(
            "Cannot export an empty drift report."
        )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_directory
        / "final_drift_report.csv"
    )

    json_path = (
        output_directory
        / "final_drift_report.json"
    )

    # --------------------------------------------
    # CSV export
    # --------------------------------------------

    report.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------
    # Operational summary
    # --------------------------------------------

    summary = calculate_operational_summary(
        report
    )

    # --------------------------------------------
    # JSON records
    # --------------------------------------------

    json_records = prepare_json_records(
        report
    )

    # --------------------------------------------
    # Complete JSON document
    # --------------------------------------------

    report_metadata = {

        "report_type":
            "Automated Data Drift Detection",

        "dataset":
            dataset_name,

        "reference_rows":
            (
                int(reference_rows)
                if reference_rows is not None
                else None
            ),

        "current_rows":
            (
                int(current_rows)
                if current_rows is not None
                else None
            ),

        "number_of_features":
            summary["number_of_features"],

        "alert_count":
            summary["alert_count"],

        "investigate_count":
            summary["investigate_count"],

        "monitor_count":
            summary["monitor_count"],

        "no_drift_count":
            summary["no_drift_count"],

        "insufficient_data_count":
            summary["insufficient_data_count"],

        "overall_status":
            summary["overall_status"],

        "features":
            json_records,
    }

    # --------------------------------------------
    # JSON export
    # --------------------------------------------

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report_metadata,
            file,
            indent=4,
            allow_nan=False,
        )

    # --------------------------------------------
    # Verify exports
    # --------------------------------------------

    if not csv_path.exists():
        raise RuntimeError(
            "CSV report was not created successfully."
        )

    if not json_path.exists():
        raise RuntimeError(
            "JSON report was not created successfully."
        )

    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "summary": summary,
        "csv_size_bytes": csv_path.stat().st_size,
        "json_size_bytes": json_path.stat().st_size,
    }


def load_exported_report(
    csv_path: str | Path,
    json_path: str | Path,
) -> dict:
    """
    Load and verify previously exported CSV and JSON reports.

    Returns
    -------
    dict
        Loaded CSV DataFrame and JSON dictionary.
    """

    csv_path = Path(csv_path)
    json_path = Path(json_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV report not found: {csv_path}"
        )

    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON report not found: {json_path}"
        )

    csv_report = pd.read_csv(
        csv_path
    )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:

        json_report = json.load(file)

    return {
        "csv": csv_report,
        "json": json_report,
    }