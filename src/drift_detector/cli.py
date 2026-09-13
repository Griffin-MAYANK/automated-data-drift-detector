"""Command-line interface for the automated data drift detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from drift_detector.config import DriftConfig
from drift_detector.detector import detect_dataset_drift
from drift_detector.features import create_reference_current_features
from drift_detector.reporting import (
    calculate_operational_summary,
    export_drift_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser used by the CLI."""

    parser = argparse.ArgumentParser(
        prog="drift-detector",
        description="Detect data drift in a reference/current dataset split.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the input dataset. Excel files are read with pandas.",
    )
    parser.add_argument(
        "--split-date",
        required=True,
        help="Split date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for CSV and JSON reports (default: reports).",
    )
    parser.add_argument(
        "--significance-threshold",
        type=float,
        default=0.05,
        help="Statistical significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum samples required per feature (default: 30).",
    )
    return parser


def _print_summary(
    data_path: Path,
    reference_rows: int,
    current_rows: int,
    summary: dict,
    csv_path: Path,
    json_path: Path,
) -> None:
    """Print the operational results in a concise human-readable format."""

    print("Drift detection completed")
    print(f"Dataset: {data_path}")
    print(f"Reference rows: {reference_rows}")
    print(f"Current rows: {current_rows}")
    print(f"Features analyzed: {summary['number_of_features']}")
    print(f"Overall status: {summary['overall_status']}")
    print(f"ALERT count: {summary['alert_count']}")
    print(f"INVESTIGATE count: {summary['investigate_count']}")
    print(f"MONITOR count: {summary['monitor_count']}")
    print(f"NO_DRIFT count: {summary['no_drift_count']}")
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run drift detection from command-line arguments."""

    parser = build_parser()
    args = parser.parse_args(argv)
    data_path = Path(args.data)

    try:
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Input dataset does not exist: {data_path}"
            )

        split_date = pd.Timestamp(args.split_date)
        if pd.isna(split_date):
            raise ValueError("Split date must be a valid date.")

        data = pd.read_excel(data_path)
        reference_data, current_data = (
            create_reference_current_features(
                data=data,
                split_date=split_date,
            )
        )

        config = DriftConfig(
            significance_threshold=args.significance_threshold,
            min_samples=args.min_samples,
        )
        report = detect_dataset_drift(
            reference_data=reference_data,
            current_data=current_data,
            config=config,
        )
        summary = calculate_operational_summary(report)
        exported = export_drift_report(
            report=report,
            output_directory=args.output_dir,
            dataset_name=data_path.name,
            reference_rows=len(reference_data),
            current_rows=len(current_data),
        )

        _print_summary(
            data_path=data_path,
            reference_rows=len(reference_data),
            current_rows=len(current_data),
            summary=summary,
            csv_path=exported["csv_path"],
            json_path=exported["json_path"],
        )
        return 0

    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: drift detection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())