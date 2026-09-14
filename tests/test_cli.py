from pathlib import Path

import pandas as pd
import pytest

from drift_detector.cli import main


@pytest.fixture(autouse=True)
def use_temporary_database(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DRIFT_DETECTOR_DATABASE_PATH",
        str(tmp_path / "history.db"),
    )


def create_cli_dataset(path: Path) -> Path:
    """Create a small local Excel dataset for CLI tests."""

    rows = []
    for index in range(40):
        rows.append({
            "InvoiceNo": str(10000 + index),
            "Quantity": 1 + (index % 3),
            "InvoiceDate": "2011-06-01 12:00:00",
            "UnitPrice": 10.0 + (index % 4),
            "Country": "United Kingdom",
        })

    for index in range(40):
        rows.append({
            "InvoiceNo": str(20000 + index),
            "Quantity": 2 + (index % 3),
            "InvoiceDate": "2011-08-01 12:00:00",
            "UnitPrice": 11.0 + (index % 4),
            "Country": "United Kingdom",
        })

    data = pd.DataFrame(rows)
    data.to_excel(path, index=False)
    return path


def test_help_works(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    assert "--data DATA" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        ["--split-date", "2011-07-01"],
        ["--data", "data.xlsx"],
    ],
)
def test_required_arguments_are_rejected(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code != 0


def test_nonexistent_dataset_path_is_rejected(capsys):
    result = main([
        "--data",
        "does-not-exist.xlsx",
        "--split-date",
        "2011-07-01",
    ])

    assert result == 1
    assert "does not exist" in capsys.readouterr().err


def test_invalid_split_date_is_rejected(tmp_path, capsys):
    dataset_path = create_cli_dataset(tmp_path / "input.xlsx")

    result = main([
        "--data",
        str(dataset_path),
        "--split-date",
        "not-a-date",
    ])

    assert result == 1
    assert "Error:" in capsys.readouterr().err


def test_valid_arguments_generate_reports_and_return_zero(tmp_path, capsys):
    dataset_path = create_cli_dataset(tmp_path / "input.xlsx")
    output_directory = tmp_path / "custom-reports"

    result = main([
        "--data",
        str(dataset_path),
        "--split-date",
        "2011-07-01",
        "--output-dir",
        str(output_directory),
        "--min-samples",
        "5",
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert "Dataset:" in output
    assert "Reference rows: 40" in output
    assert "Current rows: 40" in output
    assert "Features analyzed: 8" in output
    assert "ALERT count:" in output
    assert "INVESTIGATE count:" in output
    assert "MONITOR count:" in output
    assert "NO_DRIFT count:" in output
    assert "Run ID: drift-" in output
    assert (output_directory / "final_drift_report.csv").exists()
    assert (output_directory / "final_drift_report.json").exists()