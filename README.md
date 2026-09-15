# Automated Data Drift Detector

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![CI](https://github.com/Griffin-MAYANK/automated-data-drift-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/Griffin-MAYANK/automated-data-drift-detector/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/runtime-Docker-2496ED.svg)](https://www.docker.com/)

Automated Data Drift Detector is a production-oriented Python package for comparing a reference dataset with a current dataset and identifying changes in feature distributions that may require operational attention. It is a drift detection system for data and machine-learning pipelines, not a fraud detector or prediction model.

The current demonstration uses the UCI Online Retail dataset. The project combines statistical testing, practical magnitude, semantic feature metadata, multiple-testing correction, validation, operational decisions, a CLI, reports, Docker, and continuous integration.

## Highlights

- Modular package under `src/drift_detector/`
- Explicit numerical, categorical, and cyclical feature metadata
- Kolmogorov-Smirnov and Wasserstein analysis for numerical features
- Chi-square and Total Variation Distance analysis for categorical features
- Benjamini-Hochberg False Discovery Rate correction
- Distinction between statistical significance and practical significance
- Reproducible command-line execution
- CSV and JSON operational reports
- Non-root Docker runtime
- GitHub Actions validation for tests and Docker image behavior

## Architecture

```text
Raw Dataset
	|
	v
Feature Engineering
	|
	v
Semantic Feature Metadata
	|
	v
Input Validation
	|
	v
Statistical Drift Tests
	|
	v
FDR Correction
	|
	v
Practical Magnitude
	|
	v
Operational Decision
	|
	+------> CSV Report
	|
	+------> JSON Report
	|
	v
CLI / Docker
```

## Dataset

The current demonstration uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail), which contains 541,909 transaction records covering 2010-12-01 through 2011-12-09.

The dataset is published by the UCI Machine Learning Repository under DOI [10.24432/C5BW33](https://doi.org/10.24432/C5BW33). UCI identifies the dataset as available under a CC BY 4.0 license. Obtain the data from the UCI repository and place the file at:

```text
data/Online Retail.xlsx
```

The dataset is intentionally not included in this GitHub repository. The `data/` directory is a local/runtime directory and is ignored by Git.

## Monitoring Features

Raw transaction data is transformed into these monitoring features:

| Feature | Semantic type |
| --- | --- |
| `quantity` | Numerical |
| `unit_price` | Numerical |
| `transaction_value` | Numerical |
| `transaction_hour` | Cyclical |
| `day_of_week` | Cyclical |
| `month` | Cyclical |
| `is_cancelled` | Categorical |
| `country` | Categorical |

The detector uses explicit semantic metadata before falling back to pandas dtype inference for unconfigured features. Cyclical features currently use the categorical drift-testing path, while retaining their semantic metadata for future dedicated circular-statistics support.

## Methodology

### Numerical features

- Two-sample Kolmogorov-Smirnov test
- Wasserstein distance
- Normalized Wasserstein magnitude
- Explicit asymptotic KS calculation for predictable behavior on production-scale samples

### Categorical features

- Chi-square test
- Total Variation Distance

### Operational analysis

The detector applies Benjamini-Hochberg False Discovery Rate correction, then combines:

- Statistical significance
- Distribution distance and practical magnitude
- Severity classification
- Final operational decision

A p-value alone is not enough for operational monitoring. Very large datasets can produce tiny p-values for very small distribution changes. The detector therefore considers practical magnitude in addition to statistical significance before producing a decision.

Possible operational states are:

- `ALERT`
- `INVESTIGATE`
- `MONITOR`
- `NO_DRIFT`
- `INSUFFICIENT_DATA`

Thresholds are project configuration values. They are not universal thresholds or industry standards.

## Current Real-Data Result

Using the split date `2011-07-01`:

- Reference rows: `245,903`
- Current rows: `296,006`
- Features analyzed: `8`
- Overall result: `ALERT`

Current semantic interpretation:

| Feature | Decision |
| --- | --- |
| `month` | `ALERT` |
| `transaction_hour` | `INVESTIGATE` |
| `quantity` | `MONITOR` |
| `unit_price` | `MONITOR` |
| `transaction_value` | `MONITOR` |
| `day_of_week` | `MONITOR` |
| `country` | `MONITOR` |
| `is_cancelled` | `MONITOR` |

These results reflect the current dataset, split, and project configuration. They should not be interpreted as universal business thresholds.

## Installation

Python 3.12 or newer is required.

```bash
git clone https://github.com/Griffin-MAYANK/automated-data-drift-detector.git
cd automated-data-drift-detector

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Obtain the UCI dataset separately and place `Online Retail.xlsx` inside `data/` before running the real-data example.

## CLI Usage

The installed command is `drift-detector`.

```bash
drift-detector \
  --data "data/Online Retail.xlsx" \
  --split-date 2011-07-01
```

View all supported arguments:

```bash
drift-detector --help
```

Supported arguments:

- `--data`: required input dataset path
- `--split-date`: required reference/current split date
- `--output-dir`: report directory, defaulting to `reports`
- `--significance-threshold`: statistical threshold, defaulting to `0.05`
- `--min-samples`: minimum samples per feature, defaulting to `30`

The CLI returns exit code `0` when drift analysis completes successfully, even when drift is detected. Invalid input and runtime failures return a non-zero exit code. Drift is a business result, not a program failure.

## Monitoring Dashboard

The Streamlit monitoring dashboard presents historical drift runs stored by the Stage 3 SQLite persistence layer. It is read-only and does not require the FastAPI server to be running.

Install the optional dashboard dependency:

```bash
python -m pip install -e ".[dashboard]"
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard displays the latest operational status, alert and investigation counts, latest feature results, recent historical runs, status trends, and feature-level magnitude history. It reads through `DriftRepository` and does not duplicate SQL or detection logic.

By default it reads:

```text
reports/drift_history.db
```

To use another database, set the same configuration variable used by the API and service:

```bash
export DRIFT_DETECTOR_DATABASE_PATH=/path/to/drift_history.db
streamlit run dashboard/app.py
```

When no database or historical runs exist, the dashboard initializes the database safely and displays an empty state instead of failing. The dashboard uses built-in Streamlit charts and gracefully falls back from normalized magnitude to magnitude for categorical features.

## REST API

The project also provides a FastAPI layer around the same production drift detection engine. The API is intended for local or containerized batch analysis and does not duplicate feature engineering or detector logic.

Install the package with its runtime dependencies:

```bash
python -m pip install -e ".[test]"
```

Start the API on `0.0.0.0:8000`:

```bash
drift-detector-api
```

The API provides:

- `GET /health`: returns `{"status": "healthy"}`
- `GET /metadata`: returns package/API versions, supported feature types, and monitoring features
- `POST /detect`: runs drift detection and returns the operational summary, safe report names, and feature results

Interactive Swagger UI is available at [http://localhost:8000/docs](http://localhost:8000/docs), ReDoc is available at [http://localhost:8000/redoc](http://localhost:8000/redoc), and the OpenAPI document is available at [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json).

Example request:

```json
{
	"data_path": "data/Online Retail.xlsx",
	"split_date": "2011-07-01",
	"significance_threshold": 0.05,
	"min_samples": 30,
	"output_directory": "reports"
}
```

Example response structure:

```json
{
	"dataset": "Online Retail.xlsx",
	"split_date": "2011-07-01",
	"reference_rows": 245903,
	"current_rows": 296006,
	"number_of_features": 8,
	"overall_status": "ALERT",
	"alert_count": 1,
	"investigate_count": 1,
	"monitor_count": 6,
	"no_drift_count": 0,
	"insufficient_data_count": 0,
	"report_paths": {
		"csv": "final_drift_report.csv",
		"json": "final_drift_report.json"
	},
	"features": []
}
```

Run the API in Docker while keeping the image non-root and mounting local data and reports:

```bash
docker run --rm -p 8000:8000 \
	-v "$(pwd)/data:/app/data:ro" \
	-v "$(pwd)/reports:/app/reports" \
	drift-detector:1.0.0 \
	drift-detector-api
```

## Reports

The detector produces:

```text
reports/final_drift_report.csv
reports/final_drift_report.json
```

Reports are runtime artifacts and are intentionally ignored by Git. They can be redirected with `--output-dir`.

## Docker

The Docker image is `drift-detector:1.0.0`. It uses Python 3.12 slim, installs the package from project metadata, runs as a non-root user, and uses `/app` as its working directory.

The image provides `/app/data` and `/app/reports` as mount points. The dataset and generated reports are not baked into the image.

Build the image:

```bash
docker build -t drift-detector:1.0.0 .
```

Check the CLI:

```bash
docker run --rm drift-detector:1.0.0 --help
```

Run against the local dataset with mounted data and report directories:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/reports:/app/reports" \
  drift-detector:1.0.0 \
  --data "/app/data/Online Retail.xlsx" \
  --split-date 2011-07-01
```

## Package Modules

The installable package is located at `src/drift_detector/`:

- `config.py`: drift configuration, semantic feature types, and feature metadata
- `detector.py`: validation orchestration, drift analysis, FDR correction, and operational decisions
- `features.py`: raw retail validation, feature engineering, and reference/current splitting
- `reporting.py`: operational summaries and CSV/JSON export and reload utilities
- `statistics.py`: numerical and categorical statistical tests and severity classification
- `validation.py`: reference/current dataset validation
- `cli.py`: standard-library argument parsing and production pipeline orchestration
- `api_models.py`: Pydantic request and response contracts for the REST API
- `service.py`: API-facing orchestration of the existing production pipeline
- `api.py`: FastAPI application, routes, error handling, and Uvicorn startup
- `database.py`: SQLite schema initialization and shared database configuration
- `repository.py`: Transaction-safe historical run persistence and read access

## Testing

Run the complete suite:

```bash
python -m pytest -q
```

The current suite has `155 passed` tests covering:

- Statistical calculations
- Numerical drift
- Categorical drift
- Feature engineering
- Semantic feature metadata
- Dataset validation
- Detector behavior
- Reporting
- CLI behavior
- Historical SQLite storage and repository transactions
- Dashboard data transformations
- Real-data end-to-end pipeline behavior

## Continuous Integration

The [GitHub Actions workflow](.github/workflows/ci.yml) runs on pushes to `main` and pull requests targeting `main`. It currently:

- Installs Python 3.12 and caches pip dependencies
- Installs the package with its test dependencies
- Runs the full pytest suite
- Builds the Docker image
- Verifies CLI help
- Verifies non-root container execution
- Verifies `/app/data` and `/app/reports`
- Verifies that project data and reports are not baked into the image

The current workflow has successfully passed. It does not download the UCI dataset and does not push or deploy an image.

## Scheduled Automation

The [Scheduled Drift Detection workflow](.github/workflows/scheduled-drift-detection.yml) runs the existing `drift-detector` CLI automatically once daily at `02:00 UTC`. It can also be started manually from the GitHub Actions `workflow_dispatch` button.

The workflow installs the packaged project with Python 3.12, downloads the public UCI Online Retail dataset at runtime from the official UCI archive, extracts it to `data/Online Retail.xlsx`, verifies that the file is non-empty, and runs:

```bash
DRIFT_DETECTOR_DATABASE_PATH=reports/drift_history.db \
drift-detector \
	--data "data/Online Retail.xlsx" \
	--split-date "2011-07-01" \
	--output-dir reports
```

The dataset is not committed to Git. The workflow depends on the availability of the official UCI dataset URL at runtime. The generated CSV report, JSON report, and SQLite history database are uploaded as temporary workflow artifacts named with the GitHub Actions run number. Because GitHub-hosted runners are temporary, artifact-based history is not a permanent production database. Persistent cloud storage will be addressed in a later deployment stage.

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── scheduled-drift-detection.yml
├── data/                         # Runtime/local dataset directory; not committed
├── notebooks/
│   └── 01_understanding_data_drift.ipynb
├── reports/                      # Runtime/local report directory; not committed
├── dashboard/
│   ├── __init__.py
│   ├── app.py
│   ├── components.py
│   └── data.py
├── src/
│   └── drift_detector/
│       ├── __init__.py
│       ├── cli.py
│       ├── database.py
│       ├── config.py
│       ├── detector.py
│       ├── features.py
│       ├── reporting.py
│       ├── repository.py
│       ├── statistics.py
│       └── validation.py
├── tests/
│   ├── test_cli.py
│   ├── test_dashboard_data.py
│   ├── test_database.py
│   ├── test_detector.py
│   ├── test_features.py
│   ├── test_pipeline.py
│   ├── test_reporting.py
│   ├── test_repository.py
│   ├── test_statistics.py
│   └── test_validation.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Limitations

- Cyclical features currently use categorical drift testing rather than dedicated circular statistical methods.
- Thresholds are configurable and project-specific.
- Drift detection does not prove model performance degradation.
- Detected drift does not automatically mean retraining is required.
- The current project performs batch reference-versus-current analysis.
- There is no production database, scheduler, REST API, dashboard, or cloud deployment yet.

## Roadmap

### Phase 1 — Current

- Statistical drift engine
- Semantic feature metadata
- Dataset validation
- CLI
- CSV and JSON reports
- Docker runtime
- GitHub Actions CI

### Phase 2 — Future

- REST API
- Scheduled monitoring
- Persistent drift history
- Dashboard

### Phase 3 — Future

- Model performance monitoring
- Feature-level alert history
- Production data connectors
- Cloud deployment

### Phase 4 — Future

- Dedicated cyclic drift statistics
- Automated retraining or retraining recommendations
- Monitoring integrations

## Engineering Highlights

- Modular architecture with clear package boundaries
- Testable statistical and orchestration components
- Semantic typing instead of relying only on pandas dtype
- Multiple-testing correction for feature-level decisions
- Reproducible CLI execution
- Dockerized non-root runtime
- CI validation for Python and Docker behavior
- Separation of local data and reports from the application image

## License

No license file is currently included in this repository. Licensing can be added separately if appropriate.

## Author

Mayank Sharma

[GitHub repository](https://github.com/Griffin-MAYANK/automated-data-drift-detector)
