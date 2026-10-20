# DemandPulse

Retail revenue and demand intelligence on the dunnhumby *Let's Get Sort-of-Real* dataset: data engineering, SQL analytics, demand forecasting, revenue-shortfall risk ranking and an executive dashboard.

The dataset is synthetic and historical. Results describe the dataset and historical backtests, not live retail operations or verified financial gains.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Repository foundation and CI | in review |
| 1 | Data engineering and quality | planned |
| 2 | SQL analytics and business KPIs | planned |
| 3 | Forecasting and model evaluation | planned |
| 4 | Revenue risk and decision intelligence | planned |
| 5 | Streamlit dashboard | planned |
| 6 | Production hardening and release | planned |

## Dataset

Source: [dunnhumby source files](https://www.dunnhumby.com/source-files/). About 300M transaction rows, 47M baskets, ~500K customers, ~5,000 products, ~760 stores and 117 fiscal weeks, delivered as weekly files in nine zips plus samples.

It is one denormalized fact table (one row per product per basket) plus a `time.csv` week calendar. There are no product names, unit prices, costs or promotion flags, so revenue means `spend` and margin or promotion analysis is out of scope. Baskets without a loyalty card have no customer code.

Data is never committed. The pipeline streams the public zips over HTTP and writes Parquet to a private S3 bucket (or a local directory for tests and offline work).

## Stack

Python 3.12, uv, Polars, DuckDB, PyArrow, Parquet, LightGBM, statsmodels, Streamlit, Plotly, pytest, Ruff, GitHub Actions. Greykite and NeuralProphet are planned for the forecasting phase and are not installed yet.

## Setup

```bash
uv sync --locked
cp .env.example .env     # then fill in the values
uv run pytest
```

Configuration is read from environment variables (`uv run --env-file .env ...`). AWS credentials come from the named profile (`AWS_PROFILE`, default `demandpulse`) and are never stored in this repo.

| Variable | Purpose |
|---|---|
| `AWS_PROFILE`, `AWS_REGION` | AWS profile and region (default `us-east-1`) |
| `DEMANDPULSE_S3_BUCKET` | Private bucket holding Bronze/Silver Parquet |
| `DEMANDPULSE_LOCAL_DATA_ROOT` | Local directory instead of S3 (set one, not both) |

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m demandpulse.hygiene   # no data files or secrets in Git
uv run pytest
```

Work happens on one branch per phase, merged to `main` through a pull request after CI passes. CI runs the checks above on every pull request and never downloads the full dataset. Tests marked `integration` need network or AWS and are excluded from CI.

## Layout

```
src/demandpulse/   settings, logging, repository hygiene guard
tests/             unit and smoke tests
.github/           CI workflow and PR template
```
