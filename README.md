# DemandPulse

Retail revenue and demand intelligence on the dunnhumby *Let's Get Sort-of-Real* dataset: data engineering, SQL analytics, demand forecasting, revenue-shortfall risk ranking and an executive dashboard.

The dataset is synthetic and historical. Results describe the dataset and historical backtests, not live retail operations or verified financial gains.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Repository foundation and CI | merged |
| 1 | Data engineering and quality | merged |
| 2 | SQL analytics and business KPIs | merged |
| 3 | Forecasting and model evaluation | merged |
| 4 | Revenue risk and decision intelligence | merged |
| 5 | Streamlit dashboard | merged |
| 6 | Production hardening and release | in review |

## Try it in a few minutes (no AWS, no download)

```bash
uv sync --locked
uv run demandpulse demo --out data/demo         # synthetic data through every stage, with timings
DEMANDPULSE_APP_DATA=data/demo/app uv run streamlit run src/demandpulse/app/main.py
```

`demo` writes invented data with the real Silver schema, builds the Gold marts, runs three
forecasters and their average, scores revenue risk and exports the dashboard snapshot. It takes a
few seconds. `uv run demandpulse benchmark` times each stage at several data sizes. The synthetic data
shows the machinery works; it says nothing about the real dataset.

With Docker, the same demo runs in the dashboard image:

```bash
docker build -t demandpulse .
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD/data/demo:/demo" \
  --entrypoint demandpulse demandpulse demo --out /demo/lake
docker run --rm -p 8501:8501 -v "$PWD/data/demo/lake/app:/data/app:ro" demandpulse
```

## Dataset

Source: [dunnhumby source files](https://www.dunnhumby.com/source-files/). About 300M transaction rows, 47M baskets, ~500K customers, ~5,000 products, ~760 stores and 117 fiscal weeks, delivered as weekly files in nine zips plus samples.

It is one denormalized fact table (one row per product per basket) plus a `time.csv` week calendar. There are no product names, unit prices, costs or promotion flags, so revenue means `spend` and margin or promotion analysis is out of scope. Baskets without a loyalty card have no customer code.

Data is never committed. The pipeline streams the public zips over HTTP and writes Parquet to a private S3 bucket (or a local directory for tests and offline work).

## Data pipeline

```bash
uv run demandpulse sources
uv run demandpulse run sample-5k-customers      # ingest -> Silver -> quality report
uv run demandpulse --local-root /tmp/lake run sample-5k-customers   # no AWS needed
```

Zips are streamed over HTTP and converted to Bronze and Silver Parquet one week at a time, with
resumable manifests and a reconciliation report. Details, verified data findings and results
are in [docs/data-pipeline.md](docs/data-pipeline.md).

## Analytics warehouse (Gold)

```bash
uv run demandpulse gold build        # DuckDB marts over Silver, then validation checks
uv run demandpulse gold validate     # re-run the checks against the built marts
```

Dimensions (date, store, product hierarchy, customer) and marts for store and category KPIs,
regional and peer benchmarking, like-for-like growth, customer RFM and cohorts, basket mix and
revenue variance decomposition. KPI definitions, method notes and results are in
[docs/gold-layer.md](docs/gold-layer.md).

## Forecasting

```bash
uv run demandpulse forecast prepare              # weekly store x category panel from Gold
uv run demandpulse forecast backtest all --run main
uv run demandpulse forecast report --run main
```

Eight forecasters (seasonal naive, trailing mean, ETS, Holt-Winters, global LightGBM,
NeuralProphet, Greykite and equal-weight ensembles) are compared on identical series, cutoffs and
a 13-week horizon, with leakage guards and bootstrap confidence intervals. NeuralProphet and
Greykite run in their own locked environments under `envs/`. Method, results and caveats are in
[docs/forecasting.md](docs/forecasting.md).

## Revenue risk

```bash
uv run demandpulse risk run                      # score forecasts against hypothetical targets
```

Stores and store x category series are ranked by how likely they are to miss a revenue target
over the next 13 weeks, using only data known at each forecast origin, and the ranking is scored
against what happened (precision@K, recall@K, lift, shortfall captured) with an own-trend baseline
and an exact random baseline. A calibrated LightGBM classifier is evaluated as an exploratory
check, and a scenario planner shows what driver changes would close a gap. Targets are
hypothetical and all windows are historical. Method, results and caveats are in
[docs/revenue-risk.md](docs/revenue-risk.md).

## Dashboard

```bash
uv run demandpulse app export                      # copy the small marts and outputs from the lake
uv run streamlit run src/demandpulse/app/main.py   # executive overview, stores, forecasts, risk, customers
```

The app reads only a local snapshot (about 4 MB) and holds no credentials; it never reads the
basket fact. Pages, design and limitations are in [docs/dashboard.md](docs/dashboard.md).

## Documentation

| Doc | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Data flow, layers, lake layout, design decisions and trade-offs |
| [docs/data-quality.md](docs/data-quality.md) | Every check, and what the full run found |
| [docs/model-evaluation.md](docs/model-evaluation.md) | How well each model worked and what is not established |
| [docs/case-study.md](docs/case-study.md) | A worked business example, labelled observed, backtest, scenario and hypothetical |
| [docs/teardown.md](docs/teardown.md) | How to remove the AWS resources |

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
| `DEMANDPULSE_LOG_LEVEL`, `DEMANDPULSE_LOG_FORMAT` | Log level, and `text` or `json` (one object per line on stderr) |
| `DEMANDPULSE_APP_DATA` | Folder the dashboard reads (default `data/app`) |

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m demandpulse.hygiene   # no data files or secrets in Git
uv run pytest
```

Work happens on one branch per phase, merged to `main` through a pull request after CI passes. CI runs the checks above (with a 90% coverage gate) on every pull request, builds and smoke-tests the dashboard image, and never downloads the full dataset. Tests marked `integration` hit the real CDN and are excluded by default and in CI; run them with `uv run pytest -m integration`.

## Layout

```
src/demandpulse/   settings, storage, CLI, quality checks, hygiene guard
src/demandpulse/ingest/   source registry, remote zip reader, Bronze and Silver
src/demandpulse/warehouse/   DuckDB runner, SQL marts (sql/), validation checks
src/demandpulse/forecasting/   panel, metrics, backtest harness, models, runner and report
src/demandpulse/risk/   targets, shortfall probability, ranking backtest, classifier, scenarios, runner
src/demandpulse/app/   snapshot export and loader, view logic, charts and Streamlit pages
src/demandpulse/synthetic.py, demo.py   invented Silver data, the end-to-end demo and benchmark
envs/              isolated locked environments for NeuralProphet and Greykite
config/            dunnhumby source registry
docs/              pipeline, Gold, forecasting, risk, dashboard, architecture and quality docs
tests/             unit tests (CI) and opt-in integration tests
.github/           CI workflow and PR template
```
