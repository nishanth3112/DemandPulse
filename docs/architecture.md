# Architecture

DemandPulse turns a 307-million-row retail transaction file into KPIs, forecasts, a revenue-risk
ranking and a dashboard, on a 16 GB laptop and a few dollars of cloud storage.

## Constraints that shaped it

- **Memory and disk.** About 16 GB of RAM and 50 GB of disk, against about 307M basket lines. Nothing
  loads the whole dataset: work is per week or per partition.
- **Cost.** One private S3 bucket inside the AWS Free Tier credit, and no other paid service.
- **Honesty about the data.** It is synthetic and historical, with no prices, costs or promotions,
  so the design never offers margin or promotion analysis.
- **No leakage.** Forecasts, probabilities and classifiers only see data available at their origin.

## Data flow

```
dunnhumby zips (public CDN)
   | HTTP range reads, one week at a time; the zip is never written to disk
   v
Bronze   typed weekly Parquet, one file per shop_week, with a manifest      (S3)
   v
Silver   cleaned: exact duplicates dropped and counted, loyalty flag added  (S3)
   |   quality checks reconcile each layer to the one before
   v
Gold     DuckDB builds dimensions and marts from Silver, 24 validation checks (S3)
   |
   +--> forecasting panel (store x category x week) --> backtests --> forecasts
   |                                                         |
   |                                                         v
   |                                    revenue risk: targets, shortfall probability,
   |                                    ranking backtest, classifier, scenarios  (S3: risk/)
   v                                                         |
snapshot export  (small marts + forecast + risk tables, about 5 MB, local)  <--+
   v
Streamlit dashboard (reads only the snapshot, no credentials)
```

`demandpulse demo` runs everything from Silver onward on synthetic data in a local folder, with no
AWS, which is also what the end-to-end test does.

## Layers and modules

| Path | Responsibility |
|---|---|
| `ingest/` | Source registry, HTTP range reader for zips, Bronze and Silver builders with resumable manifests |
| `quality.py` | 22 per-source checks reconciling Bronze and Silver |
| `warehouse/` | DuckDB runner, SQL marts in `sql/`, the 24 Gold validation checks |
| `forecasting/` | Panel and eligibility rules, backtest harness, metrics, models, runner and report |
| `forecasting/models/` | Baselines, ETS, LightGBM; Greykite and NeuralProphet through subprocess workers |
| `risk/` | Targets, shortfall probability, ranking backtest, classifier, scenarios, recommendations |
| `app/` | Snapshot export and loader, view logic, chart builders, Streamlit pages |
| `synthetic.py`, `demo.py` | Invented Silver data and the end-to-end demo and benchmark |
| `storage.py`, `config.py` | One `Store` interface over S3 and a local folder; settings from the environment |
| `cli.py` | The `demandpulse` command |

## The lake

```
bronze/source=<name>/shop_week=<YYYYWW>/        silver/source=<name>/shop_week=<YYYYWW>/
gold/<table>.parquet   gold/fct_basket/shop_week=*/   gold/_manifest.json
quality/<source>/report.{md,json}
forecast/panel.parquet  forecast/subset.parquet  forecast/runs/<run>/{forecasts,scales,meta}/
risk/*.parquet  risk/classifier_*.json  risk/report.md
```

Every stage talks to a `Store`, so the same code writes to S3 for real runs and to a temporary
folder in tests.

## Decisions and their trade-offs

- **Stream zips instead of downloading them.** The raw zips are never stored, and ingestion is
  resumable by week, at the cost of depending on the CDN supporting HTTP range requests.
- **Parquet in S3 plus DuckDB for the marts.** DuckDB reads and writes S3 directly and is fast on a
  laptop. Large steps run one week at a time because a single query over all weeks ran out of
  memory. The cost is a more fiddly build than one big SQL statement.
- **Global LightGBM plus cheap baselines for the full panel; slower models on a 240-series
  subset.** Per-series ETS, Greykite and NeuralProphet on about 20,000 series would have taken far
  longer than the deadline allowed (Greykite alone took 260 to 310 seconds per origin for 240
  series). The report states which scope each model ran on, and never extends a result past it.
- **Greykite and NeuralProphet live in their own locked environments.** Greykite pins pandas
  1.5.3 and NumPy 1.26, which the main environment cannot use. The main package talks to them
  through Parquet files and a subprocess.
- **A snapshot for the dashboard instead of live queries.** The app needs no credentials, starts
  fast, and cannot scan the basket fact, which is enforced by a test. The cost is that data is a
  point-in-time copy and must be re-exported.
- **Hypothetical targets.** There are no budgets, so targets are last year's revenue, flat or
  grown by recent network growth. This is stated wherever a target appears.

## Correctness and reproducibility

- Reconciliation at every layer: Bronze to manifest, Silver to Bronze, Gold to Silver.
- Seeds fixed for sampling and models, and polars group and join results are sorted before any
  seeded resampling because their row order is not stable.
- Locked dependencies (`uv.lock`) and `uv sync --locked` in CI and in the Docker image.
- Leakage controls are tested: each feature, growth rate and error distribution is checked to be
  unchanged when data after the origin changes.

## Testing and delivery

- About 500 unit and end-to-end tests run in CI. Coverage is gated at 90% (it was 96% when the gate
  was added). Tests marked `integration` hit the real CDN and are opt-in.
- The end-to-end test builds a synthetic lake, runs every stage, validates Gold with its 24 checks,
  and renders every dashboard page with Streamlit's `AppTest`.
- CI also builds the dashboard image, runs the demo inside it and checks the server's health.
- Branch protection: pull request and passing CI required on `main`, no force push.

## Security

- No credentials, data, models or generated artifacts in Git; a hygiene check enforces it.
- The S3 bucket is private and the pipeline's IAM user is limited to it.
- The dashboard and its container hold no credentials, run as a non-root user and never touch S3 or
  the basket fact.

## Logging and observability

Every CLI command logs its start, duration and failures. `--log-format json` (or
`DEMANDPULSE_LOG_FORMAT=json`) writes one JSON object per line to stderr, and
`DEMANDPULSE_LOG_LEVEL` sets the level. Long steps log per-stage timings, and `demandpulse benchmark`
times each stage on synthetic data.
