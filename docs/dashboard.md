# Executive dashboard

A Streamlit app over the Gold marts, the forecast comparison and the revenue-risk outputs.

```bash
uv run demandpulse app export                           # lake -> data/app (about 5 MB, once)
uv run streamlit run src/demandpulse/app/main.py        # open the dashboard
```

`DEMANDPULSE_APP_DATA` points the app at a different snapshot folder (default `data/app`).

## How it gets its data

The app never touches S3, DuckDB or the basket fact, and holds no credentials. `app export`
copies a fixed list of small tables out of the lake into a local folder of Parquet files plus a
`snapshot.json` manifest, and the app reads only that folder through `app/snapshot.py`.

- **Exported:** the Gold dimensions and marts (store scorecard, region and category benchmarks,
  revenue variance, RFM and cohorts, trading pattern, basket mix), weekly store revenue / baskets
  / units, forecast comparison tables for the `main` and `robustness` runs, network-level and
  per-store forecast against actual by week (store totals only, categories summed away), and every `risk/` output.
- **Never exported:** `fct_basket` and `fct_store_category_week`. A test checks the snapshot
  contains neither, and another checks the app modules cannot import `boto3`, `duckdb`, the
  storage, ingest or warehouse layers, or mention `fct_basket`.
- The manifest is written last, so a failed export leaves a folder the app refuses to open and
  points at the command that fixes it.

The export code lives in `app/export.py` and is never imported by the app itself.

## Pages

| Page | What it shows |
|---|---|
| Executive overview | Network revenue, like-for-like growth, baskets and basket value for a chosen comparison window; weekly trend with region/format filters; revenue bridge (traffic, basket size, price/mix, trading weeks, new and closed stores); category contribution; basket missions; trading pattern heatmap. |
| Stores and regions | Region like-for-like growth and category mix; filterable store scorecard (region, format, like-for-like only, search); growth scatter; single-store drilldown with weekly history and revenue-driver split. |
| Forecast accuracy | Model leaderboard per run, accuracy at series / store / category / network level, network forecast against actual for any origin and model. Store explorer: pick any store (filter by region and format) and origin to see its weekly revenue history, then actual against each chosen model's forecast, with the store's WAPE and bias per model and where it ranks among all stores. |
| Revenue risk | Hypothetical targets by origin and model; riskiest stores with probability, revenue below target and (for the backtest) what really happened; ranking evidence against the own-trend and random baselines; classifier verdict and reliability; scenario planner. |
| Customers | RFM segments and cohort retention, stated as loyalty-card baskets only, with the loyalty share of revenue shown. |

The comparison window (last 13 weeks against a year earlier, against the previous 13 weeks, or
52 weeks against the prior 52) is a sidebar control shared by the pages that use it.

## Design notes

- All table logic is in `app/views.py` and `app/risk_views.py` as plain Polars functions, and
  chart builders are in `app/charts.py` and `app/risk_charts.py`, so each is unit tested without
  a browser. `app/pages.py` only lays them out.
- The risk page repeats the framing of [revenue-risk.md](revenue-risk.md): targets are
  hypothetical, the windows are historical, "actually missed" is hindsight, the probabilities
  rank stores but are not calibrated chances, and the classifier verdict is computed from the
  stored Brier scores rather than written in advance.
- Revenue is the dataset's `spend`. It has no currency, cost, margin or promotions, so nothing
  on the dashboard is labelled profit, margin or promotion lift.
- Colour carries meaning only for growth up (green) and down (orange-red), and values are always
  labelled as well.

## Testing

Streamlit's `AppTest` runs every page against a snapshot exported from a synthetic lake (the real
risk pipeline plus fake Gold marts with the real schemas). The tests drive the widgets: window
selection, filters, origin and target changes, the scenario sliders, and the missing-snapshot
error. They also run the full app entry point.

## Limitations

- The store explorer shows store-level totals only. Store x category forecasts are not exported, so
  there is no category-level series view. Stores with no revenue in the forecast weeks (closures)
  show undefined error rates, and stores with no forecast rows are not listed.
- The tests cover behaviour and data, not appearance. Layout, spacing and chart readability have
  not been checked in a browser by the automated tests, so look at each page when you first run it.
- The snapshot is a point-in-time copy. Re-run `app export` after rebuilding Gold or the risk
  outputs; the sidebar shows the snapshot date.
- Store x category risk results appear as backtest evidence only. The store x category windows
  are not exported, so there is no category-level risk table or recommendation list.
- Store-level revenue-driver splits need a full 13 trading weeks in both windows, so a store that
  was closed for part of the window shows no split.
