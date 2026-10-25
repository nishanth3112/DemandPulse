# Tableau Public workbook: data export and build guide

The repo produces the data and this guide. You build and publish the workbook yourself in Tableau
Public (free). Nothing in the repo has opened or built a workbook, so the sheet recipes below are
a plan, not a verified result.

```bash
uv run demandpulse tableau export                         # data/app -> data/tableau (about 40 MB)
uv run demandpulse tableau export --snapshot D --out O    # other folders
```

The export reads only the dashboard snapshot in `data/app` (create it with `demandpulse app export`
while the lake still exists). It needs no AWS access, so it keeps working after the teardown in
[teardown.md](teardown.md). It writes 23 tidy CSV files (about 417,000 rows in total), a
`_data_dictionary.csv` listing every table, column, type and row count, and `tableau_export.json`.
`data/tableau/` is gitignored; regenerate it whenever the snapshot changes.

## Before you publish: this is public

Tableau Public cannot read private data and everything published is visible to anyone. The files
hold only aggregates of the synthetic dunnhumby dataset: no basket lines, no customer codes, no
product-level rows (a test checks that `cust_code`, `basket_id` and `prod_code` never appear). Two
checks are yours, because the repo cannot make them:

- **Dataset terms.** Read dunnhumby's terms of use for "Let's Get Sort-of-Real" and confirm that
  publishing derived aggregates is allowed. I have not verified this.
- **Tableau limits.** Confirm the current Tableau Public row and size limits. The largest file is
  `fact_forecast_store_week.csv` (195,780 rows, 18 MB), well under the limits as I know them.

## Files

Dimensions are small lookups. Facts have one row per the key shown. Model, comparison and target
columns come with a readable `*_label`; use the label in views and the code for filters.

| File | Rows | One row per | Join to |
|---|---|---|---|
| `dim_date` | 117 | `week_idx` | `week_idx` |
| `dim_store` | 761 | `store_code` | `store_code` |
| `dim_model` | 6 | `model` (label, type, whether it is the risk model) | `model` |
| `dim_comparison` | 3 | `comparison` (window label, week ranges) | `comparison` |
| `fact_store_week` | 88,555 | `store_code`, `week_idx` | store, date |
| `fact_region_benchmark` | 39 | `comparison`, `store_region` (`ALL` is the network) | comparison |
| `fact_store_scorecard` | 2,283 | `comparison`, `store_code` | store, comparison |
| `fact_revenue_bridge` | 2,322 entities x 8 steps | entity (`level` region or store) and `step` | comparison, store |
| `fact_category_benchmark` | 93 | `comparison`, `prod_code_30` | comparison |
| `fact_region_category_mix` | 1,116 | `comparison`, `store_region`, `prod_code_30` | comparison |
| `fact_trading_pattern` | 414 | `store_format`, `shop_weekday`, `shop_hour` | none |
| `fact_basket_mix` | 70,813 | week, format, basket type/size/mission/price sensitivity, loyalty flag | date (`shop_week`) |
| `fact_forecast_accuracy` | 149 | `run`, `model`, `slice`, `slice_value` | model |
| `fact_forecast_vs_baseline` | 9 | `run`, `model`, `baseline` | model |
| `fact_forecast_network_week` | 260 | `run`, `model`, `origin`, `week_idx` | model, date |
| `fact_forecast_store_week` | 195,780 | `run`, `model`, `origin`, `week_idx`, `store_code` | model, store, date |
| `fact_risk_store_window` | 37,650 | `model`, `origin`, `store_code`, `target_def` | store |
| `fact_risk_ranking` | 252 | `grain`, `model`, `score`, `target_def`, `k_fraction` | none |
| `fact_risk_growth` | 10 | `origin`, `target_def` | none |
| `fact_risk_recommendations` | 40 | `model`, `origin`, `store_code`, `target_def` | store |
| `fact_risk_scenarios` | 160 | `store_code`, `target_def`, `scenario` | store |
| `fact_rfm_segment` | 7 | `rfm_segment` | none |
| `fact_customer_cohort` | 459 | `cohort_period`, `periods_since_first` | none |

Notes on the shapes:

- **`fact_forecast_accuracy`** stacks four slices so one sheet can switch between them with a
  filter: `level` (series, store, category, network), `origin`, `horizon` (week buckets) and
  `format`. `origin` is filled only for the `origin` slice. RMSE, MASE and series counts exist for
  the series level and the origin slice only. The series level is the overall pooled accuracy.
- **`fact_revenue_bridge`** is long: base period, six changes, current period, with `step_order`
  and `step_kind` (`total` or `change`). On the real data base plus changes equals current for
  every one of the 2,322 entities, to within 1e-6.
- **`fact_risk_ranking`** covers both the store grain and the store x category grain, with the
  random and own-trend baselines as ordinary `model` values.
- Revenue is the dataset's `spend`. It has no currency, cost, margin or promotion data.

## Building the workbook

Use one data source per theme and relate tables on the keys above (Tableau relationships), or
connect each CSV and keep dashboards separate. Four dashboards mirror the Streamlit pages.

Palette (same as the app): teal `#0F766E` for the model or measure of interest, slate `#94A3B8`
for context, green `#15803D` and orange-red `#C2410C` only where up or down carries meaning,
indigo `#6366F1`, amber `#F59E0B`, pink `#EC4899` for the other models. Give each model the same
colour on every sheet. As far as I know Tableau Public has no dark mode, so build the light version.

### 1. Forecast performance

| Sheet | Data and recipe |
|---|---|
| Headline cards | `fact_forecast_accuracy`, slice `level`, value `Series (store x category)`; run parameter; best model's `wape`, and its cut versus Seasonal naive. |
| Leaderboard | Bars of `wape` by `model_label` for the chosen run, sorted ascending. Add `bias` and `mase_mean` as text. |
| Accuracy by level | Slice `level`, `slice_value` on columns, `wape` on rows, colour by `model_label`. Shows errors falling as forecasts are summed. |
| By horizon and format | Slices `horizon` and `format` as two small multiples. |
| Network forecast against actual | `fact_forecast_network_week`; filters run, origin, model; lines for `actual` and `forecast` by `week_start`. Do not start the axis at zero. |
| Store explorer | `fact_forecast_store_week` for the forecast, `fact_store_week` for history; filters store, origin, model. |
| Versus baseline | `fact_forecast_vs_baseline`: `improvement` with `ci_low` and `ci_high` as error bars, `win_rate` as text. |

Calculated fields for store or network rows: WAPE `SUM(ABS([actual]-[forecast]))/SUM([actual])` and
bias `(SUM([forecast])-SUM([actual]))/SUM([actual])`. Annotate: forecasts are historical backtests
from rolling origins; the ensemble was chosen after seeing these results and is not independently
validated.

### 2. Revenue and demand overview

| Sheet | Data and recipe |
|---|---|
| KPI tiles | `fact_region_benchmark`, `store_region = ALL`, `comparison` parameter: revenue, like-for-like growth, baskets, average basket value. |
| Weekly revenue | `fact_store_week`, `SUM(revenue)` by `week_start`, plus a 13-week moving average (table calculation). Filters for region and format. |
| What changed revenue | `fact_revenue_bridge`, `level = region`, region `ALL`: base period, the six changes, current period. |
| Region growth | `fact_region_benchmark`, `lfl_revenue_growth` by region, green/orange by sign. |
| Category contribution | `fact_category_benchmark`, `contribution_to_growth_pp` by `prod_code_30`. |
| Region x category mix | `fact_region_category_mix`, `mix_index` as a diverging colour centred on 100 and clipped at 70 and 130 (100 is the national share). |
| Trading pattern | `fact_trading_pattern`, weekday by hour heatmap per store format. |
| Basket missions | `fact_basket_mix`, `SUM(revenue)` by mission and basket type. |

Annotate: categories are product hierarchy level 30; the data has no product names; price and mix
cannot be separated because there are no list prices.

### 3. Revenue risk

| Sheet | Data and recipe |
|---|---|
| Stores at risk | `fact_risk_store_window`; filters model, origin, `target_label`; bars of `revenue_at_risk` for the top stores (fall back to `forecast_shortfall` where it is empty). |
| Where to look first | `fact_risk_recommendations`: store, gap, probability, dominant driver, `evidence` text. Origin 104 only. |
| How well the ranking works | `fact_risk_ranking`: `precision` by `model_label` and `score_label` at the chosen `k_fraction`, with the random and own-trend baselines. |
| Scenarios | `fact_risk_scenarios`, filtered by `scenario`: `gap_before` against `gap_after` per store. |

Annotate on the dashboard itself: targets are hypothetical (last year's revenue, flat or grown by
recent network growth, with no budgets in the data); every window is historical, so "missed" is
hindsight; the probabilities rank stores but are not calibrated chances; the LightGBM classifier
did not beat the forecast-based probability, and the ranking rests on three overlapping windows.

### 4. Customers

| Sheet | Data and recipe |
|---|---|
| Segments | `fact_rfm_segment`: `customer_share` against `revenue_share` by `rfm_segment`. |
| Segment detail | Same table as a text table. |
| Cohort retention | `fact_customer_cohort`: `periods_since_first` by `cohort_period`, colour `retention_rate`. |

Annotate: customer analysis covers loyalty-card baskets only (about 80% of revenue); the earliest
cohort mixes new and existing customers because cohorts can only be tracked from the first week.

## Refreshing

After rebuilding the lake outputs: `demandpulse app export`, then `demandpulse tableau export`,
then refresh the data sources in Tableau Public Desktop and re-publish the workbook.
