# Forecasting engine

Weekly revenue forecasts for every store x category (product hierarchy level 30), compared across
seven models on identical series, cutoffs and horizon.

```bash
uv run demandpulse forecast prepare                      # panel + representative subset from Gold
uv run demandpulse forecast backtest seasonal_naive trailing_mean_13 lightgbm --run main
uv run demandpulse forecast backtest ets_damped holt_winters neuralprophet greykite --run main
uv run demandpulse forecast report --run main            # writes report.md and summary.json
```

The two isolated models need their environments first:
`uv sync --project envs/neuralprophet` and `uv sync --project envs/greykite`.

## Setup

- **Target.** Weekly revenue (`spend`) per store x category, from the Gold `fct_store_category_week`
  table. The panel is dense: a category with no sales in a week is an explicit zero, and weeks
  before a store opened are absent, so a new store is never mistaken for a collapsing one.
  23,591 series over 117 weeks.
- **Horizon and origins.** 13 weeks ahead. Rolling origins after weeks 78, 91 and 104, so the
  test windows are weeks 79-91, 92-104 and 105-117. Origins 85 and 98 are used only as a
  robustness check (see "Choosing the LightGBM feature set").
- **Eligibility.** A series is forecast at an origin if it has at least 65 weeks of history (a
  year plus the 13 weeks the growth features look back over), sold in at least 26 of the last 52
  weeks, and is still trading through the test window. The rules use only data up to the origin.
  About 20,400 series qualify at each origin. The 13 stores opened during the data period qualify
  only once they have enough history.
- **Representative subset.** Per-series statistical and neural models run on 240 series: 20 from
  each of 12 strata (store format x revenue tier, terciles of last-year revenue), drawn with a
  fixed seed from series eligible at every origin. Cheap models run on every eligible series and
  are also scored on the subset, so every model is compared on the same 240 series.
- **Negative forecasts** are clipped to zero for every model.

## Metrics

- **WAPE**: total absolute error divided by total actual revenue. It stays defined when single
  weeks sell nothing and weights errors by revenue. The primary metric.
- **Bias**: (forecast - actual) / actual in total. Positive means over-forecasting.
- **MASE**: mean absolute error over the in-sample error of a one-year seasonal naive forecast,
  per series. Below 1 beats that benchmark. Reported as a median across series.
- **Aggregated accuracy**: forecasts and actuals are summed to store, category or network level
  before measuring error. Errors partly cancel when summed, so these are higher than series level
  and show which decisions a model can support.
- **Paired bootstrap**: improvements over the seasonal naive baseline come with 95% intervals from
  resampling (origin, series) pairs 1,000 times. A difference is called real only if the interval
  excludes zero.

## Models

| model | what it is | scope |
|---|---|---|
| `seasonal_naive` | the same week one year earlier | all series |
| `trailing_mean_13` | flat line at the mean of the last 13 weeks | all series |
| `ets_damped` | exponential smoothing, additive damped trend, no seasonality | subset |
| `holt_winters` | additive damped-trend Holt-Winters with a yearly cycle | subset, final origin only (needs two full years) |
| `lightgbm` | one global gradient-boosted model, feature set v2 | all series |
| `lightgbm_v1` | the same with the first feature set, kept as an ablation | all series |
| `neuralprophet` | NeuralProphet with local trend and yearly seasonality per series, trained jointly | subset |
| `greykite` | Greykite Silverkite, one model per series | subset |

**LightGBM design.** Direct multi-horizon: a training row is (series, origin t, horizon h), with
features from weeks up to t and the target revenue in week t+h. Targets and revenue features are
divided by the series' mean over the 52 weeks before t, so one model learns from stores that differ
100-fold in size; the loss is L1 with sample weight equal to that scale, which makes the training
objective the revenue-weighted absolute error (the numerator of WAPE). At each backtest origin the
model is fit only on rows whose targets are observed by then. An inner validation on the final 13
weeks picks the number of trees before a refit on all rows. Features: recent lags and rolling
means, year-ago values around the target week, fiscal week, series and store scale, category and
network seasonal indices, and (v2) recent-versus-year-ago momentum at series, category and network
level plus medians.

**Isolated models.** NeuralProphet 0.9 needs pandas 2 and numpy 1; Greykite 1.1 needs pandas 1.5,
numpy 1.26 and Python 3.10. Each therefore has its own locked uv project under `envs/`, and the
main package talks to it through Parquet files and a subprocess. A worker that fails raises an
error with its output; nothing falls back silently.

## Leakage controls

- The backtest harness gives a model only `history(panel, origin)` plus the calendar of the target
  weeks (week index and fiscal week). Revenue after the origin is never passed in.
- Feature code works on a matrix sliced to the weeks observable at the origin.
- Tests alter every revenue value after the origin (multiplied by 1000) and require forecasts to
  be unchanged, for the harness, the global LightGBM model and the isolated-environment protocol.
- Training rows are checked to contain only targets observed by the cutoff, and eligibility
  ignores test-window values.
- The seasonal naive and ETS models only use a series' own past.

## Limitations

- 117 weeks is a short history: only the final origin has two full years for Holt-Winters, and
  there are only three non-overlapping 13-week test windows.
- The data is synthetic, so conclusions describe model behaviour on this dataset, not on a real
  retailer.
- Results for the expensive models rest on a 240-series subset; they are not full-scale training.
