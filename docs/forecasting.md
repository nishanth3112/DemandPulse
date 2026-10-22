# Forecasting engine

Weekly revenue forecasts for every store x category (product hierarchy level 30), compared across
seven models on identical series, cutoffs and horizon.

```bash
uv run demandpulse forecast prepare                      # panel + representative subset from Gold
uv run demandpulse forecast backtest seasonal_naive trailing_mean_13 lightgbm lightgbm_v2 --run main
uv run demandpulse forecast backtest ets_damped holt_winters neuralprophet greykite --run main
uv run demandpulse forecast combine ensemble seasonal_naive trailing_mean_13 lightgbm --run main
uv run demandpulse forecast report --run main            # writes report.md and summary.json
uv run demandpulse forecast backtest seasonal_naive trailing_mean_13 lightgbm lightgbm_v2 \
    --run robustness --origins 85,98                     # origins not used to choose a feature set
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
| `holt_winters` | additive damped-trend Holt-Winters with a yearly cycle | subset series with two full years of history (238 of 240), final origin only |
| `lightgbm` | one global gradient-boosted model, feature set v1 (adopted) | all series |
| `lightgbm_v2` | the same with added momentum features; not adopted, kept as an ablation | all series |
| `neuralprophet` | NeuralProphet with local trend and yearly seasonality per series, trained jointly | subset |
| `greykite` | Greykite Silverkite, one model per series | subset |
| `ensemble` | equal-weight mean of seasonal naive, trailing mean and LightGBM. Added after seeing results | all series |
| `ensemble_baselines` | equal-weight mean of seasonal naive and trailing mean. Added after seeing results | all series |

**LightGBM design.** Direct multi-horizon: a training row is (series, origin t, horizon h), with
features from weeks up to t and the target revenue in week t+h. Targets and revenue features are
divided by the series' mean over the 52 weeks before t, so one model learns from stores that differ
100-fold in size; the loss is L1 with sample weight equal to that scale, which makes the training
objective the revenue-weighted absolute error (the numerator of WAPE). At each backtest origin the
model is fit only on rows whose targets are observed by then. An inner validation on the final 13
weeks picks the number of trees before a refit on all rows. Features: recent lags and rolling
means, year-ago values around the target week, fiscal week, series and store scale, and category
and network seasonal indices. Feature set v2 adds recent-versus-year-ago momentum at series,
category and network level plus medians.

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

## Results

All numbers come from `docs/reports/forecast-main.md` (origins 78, 91, 104) and
`docs/reports/forecast-robustness.md` (origins 85 and 98), generated by the commands above on
the full dataset. They describe this synthetic dataset and 117 weeks of history. Three
non-overlapping test windows is little evidence: model rankings change from one window to
the next, so read differences with the confidence intervals, not the point estimates.

### 1. No single model is best

Series-level WAPE on the 240-series subset, by origin:

| model | origin 78 | origin 91 | origin 104 |
|---|---|---|---|
| seasonal naive | 21.6% | 22.1% | 18.2% |
| trailing mean (13 wk) | 22.5% | 18.7% | 15.5% |
| ETS (damped trend) | 21.8% | 18.1% | 16.1% |
| Greykite | 22.5% | 17.9% | 14.9% |
| NeuralProphet | 23.8% | 23.4% | 18.2% |
| LightGBM | 19.8% | 15.9% | 18.1% |
| ensemble (3-way mean) | 18.3% | 17.1% | 14.5% |

LightGBM is best at origins 78 and 91 and mediocre at 104; Greykite is the opposite. Pooled over
the three origins on the subset, the improvement over seasonal naive with a 95% paired-bootstrap
interval is: ensemble 19.4% [15.9, 22.4], LightGBM 12.8% [7.8, 17.5], Greykite 10.4% [5.2, 15.1],
damped ETS 9.6% [4.5, 14.4], trailing mean 8.4% [3.1, 13.5]. **NeuralProphet is not better than
seasonal naive** (-5.6% [-11.7, 0.1]) and **Holt-Winters is not either** (-5.6% [-14.5, 1.1] on
the 238 series it can fit): with about two years of weekly data, 52 seasonal parameters are
estimated from two observations each.

### 2. Equal-weight averages were the most accurate and the most stable

On the full panel (about 20,400 series per origin), pooled over the three origins:

| model | WAPE | bias | improvement over seasonal naive [95% CI] |
|---|---|---|---|
| ensemble (naive + trailing + LightGBM) | 17.8% | -1.9% | 18.2% [17.8, 18.7] |
| ensemble_baselines (naive + trailing) | 18.2% | +0.3% | 16.1% [15.7, 16.5] |
| LightGBM | 19.3% | -6.5% | 11.3% [10.5, 11.9] |
| trailing mean | 20.9% | -0.1% | 3.8% [2.9, 4.7] |
| seasonal naive | 21.7% | +0.7% | |

**These two averages were added after the single-model results were seen**, so they are
not an independent validation: I looked at three combinations on the same origins used for model
comparison, and reported two. They have no fitted weights, which limits the risk of overfitting,
but the claim that averaging helps would need new origins to confirm. A plausible reason is
that the components err in different ways (trailing mean tracks recent level, seasonal naive
tracks the calendar), but the correlation of their errors was not measured.

### 3. Accuracy at coarser levels tells a different story

Summing forecasts to a coarser level cancels independent errors but not a shared bias:

| level (pooled, 3 origins) | seasonal naive | trailing mean | LightGBM | ensemble | ensemble_baselines |
|---|---|---|---|---|---|
| series | 21.7% | 20.9% | 19.3% | 17.8% | 18.2% |
| store | 10.6% | 11.9% | 12.3% | 9.8% | 9.4% |
| category | 9.8% | 13.1% | 12.5% | 9.6% | 9.2% |
| network | 3.7% | 7.6% | 9.2% | 5.8% | 4.4% |

LightGBM's series-level advantage disappears when forecasts are summed: at store level it is
worse than seasonal naive (12.3% against 10.6%), and at network level much worse (9.2% against
3.7%), partly because its -6.5% bias does not cancel when summed. The simplest forecast, last
year's value, is the hardest to beat for total revenue.

### 4. LightGBM's under-forecast belongs to one origin

At origin 104 LightGBM under-forecasts by 12.4% in every one of the 13 weeks, starting from
week 1, and does not beat a 13-week trailing mean (19.6% against 17.1%). The last week it saw
was an abnormal dip (network revenue 4.35M against about 5.1M in the weeks around it), which it
treated as a change of level; revenue returned to normal immediately afterwards. On the two
robustness origins its bias is +0.2%, so this is a weakness of the model at one origin, not a
systematic one.

### 5. Choosing the LightGBM feature set

Feature set v2 was designed after the origin-104 result above. To avoid tuning to that window,
the adoption rule was set before looking: adopt v2 only if, on origins 85 and 98 (not used to
design it), its pooled WAPE is no worse than v1 and its bias is smaller. On the full panel v2
scored 18.9% against 18.6% for v1, with no smaller bias, so **v1 is the adopted feature
set**. Across all five origins the two are effectively tied. The one-week-dip hypothesis behind
v2 was also not confirmed by a synthetic unit test: v2 reacts more to an abnormal dip, because
its year-over-year features are designed to flag it.

### 6. By horizon and store size

WAPE rises with the horizon: the ensemble goes from 16.3% in weeks 1-4 to 19.3% in weeks 9-13,
LightGBM from 16.8% to 21.8%. Small stores are the hardest: for format SS the ensemble's
series-level WAPE is 25.2% (seasonal naive 34.1%), for XLS 15.1%.

### Run costs on a laptop

LightGBM fits in 20 s to 140 s per origin on 1.0 to 4.4 million training rows (feature set v2 up
to 270 s). Greykite takes 260 to 310 s per origin for 240 series on 8 processes, NeuralProphet
47 to 65 s, damped ETS 3 to 5 s and Holt-Winters 10 s.

### What this means for revenue-risk work

- Use the ensemble forecasts, not LightGBM alone, for store and category level decisions.
- Expect errors of roughly 15% to 19% per store-category series, about 9% to 10% per store, and
  4% to 6% for the whole network over a 13-week horizon. Shortfall thresholds finer than that
  will mostly flag noise.
- The weakest groups are small stores and long horizons.

## Known problems found while building this

- A bare `models/` entry in `.gitignore` hid the `forecasting/models` source package from git. It
  is now anchored to the repository root, and every commit on the branch was tested separately.
- uv resolved a pyarrow that needs NumPy 2 into the NumPy 1 NeuralProphet environment. The pin is
  now capped and the worker was verified on the real subset.
- Greykite workers oversubscribed the CPU with BLAS threads (850 s for one origin, now about
  300 s per origin for three).
- Holt-Winters failed on the real run because two subset series come from recently opened
  stores. It now runs on the series it can cover and the report states how many are excluded.
