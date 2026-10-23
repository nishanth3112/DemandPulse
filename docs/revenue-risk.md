# Revenue risk

Which stores and store x category series are most likely to fall short of a revenue target over
the next 13 weeks, and by how much. Built on the ensemble forecasts from the
[forecasting engine](forecasting.md) and scored against what actually happened.

```bash
uv run demandpulse risk run                        # reads forecast runs main + robustness
uv run demandpulse risk run --forecast-run main --primary-model lightgbm --top-k 10
```

The command reads `forecast/panel.parquet` and the forecast runs from the lake, reads store-week
revenue, baskets and units from Gold, and writes everything under `risk/` (parquet tables,
classifier JSON, `report.md`, `summary.json`). The full generated report is committed at
[reports/risk.md](reports/risk.md).

## Honest framing

The dataset has no budgets, prices, margins or promotions, and every window is historical. So:

- **Targets are hypothetical.** Target = revenue of the same weeks and series last year, either
  unchanged (`flat_yoy`) or grown by the network's like-for-like growth over the 13 weeks up to
  the forecast origin (`trailing_growth`: 1.7%, 2.6%, 1.4%, -2.2% and -2.0% at origins 78, 85,
  91, 98 and 104). They are definitions chosen for this project, not company goals.
- **Observed** numbers are realised outcomes in the data (for example the share of stores whose
  actual revenue was below target). **Backtest** numbers are how well a ranking or probability
  would have predicted those outcomes using only data known at each origin. **Scenario** numbers
  are arithmetic on forecasts. None of them is a claim about profit, savings or the effect of any
  action.
- The `ensemble` (equal mean of seasonal naive, trailing mean and LightGBM) was chosen after
  seeing the forecast comparison and is not independently validated.

## Method

1. **Targets and shortfall** (`risk/targets.py`). For each model, origin and entity, sum the
   13-week actual, forecast and last-year revenue, build each target, and record the forecast
   shortfall (target - forecast) and realised shortfall (target - actual). Targets are built from
   the same series the forecast covers, so store totals add up the same categories. This is
   751-756 of the 761 stores per origin, because series must pass the forecasting eligibility
   rules.
2. **Probability and revenue at risk** (`risk/uncertainty.py`). The relative error
   `actual / forecast - 1` of every entity in earlier windows gives an empirical error
   distribution per model and level. Applied to a forecast F and target T it yields
   P(shortfall) and the expected revenue below target. A window counts as history only once it
   has finished (origin' + 13 <= origin), so estimates exist for origins 91, 98 and 104, not for
   78 and 85.
3. **Ranking backtest** (`risk/ranking.py`). At each origin entities are ranked by a score and the
   top 5%, 10% and 20% are compared with the realised outcome: precision@K, recall@K, lift
   (precision over the overall miss rate) and capture (share of the total revenue shortfall that
   sits in the top group). Baselines are `own_trend` (last year grown by the entity's own recent
   year-over-year growth, no model) and `random`, whose expected values are exact. A score on
   which every entity ties is reported empty rather than given an arbitrary order.
4. **Classifier** (`risk/classifier.py`). LightGBM on forecast gap, size, own 13- and 26-week
   growth and volatility, with isotonic calibration on a later held-out origin. Train, calibration
   and test windows are ordered so labels are always known in time. Hyperparameters are fixed in
   advance.
5. **Scenarios and recommendations** (`risk/scenario.py`, `risk/recommend.py`). Top stores by
   revenue at risk at the final origin, each with measured evidence: forecast gap, probability,
   own recent growth and which of traffic, basket size or price/mix fell most (`risk/drivers.py`,
   from Gold store-week baskets, because baskets do not add up across categories). Scenarios
   multiply the forecast by (1 + traffic)(1 + basket size)(1 + price/mix).

## Results

Ranking at the top 10%, mean over origins 91, 98 and 104 (`ensemble` unless stated). Overall
miss rates are 65.0% (`flat_yoy`) and 53.9% (`trailing_growth`) of stores.

| Level | Target | Score | Precision | Lift | Capture of shortfall revenue |
|---|---|---|---|---|---|
| Store | flat | relative gap | 96.1% | 1.48 | 22.6% |
| Store | flat | forecast gap (revenue) | 92.5% | 1.43 | 34.5% |
| Store | flat | own trend baseline (gap) | 84.6% | 1.30 | 29.9% |
| Store | flat | random | 65.0% | 1.00 | 10.1% |
| Store | trailing | probability | 91.7% | 1.80 | 28.1% |
| Store | trailing | forecast gap (revenue) | 84.2% | 1.63 | 37.5% |
| Store | trailing | own trend baseline (gap) | 71.5% | 1.34 | 31.9% |
| Store | trailing | random | 53.9% | 1.00 | 10.1% |
| Store x category | flat | forecast gap | 89.2% | 1.66 | 52.6% |
| Store x category | flat | own trend baseline (gap) | 77.6% | 1.44 | 38.3% |
| Store x category | trailing | forecast gap | 85.4% | 1.70 | 52.5% |
| Store x category | trailing | own trend baseline (gap) | 71.8% | 1.41 | 37.4% |

1. **The forecast-based ranking beats both baselines, by a modest margin.** It puts clearly more
   real misses in the top group than random and than ranking by each entity's own recent growth.
   The lift is limited by high base rates: at store level most stores miss a flat target in any
   window.
2. **Precision and revenue captured pull in different directions.** Ranking by relative gap or
   probability finds the stores most certain to miss (precision up to 96%) but they are small
   misses; ranking by revenue gap or revenue at risk finds the largest shortfalls (capture 34-37%
   at store level against 22-28%) at lower precision. Which to use depends on whether the
   decision is about certainty or money.
3. **The probabilities are not well calibrated.** For `flat_yoy` at store level the Brier score is
   0.207 against 0.227 for always predicting the miss rate; for `trailing_growth` it is 0.248
   against 0.249, i.e. no skill. The reliability tables in the report show overconfidence at the
   top (stores given 90-100% shortfall probability missed 80.5% and 70.8% of the time). The
   error distribution of one earlier window carries that window's common shock, so use the
   probabilities to order entities, not as stated chances. Store x category probabilities are
   better (Brier 0.190 and 0.196 against 0.249 and 0.250) but also overconfident in the tails.
4. **The classifier did not beat the forecast-based probability, so the forecast-based ranking is
   the primary result.** Train origin 78, calibration origin 91, test origin 104 is the only
   ordered split the five origins allow. At store level it is worse than the forecast
   probability (Brier 0.266 against 0.252 for `flat_yoy`, 0.424 against 0.375 for
   `trailing_growth`; precision 65.8% against a 64.8% base rate). At store x category level it is
   close but still behind (Brier 0.213 against 0.208, precision 80.5% against 82.8%). With one
   test window this is a single check, not a verdict on the method; it is documented as
   exploratory.
5. **At the last origin the top of the list was right 85% of the time.** For `flat_yoy` at origin
   104, 17 of the 20 stores listed by revenue at risk really missed target (base rate 64.8%).
   The three that did not include a false alarm described below.
6. **Closing the gap needs far more than small driver moves.** For those 20 stores the median
   uplift needed to reach target is 13.9% (`flat_yoy`) or 13.0% (`trailing_growth`). Traffic +2%
   closes on average 12.5% of the gap, all three scenario drivers together (+2%, +2%, +1%)
   31.8%, and no store reaches target in any scenario.

## Limitations

- **Three scorable origins.** Origins 91, 98 and 104 have probability estimates, and their
  13-week windows overlap and share market-wide conditions, so the means above are not three
  independent samples. No confidence intervals are given for the ranking metrics.
- **The last origin is unusual.** At origin 104 the ensemble forecast was below target for 92% of
  stores while 65% really missed, the known under-forecast already described in the forecasting
  report. The probabilities are biased there for the same reason.
- **Temporary closures fool the forecast.** STORE00030 ranked first at origin 104 (forecast 54,560
  against a target of 144,299, shortfall probability 99.9%) but sold 149,232 and beat its target:
  it was closed for about ten weeks (weeks 94-103) and reopened at normal volume, which no
  history-based forecast can anticipate. STORE02495 and STORE00103 were forecast at about half
  of target and missed by far less than that. They are among the stores with abrupt level
  shifts noted in the data facts. Treat large forecast gaps as "check this store" prompts.
- **Driver splits are partial.** The traffic / basket size / price-mix split needs a full 13
  trading weeks in both windows, so STORE00030 and other stores with gaps have none. Price and
  mix cannot be separated; the data has no list prices or promotions.
- **Recommendations are store level only.** Store x category is ranked and evaluated but baskets
  are not additive across categories, so no category driver split exists. Store x category
  windows are saved for the primary model only.
- **Everything uses the ensemble chosen post hoc,** and the targets are hypothetical. Different
  target definitions would change the base rates (42% to 81% of stores missed depending on target
  and origin) and therefore precision.

## Leakage controls

- Target growth, own-trend growth, volatility and error distributions use only data up to the
  origin; tests change post-origin revenue and check nothing moves.
- A window's errors are used only once the window has finished.
- The classifier's train, calibration and test origins are validated to be ordered in time, and
  its hyperparameters are fixed in code.
- All models and baselines are scored on identical entities and windows. A score that is null for
  an entity is dropped for that score only, and a fully tied score is shown empty.
