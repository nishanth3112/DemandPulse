# Model evaluation

One page on how well each model worked and how far to trust it. The full tables and method are in
[forecasting.md](forecasting.md), [revenue-risk.md](revenue-risk.md) and the generated reports in
[reports/](reports/). Everything here is a backtest on 117 weeks of synthetic data.

## What was evaluated

| Model family | Question | Scored on |
|---|---|---|
| Forecasters (8) | How accurately can weekly revenue per store and category be forecast 13 weeks ahead? | Three rolling origins (weeks 78, 91, 104), plus two robustness origins (85, 98), identical series and cutoffs for every model |
| Shortfall probability | Can past forecast errors say which stores will miss a target? | Origins 91, 98 and 104, using only windows that had finished |
| Ranking | Does ranking stores by that risk put the real misses first? | Precision, recall, lift and shortfall captured at the top 5%, 10% and 20%, against own-trend and random baselines |
| Classifier | Does a calibrated LightGBM do better than the forecast-based probability? | One test window (origin 104), after training and calibrating on earlier ones |

## Headline results

Forecast accuracy, full panel (about 20,400 series per origin), WAPE pooled over three origins
(lower is better):

| Level | Seasonal naive | Ensemble |
|---|---|---|
| Store x category | 21.7% | 17.8% |
| Store | 10.6% | 9.8% |
| Network | 3.7% | 5.8% |

The ensemble is the most accurate at series level, slightly better than seasonal naive at store
level and worse than it for the network total, where last year's value is hard to beat.

Risk ranking, store level, top 10% of stores, mean over origins 91, 98 and 104:

| Target | Ranking by relative gap or probability | Own-trend baseline | Random |
|---|---|---|---|
| Flat vs last year | 96.1% precision | 85.1% | 65.0% |
| Trailing growth | 91.7% precision | 74.1% | 53.9% |

The forecast-based ranking beats both baselines by a modest margin; the miss rates are high, so lift
is limited (1.48 and 1.80).

## What is not established

- **No independent validation of the ensemble.** The two averages were added after the single-model
  results were seen, on the same origins.
- **Three or five overlapping windows.** Rankings change from window to window, and no confidence
  intervals are given for the ranking metrics.
- **Probabilities are not calibrated chances.** Store-level Brier scores are 0.207 against 0.227
  for a constant (flat target) and 0.248 against 0.249 (trailing target), so use them to order
  stores, not as stated probabilities.
- **The classifier did not beat the forecast-based probability.** It is documented as exploratory.
- **Forecast bias at the last origin.** At origin 104 the forecasts were below target for 92% of
  stores while 65% actually missed.
- **Temporary closures fool a history-based forecast.** The first store listed at origin 104 beat
  its target after a ten-week closure.
- **Synthetic data.** Results describe model behaviour on this dataset, not on a real retailer.
- **Targets are hypothetical** and there is no profit, margin or promotion information.

## Reproducing it

```bash
uv run demandpulse forecast backtest all --run main
uv run demandpulse forecast report --run main
uv run demandpulse risk run
```

The end-to-end test and `demandpulse demo` exercise the same code on a small synthetic dataset, so
they check the machinery, not these results.
