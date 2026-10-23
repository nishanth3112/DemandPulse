# Revenue risk report

Forecast runs: `main`, `robustness`. Primary model: `ensemble`. Origins: 78, 85, 91, 98, 104 (13-week windows 79-91, 86-98, 92-104, 99-111, 105-117 of the 117 data weeks).

**How to read this.** The data has no budgets, so every target is hypothetical: last year's revenue for the same weeks, either unchanged (`flat_yoy`) or grown by the network's like-for-like growth over the 13 weeks before the origin (`trailing_growth`). All windows are historical and already observed, so results are backtest measurements on synthetic data. They are not live results, and none of them is a claim about profit, margin or promotion effects. The scenario section is arithmetic on forecasts, not a causal estimate.

Estimated shortfall probabilities use only forecast errors from windows that had finished by each origin, so they exist for origins [91, 98, 104]; the earlier origins have no history to learn from. The `ensemble` was chosen after seeing the forecast comparison and is not independently validated.

## Target growth assumptions (observed at each origin)

| origin | target_def | growth |
|---|---|---|
| 78 | flat_yoy | 0.0% |
| 78 | trailing_growth | 1.7% |
| 85 | flat_yoy | 0.0% |
| 85 | trailing_growth | 2.6% |
| 91 | flat_yoy | 0.0% |
| 91 | trailing_growth | 1.4% |
| 98 | flat_yoy | 0.0% |
| 98 | trailing_growth | -2.2% |
| 104 | flat_yoy | 0.0% |
| 104 | trailing_growth | -2.0% |

## Store level

### Realised outcomes (observed in the backtest)

| target_def | origin | entities | share_missed_target | share_forecast_below_target |
|---|---|---|---|---|
| flat_yoy | 78 | 751 | 48.5% | 63.4% |
| flat_yoy | 85 | 751 | 69.9% | 55.8% |
| flat_yoy | 91 | 752 | 69.4% | 59.7% |
| flat_yoy | 98 | 755 | 60.8% | 76.4% |
| flat_yoy | 104 | 756 | 64.8% | 92.3% |
| trailing_growth | 78 | 751 | 58.1% | 72.7% |
| trailing_growth | 85 | 751 | 81.1% | 76.4% |
| trailing_growth | 91 | 752 | 75.5% | 70.6% |
| trailing_growth | 98 | 755 | 41.6% | 55.9% |
| trailing_growth | 104 | 756 | 44.6% | 89.2% |

### Probability calibration, `ensemble`, origins with an estimate

`flat_yoy`: Brier score 0.207 against 0.227 for always predicting the pooled miss rate (lower is better; the pooled rate uses the answers, so it is an optimistic reference).

| bin_low | bin_high | n | mean_predicted | observed_rate |
|---|---|---|---|---|
| 0.0% | 10.0% | 121 | 3.4% | 5.8% |
| 10.0% | 20.0% | 79 | 15.6% | 27.8% |
| 20.0% | 30.0% | 109 | 24.4% | 42.2% |
| 30.0% | 40.0% | 119 | 34.4% | 54.6% |
| 40.0% | 50.0% | 141 | 45.5% | 67.4% |
| 50.0% | 60.0% | 187 | 55.1% | 60.4% |
| 60.0% | 70.0% | 197 | 65.4% | 62.9% |
| 70.0% | 80.0% | 286 | 75.1% | 71.0% |
| 80.0% | 90.0% | 424 | 85.7% | 73.8% |
| 90.0% | 100.0% | 600 | 93.6% | 80.5% |

`trailing_growth`: Brier score 0.248 against 0.249 for always predicting the pooled miss rate (lower is better; the pooled rate uses the answers, so it is an optimistic reference).

| bin_low | bin_high | n | mean_predicted | observed_rate |
|---|---|---|---|---|
| 0.0% | 10.0% | 104 | 2.4% | 5.8% |
| 10.0% | 20.0% | 81 | 14.8% | 16.0% |
| 20.0% | 30.0% | 95 | 25.1% | 40.0% |
| 30.0% | 40.0% | 155 | 35.0% | 40.0% |
| 40.0% | 50.0% | 178 | 44.9% | 38.8% |
| 50.0% | 60.0% | 235 | 55.3% | 57.4% |
| 60.0% | 70.0% | 242 | 65.1% | 55.8% |
| 70.0% | 80.0% | 317 | 75.1% | 65.3% |
| 80.0% | 90.0% | 462 | 85.6% | 59.5% |
| 90.0% | 100.0% | 394 | 93.4% | 70.8% |

### Ranking at the top 10% (mean over origins [91, 98, 104])

precision = share of the top group that really missed target; recall = share of all misses caught; lift = precision over the overall miss rate; capture = share of total revenue shortfall in the top group. `random` has exact expected values. `own_trend` forecasts last year grown by the entity's own recent growth. A score with every entity tied has no row values.

| target_def | model | score | origins | base_rate | precision | recall | lift | capture |
|---|---|---|---|---|---|---|---|---|
| flat_yoy | ensemble | p_shortfall | 3 | 65.0% | 96.1% | 14.9% | 1.482 | 22.7% |
| flat_yoy | ensemble | relative_gap | 3 | 65.0% | 96.1% | 14.9% | 1.482 | 22.6% |
| flat_yoy | trailing_mean_13 | p_shortfall | 3 | 65.0% | 95.6% | 14.9% | 1.475 | 23.4% |
| flat_yoy | trailing_mean_13 | relative_gap | 3 | 65.0% | 95.6% | 14.9% | 1.475 | 23.4% |
| flat_yoy | lightgbm | p_shortfall | 3 | 65.0% | 93.0% | 14.5% | 1.435 | 23.4% |
| flat_yoy | lightgbm | relative_gap | 3 | 65.0% | 93.0% | 14.5% | 1.435 | 22.3% |
| flat_yoy | ensemble | forecast_gap | 3 | 65.0% | 92.5% | 14.4% | 1.426 | 34.5% |
| flat_yoy | lightgbm | revenue_at_risk | 3 | 65.0% | 92.1% | 14.3% | 1.419 | 35.2% |
| flat_yoy | lightgbm | forecast_gap | 3 | 65.0% | 91.7% | 14.2% | 1.412 | 35.5% |
| flat_yoy | trailing_mean_13 | forecast_gap | 3 | 65.0% | 91.7% | 14.2% | 1.413 | 32.7% |
| flat_yoy | trailing_mean_13 | revenue_at_risk | 3 | 65.0% | 91.2% | 14.2% | 1.406 | 34.3% |
| flat_yoy | ensemble | revenue_at_risk | 3 | 65.0% | 90.4% | 14.0% | 1.392 | 34.7% |
| flat_yoy | own_trend | p_shortfall | 3 | 65.0% | 85.1% | 13.2% | 1.310 | 22.9% |
| flat_yoy | own_trend | relative_gap | 3 | 65.0% | 85.1% | 13.2% | 1.310 | 22.9% |
| flat_yoy | own_trend | forecast_gap | 3 | 65.0% | 84.6% | 13.1% | 1.303 | 29.9% |
| flat_yoy | own_trend | revenue_at_risk | 3 | 65.0% | 84.2% | 13.1% | 1.296 | 30.3% |
| flat_yoy | seasonal_naive | revenue_at_risk | 3 | 65.0% | 70.2% | 10.9% | 1.079 | 19.8% |
| flat_yoy | random | random | 3 | 65.0% | 65.0% | 10.1% | 1.000 | 10.1% |
| flat_yoy | seasonal_naive | forecast_gap | 0 | 65.0% |  |  |  |  |
| flat_yoy | seasonal_naive | p_shortfall | 0 | 65.0% |  |  |  |  |
| flat_yoy | seasonal_naive | relative_gap | 0 | 65.0% |  |  |  |  |
| trailing_growth | ensemble | p_shortfall | 3 | 53.9% | 91.7% | 18.2% | 1.803 | 28.1% |
| trailing_growth | ensemble | relative_gap | 3 | 53.9% | 91.7% | 18.2% | 1.803 | 28.1% |
| trailing_growth | trailing_mean_13 | p_shortfall | 3 | 53.9% | 91.7% | 18.1% | 1.798 | 28.8% |
| trailing_growth | trailing_mean_13 | relative_gap | 3 | 53.9% | 91.2% | 18.0% | 1.787 | 28.7% |
| trailing_growth | lightgbm | p_shortfall | 3 | 53.9% | 90.4% | 17.9% | 1.775 | 29.0% |
| trailing_growth | lightgbm | relative_gap | 3 | 53.9% | 90.4% | 17.9% | 1.775 | 28.1% |
| trailing_growth | ensemble | forecast_gap | 3 | 53.9% | 84.2% | 16.5% | 1.634 | 37.5% |
| trailing_growth | lightgbm | forecast_gap | 3 | 53.9% | 82.9% | 16.1% | 1.600 | 37.2% |
| trailing_growth | lightgbm | revenue_at_risk | 3 | 53.9% | 82.0% | 15.9% | 1.579 | 37.0% |
| trailing_growth | trailing_mean_13 | forecast_gap | 3 | 53.9% | 81.6% | 15.9% | 1.576 | 35.2% |
| trailing_growth | ensemble | revenue_at_risk | 3 | 53.9% | 81.6% | 15.8% | 1.572 | 36.2% |
| trailing_growth | trailing_mean_13 | revenue_at_risk | 3 | 53.9% | 81.6% | 15.8% | 1.573 | 36.2% |
| trailing_growth | own_trend | p_shortfall | 3 | 53.9% | 74.1% | 14.2% | 1.405 | 26.6% |
| trailing_growth | own_trend | relative_gap | 3 | 53.9% | 74.1% | 14.2% | 1.405 | 26.5% |
| trailing_growth | seasonal_naive | forecast_gap | 3 | 53.9% | 72.8% | 14.1% | 1.400 | 11.0% |
| trailing_growth | own_trend | forecast_gap | 3 | 53.9% | 71.5% | 13.5% | 1.344 | 31.9% |
| trailing_growth | own_trend | revenue_at_risk | 3 | 53.9% | 71.1% | 13.4% | 1.327 | 31.2% |
| trailing_growth | seasonal_naive | relative_gap | 3 | 53.9% | 57.0% | 10.6% | 1.056 | 12.2% |
| trailing_growth | seasonal_naive | revenue_at_risk | 3 | 53.9% | 57.0% | 10.4% | 1.034 | 17.3% |
| trailing_growth | random | random | 3 | 53.9% | 53.9% | 10.1% | 1.000 | 10.1% |
| trailing_growth | seasonal_naive | p_shortfall | 0 | 53.9% |  |  |  |  |

### Classifier (store)

Train origins [78], calibration origin 91, test origin 104. This is a single test window: it shows whether the classifier worked once, not that it is better than the forecast-based probability. Hyperparameters were fixed before the test window was scored.

`flat_yoy`

| metric | value |
|---|---|
| n_train | 751.000 |
| n_calibration | 752.000 |
| n_test | 756.000 |
| base_rate_test | 0.648 |
| brier_constant | 0.230 |
| brier_raw | 0.251 |
| brier_calibrated | 0.266 |
| n_comparable | 756.000 |
| brier_forecast_probability | 0.252 |
| brier_calibrated_comparable | 0.266 |

Top-group ranking on the test window (K = 10%):

| score | base_rate | precision | recall | lift | capture | tied |
|---|---|---|---|---|---|---|
| p_classifier | 64.8% | 65.8% | 10.2% | 1.015 | 6.7% | no |
| p_shortfall | 64.8% | 94.7% | 14.7% | 1.462 | 18.4% | no |
| relative_gap | 64.8% | 94.7% | 14.7% | 1.462 | 18.4% | no |

Feature importance (gain):

| feature | gain |
|---|---|
| relative_gap | 2,100.390 |
| growth_13 | 998.140 |
| log_target | 699.027 |
| growth_26 | 677.758 |
| volatility | 572.897 |

`trailing_growth`

| metric | value |
|---|---|
| n_train | 751.000 |
| n_calibration | 752.000 |
| n_test | 756.000 |
| base_rate_test | 0.446 |
| brier_constant | 0.343 |
| brier_raw | 0.325 |
| brier_calibrated | 0.424 |
| n_comparable | 756.000 |
| brier_forecast_probability | 0.375 |
| brier_calibrated_comparable | 0.424 |

Top-group ranking on the test window (K = 10%):

| score | base_rate | precision | recall | lift | capture | tied |
|---|---|---|---|---|---|---|
| p_classifier | 44.6% | 48.7% | 11.0% | 1.092 | 8.8% | no |
| p_shortfall | 44.6% | 85.5% | 19.3% | 1.919 | 23.5% | no |
| relative_gap | 44.6% | 85.5% | 19.3% | 1.919 | 23.5% | no |

Feature importance (gain):

| feature | gain |
|---|---|
| relative_gap | 2,357.497 |
| growth_13 | 1,061.980 |
| growth_26 | 724.045 |
| log_target | 587.460 |
| volatility | 496.714 |

## Store x category level

### Realised outcomes (observed in the backtest)

| target_def | origin | entities | share_missed_target | share_forecast_below_target |
|---|---|---|---|---|
| flat_yoy | 78 | 20,447 | 53.8% | 53.1% |
| flat_yoy | 85 | 20,447 | 59.0% | 53.0% |
| flat_yoy | 91 | 20,416 | 55.8% | 58.2% |
| flat_yoy | 98 | 20,445 | 52.3% | 58.8% |
| flat_yoy | 104 | 20,453 | 53.5% | 71.2% |
| trailing_growth | 78 | 20,447 | 57.5% | 58.3% |
| trailing_growth | 85 | 20,447 | 64.5% | 63.5% |
| trailing_growth | 91 | 20,416 | 58.5% | 62.0% |
| trailing_growth | 98 | 20,445 | 46.2% | 51.2% |
| trailing_growth | 104 | 20,453 | 47.0% | 66.6% |

### Probability calibration, `ensemble`, origins with an estimate

`flat_yoy`: Brier score 0.190 against 0.249 for always predicting the pooled miss rate (lower is better; the pooled rate uses the answers, so it is an optimistic reference).

| bin_low | bin_high | n | mean_predicted | observed_rate |
|---|---|---|---|---|
| 0.0% | 10.0% | 5,487 | 5.4% | 13.1% |
| 10.0% | 20.0% | 3,184 | 14.6% | 20.9% |
| 20.0% | 30.0% | 2,761 | 25.1% | 21.7% |
| 30.0% | 40.0% | 3,424 | 35.2% | 23.1% |
| 40.0% | 50.0% | 4,497 | 45.3% | 30.8% |
| 50.0% | 60.0% | 6,250 | 55.3% | 42.5% |
| 60.0% | 70.0% | 8,986 | 65.2% | 58.2% |
| 70.0% | 80.0% | 12,603 | 75.2% | 73.4% |
| 80.0% | 90.0% | 10,741 | 84.4% | 82.5% |
| 90.0% | 100.0% | 3,381 | 94.0% | 84.9% |

`trailing_growth`: Brier score 0.196 against 0.250 for always predicting the pooled miss rate (lower is better; the pooled rate uses the answers, so it is an optimistic reference).

| bin_low | bin_high | n | mean_predicted | observed_rate |
|---|---|---|---|---|
| 0.0% | 10.0% | 5,665 | 5.4% | 12.5% |
| 10.0% | 20.0% | 3,328 | 14.6% | 19.9% |
| 20.0% | 30.0% | 3,108 | 25.1% | 20.5% |
| 30.0% | 40.0% | 3,773 | 35.3% | 21.0% |
| 40.0% | 50.0% | 4,828 | 45.3% | 28.0% |
| 50.0% | 60.0% | 6,652 | 55.2% | 40.6% |
| 60.0% | 70.0% | 9,344 | 65.2% | 56.7% |
| 70.0% | 80.0% | 12,119 | 75.2% | 71.3% |
| 80.0% | 90.0% | 9,435 | 84.3% | 81.2% |
| 90.0% | 100.0% | 3,062 | 94.1% | 83.4% |

### Ranking at the top 10% (mean over origins [91, 98, 104])

precision = share of the top group that really missed target; recall = share of all misses caught; lift = precision over the overall miss rate; capture = share of total revenue shortfall in the top group. `random` has exact expected values. `own_trend` forecasts last year grown by the entity's own recent growth. A score with every entity tied has no row values.

| target_def | model | score | origins | base_rate | precision | recall | lift | capture |
|---|---|---|---|---|---|---|---|---|
| flat_yoy | ensemble | forecast_gap | 3 | 53.9% | 89.2% | 16.6% | 1.657 | 52.6% |
| flat_yoy | trailing_mean_13 | forecast_gap | 3 | 53.9% | 88.6% | 16.5% | 1.647 | 51.8% |
| flat_yoy | lightgbm | forecast_gap | 3 | 53.9% | 88.4% | 16.4% | 1.641 | 52.5% |
| flat_yoy | trailing_mean_13 | relative_gap | 3 | 53.9% | 85.2% | 15.8% | 1.584 | 17.2% |
| flat_yoy | trailing_mean_13 | p_shortfall | 3 | 53.8% | 85.2% | 15.8% | 1.584 | 17.3% |
| flat_yoy | ensemble | p_shortfall | 3 | 53.9% | 84.2% | 15.7% | 1.566 | 16.7% |
| flat_yoy | ensemble | relative_gap | 3 | 53.9% | 84.2% | 15.7% | 1.565 | 16.7% |
| flat_yoy | trailing_mean_13 | revenue_at_risk | 3 | 53.8% | 83.4% | 15.5% | 1.549 | 52.7% |
| flat_yoy | lightgbm | revenue_at_risk | 3 | 53.9% | 82.8% | 15.4% | 1.538 | 52.5% |
| flat_yoy | ensemble | revenue_at_risk | 3 | 53.9% | 81.0% | 15.0% | 1.504 | 52.0% |
| flat_yoy | lightgbm | p_shortfall | 3 | 53.9% | 79.3% | 14.8% | 1.475 | 15.1% |
| flat_yoy | lightgbm | relative_gap | 3 | 53.9% | 79.3% | 14.7% | 1.475 | 15.1% |
| flat_yoy | own_trend | forecast_gap | 3 | 53.9% | 77.6% | 14.4% | 1.439 | 38.3% |
| flat_yoy | own_trend | revenue_at_risk | 3 | 53.9% | 75.0% | 13.9% | 1.392 | 47.5% |
| flat_yoy | seasonal_naive | revenue_at_risk | 3 | 53.9% | 67.1% | 12.5% | 1.247 | 43.6% |
| flat_yoy | own_trend | relative_gap | 3 | 53.9% | 61.0% | 11.3% | 1.135 | 9.3% |
| flat_yoy | own_trend | p_shortfall | 3 | 53.9% | 60.9% | 11.3% | 1.134 | 9.4% |
| flat_yoy | random | random | 3 | 53.9% | 53.9% | 10.0% | 1.000 | 10.0% |
| flat_yoy | seasonal_naive | forecast_gap | 0 | 53.9% |  |  |  |  |
| flat_yoy | seasonal_naive | p_shortfall | 0 | 53.9% |  |  |  |  |
| flat_yoy | seasonal_naive | relative_gap | 0 | 53.9% |  |  |  |  |
| trailing_growth | ensemble | forecast_gap | 3 | 50.6% | 85.4% | 17.0% | 1.701 | 52.5% |
| trailing_growth | trailing_mean_13 | forecast_gap | 3 | 50.6% | 84.7% | 16.9% | 1.688 | 51.7% |
| trailing_growth | trailing_mean_13 | relative_gap | 3 | 50.6% | 84.1% | 16.9% | 1.686 | 19.4% |
| trailing_growth | trailing_mean_13 | p_shortfall | 3 | 50.5% | 84.1% | 16.9% | 1.686 | 19.5% |
| trailing_growth | lightgbm | forecast_gap | 3 | 50.6% | 83.4% | 16.5% | 1.654 | 51.9% |
| trailing_growth | ensemble | p_shortfall | 3 | 50.6% | 83.2% | 16.7% | 1.670 | 19.0% |
| trailing_growth | ensemble | relative_gap | 3 | 50.6% | 83.2% | 16.7% | 1.670 | 19.0% |
| trailing_growth | lightgbm | relative_gap | 3 | 50.6% | 78.2% | 15.7% | 1.571 | 17.4% |
| trailing_growth | lightgbm | p_shortfall | 3 | 50.6% | 78.2% | 15.7% | 1.571 | 17.4% |
| trailing_growth | trailing_mean_13 | revenue_at_risk | 3 | 50.5% | 77.5% | 15.4% | 1.538 | 51.6% |
| trailing_growth | lightgbm | revenue_at_risk | 3 | 50.6% | 76.6% | 15.2% | 1.517 | 51.3% |
| trailing_growth | ensemble | revenue_at_risk | 3 | 50.6% | 74.7% | 14.8% | 1.480 | 50.7% |
| trailing_growth | own_trend | forecast_gap | 3 | 50.6% | 71.8% | 14.1% | 1.412 | 37.4% |
| trailing_growth | own_trend | revenue_at_risk | 3 | 50.6% | 68.2% | 13.5% | 1.347 | 45.6% |
| trailing_growth | seasonal_naive | revenue_at_risk | 3 | 50.6% | 60.0% | 11.8% | 1.184 | 41.2% |
| trailing_growth | own_trend | relative_gap | 3 | 50.6% | 59.2% | 11.9% | 1.190 | 10.4% |
| trailing_growth | own_trend | p_shortfall | 3 | 50.6% | 59.1% | 11.9% | 1.187 | 10.4% |
| trailing_growth | seasonal_naive | relative_gap | 3 | 50.6% | 51.1% | 10.1% | 1.010 | 10.1% |
| trailing_growth | random | random | 3 | 50.6% | 50.6% | 10.0% | 1.000 | 10.0% |
| trailing_growth | seasonal_naive | forecast_gap | 3 | 50.6% | 48.4% | 9.4% | 0.936 | 16.1% |
| trailing_growth | seasonal_naive | p_shortfall | 0 | 50.6% |  |  |  |  |

### Classifier (store x category)

Train origins [78], calibration origin 91, test origin 104. This is a single test window: it shows whether the classifier worked once, not that it is better than the forecast-based probability. Hyperparameters were fixed before the test window was scored.

`flat_yoy`

| metric | value |
|---|---|
| n_train | 20,447.000 |
| n_calibration | 20,416.000 |
| n_test | 20,453.000 |
| base_rate_test | 0.535 |
| brier_constant | 0.249 |
| brier_raw | 0.216 |
| brier_calibrated | 0.213 |
| n_comparable | 20,453.000 |
| brier_forecast_probability | 0.208 |
| brier_calibrated_comparable | 0.213 |

Top-group ranking on the test window (K = 10%):

| score | base_rate | precision | recall | lift | capture | tied |
|---|---|---|---|---|---|---|
| p_classifier | 53.5% | 80.5% | 15.0% | 1.504 | 12.6% | no |
| p_shortfall | 53.5% | 82.8% | 15.5% | 1.547 | 20.5% | no |
| relative_gap | 53.5% | 82.8% | 15.5% | 1.547 | 20.4% | no |

Feature importance (gain):

| feature | gain |
|---|---|
| relative_gap | 42,618.120 |
| log_target | 5,945.712 |
| growth_13 | 5,834.921 |
| volatility | 5,566.833 |
| growth_26 | 2,760.918 |

`trailing_growth`

| metric | value |
|---|---|
| n_train | 20,447.000 |
| n_calibration | 20,416.000 |
| n_test | 20,453.000 |
| base_rate_test | 0.470 |
| brier_constant | 0.262 |
| brier_raw | 0.234 |
| brier_calibrated | 0.228 |
| n_comparable | 20,453.000 |
| brier_forecast_probability | 0.220 |
| brier_calibrated_comparable | 0.228 |

Top-group ranking on the test window (K = 10%):

| score | base_rate | precision | recall | lift | capture | tied |
|---|---|---|---|---|---|---|
| p_classifier | 47.0% | 78.0% | 16.6% | 1.659 | 18.1% | no |
| p_shortfall | 47.0% | 80.8% | 17.2% | 1.718 | 24.0% | no |
| relative_gap | 47.0% | 80.8% | 17.2% | 1.717 | 24.0% | no |

Feature importance (gain):

| feature | gain |
|---|---|
| relative_gap | 42,152.424 |
| growth_13 | 6,279.465 |
| volatility | 6,044.315 |
| log_target | 5,686.103 |
| growth_26 | 2,459.334 |

## Stores to look at first (final origin, store level)

`flat_yoy`: top 10 of 20 by expected revenue below target.

1. STORE00030: forecast is 89,739 (62.2%) below target; probability of a shortfall 100%; driver split unavailable
2. STORE02495: forecast is 79,830 (50.4%) below target; probability of a shortfall 100%; last 13 weeks -51.3% on a year earlier; weakest driver baskets per week (-44.9%)
3. STORE00696: forecast is 31,562 (11.7%) below target; probability of a shortfall 96%; last 13 weeks -9.6% on a year earlier; weakest driver baskets per week (-9.0%)
4. STORE00476: forecast is 31,018 (17.2%) below target; probability of a shortfall 97%; last 13 weeks -18.3% on a year earlier; weakest driver baskets per week (-18.4%)
5. STORE01300: forecast is 30,845 (17.3%) below target; probability of a shortfall 97%; last 13 weeks -19.0% on a year earlier; weakest driver baskets per week (-14.5%)
6. STORE02858: forecast is 29,643 (16.6%) below target; probability of a shortfall 97%; last 13 weeks -19.9% on a year earlier; weakest driver baskets per week (-15.7%)
7. STORE01604: forecast is 28,013 (11.7%) below target; probability of a shortfall 96%; last 13 weeks -9.5% on a year earlier; weakest driver baskets per week (-6.1%)
8. STORE00602: forecast is 30,046 (31.5%) below target; probability of a shortfall 99%; last 13 weeks -20.2% on a year earlier; weakest driver baskets per week (-18.3%)
9. STORE00103: forecast is 29,880 (54.6%) below target; probability of a shortfall 100%; last 13 weeks -80.9% on a year earlier; weakest driver baskets per week (-62.5%)
10. STORE01423: forecast is 25,862 (10.4%) below target; probability of a shortfall 95%; last 13 weeks -5.9% on a year earlier; weakest driver baskets per week (-4.2%)

`trailing_growth`: top 10 of 20 by expected revenue below target.

1. STORE00030: forecast is 86,891 (61.4%) below target; probability of a shortfall 100%; driver split unavailable
2. STORE02495: forecast is 76,705 (49.4%) below target; probability of a shortfall 100%; last 13 weeks -51.3% on a year earlier; weakest driver baskets per week (-44.9%)
3. STORE00696: forecast is 26,217 (9.9%) below target; probability of a shortfall 94%; last 13 weeks -9.6% on a year earlier; weakest driver baskets per week (-9.0%)
4. STORE00476: forecast is 27,464 (15.6%) below target; probability of a shortfall 97%; last 13 weeks -18.3% on a year earlier; weakest driver baskets per week (-18.4%)
5. STORE01300: forecast is 27,326 (15.6%) below target; probability of a shortfall 97%; last 13 weeks -19.0% on a year earlier; weakest driver baskets per week (-14.5%)
6. STORE00103: forecast is 28,800 (53.7%) below target; probability of a shortfall 100%; last 13 weeks -80.9% on a year earlier; weakest driver baskets per week (-62.5%)
7. STORE00602: forecast is 28,165 (30.2%) below target; probability of a shortfall 99%; last 13 weeks -20.2% on a year earlier; weakest driver baskets per week (-18.3%)
8. STORE02858: forecast is 26,118 (14.9%) below target; probability of a shortfall 97%; last 13 weeks -19.9% on a year earlier; weakest driver baskets per week (-15.7%)
9. STORE01604: forecast is 23,291 (9.9%) below target; probability of a shortfall 94%; last 13 weeks -9.5% on a year earlier; weakest driver baskets per week (-6.1%)
10. STORE02528: forecast is 22,229 (11.2%) below target; probability of a shortfall 95%; last 13 weeks -11.1% on a year earlier; weakest driver baskets per week (-8.7%)

## Scenario arithmetic (hypothetical, for the stores above)

Each scenario multiplies the store's forecast by (1 + traffic) x (1 + basket size) x (1 + price/mix). It shows how large a driver change would be needed, not that any action can deliver it.

| target_def | scenario | stores | share_meeting_target | mean_gap_closed | median_uplift_needed |
|---|---|---|---|---|---|
| flat_yoy | all_three | 20 | 0.0% | 31.8% | 13.9% |
| flat_yoy | basket_size_+2% | 20 | 0.0% | 12.5% | 13.9% |
| flat_yoy | price_mix_+1% | 20 | 0.0% | 6.3% | 13.9% |
| flat_yoy | traffic_+2% | 20 | 0.0% | 12.5% | 13.9% |
| trailing_growth | all_three | 20 | 0.0% | 33.3% | 13.0% |
| trailing_growth | basket_size_+2% | 20 | 0.0% | 13.1% | 13.0% |
| trailing_growth | price_mix_+1% | 20 | 0.0% | 6.6% | 13.0% |
| trailing_growth | traffic_+2% | 20 | 0.0% | 13.1% | 13.0% |

Scenarios: traffic_+2% = traffic +2%, basket size +0%, price/mix +0%; basket_size_+2% = traffic +0%, basket size +2%, price/mix +0%; price_mix_+1% = traffic +0%, basket size +0%, price/mix +1%; all_three = traffic +2%, basket size +2%, price/mix +1%.


