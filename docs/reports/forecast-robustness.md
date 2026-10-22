# Forecast model comparison: run `robustness`

Weekly revenue per store x category (product hierarchy level 30). Forecast horizon 13 weeks from rolling origins at weeks 85, 98 of the 117 data weeks (test windows weeks 86-98, 99-111). Models see only revenue up to the origin. Negative forecasts are clipped to zero for every model.

Panel: 23,591 series. Representative subset: 240 series, stratified by store format x revenue tier (seed 20061).

WAPE = total absolute error / total actual revenue. Bias = (forecast - actual) / actual, positive means over-forecasting. MASE below 1 beats a one-year seasonal naive in-sample.

## Subset, final origin (all models, identical series and weeks)

| model | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|
| lightgbm | 3,120 | 14.7% | -2.1% | 67.646 | 767,714.720 | 240 | 0.691 | 0.660 |
| lightgbm_v2 | 3,120 | 14.8% | -2.2% | 68.050 | 767,714.720 | 240 | 0.701 | 0.668 |
| trailing_mean_13 | 3,120 | 15.0% | -1.0% | 64.648 | 767,714.720 | 240 | 0.737 | 0.689 |
| ensemble | 3,120 | 15.1% | -0.8% | 68.333 | 767,714.720 | 240 | 0.716 | 0.689 |
| seasonal_naive | 3,120 | 20.7% | 0.7% | 94.434 | 767,714.720 | 240 | 0.977 | 0.951 |

## Improvement over seasonal naive, subset final origin (paired bootstrap, 95% CI)

| model | baseline | units | improvement | ci_low | ci_high | win_rate | distinguishable |
|---|---|---|---|---|---|---|---|
| ensemble | seasonal_naive | 240 | 27.2% | 25.0% | 29.5% | 93.8% | yes |
| lightgbm | seasonal_naive | 240 | 29.1% | 26.1% | 32.0% | 89.6% | yes |
| lightgbm_v2 | seasonal_naive | 240 | 28.7% | 25.5% | 31.6% | 88.3% | yes |
| trailing_mean_13 | seasonal_naive | 240 | 27.7% | 23.9% | 31.1% | 82.1% | yes |

## Subset, all rolling origins pooled

| model | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|
| ensemble | 6,240 | 16.6% | 1.2% | 78.220 | 1,528,125.580 | 240 | 0.796 | 0.731 |
| lightgbm | 6,240 | 17.1% | 1.2% | 84.452 | 1,528,125.580 | 240 | 0.791 | 0.714 |
| lightgbm_v2 | 6,240 | 17.3% | 0.8% | 86.660 | 1,528,125.580 | 240 | 0.799 | 0.708 |
| trailing_mean_13 | 6,240 | 18.4% | 1.2% | 93.951 | 1,528,125.580 | 240 | 0.848 | 0.767 |
| seasonal_naive | 6,240 | 20.8% | 1.2% | 100.679 | 1,528,125.580 | 240 | 1.020 | 0.954 |

## Improvement over seasonal naive, subset rolling origins

| model | baseline | units | improvement | ci_low | ci_high | win_rate | distinguishable |
|---|---|---|---|---|---|---|---|
| ensemble | seasonal_naive | 480 | 19.9% | 16.8% | 23.2% | 87.1% | yes |
| lightgbm | seasonal_naive | 480 | 17.6% | 13.5% | 21.7% | 79.0% | yes |
| lightgbm_v2 | seasonal_naive | 480 | 16.5% | 12.2% | 20.8% | 79.0% | yes |
| trailing_mean_13 | seasonal_naive | 480 | 11.6% | 6.1% | 16.9% | 71.0% | yes |

## Subset, by origin

| model | origin | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|---|
| ensemble | 85 | 3,120 | 18.2% | 3.2% | 86.990 | 760,410.860 | 240 | 0.876 | 0.804 |
| ensemble | 98 | 3,120 | 15.1% | -0.8% | 68.333 | 767,714.720 | 240 | 0.716 | 0.689 |
| lightgbm | 85 | 3,120 | 19.5% | 4.5% | 98.430 | 760,410.860 | 240 | 0.891 | 0.828 |
| lightgbm | 98 | 3,120 | 14.7% | -2.1% | 67.646 | 767,714.720 | 240 | 0.691 | 0.660 |
| lightgbm_v2 | 85 | 3,120 | 19.9% | 3.8% | 101.927 | 760,410.860 | 240 | 0.897 | 0.821 |
| lightgbm_v2 | 98 | 3,120 | 14.8% | -2.2% | 68.050 | 767,714.720 | 240 | 0.701 | 0.668 |
| seasonal_naive | 85 | 3,120 | 20.8% | 1.7% | 106.559 | 760,410.860 | 240 | 1.062 | 0.954 |
| seasonal_naive | 98 | 3,120 | 20.7% | 0.7% | 94.434 | 767,714.720 | 240 | 0.977 | 0.951 |
| trailing_mean_13 | 85 | 3,120 | 21.8% | 3.4% | 116.078 | 760,410.860 | 240 | 0.960 | 0.893 |
| trailing_mean_13 | 98 | 3,120 | 15.0% | -1.0% | 64.648 | 767,714.720 | 240 | 0.737 | 0.689 |

## Full panel, by origin

| model | origin | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|---|
| ensemble | 85 | 265,811 | 19.8% | 2.5% | 110.227 | 63,940,023.210 | 20,447 | 0.904 | 0.813 |
| ensemble | 98 | 265,785 | 15.9% | -1.3% | 79.852 | 65,214,261.040 | 20,445 | 0.733 | 0.698 |
| lightgbm | 85 | 265,811 | 21.3% | 3.5% | 122.232 | 63,940,023.210 | 20,447 | 0.926 | 0.830 |
| lightgbm | 98 | 265,785 | 16.0% | -3.0% | 83.138 | 65,214,261.040 | 20,445 | 0.720 | 0.684 |
| lightgbm_v2 | 85 | 265,811 | 21.7% | 2.9% | 125.456 | 63,940,023.210 | 20,447 | 0.933 | 0.833 |
| lightgbm_v2 | 98 | 265,785 | 16.0% | -3.4% | 83.738 | 65,214,261.040 | 20,445 | 0.724 | 0.686 |
| seasonal_naive | 85 | 265,811 | 22.7% | 2.6% | 121.813 | 63,940,023.210 | 20,447 | 1.107 | 0.986 |
| seasonal_naive | 98 | 265,785 | 21.3% | 0.2% | 106.033 | 65,214,261.040 | 20,445 | 0.973 | 0.942 |
| trailing_mean_13 | 85 | 265,811 | 23.7% | 1.3% | 140.870 | 63,940,023.210 | 20,447 | 0.998 | 0.885 |
| trailing_mean_13 | 98 | 265,785 | 17.0% | -1.0% | 85.370 | 65,214,261.040 | 20,445 | 0.773 | 0.730 |

## Full panel, pooled over origins

| model | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|
| ensemble | 531,596 | 17.8% | 0.6% | 96.246 | 129,154,284.250 | 20,626 | 0.818 | 0.748 |
| lightgbm | 531,596 | 18.6% | 0.2% | 104.530 | 129,154,284.250 | 20,626 | 0.823 | 0.744 |
| lightgbm_v2 | 531,596 | 18.9% | -0.3% | 106.657 | 129,154,284.250 | 20,626 | 0.828 | 0.746 |
| trailing_mean_13 | 531,596 | 20.3% | 0.1% | 116.475 | 129,154,284.250 | 20,626 | 0.885 | 0.792 |
| seasonal_naive | 531,596 | 22.0% | 1.4% | 114.197 | 129,154,284.250 | 20,626 | 1.040 | 0.961 |

## Improvement over seasonal naive, full panel

| model | baseline | units | improvement | ci_low | ci_high | win_rate | distinguishable |
|---|---|---|---|---|---|---|---|
| ensemble | seasonal_naive | 40,892 | 18.8% | 18.3% | 19.2% | 86.5% | yes |
| lightgbm | seasonal_naive | 40,892 | 15.1% | 14.6% | 15.7% | 79.8% | yes |
| lightgbm_v2 | seasonal_naive | 40,892 | 14.1% | 13.6% | 14.7% | 79.5% | yes |
| trailing_mean_13 | seasonal_naive | 40,892 | 7.4% | 6.6% | 8.1% | 71.6% | yes |

## Full panel, accuracy summed to store level

| model | n | wape | bias |
|---|---|---|---|
| ensemble | 19,578 | 9.9% | 0.6% |
| lightgbm | 19,578 | 10.9% | 0.2% |
| lightgbm_v2 | 19,578 | 11.1% | -0.3% |
| seasonal_naive | 19,578 | 10.9% | 1.4% |
| trailing_mean_13 | 19,578 | 11.7% | 0.1% |

## Full panel, accuracy summed to category level

| model | n | wape | bias |
|---|---|---|---|
| ensemble | 780 | 9.5% | 0.6% |
| lightgbm | 780 | 11.2% | 0.2% |
| lightgbm_v2 | 780 | 11.5% | -0.3% |
| seasonal_naive | 780 | 9.8% | 1.4% |
| trailing_mean_13 | 780 | 12.8% | 0.1% |

## Full panel, accuracy summed to the whole network

| model | n | wape | bias |
|---|---|---|---|
| ensemble | 26 | 5.6% | 0.6% |
| lightgbm | 26 | 7.2% | 0.2% |
| lightgbm_v2 | 26 | 7.3% | -0.3% |
| seasonal_naive | 26 | 3.9% | 1.4% |
| trailing_mean_13 | 26 | 7.6% | 0.1% |

## Full panel, by forecast horizon

| model | horizon_bucket | wape | bias |
|---|---|---|---|
| ensemble | weeks 1-4 | 17.9% | -5.4% |
| ensemble | weeks 5-8 | 19.4% | 6.8% |
| ensemble | weeks 9-13 | 16.6% | 1.3% |
| lightgbm | weeks 1-4 | 18.7% | -6.9% |
| lightgbm | weeks 5-8 | 20.3% | 7.3% |
| lightgbm | weeks 9-13 | 17.3% | 1.3% |
| lightgbm_v2 | weeks 1-4 | 19.0% | -7.6% |
| lightgbm_v2 | weeks 5-8 | 20.4% | 7.0% |
| lightgbm_v2 | weeks 9-13 | 17.6% | 0.9% |
| seasonal_naive | weeks 1-4 | 22.9% | -0.4% |
| seasonal_naive | weeks 5-8 | 22.8% | 3.4% |
| seasonal_naive | weeks 9-13 | 20.6% | 1.5% |
| trailing_mean_13 | weeks 1-4 | 21.1% | -8.9% |
| trailing_mean_13 | weeks 5-8 | 22.7% | 9.7% |
| trailing_mean_13 | weeks 9-13 | 17.9% | 1.1% |

## Full panel, by store format

| model | store_format | n | wape | bias |
|---|---|---|---|---|
| ensemble | LS | 244,179 | 16.6% | 0.6% |
| ensemble | MS | 160,901 | 20.6% | 0.3% |
| ensemble | SS | 105,001 | 24.9% | 1.2% |
| ensemble | XLS | 21,515 | 15.3% | 0.7% |
| lightgbm | LS | 244,179 | 17.5% | -0.0% |
| lightgbm | MS | 160,901 | 21.1% | 0.5% |
| lightgbm | SS | 105,001 | 25.0% | 2.1% |
| lightgbm | XLS | 21,515 | 16.5% | -0.1% |
| lightgbm_v2 | LS | 244,179 | 17.7% | -0.6% |
| lightgbm_v2 | MS | 160,901 | 21.2% | 0.4% |
| lightgbm_v2 | SS | 105,001 | 25.2% | 1.9% |
| lightgbm_v2 | XLS | 21,515 | 16.8% | -0.7% |
| seasonal_naive | LS | 244,179 | 19.9% | 2.0% |
| seasonal_naive | MS | 160,901 | 26.2% | -0.3% |
| seasonal_naive | SS | 105,001 | 33.3% | -0.9% |
| seasonal_naive | XLS | 21,515 | 18.1% | 2.4% |
| trailing_mean_13 | LS | 244,179 | 19.2% | -0.2% |
| trailing_mean_13 | MS | 160,901 | 22.8% | 0.6% |
| trailing_mean_13 | SS | 105,001 | 26.5% | 2.4% |
| trailing_mean_13 | XLS | 21,515 | 18.1% | -0.3% |

## Run details

- **ensemble** (full scope, origins [85, 98]): series per origin {'85': 20447, '98': 20445}, seconds {}
  - equal-weight mean of component forecasts of seasonal_naive, trailing_mean_13, lightgbm. Equal weights, nothing fitted.
- **lightgbm** (full scope, origins [85, 98]): series per origin {'85': 20447, '98': 20445}, seconds {'85': 12.84, '98': 37.26}
  - origin 85: 1,792,474 training rows, 50 rounds, inner validation WAPE 0.164, fit 11.4s
  - origin 98: 3,632,785 training rows, 50 rounds, inner validation WAPE 0.215, fit 35.3s
- **lightgbm_v2** (full scope, origins [85, 98]): series per origin {'85': 20447, '98': 20445}, seconds {'85': 12.54, '98': 101.51}
  - origin 85: 1,792,474 training rows, 50 rounds, inner validation WAPE 0.165, fit 11.1s
  - origin 98: 3,632,785 training rows, 126 rounds, inner validation WAPE 0.213, fit 99.1s
- **seasonal_naive** (full scope, origins [85, 98]): series per origin {'85': 20447, '98': 20445}, seconds {'85': 0.08, '98': 0.08}
- **trailing_mean_13** (full scope, origins [85, 98]): series per origin {'85': 20447, '98': 20445}, seconds {'85': 0.06, '98': 0.05}

