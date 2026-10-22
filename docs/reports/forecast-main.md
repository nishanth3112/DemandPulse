# Forecast model comparison: run `main`

Weekly revenue per store x category (product hierarchy level 30). Forecast horizon 13 weeks from rolling origins at weeks 78, 91, 104 of the 117 data weeks (test windows weeks 79-91, 92-104, 105-117). Models see only revenue up to the origin. Negative forecasts are clipped to zero for every model.

Panel: 23,591 series. Representative subset: 240 series, stratified by store format x revenue tier (seed 20061).

WAPE = total absolute error / total actual revenue. Bias = (forecast - actual) / actual, positive means over-forecasting. MASE below 1 beats a one-year seasonal naive in-sample.

## Subset, final origin (all models, identical series and weeks)

| model | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|
| ensemble_baselines | 3,120 | 14.2% | -2.2% | 62.401 | 785,401.510 | 240 | 0.718 | 0.708 |
| ensemble | 3,120 | 14.5% | -5.5% | 64.242 | 785,401.510 | 240 | 0.711 | 0.685 |
| greykite | 3,120 | 14.9% | -0.9% | 69.803 | 785,401.510 | 240 | 0.739 | 0.713 |
| trailing_mean_13 | 3,120 | 15.5% | -5.3% | 69.417 | 785,401.510 | 240 | 0.755 | 0.737 |
| ets_damped | 3,120 | 16.1% | -5.7% | 74.436 | 785,401.510 | 240 | 0.761 | 0.716 |
| lightgbm_v2 | 3,120 | 17.7% | -10.9% | 80.636 | 785,401.510 | 240 | 0.810 | 0.760 |
| lightgbm | 3,120 | 18.1% | -12.0% | 82.830 | 785,401.510 | 240 | 0.825 | 0.765 |
| seasonal_naive | 3,120 | 18.2% | 1.0% | 80.692 | 785,401.510 | 240 | 0.910 | 0.907 |
| neuralprophet | 3,120 | 18.2% | -2.4% | 79.698 | 785,401.510 | 240 | 0.911 | 0.852 |

## Improvement over seasonal naive, subset final origin (paired bootstrap, 95% CI)

| model | baseline | units | improvement | ci_low | ci_high | win_rate | distinguishable |
|---|---|---|---|---|---|---|---|
| ensemble | seasonal_naive | 240 | 20.4% | 15.6% | 24.9% | 84.2% | yes |
| ensemble_baselines | seasonal_naive | 240 | 21.8% | 18.6% | 24.4% | 88.8% | yes |
| ets_damped | seasonal_naive | 240 | 11.5% | 3.0% | 19.5% | 76.7% | yes |
| greykite | seasonal_naive | 240 | 18.0% | 6.6% | 26.3% | 75.8% | yes |
| lightgbm | seasonal_naive | 240 | 0.2% | -10.7% | 9.1% | 68.3% | no |
| lightgbm_v2 | seasonal_naive | 240 | 2.9% | -7.5% | 11.9% | 69.6% | no |
| neuralprophet | seasonal_naive | 240 | -0.0% | -10.8% | 9.6% | 59.6% | no |
| trailing_mean_13 | seasonal_naive | 240 | 14.9% | 7.7% | 21.0% | 74.2% | yes |

## Subset, final origin, the 238 series holt_winters could forecast

| model | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|
| greykite | 3,094 | 14.1% | -2.0% | 60.578 | 773,479.250 | 238 | 0.735 | 0.710 |
| ensemble_baselines | 3,094 | 14.2% | -2.4% | 61.922 | 773,479.250 | 238 | 0.721 | 0.708 |
| ensemble | 3,094 | 14.3% | -5.9% | 63.207 | 773,479.250 | 238 | 0.713 | 0.688 |
| trailing_mean_13 | 3,094 | 15.1% | -6.0% | 66.765 | 773,479.250 | 238 | 0.755 | 0.735 |
| ets_damped | 3,094 | 15.6% | -6.6% | 70.413 | 773,479.250 | 238 | 0.759 | 0.712 |
| lightgbm_v2 | 3,094 | 17.2% | -11.8% | 77.687 | 773,479.250 | 238 | 0.809 | 0.754 |
| neuralprophet | 3,094 | 17.6% | -3.3% | 74.594 | 773,479.250 | 238 | 0.910 | 0.852 |
| lightgbm | 3,094 | 17.8% | -12.9% | 80.667 | 773,479.250 | 238 | 0.825 | 0.764 |
| seasonal_naive | 3,094 | 18.2% | 1.1% | 80.662 | 773,479.250 | 238 | 0.914 | 0.914 |
| holt_winters | 3,094 | 19.3% | -3.7% | 87.715 | 773,479.250 | 238 | 0.921 | 0.907 |

## Improvement over seasonal naive on those 238 series

| model | baseline | units | improvement | ci_low | ci_high | win_rate | distinguishable |
|---|---|---|---|---|---|---|---|
| ensemble | seasonal_naive | 238 | 21.4% | 16.9% | 25.3% | 84.5% | yes |
| ensemble_baselines | seasonal_naive | 238 | 22.2% | 19.2% | 24.8% | 89.1% | yes |
| ets_damped | seasonal_naive | 238 | 14.6% | 7.9% | 20.7% | 77.3% | yes |
| greykite | seasonal_naive | 238 | 22.7% | 16.9% | 27.7% | 76.5% | yes |
| holt_winters | seasonal_naive | 238 | -5.6% | -14.5% | 1.1% | 58.4% | no |
| lightgbm | seasonal_naive | 238 | 2.3% | -6.6% | 10.1% | 68.9% | no |
| lightgbm_v2 | seasonal_naive | 238 | 5.5% | -4.0% | 13.5% | 70.2% | no |
| neuralprophet | seasonal_naive | 238 | 3.4% | -4.6% | 10.8% | 60.1% | no |
| trailing_mean_13 | seasonal_naive | 238 | 17.1% | 11.0% | 22.6% | 74.8% | yes |

## Subset, all rolling origins pooled

| model | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|
| ensemble | 9,360 | 16.6% | -1.9% | 78.962 | 2,313,932.050 | 240 | 0.803 | 0.734 |
| ensemble_baselines | 9,360 | 17.1% | 0.1% | 80.329 | 2,313,932.050 | 240 | 0.837 | 0.768 |
| lightgbm | 9,360 | 18.0% | -6.0% | 89.684 | 2,313,932.050 | 240 | 0.828 | 0.748 |
| lightgbm_v2 | 9,360 | 18.0% | -5.5% | 90.047 | 2,313,932.050 | 240 | 0.826 | 0.748 |
| greykite | 9,360 | 18.5% | 2.1% | 95.394 | 2,313,932.050 | 240 | 0.865 | 0.774 |
| ets_damped | 9,360 | 18.6% | -1.6% | 94.424 | 2,313,932.050 | 240 | 0.857 | 0.777 |
| trailing_mean_13 | 9,360 | 18.9% | 0.3% | 96.161 | 2,313,932.050 | 240 | 0.865 | 0.779 |
| seasonal_naive | 9,360 | 20.6% | 0.0% | 99.844 | 2,313,932.050 | 240 | 1.024 | 0.952 |
| neuralprophet | 9,360 | 21.8% | 0.4% | 108.074 | 2,313,932.050 | 240 | 1.014 | 0.918 |

## Improvement over seasonal naive, subset rolling origins

| model | baseline | units | improvement | ci_low | ci_high | win_rate | distinguishable |
|---|---|---|---|---|---|---|---|
| ensemble | seasonal_naive | 720 | 19.4% | 15.9% | 22.4% | 84.4% | yes |
| ensemble_baselines | seasonal_naive | 720 | 17.0% | 14.4% | 19.7% | 85.1% | yes |
| ets_damped | seasonal_naive | 720 | 9.6% | 4.5% | 14.4% | 72.2% | yes |
| greykite | seasonal_naive | 720 | 10.4% | 5.2% | 15.1% | 71.1% | yes |
| lightgbm | seasonal_naive | 720 | 12.8% | 7.8% | 17.5% | 75.8% | yes |
| lightgbm_v2 | seasonal_naive | 720 | 12.8% | 8.3% | 17.2% | 76.0% | yes |
| neuralprophet | seasonal_naive | 720 | -5.6% | -11.7% | 0.1% | 57.9% | no |
| trailing_mean_13 | seasonal_naive | 720 | 8.4% | 3.1% | 13.5% | 69.4% | yes |

## Subset, by origin

| model | origin | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|---|
| ensemble | 78 | 3,120 | 18.3% | -2.3% | 94.273 | 784,774.670 | 240 | 0.921 | 0.822 |
| ensemble | 91 | 3,120 | 17.1% | 2.3% | 75.436 | 743,755.870 | 240 | 0.777 | 0.727 |
| ensemble | 104 | 3,120 | 14.5% | -5.5% | 64.242 | 785,401.510 | 240 | 0.711 | 0.685 |
| ensemble_baselines | 78 | 3,120 | 18.7% | -1.0% | 94.533 | 784,774.670 | 240 | 0.953 | 0.826 |
| ensemble_baselines | 91 | 3,120 | 18.5% | 3.8% | 80.795 | 743,755.870 | 240 | 0.839 | 0.792 |
| ensemble_baselines | 104 | 3,120 | 14.2% | -2.2% | 62.401 | 785,401.510 | 240 | 0.718 | 0.708 |
| ets_damped | 78 | 3,120 | 21.8% | -0.2% | 122.890 | 784,774.670 | 240 | 1.002 | 0.881 |
| ets_damped | 91 | 3,120 | 18.1% | 1.4% | 78.134 | 743,755.870 | 240 | 0.807 | 0.757 |
| ets_damped | 104 | 3,120 | 16.1% | -5.7% | 74.436 | 785,401.510 | 240 | 0.761 | 0.716 |
| greykite | 78 | 3,120 | 22.5% | 1.7% | 126.366 | 784,774.670 | 240 | 1.073 | 0.920 |
| greykite | 91 | 3,120 | 17.9% | 5.7% | 80.369 | 743,755.870 | 240 | 0.782 | 0.731 |
| greykite | 104 | 3,120 | 14.9% | -0.9% | 69.803 | 785,401.510 | 240 | 0.739 | 0.713 |
| lightgbm | 78 | 3,120 | 19.8% | -5.0% | 110.049 | 784,774.670 | 240 | 0.935 | 0.845 |
| lightgbm | 91 | 3,120 | 15.9% | -0.7% | 71.817 | 743,755.870 | 240 | 0.723 | 0.659 |
| lightgbm | 104 | 3,120 | 18.1% | -12.0% | 82.830 | 785,401.510 | 240 | 0.825 | 0.765 |
| lightgbm_v2 | 78 | 3,120 | 19.8% | -5.2% | 110.066 | 784,774.670 | 240 | 0.936 | 0.840 |
| lightgbm_v2 | 91 | 3,120 | 16.4% | 0.0% | 75.554 | 743,755.870 | 240 | 0.732 | 0.653 |
| lightgbm_v2 | 104 | 3,120 | 17.7% | -10.9% | 80.636 | 785,401.510 | 240 | 0.810 | 0.760 |
| neuralprophet | 78 | 3,120 | 23.8% | -1.3% | 128.019 | 784,774.670 | 240 | 1.145 | 1.018 |
| neuralprophet | 91 | 3,120 | 23.4% | 5.2% | 110.903 | 743,755.870 | 240 | 0.985 | 0.877 |
| neuralprophet | 104 | 3,120 | 18.2% | -2.4% | 79.698 | 785,401.510 | 240 | 0.911 | 0.852 |
| seasonal_naive | 78 | 3,120 | 21.6% | -2.8% | 118.088 | 784,774.670 | 240 | 1.120 | 0.957 |
| seasonal_naive | 91 | 3,120 | 22.1% | 2.0% | 97.212 | 743,755.870 | 240 | 1.043 | 1.000 |
| seasonal_naive | 104 | 3,120 | 18.2% | 1.0% | 80.692 | 785,401.510 | 240 | 0.910 | 0.907 |
| trailing_mean_13 | 78 | 3,120 | 22.5% | 0.9% | 125.098 | 784,774.670 | 240 | 1.034 | 0.906 |
| trailing_mean_13 | 91 | 3,120 | 18.7% | 5.5% | 85.279 | 743,755.870 | 240 | 0.805 | 0.736 |
| trailing_mean_13 | 104 | 3,120 | 15.5% | -5.3% | 69.417 | 785,401.510 | 240 | 0.755 | 0.737 |

## Full panel, by origin

| model | origin | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|---|
| ensemble | 78 | 265,811 | 19.4% | -2.6% | 132.620 | 66,009,516.020 | 20,447 | 0.929 | 0.795 |
| ensemble | 91 | 265,408 | 18.6% | 2.7% | 90.378 | 62,159,072.220 | 20,416 | 0.821 | 0.750 |
| ensemble | 104 | 265,889 | 15.4% | -5.6% | 80.663 | 66,434,770.430 | 20,453 | 0.709 | 0.686 |
| ensemble_baselines | 78 | 265,811 | 19.6% | -1.0% | 130.064 | 66,009,516.020 | 20,447 | 0.963 | 0.823 |
| ensemble_baselines | 91 | 265,408 | 20.2% | 4.5% | 98.287 | 62,159,072.220 | 20,416 | 0.876 | 0.803 |
| ensemble_baselines | 104 | 265,889 | 15.0% | -2.2% | 75.600 | 66,434,770.430 | 20,453 | 0.719 | 0.699 |
| lightgbm | 78 | 265,811 | 21.2% | -5.8% | 149.949 | 66,009,516.020 | 20,447 | 0.951 | 0.808 |
| lightgbm | 91 | 265,408 | 17.0% | -0.7% | 82.961 | 62,159,072.220 | 20,416 | 0.771 | 0.699 |
| lightgbm | 104 | 265,889 | 19.6% | -12.4% | 106.650 | 66,434,770.430 | 20,453 | 0.821 | 0.762 |
| lightgbm_v2 | 78 | 265,811 | 21.2% | -6.3% | 150.488 | 66,009,516.020 | 20,447 | 0.950 | 0.806 |
| lightgbm_v2 | 91 | 265,408 | 17.2% | -0.1% | 84.778 | 62,159,072.220 | 20,416 | 0.775 | 0.701 |
| lightgbm_v2 | 104 | 265,889 | 19.1% | -11.6% | 104.510 | 66,434,770.430 | 20,453 | 0.810 | 0.753 |
| seasonal_naive | 78 | 265,811 | 22.9% | -1.1% | 142.000 | 66,009,516.020 | 20,447 | 1.175 | 0.984 |
| seasonal_naive | 91 | 265,408 | 23.4% | 2.2% | 115.174 | 62,159,072.220 | 20,416 | 1.051 | 0.972 |
| seasonal_naive | 104 | 265,889 | 19.1% | 1.1% | 91.141 | 66,434,770.430 | 20,453 | 0.915 | 0.904 |
| trailing_mean_13 | 78 | 265,811 | 23.7% | -0.9% | 161.056 | 66,009,516.020 | 20,447 | 1.043 | 0.891 |
| trailing_mean_13 | 91 | 265,408 | 22.0% | 6.7% | 112.836 | 62,159,072.220 | 20,416 | 0.885 | 0.772 |
| trailing_mean_13 | 104 | 265,889 | 17.1% | -5.6% | 94.965 | 66,434,770.430 | 20,453 | 0.761 | 0.713 |

## Full panel, pooled over origins

| model | n | wape | bias | rmse | actual_total | series | mase_mean | mase_median |
|---|---|---|---|---|---|---|---|---|
| ensemble | 797,108 | 17.8% | -1.9% | 103.707 | 194,603,358.670 | 20,752 | 0.820 | 0.737 |
| ensemble_baselines | 797,108 | 18.2% | 0.3% | 103.750 | 194,603,358.670 | 20,752 | 0.853 | 0.766 |
| lightgbm_v2 | 797,108 | 19.2% | -6.1% | 116.570 | 194,603,358.670 | 20,752 | 0.845 | 0.749 |
| lightgbm | 797,108 | 19.3% | -6.5% | 116.549 | 194,603,358.670 | 20,752 | 0.848 | 0.752 |
| trailing_mean_13 | 797,108 | 20.9% | -0.1% | 126.085 | 194,603,358.670 | 20,752 | 0.896 | 0.783 |
| seasonal_naive | 797,108 | 21.7% | 0.7% | 117.948 | 194,603,358.670 | 20,752 | 1.047 | 0.947 |

## Improvement over seasonal naive, full panel

| model | baseline | units | improvement | ci_low | ci_high | win_rate | distinguishable |
|---|---|---|---|---|---|---|---|
| ensemble | seasonal_naive | 61,316 | 18.2% | 17.8% | 18.7% | 84.5% | yes |
| ensemble_baselines | seasonal_naive | 61,316 | 16.1% | 15.7% | 16.5% | 85.4% | yes |
| lightgbm | seasonal_naive | 61,316 | 11.3% | 10.5% | 11.9% | 74.9% | yes |
| lightgbm_v2 | seasonal_naive | 61,316 | 11.7% | 11.0% | 12.3% | 75.5% | yes |
| trailing_mean_13 | seasonal_naive | 61,316 | 3.8% | 2.9% | 4.7% | 69.5% | yes |

## Full panel, accuracy summed to store level

| model | n | wape | bias |
|---|---|---|---|
| ensemble | 29,367 | 9.8% | -1.9% |
| ensemble_baselines | 29,367 | 9.4% | 0.3% |
| lightgbm | 29,367 | 12.3% | -6.5% |
| lightgbm_v2 | 29,367 | 12.2% | -6.1% |
| seasonal_naive | 29,367 | 10.6% | 0.7% |
| trailing_mean_13 | 29,367 | 11.9% | -0.1% |

## Full panel, accuracy summed to category level

| model | n | wape | bias |
|---|---|---|---|
| ensemble | 1,170 | 9.6% | -1.9% |
| ensemble_baselines | 1,170 | 9.2% | 0.3% |
| lightgbm | 1,170 | 12.5% | -6.5% |
| lightgbm_v2 | 1,170 | 12.2% | -6.1% |
| seasonal_naive | 1,170 | 9.8% | 0.7% |
| trailing_mean_13 | 1,170 | 13.1% | -0.1% |

## Full panel, accuracy summed to the whole network

| model | n | wape | bias |
|---|---|---|---|
| ensemble | 39 | 5.8% | -1.9% |
| ensemble_baselines | 39 | 4.4% | 0.3% |
| lightgbm | 39 | 9.2% | -6.5% |
| lightgbm_v2 | 39 | 8.9% | -6.1% |
| seasonal_naive | 39 | 3.7% | 0.7% |
| trailing_mean_13 | 39 | 7.6% | -0.1% |

## Full panel, by forecast horizon

| model | horizon_bucket | wape | bias |
|---|---|---|---|
| ensemble | weeks 1-4 | 16.3% | -0.2% |
| ensemble | weeks 5-8 | 17.3% | -3.6% |
| ensemble | weeks 9-13 | 19.3% | -2.0% |
| ensemble_baselines | weeks 1-4 | 17.3% | 2.1% |
| ensemble_baselines | weeks 5-8 | 17.9% | -1.6% |
| ensemble_baselines | weeks 9-13 | 19.2% | 0.5% |
| lightgbm | weeks 1-4 | 16.8% | -4.6% |
| lightgbm | weeks 5-8 | 18.4% | -7.6% |
| lightgbm | weeks 9-13 | 21.8% | -6.9% |
| lightgbm_v2 | weeks 1-4 | 16.6% | -4.1% |
| lightgbm_v2 | weeks 5-8 | 18.4% | -7.2% |
| lightgbm_v2 | weeks 9-13 | 21.8% | -6.8% |
| seasonal_naive | weeks 1-4 | 20.8% | 0.9% |
| seasonal_naive | weeks 5-8 | 21.6% | -1.4% |
| seasonal_naive | weeks 9-13 | 22.6% | 2.3% |
| trailing_mean_13 | weeks 1-4 | 19.2% | 3.3% |
| trailing_mean_13 | weeks 5-8 | 19.3% | -1.7% |
| trailing_mean_13 | weeks 9-13 | 23.5% | -1.3% |

## Full panel, by store format

| model | store_format | n | wape | bias |
|---|---|---|---|---|
| ensemble | LS | 365,911 | 16.5% | -2.2% |
| ensemble | MS | 241,332 | 20.5% | -1.5% |
| ensemble | SS | 157,638 | 25.2% | -1.0% |
| ensemble | XLS | 32,227 | 15.1% | -2.1% |
| ensemble_baselines | LS | 365,911 | 16.8% | 0.2% |
| ensemble_baselines | MS | 241,332 | 21.2% | 0.5% |
| ensemble_baselines | SS | 157,638 | 26.6% | 1.0% |
| ensemble_baselines | XLS | 32,227 | 15.2% | 0.3% |
| lightgbm | LS | 365,911 | 18.2% | -6.9% |
| lightgbm | MS | 241,332 | 21.5% | -5.5% |
| lightgbm | SS | 157,638 | 25.2% | -5.0% |
| lightgbm | XLS | 32,227 | 17.5% | -6.9% |
| lightgbm_v2 | LS | 365,911 | 18.1% | -6.5% |
| lightgbm_v2 | MS | 241,332 | 21.4% | -5.1% |
| lightgbm_v2 | SS | 157,638 | 25.2% | -4.5% |
| lightgbm_v2 | XLS | 32,227 | 17.5% | -6.8% |
| seasonal_naive | LS | 365,911 | 19.7% | 1.1% |
| seasonal_naive | MS | 241,332 | 25.8% | -0.5% |
| seasonal_naive | SS | 157,638 | 34.1% | -0.7% |
| seasonal_naive | XLS | 32,227 | 17.6% | 1.5% |
| trailing_mean_13 | LS | 365,911 | 19.7% | -0.8% |
| trailing_mean_13 | MS | 241,332 | 23.5% | 1.5% |
| trailing_mean_13 | SS | 157,638 | 27.3% | 2.7% |
| trailing_mean_13 | XLS | 32,227 | 18.6% | -1.0% |

## Run details

- **ensemble** (full scope, origins [78, 91, 104]): series per origin {'78': 20447, '91': 20416, '104': 20453}, seconds {}
  - equal-weight mean of component forecasts of seasonal_naive, trailing_mean_13, lightgbm. Added after the component results were seen; equal weights, nothing fitted.
- **ensemble_baselines** (full scope, origins [78, 91, 104]): series per origin {'78': 20447, '91': 20416, '104': 20453}, seconds {}
  - equal-weight mean of component forecasts of seasonal_naive, trailing_mean_13. Added after the component results were seen; equal weights, nothing fitted.
- **ets_damped** (subset scope, origins [78, 91, 104]): series per origin {'78': 240, '91': 240, '104': 240}, seconds {'78': 4.88, '91': 2.9, '104': 2.96}
  - fits 720, trailing-mean fallbacks 0
- **greykite** (subset scope, origins [78, 91, 104]): series per origin {'78': 240, '91': 240, '104': 240}, seconds {'78': 305.63, '91': 258.24, '104': 308.09}
- **holt_winters** (subset scope, origins [104]): series per origin {'104': 238}, seconds {'104': 9.71}
  - 2 subset series excluded (stores opened too recently for this model's history requirement)
  - fits 238, trailing-mean fallbacks 0
- **lightgbm** (full scope, origins [78, 91, 104]): series per origin {'78': 20447, '91': 20416, '104': 20453}, seconds {'78': 21.83, '91': 86.1, '104': 140.58}
  - origin 78: 997,835 training rows, 200 rounds, inner validation WAPE nan, fit 19.9s
  - origin 91: 2,582,022 training rows, 194 rounds, inner validation WAPE 0.211, fit 84.1s
  - origin 104: 4,423,873 training rows, 159 rounds, inner validation WAPE 0.172, fit 138.3s
- **lightgbm_v2** (full scope, origins [78, 91, 104]): series per origin {'78': 20447, '91': 20416, '104': 20453}, seconds {'78': 25.33, '91': 100.6, '104': 274.17}
  - origin 78: 997,835 training rows, 200 rounds, inner validation WAPE nan, fit 23.6s
  - origin 91: 2,582,022 training rows, 190 rounds, inner validation WAPE 0.212, fit 97.1s
  - origin 104: 4,423,873 training rows, 300 rounds, inner validation WAPE 0.170, fit 271.7s
- **neuralprophet** (subset scope, origins [78, 91, 104]): series per origin {'78': 240, '91': 240, '104': 240}, seconds {'78': 47.29, '91': 57.83, '104': 65.37}
- **seasonal_naive** (full scope, origins [78, 91, 104]): series per origin {'78': 20447, '91': 20416, '104': 20453}, seconds {'78': 0.12, '91': 0.11, '104': 0.08}
- **trailing_mean_13** (full scope, origins [78, 91, 104]): series per origin {'78': 20447, '91': 20416, '104': 20453}, seconds {'78': 0.06, '91': 0.06, '104': 0.05}

