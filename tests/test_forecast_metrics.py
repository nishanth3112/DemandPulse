import math

import numpy as np
import polars as pl
import pytest

from demandpulse.forecasting.metrics import (
    aggregate_level,
    bias,
    mase_scales,
    mase_summary,
    rmse,
    seasonal_scale,
    series_mase,
    summarise,
    wape,
)

Y = np.array([10.0, 20.0, 0.0])
YHAT = np.array([12.0, 18.0, 5.0])


def test_wape_is_total_absolute_error_over_total_actual():
    assert wape(Y, YHAT) == pytest.approx(9 / 30)


def test_wape_is_defined_when_some_actuals_are_zero():
    assert not math.isnan(wape(Y, YHAT))


def test_wape_is_nan_when_nothing_sold():
    assert math.isnan(wape(np.zeros(3), np.ones(3)))


def test_bias_is_signed_and_positive_for_over_forecasting():
    assert bias(Y, YHAT) == pytest.approx(5 / 30)
    assert bias(Y, Y * 0.5) < 0


def test_rmse_matches_definition():
    assert rmse(Y, YHAT) == pytest.approx(math.sqrt((4 + 4 + 25) / 3))


def test_seasonal_scale_is_mean_abs_seasonal_difference():
    values = np.array([1.0, 2.0, 4.0, 3.0, 6.0, 7.0])
    # season 2: |4-1|, |3-2|, |6-4|, |7-3| = 3, 1, 2, 4
    assert seasonal_scale(values, season=2) == pytest.approx(2.5)


def test_seasonal_scale_needs_more_than_one_season_of_data():
    assert math.isnan(seasonal_scale(np.arange(5.0), season=5))


def frame(rows):
    return pl.DataFrame(
        rows,
        schema={
            "model": pl.String,
            "origin": pl.Int64,
            "store_code": pl.String,
            "prod_code_30": pl.String,
            "horizon": pl.Int64,
            "week_idx": pl.Int64,
            "y": pl.Float64,
            "y_hat": pl.Float64,
        },
        orient="row",
    )


def test_aggregating_to_store_level_lets_errors_cancel():
    forecasts = frame(
        [
            ("m", 1, "S1", "A", 1, 2, 10.0, 15.0),  # +5
            ("m", 1, "S1", "B", 1, 2, 10.0, 5.0),  # -5
        ]
    )
    series = summarise(forecasts, ["model", "origin"])
    store = summarise(aggregate_level(forecasts, ["store_code"]), ["model", "origin"])
    assert series["wape"][0] == pytest.approx(10 / 20)
    assert store["wape"][0] == pytest.approx(0.0)
    assert store["n"][0] == 1


def test_summarise_groups_by_requested_columns():
    forecasts = frame(
        [
            ("a", 1, "S1", "A", 1, 2, 10.0, 12.0),
            ("a", 1, "S1", "A", 2, 3, 10.0, 8.0),
            ("b", 1, "S1", "A", 1, 2, 10.0, 10.0),
            ("b", 1, "S1", "A", 2, 3, 10.0, 10.0),
        ]
    )
    result = summarise(forecasts, ["model"])
    by_model = dict(zip(result["model"], result["wape"], strict=True))
    assert by_model == {"a": pytest.approx(0.2), "b": pytest.approx(0.0)}
    assert result["rmse"][0] == pytest.approx(2.0)


def test_mase_scales_use_only_data_up_to_the_origin():
    revenue = [1.0, 2.0, 4.0, 3.0, 6.0, 7.0, 100.0, 100.0]
    panel = pl.DataFrame(
        {
            "store_code": ["S"] * 8,
            "prod_code_30": ["A"] * 8,
            "week_idx": list(range(1, 9)),
            "revenue": revenue,
        }
    )
    at_6 = mase_scales(panel, origin=6, season=2)
    at_8 = mase_scales(panel, origin=8, season=2)
    assert at_6["scale"][0] == pytest.approx(2.5)  # unaffected by the 100s after the origin
    assert at_8["scale"][0] > at_6["scale"][0]


def test_mase_is_error_relative_to_seasonal_naive():
    forecasts = frame(
        [
            ("m", 6, "S", "A", 1, 7, 10.0, 12.0),
            ("m", 6, "S", "A", 2, 8, 10.0, 8.0),
        ]
    )
    scales = pl.DataFrame(
        {"store_code": ["S"], "prod_code_30": ["A"], "origin": [6], "scale": [2.0]}
    )
    per_series = series_mase(forecasts, scales)
    assert per_series["mase"][0] == pytest.approx(1.0)  # mean abs error 2 / scale 2
    summary = mase_summary(forecasts, scales, ["model"])
    assert summary["mase_median"][0] == pytest.approx(1.0)


def test_series_without_a_valid_scale_are_excluded_from_mase():
    forecasts = frame([("m", 6, "S", "A", 1, 7, 10.0, 12.0)])
    scales = pl.DataFrame(
        {"store_code": ["S"], "prod_code_30": ["A"], "origin": [6], "scale": [0.0]}
    )
    assert series_mase(forecasts, scales).height == 0
