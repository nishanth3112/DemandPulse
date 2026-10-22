import logging

import numpy as np
import polars as pl
import pytest

from demandpulse.forecasting.backtest import ForecastError, future_calendar, run_backtest
from demandpulse.forecasting.models import ets as ets_module
from demandpulse.forecasting.models.baselines import SeasonalNaive, TrailingMean
from demandpulse.forecasting.models.ets import ets_damped, holt_winters
from demandpulse.forecasting.panel import SERIES_KEY

SEASON = 8
WEEKS = 40
HORIZON = 4


def make_panel(values_by_series: dict[tuple[str, str], np.ndarray]) -> pl.DataFrame:
    rows = []
    for (store, category), values in values_by_series.items():
        for i, value in enumerate(values, start=1):
            rows.append((store, category, i, 200600 + i, i % 52 or 52, float(value)))
    return pl.DataFrame(
        rows,
        schema={
            "store_code": pl.String,
            "prod_code_30": pl.String,
            "week_idx": pl.Int64,
            "shop_week": pl.Int64,
            "fiscal_week": pl.Int64,
            "revenue": pl.Float64,
        },
        orient="row",
    )


@pytest.fixture
def seasonal_panel():
    weeks = np.arange(WEEKS)
    pattern = np.array([10, 12, 15, 20, 30, 20, 15, 12], dtype=float)
    return make_panel(
        {
            ("S1", "A"): pattern[weeks % SEASON],
            ("S1", "B"): 2 * pattern[weeks % SEASON] + 5,
            ("S2", "A"): 100 + weeks * 2.0,
        }
    )


def one_origin(panel, model, origin=32):
    series = panel.select(SERIES_KEY).unique()
    return run_backtest(panel, model, (origin,), HORIZON, lambda _o: series)


def test_seasonal_naive_repeats_last_seasons_values(seasonal_panel):
    result = one_origin(seasonal_panel, SeasonalNaive(season=SEASON)).forecasts
    check = result.join(
        seasonal_panel.select(
            *SERIES_KEY,
            (pl.col("week_idx") + SEASON).alias("week_idx"),
            pl.col("revenue").alias("last"),
        ),
        on=[*SERIES_KEY, "week_idx"],
    )
    assert (check["y_hat"] == check["last"]).all()
    pure = result.filter(pl.col("store_code") == "S1")
    assert (pure["y_hat"] == pure["y"]).all()  # a perfectly seasonal series is forecast exactly


def test_seasonal_naive_refuses_horizons_beyond_one_season(seasonal_panel):
    model = SeasonalNaive(season=3)
    future = future_calendar(seasonal_panel, 32, 4)
    with pytest.raises(ForecastError, match="exceeds the season"):
        model.forecast(seasonal_panel.filter(pl.col("week_idx") <= 32), future, 32, 4)


def test_seasonal_naive_fails_loudly_without_a_year_ago_value():
    panel = make_panel({("S", "A"): np.arange(1.0, 7.0)})
    model = SeasonalNaive(season=8)
    train = panel.filter(pl.col("week_idx") <= 4)
    with pytest.raises(ForecastError, match="year-ago"):
        model.forecast(train, future_calendar(panel, 4, 2), 4, 2)


def test_trailing_mean_is_flat_and_uses_the_last_window(seasonal_panel):
    result = one_origin(seasonal_panel, TrailingMean(window=4)).forecasts
    series = result.filter((pl.col("store_code") == "S2") & (pl.col("prod_code_30") == "A"))
    expected = np.mean([100 + w * 2.0 for w in range(28, 32)])  # weeks 29..32 are indices 28..31
    assert series["y_hat"].n_unique() == 1
    assert series["y_hat"][0] == pytest.approx(expected)


def test_damped_ets_follows_a_trend(seasonal_panel):
    result = one_origin(seasonal_panel, ets_damped()).forecasts
    trend = result.filter(pl.col("store_code") == "S2")
    assert trend["y_hat"].to_list() == sorted(trend["y_hat"].to_list())  # still rising
    assert abs(trend["y_hat"][0] - trend["y"][0]) < 10


def test_holt_winters_learns_a_clean_seasonal_pattern(seasonal_panel):
    model = holt_winters(season=SEASON)
    result = one_origin(seasonal_panel, model).forecasts
    seasonal = result.filter(pl.col("store_code") == "S1")
    scaled_error = (seasonal["y"] - seasonal["y_hat"]).abs().sum() / seasonal["y"].sum()
    assert scaled_error < 0.1
    assert model.fits == 3
    assert model.fallbacks == 0


def test_holt_winters_needs_two_full_seasons(seasonal_panel):
    model = holt_winters(season=SEASON)
    series = seasonal_panel.select(SERIES_KEY).unique()
    with pytest.raises(ForecastError, match="two full seasons"):
        run_backtest(seasonal_panel, model, (SEASON + 4,), HORIZON, lambda _o: series)


def test_failed_fits_use_a_logged_counted_fallback(seasonal_panel, monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(ets_module, "ExponentialSmoothing", boom)
    model = ets_damped()
    with caplog.at_level(logging.WARNING):
        result = one_origin(seasonal_panel, model).forecasts

    assert model.fallbacks == model.fits == 3
    assert "synthetic failure" in caplog.text
    flat = result.filter((pl.col("store_code") == "S2") & (pl.col("prod_code_30") == "A"))
    assert flat["y_hat"].n_unique() == 1


def test_parallel_and_serial_fits_agree(seasonal_panel):
    # chunk_series=1 puts every series in its own chunk, so the process pool really is used.
    serial = one_origin(seasonal_panel, ets_damped(workers=1, chunk_series=1)).forecasts
    parallel_model = ets_damped(workers=2, chunk_series=1)
    parallel = one_origin(seasonal_panel, parallel_model).forecasts
    order = [*SERIES_KEY, "horizon"]
    assert serial.sort(order)["y_hat"].to_list() == pytest.approx(
        parallel.sort(order)["y_hat"].to_list()
    )
    assert parallel_model.fits == 3
