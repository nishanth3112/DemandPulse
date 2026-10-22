import polars as pl
import pytest

from demandpulse.forecasting.backtest import (
    FORECAST_COLUMNS,
    ForecastError,
    future_calendar,
    run_backtest,
    validate_forecasts,
)
from demandpulse.forecasting.panel import SERIES_KEY, eligible_series

HORIZON = 2
ORIGINS = (5, 6)
SMALL = {"horizon": HORIZON, "min_history": 4, "min_nonzero": 2, "year_weeks": 4}


def select(panel):
    return lambda origin: eligible_series(panel, origin, **SMALL)


class RecentMean:
    """Forecast the mean of the last 3 observed weeks, recording what it was shown."""

    name = "recent_mean"

    def __init__(self):
        self.seen_max_week = {}

    def forecast(self, train, future, origin, horizon):
        self.seen_max_week[origin] = train["week_idx"].max()
        level = (
            train.sort("week_idx")
            .group_by(SERIES_KEY)
            .agg(pl.col("revenue").tail(3).mean().alias("y_hat"))
        )
        return level.join(future.select("horizon", "week_idx"), how="cross").select(
            FORECAST_COLUMNS
        )


def test_model_only_sees_weeks_up_to_the_origin(forecast_panel):
    model = RecentMean()
    run_backtest(forecast_panel, model, ORIGINS, HORIZON, select(forecast_panel))
    assert model.seen_max_week == {5: 5, 6: 6}


def test_forecast_rows_cover_every_series_and_horizon(forecast_panel):
    result = run_backtest(forecast_panel, RecentMean(), ORIGINS, HORIZON, select(forecast_panel))
    per_origin = result.forecasts.group_by("origin").len().sort("origin")
    for origin, count in per_origin.iter_rows():
        assert count == select(forecast_panel)(origin).height * HORIZON
    assert set(result.forecasts["horizon"]) == {1, 2}


def test_actuals_are_the_panel_revenue_in_the_target_week(forecast_panel):
    result = run_backtest(forecast_panel, RecentMean(), ORIGINS, HORIZON, select(forecast_panel))
    check = result.forecasts.join(
        forecast_panel.select(*SERIES_KEY, "week_idx", "revenue"), on=[*SERIES_KEY, "week_idx"]
    )
    assert (check["y"] == check["revenue"]).all()
    assert (
        result.forecasts["week_idx"] == result.forecasts["origin"] + result.forecasts["horizon"]
    ).all()


def test_changing_future_revenue_cannot_change_any_forecast(forecast_panel):
    """The leakage proof: wipe and inflate everything after each origin, forecasts must not move."""
    baseline = run_backtest(forecast_panel, RecentMean(), ORIGINS, HORIZON, select(forecast_panel))
    series_by_origin = {o: select(forecast_panel)(o) for o in ORIGINS}
    forecasts = []
    for origin in ORIGINS:
        tampered = forecast_panel.with_columns(
            revenue=pl.when(pl.col("week_idx") > origin)
            .then(pl.col("revenue") * 1000 + 7)
            .otherwise(pl.col("revenue"))
        )
        run = run_backtest(
            tampered, RecentMean(), (origin,), HORIZON, lambda o: series_by_origin[o]
        )
        forecasts.append(run.forecasts)
    tampered_forecasts = pl.concat(forecasts)
    keys = ["origin", *SERIES_KEY, "horizon"]
    compared = baseline.forecasts.join(tampered_forecasts, on=keys, suffix="_t")
    assert compared.height == baseline.forecasts.height
    assert (compared["y_hat"] == compared["y_hat_t"]).all()
    assert (compared["y"] != compared["y_t"]).any()  # the tampering did change the actuals


def test_mase_scales_are_returned_per_series_and_origin(forecast_panel):
    result = run_backtest(forecast_panel, RecentMean(), ORIGINS, HORIZON, select(forecast_panel))
    assert set(result.scales["origin"]) == set(ORIGINS)


def good_forecasts(panel, origin=5):
    series = select(panel)(origin).select(SERIES_KEY)
    model = RecentMean()
    train = panel.filter(pl.col("week_idx") <= origin).join(series, on=SERIES_KEY)
    future = future_calendar(panel, origin, HORIZON)
    return model.forecast(train, future, origin, HORIZON), series


def test_validation_accepts_complete_forecasts(forecast_panel):
    forecasts, series = good_forecasts(forecast_panel)
    assert validate_forecasts(forecasts, series, 5, HORIZON, "m").height == forecasts.height


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: f.head(f.height - 1), "expected"),
        (lambda f: pl.concat([f, f.head(1)]), "expected"),
        (lambda f: f.drop("y_hat"), "missing columns"),
        (lambda f: f.with_columns(week_idx=pl.col("week_idx") + 1), "origin \\+ horizon"),
        (lambda f: f.with_columns(y_hat=pl.lit(float("nan"))), "non-finite"),
        (lambda f: f.with_columns(y_hat=pl.lit(float("inf"))), "non-finite"),
    ],
)
def test_validation_rejects_untrustworthy_forecasts(forecast_panel, mutate, message):
    forecasts, series = good_forecasts(forecast_panel)
    with pytest.raises(ForecastError, match=message):
        validate_forecasts(mutate(forecasts), series, 5, HORIZON, "m")


def test_validation_rejects_series_outside_the_evaluation_set(forecast_panel):
    forecasts, series = good_forecasts(forecast_panel)
    swapped = forecasts.with_columns(prod_code_30=pl.col("prod_code_30") + "_unknown")
    with pytest.raises(ForecastError, match="outside the evaluation set"):
        validate_forecasts(swapped, series, 5, HORIZON, "m")


def test_duplicate_series_horizon_rows_are_rejected(forecast_panel):
    forecasts, series = good_forecasts(forecast_panel)
    duplicated = pl.concat([forecasts.head(1), forecasts.slice(1)]).with_columns(
        horizon=pl.lit(1), week_idx=pl.lit(6)
    )
    with pytest.raises(ForecastError, match="duplicate"):
        validate_forecasts(duplicated, series, 5, HORIZON, "m")


def test_negative_forecasts_are_clipped_to_zero(forecast_panel):
    forecasts, series = good_forecasts(forecast_panel)
    negative = forecasts.with_columns(y_hat=pl.lit(-5.0))
    clipped = validate_forecasts(negative, series, 5, HORIZON, "m")
    assert (clipped["y_hat"] == 0.0).all()


def test_future_calendar_lists_only_target_weeks(forecast_panel):
    calendar = future_calendar(forecast_panel, 5, HORIZON)
    assert calendar["week_idx"].to_list() == [6, 7]
    assert calendar["horizon"].to_list() == [1, 2]


class FlatForecaster:
    """A second, differently built model: the same constant for every series and week."""

    name = "flat"

    def forecast(self, train, future, origin, horizon):
        series = train.select(SERIES_KEY).unique()
        return (
            series.join(future.select("horizon", "week_idx"), how="cross")
            .with_columns(y_hat=pl.lit(10, dtype=pl.Int32))
            .select(FORECAST_COLUMNS)
        )


def test_results_from_different_models_share_one_schema(forecast_panel):
    first = run_backtest(forecast_panel, RecentMean(), ORIGINS, HORIZON, select(forecast_panel))
    second = run_backtest(
        forecast_panel, FlatForecaster(), ORIGINS, HORIZON, select(forecast_panel)
    )
    assert first.forecasts.schema == second.forecasts.schema
    assert first.scales.schema == second.scales.schema
    assert pl.concat([first.forecasts, second.forecasts]).height == 2 * first.forecasts.height
