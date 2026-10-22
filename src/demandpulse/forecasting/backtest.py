"""Rolling-origin backtesting with structural leakage guards.

At each origin a model receives only `history(panel, origin)` plus the calendar of the weeks to
forecast (week index and fiscal week, which are known in advance). It never sees a revenue value
after the origin. Every model is run through this same harness over the same origins, horizon and
series, so scores are directly comparable.
"""

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl

from demandpulse.forecasting.metrics import mase_scales
from demandpulse.forecasting.panel import (
    HORIZON,
    ORIGINS,
    SERIES_KEY,
    eligible_series,
    history,
)

logger = logging.getLogger(__name__)

FORECAST_COLUMNS = [*SERIES_KEY, "horizon", "week_idx", "y_hat"]


class ForecastError(ValueError):
    """A model returned forecasts the harness cannot trust."""


class Forecaster(Protocol):
    name: str

    def forecast(
        self, train: pl.DataFrame, future: pl.DataFrame, origin: int, horizon: int
    ) -> pl.DataFrame:
        """Forecast `horizon` weeks after `origin` for every series in `train`.

        `train` holds panel rows up to and including the origin. `future` lists the target
        weeks (week_idx, shop_week, fiscal_week). Return FORECAST_COLUMNS.
        """
        ...


@dataclass
class BacktestResult:
    model: str
    forecasts: pl.DataFrame
    scales: pl.DataFrame
    seconds: dict[int, float] = field(default_factory=dict)


def future_calendar(panel: pl.DataFrame, origin: int, horizon: int) -> pl.DataFrame:
    return (
        panel.filter((pl.col("week_idx") > origin) & (pl.col("week_idx") <= origin + horizon))
        .select("week_idx", "shop_week", "fiscal_week")
        .unique()
        .sort("week_idx")
        .with_columns((pl.col("week_idx") - origin).alias("horizon"))
    )


def validate_forecasts(
    forecasts: pl.DataFrame, series: pl.DataFrame, origin: int, horizon: int, model: str
) -> pl.DataFrame:
    """Check a model's output is complete, unique, correctly dated and finite."""
    missing = [c for c in FORECAST_COLUMNS if c not in forecasts.columns]
    if missing:
        raise ForecastError(f"{model}: missing columns {missing}")
    forecasts = forecasts.select(FORECAST_COLUMNS)
    expected_rows = series.height * horizon
    if forecasts.height != expected_rows:
        raise ForecastError(
            f"{model}: expected {expected_rows} forecasts ({series.height} series x {horizon} "
            f"weeks) at origin {origin}, got {forecasts.height}"
        )
    if forecasts.select(SERIES_KEY + ["horizon"]).is_duplicated().any():
        raise ForecastError(f"{model}: duplicate series/horizon rows at origin {origin}")
    wrong_week = forecasts.filter(pl.col("week_idx") != origin + pl.col("horizon"))
    if wrong_week.height or not forecasts["horizon"].is_between(1, horizon).all():
        raise ForecastError(f"{model}: week_idx does not equal origin + horizon at origin {origin}")
    unknown = forecasts.join(series, on=SERIES_KEY, how="anti")
    if unknown.height:
        raise ForecastError(f"{model}: forecasts for series outside the evaluation set")
    values = forecasts["y_hat"].to_numpy()
    if not np.isfinite(values).all():
        raise ForecastError(f"{model}: non-finite forecasts at origin {origin}")
    # Revenue cannot be negative; clipping is applied identically to every model. Integer and
    # float widths are normalised so results from different models always concatenate.
    return forecasts.with_columns(
        pl.col("horizon").cast(pl.Int64),
        pl.col("week_idx").cast(pl.Int64),
        pl.col("y_hat").cast(pl.Float64).clip(lower_bound=0.0),
    )


def run_backtest(
    panel: pl.DataFrame,
    model: Forecaster,
    origins: Iterable[int] = ORIGINS,
    horizon: int = HORIZON,
    series_for_origin: Callable[[int], pl.DataFrame] | None = None,
) -> BacktestResult:
    select = series_for_origin or (lambda origin: eligible_series(panel, origin, horizon=horizon))
    results: list[pl.DataFrame] = []
    scales: list[pl.DataFrame] = []
    seconds: dict[int, float] = {}
    for origin in origins:
        series = select(origin).select(SERIES_KEY)
        train = history(panel, origin).join(series, on=SERIES_KEY)
        if train.height and train["week_idx"].max() > origin:
            raise AssertionError("training data extends past the origin")  # pragma: no cover
        future = future_calendar(panel, origin, horizon)
        started = time.monotonic()
        raw = model.forecast(train, future, origin, horizon)
        seconds[origin] = round(time.monotonic() - started, 2)
        predicted = validate_forecasts(raw, series, origin, horizon, model.name)

        actual = panel.select(*SERIES_KEY, "week_idx", pl.col("revenue").alias("y"))
        scored = (
            predicted.join(actual, on=[*SERIES_KEY, "week_idx"], how="left")
            .with_columns(
                pl.lit(model.name).alias("model"),
                pl.lit(origin, dtype=pl.Int64).alias("origin"),
                pl.col("y").cast(pl.Float64),
            )
            .select("model", "origin", *SERIES_KEY, "horizon", "week_idx", "y", "y_hat")
        )
        if scored["y"].null_count():
            raise ForecastError(f"{model.name}: forecasts for weeks without actuals at {origin}")
        results.append(scored)
        scales.append(mase_scales(train, origin))
        logger.info(
            "%s origin %d: %d series, %.1fs", model.name, origin, series.height, seconds[origin]
        )
    return BacktestResult(
        model=model.name,
        forecasts=pl.concat(results),
        scales=pl.concat(scales),
        seconds=seconds,
    )
