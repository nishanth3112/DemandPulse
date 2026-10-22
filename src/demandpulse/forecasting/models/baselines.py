"""Benchmark forecasters every other model has to beat."""

import polars as pl

from demandpulse.forecasting.backtest import FORECAST_COLUMNS, ForecastError
from demandpulse.forecasting.panel import SERIES_KEY, YEAR_WEEKS


class SeasonalNaive:
    """Forecast each week with revenue from the same week one season (a year) earlier."""

    def __init__(self, season: int = YEAR_WEEKS) -> None:
        self.season = season
        self.name = "seasonal_naive"

    def forecast(
        self, train: pl.DataFrame, future: pl.DataFrame, origin: int, horizon: int
    ) -> pl.DataFrame:
        if horizon > self.season:
            raise ForecastError(
                f"{self.name}: horizon {horizon} exceeds the season {self.season}, so the "
                "year-ago value would not be observed yet"
            )
        series = train.select(SERIES_KEY).unique()
        targets = series.join(future.select("horizon", "week_idx"), how="cross")
        year_ago = train.select(
            *SERIES_KEY,
            (pl.col("week_idx") + self.season).alias("week_idx"),
            pl.col("revenue").alias("y_hat"),
        )
        joined = targets.join(year_ago, on=[*SERIES_KEY, "week_idx"], how="left")
        if joined["y_hat"].null_count():
            raise ForecastError(f"{self.name}: some series lack a year-ago observation")
        return joined.select(FORECAST_COLUMNS)


class TrailingMean:
    """Forecast a flat line at the mean of the most recent `window` weeks."""

    def __init__(self, window: int = 13) -> None:
        self.window = window
        self.name = f"trailing_mean_{window}"

    def forecast(
        self, train: pl.DataFrame, future: pl.DataFrame, origin: int, horizon: int
    ) -> pl.DataFrame:
        level = (
            train.sort("week_idx")
            .group_by(SERIES_KEY)
            .agg(pl.col("revenue").tail(self.window).mean().alias("y_hat"))
        )
        return level.join(future.select("horizon", "week_idx"), how="cross").select(
            FORECAST_COLUMNS
        )
