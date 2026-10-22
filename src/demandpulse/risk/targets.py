"""Revenue targets and the shortfall of forecast and actual revenue against them.

The dataset has no budgets, so every target here is synthetic and hypothetical: last year's
revenue for the same weeks and series, scaled by a growth assumption. Two assumptions are used.
`flat_yoy` expects exactly last year. `trailing_growth` expects last year grown by the network's
like-for-like year-over-year growth over the 13 weeks up to the forecast origin, so it only uses
data known at the origin.

Targets are built from the same series the forecast covers, so a store target and a store
forecast always add up the same set of categories.
"""

import polars as pl

from demandpulse.forecasting.panel import HORIZON, SERIES_KEY, YEAR_WEEKS

FLAT_YOY = "flat_yoy"
TRAILING_GROWTH = "trailing_growth"
TREND_WEEKS = HORIZON


def network_growth(panel: pl.DataFrame, origin: int, weeks: int = TREND_WEEKS) -> float:
    """Like-for-like growth of the last `weeks` weeks to the origin against a year earlier.

    Only series trading in both windows count, so a newly opened store does not look like growth.
    """
    recent = (
        panel.filter((pl.col("week_idx") > origin - weeks) & (pl.col("week_idx") <= origin))
        .group_by(SERIES_KEY)
        .agg(pl.col("revenue").sum().alias("recent"), pl.len().alias("n_recent"))
    )
    year_ago = (
        panel.filter(
            (pl.col("week_idx") > origin - weeks - YEAR_WEEKS)
            & (pl.col("week_idx") <= origin - YEAR_WEEKS)
        )
        .group_by(SERIES_KEY)
        .agg(pl.col("revenue").sum().alias("before"), pl.len().alias("n_before"))
    )
    both = recent.join(year_ago, on=SERIES_KEY).filter(
        (pl.col("n_recent") == weeks) & (pl.col("n_before") == weeks)
    )
    if not both.height or both["before"].sum() <= 0:
        raise ValueError(
            f"Not enough history to measure growth at origin {origin}: "
            f"need {weeks} weeks and the same weeks a year earlier"
        )
    return float(both["recent"].sum() / both["before"].sum() - 1)


def target_growth(panel: pl.DataFrame, origin: int) -> dict[str, float]:
    """Growth assumption per target definition at an origin."""
    return {FLAT_YOY: 0.0, TRAILING_GROWTH: network_growth(panel, origin)}


def window_table(
    forecasts: pl.DataFrame,
    panel: pl.DataFrame,
    level: list[str],
    growth: dict[str, float],
) -> pl.DataFrame:
    """One row per model, origin, entity and target definition over the 13-week window.

    `forecast_shortfall` is target minus forecast (positive = forecast falls short of target);
    `realised_shortfall` is target minus the actual revenue. The `*_miss` flags are the same
    comparisons as booleans. `growth` holds one rate per definition, so a caller with several
    origins calls this once per origin.
    """
    last_year = panel.select(
        *SERIES_KEY,
        (pl.col("week_idx") + YEAR_WEEKS).alias("week_idx"),
        pl.col("revenue").alias("last_year"),
    )
    joined = forecasts.join(last_year, on=[*SERIES_KEY, "week_idx"], how="left")
    if joined["last_year"].null_count():
        raise ValueError("Some forecast weeks have no last-year revenue to build a target from")

    windows = joined.group_by(["model", "origin", *level]).agg(
        pl.col("y").sum().alias("actual"),
        pl.col("y_hat").sum().alias("forecast"),
        pl.col("last_year").sum(),
    )
    frames = [
        windows.with_columns(
            pl.lit(name).alias("target_def"),
            pl.lit(rate).alias("growth"),
            (pl.col("last_year") * (1 + rate)).alias("target"),
        )
        for name, rate in growth.items()
    ]
    return (
        pl.concat(frames)
        .with_columns(
            (pl.col("target") - pl.col("forecast")).alias("forecast_shortfall"),
            (pl.col("target") - pl.col("actual")).alias("realised_shortfall"),
        )
        .with_columns(
            (pl.col("forecast_shortfall") > 0).alias("forecast_miss"),
            (pl.col("realised_shortfall") > 0).alias("realised_miss"),
        )
        .sort(["model", "origin", "target_def", *level])
    )
