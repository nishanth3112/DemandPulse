"""Per-entity trend and revenue-driver measures computed from history up to an origin.

`entity_growth` and `entity_volatility` work on the forecasting panel at any level (store or
store x category). `driver_changes` splits a store's year-over-year revenue change into traffic
(baskets per week), basket size (units per basket) and price/mix (revenue per unit), using true
store-week baskets from Gold. Baskets are not additive across categories, so this split exists at
store level only.
"""

import polars as pl

from demandpulse.forecasting.panel import YEAR_WEEKS

TREND_WEEKS = 13
VOLATILITY_WEEKS = 26


def _window_sums(
    frame: pl.DataFrame, level: list[str], start: int, end: int, values: list[str], suffix: str
) -> pl.DataFrame:
    """Sum `values` over weeks (start, end] per entity, with the number of distinct weeks seen."""
    return (
        frame.filter((pl.col("week_idx") > start) & (pl.col("week_idx") <= end))
        .group_by(level)
        .agg(
            *(pl.col(v).sum().alias(f"{v}_{suffix}") for v in values),
            pl.col("week_idx").n_unique().alias(f"weeks_{suffix}"),
        )
    )


def entity_growth(
    panel: pl.DataFrame, level: list[str], origin: int, weeks: int = TREND_WEEKS
) -> pl.DataFrame:
    """Own revenue growth of the last `weeks` weeks against the same weeks a year earlier.

    Null where either window is incomplete (a new entity) or last year's revenue is not positive.
    """
    recent = _window_sums(panel, level, origin - weeks, origin, ["revenue"], "now")
    before = _window_sums(
        panel, level, origin - weeks - YEAR_WEEKS, origin - YEAR_WEEKS, ["revenue"], "ago"
    )
    both = recent.join(before, on=level, how="left")
    complete = (pl.col("weeks_now") == weeks) & (pl.col("weeks_ago") == weeks)
    usable = complete & (pl.col("revenue_ago") > 0)
    return both.select(
        *level,
        pl.when(usable)
        .then(pl.col("revenue_now") / pl.col("revenue_ago") - 1)
        .otherwise(None)
        .alias("growth"),
    )


def entity_volatility(
    panel: pl.DataFrame, level: list[str], origin: int, weeks: int = VOLATILITY_WEEKS
) -> pl.DataFrame:
    """Coefficient of variation of weekly revenue over the last `weeks` weeks (null if no sales)."""
    weekly = (
        panel.filter((pl.col("week_idx") > origin - weeks) & (pl.col("week_idx") <= origin))
        .group_by([*level, "week_idx"])
        .agg(pl.col("revenue").sum())
    )
    return weekly.group_by(level).agg(
        pl.when(pl.col("revenue").mean() > 0)
        .then(pl.col("revenue").std(ddof=0) / pl.col("revenue").mean())
        .otherwise(None)
        .alias("volatility")
    )


def driver_changes(
    store_weeks: pl.DataFrame, origin: int, weeks: int = TREND_WEEKS
) -> pl.DataFrame:
    """Year-over-year change of each revenue driver per store over the last `weeks` weeks.

    revenue = baskets x units per basket x revenue per unit, so the three changes below multiply
    to `1 + revenue_growth`. `dominant_driver` is the factor with the most negative change, or
    "none" when all three are flat or growing. Stores without both windows are omitted.
    """
    values = ["revenue", "baskets", "units"]
    level = ["store_code"]
    now = _window_sums(store_weeks, level, origin - weeks, origin, values, "now")
    ago = _window_sums(
        store_weeks, level, origin - weeks - YEAR_WEEKS, origin - YEAR_WEEKS, values, "ago"
    )
    both = now.join(ago, on=level).filter(
        (pl.col("weeks_now") == weeks)
        & (pl.col("weeks_ago") == weeks)
        & (pl.col("revenue_ago") > 0)
        & (pl.col("baskets_ago") > 0)
        & (pl.col("units_ago") > 0)
        & (pl.col("baskets_now") > 0)
        & (pl.col("units_now") > 0)
    )
    changes = both.select(
        *level,
        (pl.col("revenue_now") / pl.col("revenue_ago") - 1).alias("revenue_growth"),
        (pl.col("baskets_now") / pl.col("baskets_ago") - 1).alias("traffic_change"),
        (
            (pl.col("units_now") / pl.col("baskets_now"))
            / (pl.col("units_ago") / pl.col("baskets_ago"))
            - 1
        ).alias("basket_size_change"),
        (
            (pl.col("revenue_now") / pl.col("units_now"))
            / (pl.col("revenue_ago") / pl.col("units_ago"))
            - 1
        ).alias("price_mix_change"),
    )
    factors = ["traffic_change", "basket_size_change", "price_mix_change"]
    names = {"traffic_change": "traffic", "basket_size_change": "basket_size"}
    worst = pl.concat_list(factors).list.arg_min()
    return changes.with_columns(
        pl.when(pl.min_horizontal(factors) < 0)
        .then(
            pl.when(worst == 0)
            .then(pl.lit(names["traffic_change"]))
            .when(worst == 1)
            .then(pl.lit(names["basket_size_change"]))
            .otherwise(pl.lit("price_mix"))
        )
        .otherwise(pl.lit("none"))
        .alias("dominant_driver")
    )
