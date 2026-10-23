"""What-if arithmetic: how much of a forecast gap a change in each revenue driver would close.

Revenue is baskets x units per basket x revenue per unit, so a scenario multiplies the forecast
by (1 + traffic) x (1 + basket size) x (1 + price/mix). This is hypothetical arithmetic on a
forecast, not an estimate of what any action would do: the data has no prices, promotions or
costs, so it says nothing about how a driver could be moved, what it would cost, or whether
other drivers would react.
"""

import polars as pl

# name -> (traffic, basket size, price/mix) relative changes
SCENARIOS: dict[str, tuple[float, float, float]] = {
    "traffic_+2%": (0.02, 0.0, 0.0),
    "basket_size_+2%": (0.0, 0.02, 0.0),
    "price_mix_+1%": (0.0, 0.0, 0.01),
    "all_three": (0.02, 0.02, 0.01),
}


def required_uplift(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the uniform revenue increase that would bring the forecast up to target (0 if above)."""
    return frame.with_columns(
        pl.when((pl.col("forecast") > 0) & (pl.col("target") > pl.col("forecast")))
        .then(pl.col("target") / pl.col("forecast") - 1)
        .when(pl.col("forecast") > 0)
        .then(0.0)
        .otherwise(None)
        .alias("required_uplift")
    )


def scenario_table(
    frame: pl.DataFrame, scenarios: dict[str, tuple[float, float, float]] | None = None
) -> pl.DataFrame:
    """One row per entity and scenario with the forecast, remaining gap and share of gap closed.

    `frame` needs `forecast` and `target`; every other column is carried through.
    """
    scenarios = SCENARIOS if scenarios is None else scenarios
    for name, factors in scenarios.items():
        if any(change <= -1 for change in factors):
            raise ValueError(f"Scenario {name!r} has a driver change of -100% or less")
    base = required_uplift(frame).with_columns(
        pl.max_horizontal(pl.col("target") - pl.col("forecast"), pl.lit(0.0)).alias("gap_before")
    )
    pieces = []
    for name, (traffic, basket_size, price_mix) in scenarios.items():
        multiplier = (1 + traffic) * (1 + basket_size) * (1 + price_mix)
        pieces.append(
            base.with_columns(
                pl.lit(name).alias("scenario"),
                pl.lit(traffic).alias("traffic_change"),
                pl.lit(basket_size).alias("basket_size_change"),
                pl.lit(price_mix).alias("price_mix_change"),
                (pl.col("forecast") * multiplier).alias("scenario_forecast"),
            )
        )
    return (
        pl.concat(pieces)
        .with_columns(
            pl.max_horizontal(pl.col("target") - pl.col("scenario_forecast"), pl.lit(0.0)).alias(
                "gap_after"
            ),
            (pl.col("scenario_forecast") >= pl.col("target")).alias("meets_target"),
        )
        .with_columns(
            pl.when(pl.col("gap_before") > 0)
            .then((pl.col("gap_before") - pl.col("gap_after")) / pl.col("gap_before"))
            .otherwise(None)
            .alias("gap_closed_share")
        )
    )
