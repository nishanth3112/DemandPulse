"""Probability of missing a target and expected revenue shortfall, from past forecast errors.

For each model and origin, the relative forecast error `actual / forecast - 1` of every entity
(store or store-category) in earlier windows forms an empirical error distribution. Applying
it to today's forecast gives, for a target T and forecast F:

    P(shortfall)     = share of past errors e with F * (1 + e) < T
    revenue at risk  = F * mean(max(T / F - 1 - e, 0)), the expected revenue below target

A window only counts as history once it has finished: the 13-week window of origin o' is
usable at origin o only if o' + 13 <= o. Nothing from the scored window or later ever enters,
and each model is judged by its own errors. The pooled errors share one calendar window each,
so a market-wide surprise in that window is included once per window, not averaged away.
"""

import numpy as np
import polars as pl

from demandpulse.forecasting.panel import HORIZON

MIN_POOL = 30


def shortfall_risk(
    windows: pl.DataFrame, horizon: int = HORIZON, min_pool: int = MIN_POOL
) -> pl.DataFrame:
    """Add `p_shortfall`, `revenue_at_risk` and `pool_size` to a window table.

    The estimates are null where fewer than `min_pool` past errors exist (the first origins)
    or the forecast is not positive.
    """
    first_def = windows["target_def"].min()
    pool = (
        windows.filter((pl.col("target_def") == first_def) & (pl.col("forecast") > 0))
        .with_columns((pl.col("actual") / pl.col("forecast") - 1).alias("rel_error"))
        .select("model", "origin", "rel_error")
    )
    pieces = []
    for (model, origin), part in windows.partition_by(["model", "origin"], as_dict=True).items():
        past = np.sort(
            pool.filter((pl.col("model") == model) & (pl.col("origin") <= origin - horizon))[
                "rel_error"
            ].to_numpy()
        )
        n = len(past)
        forecast = part["forecast"].to_numpy()
        target = part["target"].to_numpy()
        valid = forecast > 0
        if n < min_pool:
            prob = np.full(part.height, np.nan)
            at_risk = np.full(part.height, np.nan)
        else:
            needed = np.where(valid, target / np.where(valid, forecast, 1.0) - 1, np.nan)
            below = np.searchsorted(past, needed, side="left")
            cumulative = np.concatenate([[0.0], np.cumsum(past)])
            prob = np.where(valid, below / n, np.nan)
            at_risk = np.where(valid, forecast * (below * needed - cumulative[below]) / n, np.nan)
        pieces.append(
            part.with_columns(
                pl.Series("p_shortfall", prob, nan_to_null=True),
                pl.Series("revenue_at_risk", at_risk, nan_to_null=True),
                pl.lit(n, dtype=pl.Int64).alias("pool_size"),
            )
        )
    return pl.concat(pieces).sort(["model", "origin", "target_def"])
