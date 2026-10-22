"""Model comparison tables with uncertainty: leaderboards, paired bootstrap and breakdowns.

Differences between models are only reported as improvements when a paired bootstrap over series
shows the interval excludes zero; otherwise the report says the models are not distinguishable.
"""

import numpy as np
import polars as pl

from demandpulse.forecasting.metrics import (
    ACTUAL,
    PREDICTION,
    aggregate_level,
    mase_summary,
    summarise,
)
from demandpulse.forecasting.panel import SERIES_KEY

BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20061
HORIZON_BUCKETS = ((1, 4, "weeks 1-4"), (5, 8, "weeks 5-8"), (9, 13, "weeks 9-13"))


def leaderboard(
    forecasts: pl.DataFrame, scales: pl.DataFrame, by: list[str] | None = None
) -> pl.DataFrame:
    """Series-level accuracy per model (and per origin if requested), best WAPE first."""
    by = by or ["model"]
    table = summarise(forecasts, by)
    series = forecasts.group_by(by).agg(
        pl.struct(SERIES_KEY).n_unique().alias("series"),
    )
    table = table.join(series, on=by)
    if set(by) <= {"model", "origin"}:
        table = table.join(mase_summary(forecasts, scales, by), on=by, how="left")
    return table.sort(["wape"] if by == ["model"] else by)


def aggregated_leaderboard(
    forecasts: pl.DataFrame, level: list[str], by: list[str]
) -> pl.DataFrame:
    """Accuracy after summing series to a coarser level, e.g. store or network."""
    grouped = aggregate_level(forecasts, level) if level else aggregate_level(forecasts, [])
    return summarise(grouped, by).select(*by, "n", "wape", "bias")


def horizon_buckets(forecasts: pl.DataFrame) -> pl.DataFrame:
    labelled = forecasts.with_columns(
        pl.when(pl.col("horizon").is_between(1, 4))
        .then(pl.lit("weeks 1-4"))
        .when(pl.col("horizon").is_between(5, 8))
        .then(pl.lit("weeks 5-8"))
        .otherwise(pl.lit("weeks 9-13"))
        .alias("horizon_bucket")
    )
    return summarise(labelled, ["model", "horizon_bucket"]).select(
        "model", "horizon_bucket", "wape", "bias"
    )


def by_attribute(forecasts: pl.DataFrame, attributes: pl.DataFrame, column: str) -> pl.DataFrame:
    joined = forecasts.join(
        attributes.select("store_code", column).unique("store_code"), on="store_code"
    )
    return summarise(joined, ["model", column]).select("model", column, "n", "wape", "bias")


def paired_vs_baseline(
    forecasts: pl.DataFrame,
    baseline: str,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> pl.DataFrame:
    """Relative reduction in absolute error versus a baseline, with a paired bootstrap CI.

    The unit of resampling is a (origin, series) pair so the same series are compared for both
    models. Positive improvement means lower error than the baseline.
    """
    errors = (
        forecasts.with_columns((pl.col(ACTUAL) - pl.col(PREDICTION)).abs().alias("abs_error"))
        .group_by("model", "origin", *SERIES_KEY)
        .agg(pl.col("abs_error").sum())
    )
    base = errors.filter(pl.col("model") == baseline).select(
        "origin", *SERIES_KEY, pl.col("abs_error").alias("base_error")
    )
    rng = np.random.default_rng(seed)
    rows = []
    for model in sorted(errors["model"].unique().to_list()):
        if model == baseline:
            continue
        paired = (
            errors.filter(pl.col("model") == model)
            .join(base, on=["origin", *SERIES_KEY])
            # group_by and joins return rows in arbitrary order; without a fixed order the same
            # seed would resample different series and the interval would not be reproducible.
            .sort(["origin", *SERIES_KEY])
            .select("abs_error", "base_error")
        )
        if not paired.height:
            continue
        own = paired["abs_error"].to_numpy()
        ref = paired["base_error"].to_numpy()
        point = 1 - own.sum() / ref.sum()
        picks = rng.integers(0, len(own), size=(draws, len(own)))
        resampled = 1 - own[picks].sum(axis=1) / ref[picks].sum(axis=1)
        low, high = np.percentile(resampled, [2.5, 97.5])
        rows.append(
            {
                "model": model,
                "baseline": baseline,
                "units": len(own),
                "improvement": float(point),
                "ci_low": float(low),
                "ci_high": float(high),
                "win_rate": float((own < ref).mean()),
                "distinguishable": bool(low > 0 or high < 0),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "model": pl.String,
            "baseline": pl.String,
            "units": pl.Int64,
            "improvement": pl.Float64,
            "ci_low": pl.Float64,
            "ci_high": pl.Float64,
            "win_rate": pl.Float64,
            "distinguishable": pl.Boolean,
        },
    )


def _fmt(frame: pl.DataFrame, percent: tuple[str, ...] = ("wape", "bias")) -> str:
    """Render a small frame as a Markdown table, percentages for the named columns."""
    columns = frame.columns
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.iter_rows(named=True):
        cells = []
        for column in columns:
            value = row[column]
            if value is None:
                cells.append("")
            elif column in percent and isinstance(value, float):
                cells.append(f"{value * 100:.1f}%")
            elif isinstance(value, float):
                cells.append(f"{value:,.3f}")
            elif isinstance(value, bool):
                cells.append("yes" if value else "no")
            elif isinstance(value, int):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_table(frame: pl.DataFrame, percent: tuple[str, ...] = ("wape", "bias")) -> str:
    return _fmt(frame, percent)
