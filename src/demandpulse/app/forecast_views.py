"""Display-ready tables for the single-store forecast explorer.

Every forecast here comes from a historical backtest: models saw only data up to the origin and
the "actual" weeks were already known when this snapshot was built. Functions are pure so they
can be tested without Streamlit.
"""

import polars as pl

from demandpulse.app.risk_views import MODEL_LABELS

HISTORY_WEEKS = 52
ACTUAL_LABEL = "Actual"


def store_options(
    stores: pl.DataFrame, forecast_stores: set[str], regions: list[str], formats: list[str]
) -> list[str]:
    """Store codes that have forecasts, narrowed by region and format (empty list = no filter)."""
    kept = stores.filter(pl.col("store_code").is_in(forecast_stores))
    if regions:
        kept = kept.filter(pl.col("store_region").is_in(regions))
    if formats:
        kept = kept.filter(pl.col("store_format").is_in(formats))
    return sorted(kept["store_code"].to_list())


def _window(weeks: pl.DataFrame, run: str, origin: int) -> pl.DataFrame:
    window = weeks.filter((pl.col("run") == run) & (pl.col("origin") == origin))
    if not window.height:
        raise ValueError(f"No store forecasts for run {run!r} at origin {origin}")
    return window


def store_series(
    forecast_weeks: pl.DataFrame,
    store_week: pl.DataFrame,
    run: str,
    origin: int,
    store: str,
    models: list[str],
) -> pl.DataFrame:
    """Actual history to the origin, then actual and each model's forecast for the next weeks.

    Columns: week_start, series, y. The actual line is the store's weekly revenue for the
    `HISTORY_WEEKS` before the origin, continued with the actual revenue of the forecast weeks.
    """
    window = _window(forecast_weeks, run, origin).filter(pl.col("store_code") == store)
    if not window.height:
        raise ValueError(f"No forecasts for store {store!r} in run {run!r} at origin {origin}")
    history = store_week.filter(
        (pl.col("store_code") == store)
        & (pl.col("week_idx") <= origin)
        & (pl.col("week_idx") > origin - HISTORY_WEEKS)
    ).select("week_start", pl.lit(ACTUAL_LABEL).alias("series"), pl.col("revenue").alias("y"))
    first = window.filter(pl.col("model") == window["model"].min())
    realised = first.select(
        "week_start", pl.lit(ACTUAL_LABEL).alias("series"), pl.col("actual").alias("y")
    )
    forecasts = [
        window.filter(pl.col("model") == m).select(
            "week_start",
            pl.lit(MODEL_LABELS.get(m, m)).alias("series"),
            pl.col("forecast").alias("y"),
        )
        for m in models
        if m in set(window["model"])
    ]
    return pl.concat([history, realised, *forecasts]).sort(["series", "week_start"])


def _error_columns() -> list[pl.Expr]:
    actual = pl.col("actual").sum()
    return [
        pl.when(actual > 0)
        .then((pl.col("actual") - pl.col("forecast")).abs().sum() / actual)
        .alias("wape"),
        pl.when(actual > 0).then((pl.col("forecast").sum() - actual) / actual).alias("bias"),
        actual.alias("actual_total"),
        pl.col("forecast").sum().alias("forecast_total"),
    ]


def store_accuracy(
    forecast_weeks: pl.DataFrame, run: str, origin: int, store: str, models: list[str]
) -> pl.DataFrame:
    """WAPE and bias of each model for one store over the forecast weeks of one origin.

    WAPE and bias are null for a store with no revenue in those weeks (a closure).
    """
    window = _window(forecast_weeks, run, origin).filter(
        (pl.col("store_code") == store) & pl.col("model").is_in(models)
    )
    return (
        window.group_by("model")
        .agg(_error_columns())
        .with_columns(pl.col("model").replace(MODEL_LABELS))
        .sort("wape", nulls_last=True)
    )


def wape_by_store(forecast_weeks: pl.DataFrame, run: str, origin: int, model: str) -> pl.DataFrame:
    """WAPE per store for one model and origin, stores with no revenue in the window dropped."""
    window = _window(forecast_weeks, run, origin).filter(pl.col("model") == model)
    return (
        window.group_by("store_code")
        .agg(_error_columns())
        .drop_nulls("wape")
        .select("store_code", "wape", "bias", "actual_total")
        .sort("wape")
    )


def store_rank(by_store: pl.DataFrame, store: str) -> dict[str, float | int] | None:
    """Where one store sits among all stores for the same model and origin (1 = most accurate)."""
    row = by_store.filter(pl.col("store_code") == store)
    if not row.height:
        return None
    wape = row["wape"][0]
    return {
        "rank": int((by_store["wape"] < wape).sum()) + 1,
        "of": by_store.height,
        "median_wape": float(by_store["wape"].median()),
        "wape": float(wape),
    }
