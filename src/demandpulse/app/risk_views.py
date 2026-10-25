"""Display-ready tables for the forecasting and revenue-risk pages.

Targets here are hypothetical (last year's revenue, flat or grown by network growth) and every
window is historical, so the "realised" columns are hindsight from the backtest, not live
tracking. Functions are pure so they can be tested without Streamlit.
"""

import polars as pl

from demandpulse.risk.scenario import scenario_table

LEVELS = (
    ("Series (store x category)", "forecast_full_pooled"),
    ("Store", "forecast_full_store_level"),
    ("Category", "forecast_full_category_level"),
    ("Network", "forecast_full_network_level"),
)
TARGET_LABELS = {
    "flat_yoy": "Flat: same weeks last year",
    "trailing_growth": "Trailing growth: last year grown by recent network growth",
}
MODEL_LABELS = {
    "ensemble": "Ensemble",
    "ensemble_baselines": "Ensemble (baselines only)",
    "lightgbm": "LightGBM",
    "lightgbm_v2": "LightGBM v2",
    "seasonal_naive": "Seasonal naive",
    "trailing_mean_13": "13-week trailing mean",
    "own_trend": "Own-trend baseline",
    "random": "Random",
}
SCORE_LABELS = {
    "forecast_gap": "Revenue gap to target",
    "relative_gap": "Gap as share of target",
    "p_shortfall": "Probability of shortfall",
    "revenue_at_risk": "Expected revenue below target",
    "p_classifier": "Classifier probability",
    "random": "Random",
}


def model_leaderboard(pooled: pl.DataFrame, run: str) -> pl.DataFrame:
    """Pooled accuracy per model over all origins and series of a forecast run."""
    return (
        pooled.filter(pl.col("run") == run)
        .select("model", "wape", "bias", "mase_mean", "series")
        .sort("wape")
    )


def accuracy_by_level(tables: dict[str, pl.DataFrame], run: str) -> pl.DataFrame:
    """WAPE per model after summing forecasts to each level (errors cancel as levels coarsen)."""
    frames = [
        tables[name]
        .filter(pl.col("run") == run)
        .select("model", "wape")
        .with_columns(pl.lit(label).alias("level"))
        for label, name in LEVELS
        if name in tables
    ]
    return pl.concat(frames)


def network_forecast(weeks: pl.DataFrame, run: str, origin: int, models: list[str]) -> pl.DataFrame:
    """Actual and forecast network revenue by week for one origin: columns week_start, series, y."""
    window = weeks.filter((pl.col("run") == run) & (pl.col("origin") == origin))
    if not window.height:
        raise ValueError(f"No forecasts for run {run!r} at origin {origin}")
    actual = (
        window.filter(pl.col("model") == window["model"].min())
        .select("week_start", pl.lit("Actual").alias("series"), pl.col("actual").alias("y"))
        .sort("week_start")
    )
    forecasts = [
        window.filter(pl.col("model") == m)
        .select(
            "week_start",
            pl.lit(MODEL_LABELS.get(m, m)).alias("series"),
            pl.col("forecast").alias("y"),
        )
        .sort("week_start")
        for m in models
        if m in set(window["model"])
    ]
    return pl.concat([actual, *forecasts])


def risk_origins(windows: pl.DataFrame, model: str) -> list[int]:
    return sorted(windows.filter(pl.col("model") == model)["origin"].unique().to_list())


def risk_table(
    windows: pl.DataFrame,
    stores: pl.DataFrame,
    model: str,
    origin: int,
    target_def: str,
    regions: list[str] | None = None,
    formats: list[str] | None = None,
) -> pl.DataFrame:
    """Stores for one model, origin and target, riskiest first.

    Ordered by expected revenue below target; origins with no probability estimate (the first
    two) fall back to the forecast gap. `actual` and `realised_miss` are backtest hindsight.
    """
    frame = windows.filter(
        (pl.col("model") == model)
        & (pl.col("origin") == origin)
        & (pl.col("target_def") == target_def)
    ).join(stores.select("store_code", "store_region", "store_format"), on="store_code")
    if regions:
        frame = frame.filter(pl.col("store_region").is_in(regions))
    if formats:
        frame = frame.filter(pl.col("store_format").is_in(formats))
    by = (
        "revenue_at_risk"
        if frame["revenue_at_risk"].null_count() < frame.height
        else "forecast_shortfall"
    )
    return frame.select(
        "store_code", "store_region", "store_format", "target", "forecast", "forecast_shortfall",
        "relative_gap", "p_shortfall", "revenue_at_risk", "actual", "realised_miss",
    ).sort([by, "store_code"], descending=[True, False], nulls_last=True)  # fmt: skip


def risk_headline(table: pl.DataFrame) -> dict[str, float | None]:
    """Counts for the selected stores, with expected revenue at risk and the hindsight miss rate."""
    n = table.height
    return {
        "stores": n,
        "forecast_below_target": int((table["forecast_shortfall"] > 0).sum()) if n else 0,
        "revenue_at_risk": table["revenue_at_risk"].sum()
        if table["revenue_at_risk"].count()
        else None,
        "missed_in_hindsight": float(table["realised_miss"].mean()) if n else None,
    }


def ranking_view(summary: pl.DataFrame, target_def: str, k_fraction: float) -> pl.DataFrame:
    """Backtest precision, lift and revenue captured by model and score, best precision first."""
    return (
        summary.filter(
            (pl.col("target_def") == target_def)
            & (pl.col("k_fraction") == k_fraction)
            & pl.col("precision").is_not_null()
        )
        .with_columns(
            pl.col("model").replace(MODEL_LABELS).alias("model_label"),
            pl.col("score").replace(SCORE_LABELS).alias("score_label"),
        )
        .select(
            "model", "score", "model_label", "score_label", "origins", "base_rate", "precision",
            "recall", "lift", "capture",
        )
        .sort("precision", descending=True)
    )  # fmt: skip


def classifier_metrics(document: dict, target_def: str) -> dict[str, float]:
    if "skipped" in document:
        raise ValueError(f"Classifier was skipped: {document['skipped']}")
    return document[target_def]["metrics"]


def classifier_verdict(metrics: dict[str, float]) -> str:
    """One honest sentence comparing the calibrated classifier with the forecast probability."""
    ours = metrics["brier_calibrated_comparable"]
    theirs = metrics["brier_forecast_probability"]
    if ours < theirs:
        return f"Calibrated classifier Brier {ours:.3f} beat the forecast probability {theirs:.3f}."
    return (
        f"Calibrated classifier Brier {ours:.3f} did not beat the forecast probability "
        f"{theirs:.3f} (lower is better); on one test window this is a single check."
    )


def custom_scenario(
    stores: pl.DataFrame, traffic: float, basket_size: float, price_mix: float
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Apply one what-if to stores with `forecast` and `target`; returns rows and totals."""
    rows = scenario_table(stores, {"Custom": (traffic, basket_size, price_mix)})
    before = float(rows["gap_before"].sum())
    after = float(rows["gap_after"].sum())
    return rows, {
        "gap_before": before,
        "gap_after": after,
        "gap_closed_share": (before - after) / before if before else 0.0,
        "stores_meeting_target": int(rows["meets_target"].sum()),
        "stores": rows.height,
    }
