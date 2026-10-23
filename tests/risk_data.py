"""Synthetic panel, forecasts and Gold store-weeks for revenue-risk tests."""

import numpy as np
import polars as pl

ORIGINS = (78, 85, 91, 98, 104)
N_STORES = 45
CATEGORIES = ("C1", "C2")


def make_panel() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for s in range(N_STORES):
        for c, category in enumerate(CATEGORIES):
            level = 100.0 * (1 + s % 5) * (1 + c)
            drift = rng.normal(0.0, 0.002)
            for week in range(1, 118):
                seasonal = 1 + 0.3 * np.sin(2 * np.pi * week / 52 + c)
                revenue = level * seasonal * (1 + drift * week) * rng.lognormal(0, 0.05)
                rows.append(
                    (
                        f"S{s:02d}",
                        category,
                        "LS" if s % 2 else "SS",
                        "E01",
                        week,
                        revenue,
                        revenue / 2,
                        revenue / 10,
                    )
                )
    return pl.DataFrame(
        rows,
        schema={
            "store_code": pl.String,
            "prod_code_30": pl.String,
            "store_format": pl.String,
            "store_region": pl.String,
            "week_idx": pl.Int64,
            "revenue": pl.Float64,
            "units": pl.Float64,
            "baskets": pl.Float64,
        },
        orient="row",
    )


def make_forecasts(panel: pl.DataFrame, models=("ensemble", "lightgbm"), seed=5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for model in models:
        for origin in ORIGINS:
            window = panel.filter(
                (pl.col("week_idx") > origin) & (pl.col("week_idx") <= origin + 13)
            ).select("store_code", "prod_code_30", "week_idx", pl.col("revenue").alias("y"))
            frames.append(
                window.with_columns(
                    pl.lit(model).alias("model"),
                    pl.lit(origin, dtype=pl.Int64).alias("origin"),
                    (pl.col("week_idx") - origin).alias("horizon"),
                    (pl.col("y") * pl.Series(rng.lognormal(0.0, 0.08, window.height))).alias(
                        "y_hat"
                    ),
                ).select(
                    "model",
                    "origin",
                    "store_code",
                    "prod_code_30",
                    "horizon",
                    "week_idx",
                    "y",
                    "y_hat",
                )
            )
    return pl.concat(frames)


def store_weeks(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.group_by(["store_code", "week_idx"])
        .agg(pl.col("revenue").sum(), pl.col("baskets").sum(), pl.col("units").sum())
        .sort("store_code", "week_idx")
    )


def save_forecasts(store, forecasts: pl.DataFrame, run: str, origins) -> None:
    for model in forecasts["model"].unique().to_list():
        part = forecasts.filter((pl.col("model") == model) & pl.col("origin").is_in(origins))
        part.write_parquet(_path(store, f"forecast/runs/{run}/forecasts/{model}.parquet"))


def _path(store, key):
    path = store.root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
