"""Export the small tables the dashboard reads from the lake into a local snapshot folder.

Run once with `demandpulse app export`; the app itself never imports this module. Only a fixed
list of small Gold marts, forecast summaries and risk outputs is copied. The basket fact and the
store x category x week fact are never exported. The manifest is written last.
"""

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from demandpulse.app.snapshot import MANIFEST
from demandpulse.storage import Store

GOLD_TABLES = (
    "dim_store",
    "dim_date",
    "dim_comparison",
    "mart_store_scorecard",
    "mart_category_benchmark",
    "mart_region_benchmark",
    "mart_region_category_mix",
    "mart_revenue_variance_store",
    "mart_revenue_variance_region",
    "mart_rfm_segment_summary",
    "mart_customer_cohort",
    "mart_trading_pattern",
    "mart_basket_mix",
)
STORE_WEEK_COLUMNS = (
    "shop_week",
    "store_code",
    "baskets",
    "revenue",
    "units",
    "customers",
    "loyalty_baskets",
    "loyalty_revenue",
)
FORECAST_RUNS = ("main", "robustness")
FORECAST_MODELS = ("ensemble", "lightgbm", "seasonal_naive", "trailing_mean_13")
FORECAST_SUMMARY_TABLES = (
    "full_pooled",
    "full_by_origin",
    "full_store_level",
    "full_category_level",
    "full_network_level",
    "full_by_horizon",
    "full_by_format",
    "full_vs_baseline",
    "subset_rolling",
)
RISK_TABLES = (
    "growth",
    "windows_store",
    "ranking_store",
    "ranking_summary_store",
    "ranking_summary_store_category",
)
RISK_OPTIONAL_TABLES = ("recommendations", "scenarios")
RISK_JSON = ("classifier_store", "classifier_store_category", "summary")


def _read(store: Store, key: str) -> pl.DataFrame:
    if not store.exists(key):
        raise FileNotFoundError(f"{key} is not in the lake; build the earlier phases first")
    return pl.read_parquet(io.BytesIO(store.read_bytes(key)))


def _floats(frame: pl.DataFrame) -> pl.DataFrame:
    """Money columns are exact decimals in Gold; the app only charts them, so use floats."""
    return frame.with_columns(pl.col(pl.Decimal).cast(pl.Float64))


def _store_week(store: Store, dates: pl.DataFrame, stores: pl.DataFrame) -> pl.DataFrame:
    weekly = _read(store, "gold/fct_store_week.parquet").select(*STORE_WEEK_COLUMNS)
    return (
        _floats(weekly)
        .join(dates.select("shop_week", "week_idx", "week_start"), on="shop_week")
        .join(stores.select("store_code", "store_format", "store_region"), on="store_code")
        .sort(["store_code", "week_idx"])
    )


def _forecast_weeks(store: Store, dates: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Forecast and actual revenue per model, origin and week, by store and for the network.

    Categories are summed away per store; the network table sums those store totals, so both
    come from one read of each forecast file.
    """
    frames = []
    for run in FORECAST_RUNS:
        for model in FORECAST_MODELS:
            key = f"forecast/runs/{run}/forecasts/{model}.parquet"
            if not store.exists(key):
                continue
            by_store = (
                pl.read_parquet(io.BytesIO(store.read_bytes(key)))
                .group_by(["model", "origin", "week_idx", "store_code"])
                .agg(pl.col("y").sum().alias("actual"), pl.col("y_hat").sum().alias("forecast"))
                .with_columns(pl.lit(run).alias("run"))
            )
            frames.append(by_store)
    if not frames:
        raise FileNotFoundError("No forecast runs in the lake; run `demandpulse forecast backtest`")
    week_dates = dates.select("week_idx", "week_start")
    stores = (
        pl.concat(frames)
        .with_columns(pl.col("week_idx").cast(pl.Int32))
        .join(week_dates, on="week_idx", how="left")
        .sort(["run", "model", "origin", "week_idx", "store_code"])
    )
    network = (
        stores.group_by(["run", "model", "origin", "week_idx", "week_start"])
        .agg(pl.col("actual").sum(), pl.col("forecast").sum())
        .sort(["model", "origin", "week_idx"])
    )
    return network, stores


def _forecast_summaries(store: Store) -> dict[str, pl.DataFrame]:
    collected: dict[str, list[pl.DataFrame]] = {name: [] for name in FORECAST_SUMMARY_TABLES}
    for run in FORECAST_RUNS:
        key = f"forecast/runs/{run}/summary.json"
        if not store.exists(key):
            continue
        summary = json.loads(store.read_bytes(key))
        for name in FORECAST_SUMMARY_TABLES:
            if summary.get(name):
                collected[name].append(pl.DataFrame(summary[name]).with_columns(run=pl.lit(run)))
    return {
        f"forecast_{name}": pl.concat(frames, how="diagonal_relaxed")
        for name, frames in collected.items()
        if frames
    }


def export_snapshot(store: Store, out_dir: Path) -> dict[str, Any]:
    """Copy everything the dashboard needs from the lake into `out_dir` and return the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST).unlink(missing_ok=True)
    rows: dict[str, int] = {}

    def put(name: str, frame: pl.DataFrame) -> None:
        _floats(frame).write_parquet(out_dir / f"{name}.parquet")
        rows[name] = frame.height

    gold = {name: _read(store, f"gold/{name}.parquet") for name in GOLD_TABLES}
    for name, frame in gold.items():
        put(name, frame)
    put("store_week", _store_week(store, gold["dim_date"], gold["dim_store"]))
    network_weeks, store_weeks = _forecast_weeks(store, gold["dim_date"])
    put("forecast_network_weeks", network_weeks)
    put("forecast_store_weeks", store_weeks)
    for name, frame in _forecast_summaries(store).items():
        put(name, frame)

    for name in RISK_TABLES:
        put(f"risk_{name}", _read(store, f"risk/{name}.parquet"))
    for name in RISK_OPTIONAL_TABLES:
        if store.exists(f"risk/{name}.parquet"):
            put(f"risk_{name}", _read(store, f"risk/{name}.parquet"))
    for name in RISK_JSON:
        key = f"risk/{name}.json"
        if not store.exists(key):
            raise FileNotFoundError(f"{key} is not in the lake; run `demandpulse risk run`")
        (out_dir / f"risk_{name}.json").write_bytes(store.read_bytes(key))

    gold_manifest = json.loads(store.read_bytes("gold/_manifest.json"))
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tables": rows,
        "gold": {
            key: gold_manifest[key]
            for key in ("built_at", "silver_first_week", "silver_last_week", "silver_weeks")
            if key in gold_manifest
        },
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest
