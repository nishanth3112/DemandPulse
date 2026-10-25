"""Export the dashboard snapshot as tidy CSV files for a Tableau Public workbook.

Tableau Public cannot read private data and everything published is public, so this reads only
the local dashboard snapshot (never the lake or the basket fact): aggregated tables of the
synthetic dataset, with readable labels, ISO dates and one grain per file. The snapshot is the
only input, so the export still works after the AWS resources are torn down.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from demandpulse.app.risk_views import MODEL_LABELS, SCORE_LABELS, TARGET_LABELS
from demandpulse.app.snapshot import Snapshot
from demandpulse.app.views import COMPARISON_LABELS

DEFAULT_TABLEAU_DIR = Path("data/tableau")
MANIFEST = "tableau_export.json"
DICTIONARY = "_data_dictionary.csv"
FLOAT_PRECISION = 6

BRIDGE_STEPS = (
    ("base_revenue", "Base period", "total"),
    ("trading_weeks_effect", "Trading weeks", "change"),
    ("traffic_effect", "Traffic (baskets per week)", "change"),
    ("basket_size_effect", "Basket size (units per basket)", "change"),
    ("price_mix_effect", "Price and mix (revenue per unit)", "change"),
    ("new_store_effect", "New stores", "change"),
    ("closed_store_effect", "Closed stores", "change"),
    ("current_revenue", "Current period", "total"),
)
ACCURACY_LEVELS = (
    ("Series (store x category)", "forecast_full_pooled"),
    ("Store", "forecast_full_store_level"),
    ("Category", "forecast_full_category_level"),
    ("Network", "forecast_full_network_level"),
)
ACCURACY_COLUMNS = (
    "run", "model", "slice", "slice_value", "origin", "n", "wape", "bias", "rmse",
    "mase_mean", "mase_median", "actual_total", "series",
)  # fmt: skip
OPTIONAL = ("risk_recommendations", "risk_scenarios")


def _model_label(column: str = "model", name: str = "model_label") -> pl.Expr:
    return pl.col(column).replace_strict(MODEL_LABELS, default=pl.col(column)).alias(name)


def _comparison_label() -> pl.Expr:
    return (
        pl.col("comparison")
        .replace_strict(COMPARISON_LABELS, default=pl.col("comparison"))
        .alias("comparison_label")
    )


def _target_label() -> pl.Expr:
    return (
        pl.col("target_def")
        .replace_strict(TARGET_LABELS, default=pl.col("target_def"))
        .alias("target_label")
    )


def _model_type(model: str) -> str:
    if model.startswith("ensemble"):
        return "ensemble"
    if model.startswith("lightgbm"):
        return "machine learning"
    if model in {"seasonal_naive", "trailing_mean_13"}:
        return "baseline"
    return "statistical"


def _dim_model(snap: Snapshot) -> pl.DataFrame:
    models = sorted(snap.table("forecast_full_pooled")["model"].unique().to_list())
    primary = snap.json("risk_summary").get("primary")
    return pl.DataFrame(
        {
            "model": models,
            "model_label": [MODEL_LABELS.get(m, m) for m in models],
            "model_type": [_model_type(m) for m in models],
            "is_risk_primary": [m == primary for m in models],
        }
    )


def _bridge(snap: Snapshot) -> pl.DataFrame:
    frames = []
    for level, name in (
        ("region", "mart_revenue_variance_region"),
        ("store", "mart_revenue_variance_store"),
    ):
        table = snap.table(name)
        keys = [c for c in ("store_region", "store_code", "store_status") if c in table.columns]
        steps = pl.DataFrame(
            {
                "column": [c for c, _, _ in BRIDGE_STEPS],
                "step": [s for _, s, _ in BRIDGE_STEPS],
                "step_order": list(range(1, len(BRIDGE_STEPS) + 1)),
                "step_kind": [k for _, _, k in BRIDGE_STEPS],
            }
        )
        long = (
            table.select("comparison", *keys, *[c for c, _, _ in BRIDGE_STEPS])
            .unpivot(index=["comparison", *keys], variable_name="column", value_name="value")
            .join(steps, on="column")
            .with_columns(pl.lit(level).alias("level"))
            .drop("column")
        )
        frames.append(long)
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(_comparison_label())
        .select(
            "comparison",
            "comparison_label",
            "level",
            "store_region",
            "store_code",
            "store_status",
            "step",
            "step_order",
            "step_kind",
            "value",
        )  # fmt: skip
        .sort("comparison", "level", "store_region", "store_code", "step_order", nulls_last=True)
    )


def _accuracy(snap: Snapshot) -> pl.DataFrame:
    frames = []
    for label, name in ACCURACY_LEVELS:
        frames.append(
            snap.table(name).with_columns(
                pl.lit("level").alias("slice"), pl.lit(label).alias("slice_value")
            )
        )
    by_origin = snap.table("forecast_full_by_origin")
    frames.append(
        by_origin.with_columns(
            pl.lit("origin").alias("slice"), pl.col("origin").cast(pl.String).alias("slice_value")
        )
    )
    for name, column, kind in (
        ("forecast_full_by_horizon", "horizon_bucket", "horizon"),
        ("forecast_full_by_format", "store_format", "format"),
    ):
        frames.append(
            snap.table(name).with_columns(
                pl.lit(kind).alias("slice"), pl.col(column).alias("slice_value")
            )
        )
    stacked = pl.concat(frames, how="diagonal_relaxed")
    for column in ACCURACY_COLUMNS:
        if column not in stacked.columns:
            stacked = stacked.with_columns(pl.lit(None).alias(column))
    return (
        stacked.select(ACCURACY_COLUMNS)
        .with_columns(_model_label())
        .select("run", "model", "model_label", *ACCURACY_COLUMNS[2:])
        .sort("run", "slice", "slice_value", "model")
    )


def _ranking(snap: Snapshot) -> pl.DataFrame:
    frames = [
        snap.table("risk_ranking_summary_store").with_columns(pl.lit("store").alias("grain")),
        snap.table("risk_ranking_summary_store_category").with_columns(
            pl.lit("store x category").alias("grain")
        ),
    ]
    return (
        pl.concat(frames)
        .with_columns(
            _model_label(),
            pl.col("score")
            .replace_strict(SCORE_LABELS, default=pl.col("score"))
            .alias("score_label"),
            _target_label(),
        )
        .sort("grain", "target_def", "k_fraction", "model", "score")
    )  # fmt: skip


def build_tables(snap: Snapshot) -> dict[str, pl.DataFrame]:
    """Every exported table by file name (without extension), built from the snapshot only."""
    stores = snap.table("dim_store").select("store_code", "store_format", "store_region")
    comparison = snap.table("dim_comparison").with_columns(_comparison_label())
    tables: dict[str, pl.DataFrame] = {
        "dim_date": snap.table("dim_date"),
        "dim_store": snap.table("dim_store"),
        "dim_model": _dim_model(snap),
        "dim_comparison": comparison,
        "fact_store_week": snap.table("store_week").drop("shop_week"),
        "fact_region_benchmark": snap.table("mart_region_benchmark").with_columns(
            _comparison_label()
        ),
        "fact_store_scorecard": snap.table("mart_store_scorecard").with_columns(
            _comparison_label()
        ),
        "fact_revenue_bridge": _bridge(snap),
        "fact_category_benchmark": snap.table("mart_category_benchmark").with_columns(
            _comparison_label()
        ),
        "fact_region_category_mix": snap.table("mart_region_category_mix").with_columns(
            _comparison_label()
        ),
        "fact_trading_pattern": snap.table("mart_trading_pattern"),
        "fact_basket_mix": snap.table("mart_basket_mix"),
        "fact_forecast_accuracy": _accuracy(snap),
        "fact_forecast_vs_baseline": snap.table("forecast_full_vs_baseline").with_columns(
            _model_label(), _model_label("baseline", "baseline_label")
        ),
        "fact_forecast_network_week": snap.table("forecast_network_weeks")
        .with_columns(_model_label())
        .select(
            "run", "model", "model_label", "origin", "week_idx", "week_start", "actual", "forecast"
        )
        .sort("run", "model", "origin", "week_idx"),
        "fact_forecast_store_week": snap.table("forecast_store_weeks")
        .join(stores, on="store_code", how="left")
        .with_columns(_model_label())
        .select(
            "run",
            "model",
            "model_label",
            "origin",
            "week_idx",
            "week_start",
            "store_code",
            "store_format",
            "store_region",
            "actual",
            "forecast",
        )
        .sort("run", "model", "origin", "week_idx", "store_code"),  # fmt: skip
        "fact_risk_store_window": snap.table("risk_windows_store")
        .join(stores, on="store_code", how="left")
        .with_columns(_model_label(), _target_label()),
        "fact_risk_ranking": _ranking(snap),
        "fact_risk_growth": snap.table("risk_growth").with_columns(_target_label()),
        "fact_rfm_segment": snap.table("mart_rfm_segment_summary"),
        "fact_customer_cohort": snap.table("mart_customer_cohort"),
    }
    if snap.has("risk_recommendations"):
        tables["fact_risk_recommendations"] = snap.table("risk_recommendations").with_columns(
            _model_label(), _target_label()
        )
    if snap.has("risk_scenarios"):
        tables["fact_risk_scenarios"] = snap.table("risk_scenarios").with_columns(_target_label())
    return tables


def export_tableau(snapshot_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Write every table as CSV plus a data dictionary and a manifest (written last)."""
    snap = Snapshot(snapshot_dir)
    tables = build_tables(snap)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST).unlink(missing_ok=True)
    dictionary = []
    for name, frame in tables.items():
        frame.write_csv(out_dir / f"{name}.csv", float_precision=FLOAT_PRECISION)
        dictionary.extend(
            {"table": name, "column": column, "type": str(dtype), "rows": frame.height}
            for column, dtype in frame.schema.items()
        )
    pl.DataFrame(dictionary).write_csv(out_dir / DICTIONARY)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "snapshot_created_at": snap.manifest["created_at"],
        "tables": {name: frame.height for name, frame in tables.items()},
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest
