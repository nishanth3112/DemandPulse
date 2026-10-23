"""A small synthetic lake with every table the dashboard snapshot exports.

Risk outputs come from the real risk pipeline run over the synthetic panel; the Gold marts are
generated with the real column names and plausible values, so views and pages are exercised
against the same schemas as production.
"""

import json
from datetime import date, timedelta

import numpy as np
import polars as pl

from demandpulse.risk.runner import compute_risk, save_results
from demandpulse.storage import LocalStore
from risk_data import (
    CATEGORIES,
    REGIONS,
    make_forecasts,
    make_panel,
    save_forecasts,
    store_weeks,
)

COMPARISONS = ("window_vs_prior", "window_vs_year_ago", "year_vs_prior_year")
SEGMENTS = (
    "Champions", "Loyal", "Potential loyalists", "New or returning", "Needs attention",
    "At risk", "Hibernating",
)  # fmt: skip
FIRST_MONDAY = date(2006, 4, 10)
EFFECTS = ("traffic_effect", "basket_size_effect", "price_mix_effect")


def _dim_date() -> pl.DataFrame:
    rows = []
    for idx in range(1, 118):
        start = FIRST_MONDAY + timedelta(days=7 * (idx - 1))
        rows.append(
            (200600 + idx, start, start + timedelta(days=6), 2006 + (idx - 1) // 52,
             (idx - 1) % 52 + 1, idx, (idx - 1) // 4 + 1)
        )  # fmt: skip
    return pl.DataFrame(
        rows,
        schema={
            "shop_week": pl.Int32, "week_start": pl.Date, "week_end": pl.Date,
            "fiscal_year": pl.Int32, "fiscal_week": pl.Int32, "week_idx": pl.Int32,
            "period_idx": pl.Int32,
        },
        orient="row",
    )  # fmt: skip


def gold_tables(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(4)
    stores = panel.select("store_code", "store_format", "store_region").unique("store_code")
    stores = stores.sort("store_code")
    codes = stores["store_code"].to_list()
    dim_store = stores.with_columns(
        first_week=pl.lit(200601, dtype=pl.Int32),
        last_week=pl.lit(200717, dtype=pl.Int32),
        first_week_idx=pl.lit(1, dtype=pl.Int32),
        last_week_idx=pl.lit(117, dtype=pl.Int32),
        weeks_active=pl.lit(117, dtype=pl.Int32),
        lifecycle=pl.when(pl.col("store_code") == codes[-1])
        .then(pl.lit("opened_in_period"))
        .otherwise(pl.lit("open_throughout")),
    )
    weekly = (
        store_weeks(panel)
        .with_columns(
            (pl.col("week_idx") + 200600).cast(pl.Int32).alias("shop_week"),
            pl.col("baskets").round().cast(pl.Int32),
            pl.col("units").round().cast(pl.Int64),
        )
        .join(stores, on="store_code")
        .with_columns(
            (pl.col("baskets") * 0.8).round().cast(pl.Int32).alias("loyalty_baskets"),
            (pl.col("revenue") * 0.8).alias("loyalty_revenue"),
            (pl.col("baskets") * 0.6).round().cast(pl.Int32).alias("customers"),
        )
    )
    fct_store_week = weekly.select(
        "shop_week", "store_code", "store_format", "store_region", "baskets", "revenue",
        "units", "customers", "loyalty_baskets", "loyalty_revenue",
    )  # fmt: skip

    def per_comparison(builder):
        return pl.concat([builder(c) for c in COMPARISONS])

    def scorecard(comparison: str) -> pl.DataFrame:
        n = stores.height
        base = rng.uniform(5e4, 2e5, n)
        growth = rng.normal(0.0, 0.05, n)
        return stores.with_columns(
            pl.lit(comparison).alias("comparison"),
            pl.when(pl.col("store_code") == codes[-1])
            .then(pl.lit("opened_in_period"))
            .otherwise(pl.lit("open_throughout"))
            .alias("lifecycle"),
            pl.lit(13, dtype=pl.Int64).alias("current_weeks"),
            pl.lit(13, dtype=pl.Int64).alias("base_weeks"),
            (pl.col("store_code") != codes[-1]).alias("is_lfl"),
            pl.Series("current_revenue", base * (1 + growth)),
            pl.Series("base_revenue", base),
            pl.Series("revenue_growth", growth),
            pl.Series("current_revenue_per_week", base * (1 + growth) / 13),
            pl.Series("base_revenue_per_week", base / 13),
            pl.Series("current_baskets", base / 20),
            pl.Series("base_baskets", base / 21),
            pl.Series("basket_growth", rng.normal(0.0, 0.03, n)),
            pl.Series("current_avg_basket_value", rng.uniform(15, 30, n)),
            pl.Series("base_avg_basket_value", rng.uniform(15, 30, n)),
            pl.Series("current_units_per_basket", rng.uniform(8, 14, n)),
            pl.Series("current_avg_unit_price", rng.uniform(1.5, 3, n)),
            pl.Series("current_loyalty_share", rng.uniform(0.7, 0.85, n)),
            pl.Series("revenue_index_vs_format", rng.uniform(0.6, 1.4, n)),
            pl.Series("basket_value_index_vs_format", rng.uniform(0.8, 1.2, n)),
            pl.Series("revenue_rank_in_region", rng.integers(1, 10, n), dtype=pl.Int64),
            pl.lit(15, dtype=pl.Int32).alias("trading_stores_in_region"),
            pl.Series("lfl_growth_gap_pp", rng.normal(0, 2, n)),
            pl.Series("lfl_growth_zscore", rng.normal(0, 1, n)),
        )

    def variance(comparison: str) -> pl.DataFrame:
        n = stores.height
        effects = {c: rng.normal(0, 500, n) for c in EFFECTS}
        return stores.with_columns(
            pl.lit(comparison).alias("comparison"),
            pl.when(pl.col("store_code") == codes[-1])
            .then(pl.lit("new"))
            .otherwise(pl.lit("continuing"))
            .alias("store_status"),
            pl.lit(1e5).alias("base_revenue"),
            pl.lit(1.01e5).alias("current_revenue"),
            pl.lit(1000.0).alias("revenue_change"),
            pl.lit(0.0).alias("trading_weeks_effect"),
            *(pl.Series(c, v) for c, v in effects.items()),
            pl.lit(0.0).alias("new_store_effect"),
            pl.lit(0.0).alias("closed_store_effect"),
        )

    regions = [*REGIONS, "ALL"]

    def region_benchmark(comparison: str) -> pl.DataFrame:
        n = len(regions)
        growth = rng.normal(0, 0.03, n)
        return pl.DataFrame(
            {
                "comparison": comparison,
                "store_region": regions,
                "stores_current": 15,
                "stores_base": 15,
                "lfl_stores": 14,
                "current_revenue": rng.uniform(5e6, 7e6, n),
                "base_revenue": rng.uniform(5e6, 7e6, n),
                "revenue_growth": growth,
                "lfl_current_revenue": rng.uniform(5e6, 7e6, n),
                "lfl_base_revenue": rng.uniform(5e6, 7e6, n),
                "lfl_revenue_growth": growth - 0.002,
                "current_baskets": rng.uniform(2e5, 3e5, n),
                "base_baskets": rng.uniform(2e5, 3e5, n),
                "basket_growth": rng.normal(0, 0.02, n),
                "current_avg_basket_value": rng.uniform(20, 25, n),
                "base_avg_basket_value": rng.uniform(20, 25, n),
            }
        ).cast({"stores_current": pl.Int32, "stores_base": pl.Int32, "lfl_stores": pl.Int32})

    def variance_region(comparison: str) -> pl.DataFrame:
        n = len(regions)
        return pl.DataFrame(
            {
                "comparison": comparison,
                "store_region": regions,
                "base_revenue": 1e6,
                "current_revenue": 1.01e6,
                "revenue_change": 1e4,
                "trading_weeks_effect": 0.0,
                "traffic_effect": rng.normal(0, 4000, n),
                "basket_size_effect": rng.normal(0, 3000, n),
                "price_mix_effect": rng.normal(0, 3000, n),
                "new_store_effect": 1500.0,
                "closed_store_effect": -500.0,
            }
        )

    def category_benchmark(comparison: str) -> pl.DataFrame:
        n = len(CATEGORIES)
        return pl.DataFrame(
            {
                "comparison": comparison,
                "prod_code_30": list(CATEGORIES),
                "current_revenue": rng.uniform(1e6, 2e6, n),
                "base_revenue": rng.uniform(1e6, 2e6, n),
                "revenue_growth": rng.normal(0, 0.05, n),
                "current_share": [0.6, 0.4],
                "base_share": [0.58, 0.42],
                "share_change_pp": [0.2, -0.2],
                "contribution_to_growth_pp": [0.3, -0.1],
                "current_penetration": [0.8, 0.5],
                "base_penetration": [0.79, 0.52],
                "current_avg_unit_price": [2.0, 3.0],
                "base_avg_unit_price": [1.9, 3.1],
                "avg_unit_price_change": [0.05, -0.03],
            }
        )

    def region_category_mix(comparison: str) -> pl.DataFrame:
        rows = [
            (comparison, r, c, 1e5 * (1 + i), 0.5, 0.5, 1.0 + 0.1 * i)
            for r in REGIONS
            for i, c in enumerate(CATEGORIES)
        ]
        return pl.DataFrame(
            rows,
            schema=["comparison", "store_region", "prod_code_30", "revenue", "region_share",
                    "national_share", "mix_index"],
            orient="row",
        )  # fmt: skip

    segments = pl.DataFrame(
        {
            "rfm_segment": list(SEGMENTS),
            "customers": [100, 200, 150, 80, 90, 120, 60],
            "customer_share": [0.1, 0.2, 0.15, 0.08, 0.09, 0.12, 0.06],
            "revenue": [5e5, 6e5, 3e5, 1e5, 1.5e5, 1e5, 2e4],
            "revenue_share": [0.3, 0.3, 0.15, 0.05, 0.08, 0.05, 0.01],
            "avg_baskets": [40.0, 30.0, 20.0, 8.0, 15.0, 10.0, 4.0],
            "avg_revenue": [5e3, 3e3, 2e3, 1.2e3, 1.6e3, 8e2, 3e2],
            "avg_recency_weeks": [1.0, 3.0, 5.0, 2.0, 20.0, 40.0, 80.0],
        }
    ).cast({"customers": pl.Int32})

    cohort_rows = []
    for cohort in (1, 2, 3):
        for since in range(0, 4 - cohort + 1):
            cohort_rows.append(
                (cohort, cohort + since, since, FIRST_MONDAY, FIRST_MONDAY, 4, 1000,
                 int(1000 * 0.9**since), 0.9**since, 5000, 1e5, 20.0, cohort == 1)
            )  # fmt: skip
    cohort = pl.DataFrame(
        cohort_rows,
        schema={
            "cohort_period": pl.Int32, "period_idx": pl.Int32, "periods_since_first": pl.Int32,
            "cohort_start": pl.Date, "period_start": pl.Date, "weeks_in_period": pl.Int32,
            "cohort_customers": pl.Int32, "active_customers": pl.Int32,
            "retention_rate": pl.Float64, "baskets": pl.Int64, "revenue": pl.Float64,
            "revenue_per_active_customer": pl.Float64, "is_baseline_cohort": pl.Boolean,
        },
        orient="row",
    )  # fmt: skip

    formats = sorted(stores["store_format"].unique().to_list())
    trading = pl.DataFrame(
        [
            (f, d, h, int(rng.integers(100, 1000)), float(rng.uniform(2e3, 2e4)), 0.01)
            for f in formats
            for d in range(1, 8)
            for h in range(6, 22)
        ],
        schema={
            "store_format": pl.String, "shop_weekday": pl.Int8, "shop_hour": pl.Int8,
            "baskets": pl.Int64, "revenue": pl.Float64, "revenue_share_of_format": pl.Float64,
        },
        orient="row",
    )  # fmt: skip
    mix_rows = [
        (200600 + w, f, t, s, m, "UM", True, 100, 2e3, 800, 900)
        for w in range(1, 118, 8)
        for f in formats
        for t in ("Top Up", "Full Shop")
        for s, m in (("S", "Fresh"), ("L", "Grocery"))
    ]
    mix = pl.DataFrame(
        mix_rows,
        schema={
            "shop_week": pl.Int32, "store_format": pl.String, "basket_type": pl.String,
            "basket_size": pl.String, "basket_dominant_mission": pl.String,
            "basket_price_sensitivity": pl.String, "is_loyalty": pl.Boolean,
            "baskets": pl.Int32, "revenue": pl.Float64, "units": pl.Int64, "lines": pl.Int64,
        },
        orient="row",
    )  # fmt: skip

    return {
        "dim_store": dim_store,
        "dim_date": _dim_date(),
        "dim_comparison": pl.DataFrame(
            {
                "comparison": list(COMPARISONS),
                "current_from": [105, 105, 66],
                "current_to": [117, 117, 117],
                "base_from": [92, 53, 14],
                "base_to": [104, 65, 65],
            }
        ).cast(dict.fromkeys(("current_from", "current_to", "base_from", "base_to"), pl.Int32)),
        "fct_store_week": fct_store_week,
        "mart_store_scorecard": per_comparison(scorecard),
        "mart_category_benchmark": per_comparison(category_benchmark),
        "mart_region_benchmark": per_comparison(region_benchmark),
        "mart_region_category_mix": per_comparison(region_category_mix),
        "mart_revenue_variance_store": per_comparison(variance),
        "mart_revenue_variance_region": per_comparison(variance_region),
        "mart_rfm_segment_summary": segments,
        "mart_customer_cohort": cohort,
        "mart_trading_pattern": trading,
        "mart_basket_mix": mix,
    }


def summary_rows(models: list[str]) -> dict[str, list[dict]]:
    def table(*extra, **fixed):
        return [
            {"model": m, **dict.fromkeys(extra, 0.1), "wape": 0.15 + 0.01 * i, "bias": 0.0, **fixed}
            for i, m in enumerate(models)
        ]

    return {
        "full_pooled": table("rmse", n=100, series=10, mase_mean=1.0, mase_median=0.9),
        "full_by_origin": [{**r, "origin": o} for o in (78, 91, 104) for r in table()],
        "full_store_level": table(n=10),
        "full_category_level": table(n=2),
        "full_network_level": table(n=1),
        "full_by_horizon": [{**r, "horizon_bucket": b} for b in ("weeks 1-4",) for r in table()],
        "full_by_format": [{**r, "store_format": "LS", "n": 5} for r in table()],
        "full_vs_baseline": [
            {"model": m, "baseline": "seasonal_naive", "units": 5, "improvement": 0.02,
             "ci_low": -0.01, "ci_high": 0.05, "win_rate": 0.6, "distinguishable": False}
            for m in models
        ],
        "subset_rolling": table("rmse", n=50, series=5, mase_mean=1.0, mase_median=0.9),
    }  # fmt: skip


def write_lake(root) -> LocalStore:
    store = LocalStore(root)
    panel = make_panel()
    forecasts = make_forecasts(panel, models=("ensemble", "lightgbm"))
    for name, frame in gold_tables(panel).items():
        path = store.root / f"gold/{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
    store.write_bytes(
        "gold/_manifest.json",
        json.dumps(
            {"built_at": "2026-10-22T00:00:00+00:00", "silver_first_week": 200607,
             "silver_last_week": 200819, "silver_weeks": 117, "tables": {}}
        ).encode(),
    )  # fmt: skip
    save_forecasts(store, forecasts, "main", [78, 91, 104])
    save_forecasts(store, forecasts, "robustness", [85, 98])
    for run in ("main", "robustness"):
        store.write_bytes(
            f"forecast/runs/{run}/summary.json",
            json.dumps(summary_rows(["ensemble", "lightgbm"])).encode(),
        )
    results = compute_risk(forecasts, panel, store_weeks(panel), ["main", "robustness"], top_k=5)
    save_results(store, results)
    return store
