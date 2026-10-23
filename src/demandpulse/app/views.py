"""Display-ready tables for the overview, store, region and customer pages.

Every function takes plain Polars frames from the snapshot and returns plain frames or dicts, so
the logic is testable without Streamlit. Revenue is the dataset's `spend`; it has no currency,
cost or margin, so nothing here is labelled profit or in a currency.
"""

from dataclasses import dataclass

import polars as pl

COMPARISON_LABELS = {
    "window_vs_year_ago": "Last 13 weeks vs the same weeks a year earlier",
    "window_vs_prior": "Last 13 weeks vs the previous 13 weeks",
    "year_vs_prior_year": "Last 52 weeks vs the 52 weeks before",
}
NETWORK = "ALL"
TREND_WINDOW = 13
SEGMENT_ORDER = ("Champions", "Loyal", "Potential", "New", "At risk", "Hibernating", "Lost")
BRIDGE_STEPS = (
    ("trading_weeks_effect", "Trading weeks"),
    ("traffic_effect", "Traffic (baskets per week)"),
    ("basket_size_effect", "Basket size (units per basket)"),
    ("price_mix_effect", "Price and mix (revenue per unit)"),
    ("new_store_effect", "New stores"),
    ("closed_store_effect", "Closed stores"),
)
RECONCILIATION_TOLERANCE = 0.5


def network_kpis(region_benchmark: pl.DataFrame, comparison: str) -> dict[str, float]:
    """Headline numbers for the whole network in one comparison window."""
    rows = region_benchmark.filter(
        (pl.col("comparison") == comparison) & (pl.col("store_region") == NETWORK)
    )
    if rows.height != 1:
        raise ValueError(f"Expected one network row for {comparison!r}, found {rows.height}")
    row = rows.row(0, named=True)
    return {
        "revenue": row["current_revenue"],
        "base_revenue": row["base_revenue"],
        "revenue_growth": row["revenue_growth"],
        "lfl_revenue_growth": row["lfl_revenue_growth"],
        "baskets": row["current_baskets"],
        "basket_growth": row["basket_growth"],
        "avg_basket_value": row["current_avg_basket_value"],
        "avg_basket_value_growth": row["current_avg_basket_value"] / row["base_avg_basket_value"]
        - 1,
        "stores": row["stores_current"],
        "lfl_stores": row["lfl_stores"],
    }


def weekly_trend(
    store_week: pl.DataFrame, regions: list[str] | None = None, formats: list[str] | None = None
) -> pl.DataFrame:
    """Weekly revenue and baskets for the selected stores, with a 13-week rolling mean."""
    frame = store_week
    if regions:
        frame = frame.filter(pl.col("store_region").is_in(regions))
    if formats:
        frame = frame.filter(pl.col("store_format").is_in(formats))
    return (
        frame.group_by(["week_idx", "week_start"])
        .agg(
            pl.col("revenue").sum(),
            pl.col("baskets").sum(),
            pl.col("store_code").n_unique().alias("stores"),
        )
        .sort("week_idx")
        .with_columns(pl.col("revenue").rolling_mean(window_size=TREND_WINDOW).alias("rolling_13"))
    )


def revenue_bridge(
    variance_region: pl.DataFrame, comparison: str, region: str = NETWORK
) -> pl.DataFrame:
    """Steps from base to current revenue. Effects add up exactly in Gold; any gap is shown."""
    rows = variance_region.filter(
        (pl.col("comparison") == comparison) & (pl.col("store_region") == region)
    )
    if rows.height != 1:
        raise ValueError(f"Expected one variance row for {comparison!r}/{region!r}")
    row = rows.row(0, named=True)
    steps = [("Base period", row["base_revenue"], "total")]
    steps += [(label, row[column], "delta") for column, label in BRIDGE_STEPS]
    explained = row["base_revenue"] + sum(row[column] for column, _ in BRIDGE_STEPS)
    gap = row["current_revenue"] - explained
    if abs(gap) > RECONCILIATION_TOLERANCE:
        steps.append(("Unreconciled", gap, "delta"))
    steps.append(("Current period", row["current_revenue"], "total"))
    return pl.DataFrame(steps, schema=["step", "value", "kind"], orient="row")


def category_contribution(
    category_benchmark: pl.DataFrame, comparison: str, top: int = 8
) -> pl.DataFrame:
    """The categories that added most to, and took most from, revenue growth (in points)."""
    rows = category_benchmark.filter(pl.col("comparison") == comparison).select(
        "prod_code_30", "revenue_growth", "current_share", "contribution_to_growth_pp"
    )
    ordered = rows.sort("contribution_to_growth_pp")
    return (
        pl.concat([ordered.head(top), ordered.tail(top)])
        .unique("prod_code_30")
        .sort("contribution_to_growth_pp")
    )


def mission_mix(basket_mix: pl.DataFrame, formats: list[str] | None = None) -> pl.DataFrame:
    """Share of sampled-week revenue by basket mission and type."""
    frame = basket_mix.filter(pl.col("store_format").is_in(formats)) if formats else basket_mix
    grouped = frame.group_by(["basket_dominant_mission", "basket_type"]).agg(
        pl.col("revenue").sum()
    )
    total = grouped["revenue"].sum()
    return grouped.with_columns((pl.col("revenue") / total).alias("revenue_share")).sort(
        "revenue_share", descending=True
    )


def trading_matrix(trading_pattern: pl.DataFrame, store_format: str) -> pl.DataFrame:
    """Revenue share of a format by weekday (rows) and hour (columns)."""
    frame = trading_pattern.filter(pl.col("store_format") == store_format).sort("shop_hour")
    return frame.pivot(
        on="shop_hour",
        index="shop_weekday",
        values="revenue_share_of_format",
        aggregate_function="sum",
    ).sort("shop_weekday")


def region_table(region_benchmark: pl.DataFrame, comparison: str) -> pl.DataFrame:
    """Regions (without the network row) ordered by like-for-like growth."""
    return (
        region_benchmark.filter(
            (pl.col("comparison") == comparison) & (pl.col("store_region") != NETWORK)
        )
        .select(
            "store_region",
            "stores_current",
            "lfl_stores",
            "current_revenue",
            "revenue_growth",
            "lfl_revenue_growth",
            "basket_growth",
            "current_avg_basket_value",
        )
        .sort("lfl_revenue_growth", descending=True)
    )


def region_mix_matrix(region_category_mix: pl.DataFrame, comparison: str) -> pl.DataFrame:
    """Mix index (1.0 = national share) by region (rows) and category (columns)."""
    return (
        region_category_mix.filter(pl.col("comparison") == comparison)
        .sort("prod_code_30")
        .pivot(
            on="prod_code_30", index="store_region", values="mix_index", aggregate_function="first"
        )
        .sort("store_region")
    )


def scorecard(
    store_scorecard: pl.DataFrame,
    comparison: str,
    regions: list[str] | None = None,
    formats: list[str] | None = None,
    lfl_only: bool = False,
    search: str = "",
) -> pl.DataFrame:
    """Store scorecard rows for the selected window, filtered, weakest growth first."""
    frame = store_scorecard.filter(pl.col("comparison") == comparison)
    if regions:
        frame = frame.filter(pl.col("store_region").is_in(regions))
    if formats:
        frame = frame.filter(pl.col("store_format").is_in(formats))
    if lfl_only:
        frame = frame.filter(pl.col("is_lfl"))
    if search.strip():
        frame = frame.filter(
            pl.col("store_code").str.contains(search.strip().upper(), literal=True)
        )
    return frame.select(
        "store_code",
        "store_region",
        "store_format",
        "lifecycle",
        "is_lfl",
        "current_revenue",
        "revenue_growth",
        "basket_growth",
        "current_avg_basket_value",
        "revenue_index_vs_format",
        "lfl_growth_zscore",
    ).sort("revenue_growth", nulls_last=True)


@dataclass(frozen=True)
class StoreDetail:
    profile: dict
    drivers: pl.DataFrame
    weekly: pl.DataFrame


def store_detail(
    store_scorecard: pl.DataFrame,
    variance_store: pl.DataFrame,
    store_week: pl.DataFrame,
    store_code: str,
    comparison: str,
) -> StoreDetail:
    """Scorecard row, revenue-driver split and weekly history for one store."""
    card = store_scorecard.filter(
        (pl.col("store_code") == store_code) & (pl.col("comparison") == comparison)
    )
    if card.height != 1:
        raise KeyError(f"No scorecard for {store_code!r} in {comparison!r}")
    variance = variance_store.filter(
        (pl.col("store_code") == store_code) & (pl.col("comparison") == comparison)
    )
    drivers = pl.DataFrame(
        [(label, variance[column][0]) for column, label in BRIDGE_STEPS] if variance.height else [],
        schema={"driver": pl.String, "effect": pl.Float64},
        orient="row",
    )
    weekly = (
        store_week.filter(pl.col("store_code") == store_code)
        .select("week_idx", "week_start", "revenue", "baskets", "units")
        .sort("week_idx")
        .with_columns(pl.col("revenue").rolling_mean(window_size=TREND_WINDOW).alias("rolling_13"))
    )
    return StoreDetail(profile=card.row(0, named=True), drivers=drivers, weekly=weekly)


def loyalty_coverage(store_week: pl.DataFrame) -> dict[str, float]:
    """How much of revenue and baskets carries a loyalty card; customer pages cover only these."""
    totals = store_week.select(
        pl.col("revenue").sum(),
        pl.col("loyalty_revenue").sum(),
        pl.col("baskets").sum().cast(pl.Float64),
        pl.col("loyalty_baskets").sum().cast(pl.Float64),
    ).row(0, named=True)
    return {
        "revenue_share": totals["loyalty_revenue"] / totals["revenue"],
        "basket_share": totals["loyalty_baskets"] / totals["baskets"],
    }


def segment_table(rfm_summary: pl.DataFrame) -> pl.DataFrame:
    """RFM segments in the order of the customer journey, with their customer and revenue shares."""
    order = {name: i for i, name in enumerate(SEGMENT_ORDER)}
    return (
        rfm_summary.with_columns(
            pl.col("rfm_segment").replace_strict(order, default=len(order)).alias("_order")
        )
        .sort("_order")
        .drop("_order")
    )


def retention_matrix(cohort: pl.DataFrame) -> pl.DataFrame:
    """Retention rate by cohort (rows) and periods since first purchase (columns)."""
    return (
        cohort.sort("periods_since_first")
        .with_columns(pl.col("periods_since_first").cast(pl.String))
        .pivot(
            on="periods_since_first",
            index="cohort_period",
            values="retention_rate",
            aggregate_function="first",
        )
        .sort("cohort_period")
    )
