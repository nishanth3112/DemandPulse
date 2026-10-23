"""Stores to look at first, with the measured evidence behind each.

Ranked by expected revenue below target. Each row carries only numbers computed from the data:
the forecast gap, the probability of a shortfall, the store's own recent growth and which of
traffic, basket size and price/mix fell most. It names where to look, not what to do: nothing
here is a causal finding.
"""

import polars as pl

DRIVER_LABELS = {
    "traffic": "baskets per week",
    "basket_size": "units per basket",
    "price_mix": "revenue per unit",
}
DRIVER_COLUMNS = {
    "traffic": "traffic_change",
    "basket_size": "basket_size_change",
    "price_mix": "price_mix_change",
}


def evidence_text(row: dict) -> str:
    gap = row["forecast_shortfall"]
    side = "below" if gap > 0 else "above"
    parts = [
        f"{row['store_code']}: forecast is {abs(gap):,.0f} ({abs(gap) / row['target']:.1%}) "
        f"{side} target",
        f"probability of a shortfall {row['p_shortfall']:.0%}",
    ]
    if row.get("revenue_growth") is not None:
        parts.append(f"last 13 weeks {row['revenue_growth']:+.1%} on a year earlier")
    driver = row.get("dominant_driver")
    if driver in DRIVER_COLUMNS:
        change = row[DRIVER_COLUMNS[driver]]
        parts.append(f"weakest driver {DRIVER_LABELS[driver]} ({change:+.1%})")
    elif driver == "none":
        parts.append("no driver is below last year")
    else:
        parts.append("driver split unavailable")
    return "; ".join(parts)


def recommendations(risk: pl.DataFrame, drivers: pl.DataFrame, top_k: int = 20) -> pl.DataFrame:
    """Top `top_k` stores by revenue at risk for one model, origin and target definition."""
    scored = risk.filter(pl.col("revenue_at_risk").is_not_null())
    if not scored.height:
        raise ValueError("No revenue-at-risk estimates; the origin has no earlier windows")
    top = (
        scored.sort(["revenue_at_risk", "store_code"], descending=[True, False])
        .head(top_k)
        .join(drivers, on="store_code", how="left")
        .sort(["revenue_at_risk", "store_code"], descending=[True, False])
    )
    rows = top.to_dicts()
    return top.with_columns(
        pl.Series("rank", range(1, top.height + 1), dtype=pl.Int64),
        pl.Series("evidence", [evidence_text(r) for r in rows], dtype=pl.String),
    )
