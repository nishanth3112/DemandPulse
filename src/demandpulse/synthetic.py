"""Deterministic synthetic Silver data with the real schema, for demos, benchmarks and tests.

It is invented data with realistic structure, not the dunnhumby dataset: a 117-week fiscal
calendar starting at 200607, stores of four formats in twelve regions with yearly seasonality and
a trend, a store that opens late, one that closes, one with an abrupt level shift, loyalty and
anonymous baskets, and a three-level product hierarchy. Nothing about its results says anything
about the real dataset.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from demandpulse.ingest.schema import TRANSACTION_SCHEMA

SOURCE = "synthetic"
FIRST_MONDAY = date(2006, 4, 10)
FIRST_FISCAL_WEEK = (2006, 7)
FISCAL_WEEKS_PER_YEAR = 52
FORMATS = ("LS", "MS", "SS", "XLS")
BASE_BASKETS = {"LS": 30, "MS": 22, "SS": 14, "XLS": 40}
REGIONS = ("E01", "E02", "E03", "N01", "N02", "N03", "S01", "S02", "S03", "W01", "W02", "W03")
PRODUCTS_PER_CATEGORY = 4
CUSTOMER_POOL = 800
LOYALTY_SHARE = 0.75
# Store indices (1-based) that get a non-trivial history, so the dashboard and risk model have
# something to find: late opening, closure, and a sudden drop in volume.
LATE_OPENING = (4, 30)
CLOSURE = (6, 80)
LEVEL_SHIFT = (8, 90, 0.55)


@dataclass(frozen=True)
class SyntheticSummary:
    rows: int
    baskets: int
    weeks: int
    stores: int


def fiscal_week(index: int) -> int:
    """YYYYWW of the `index`-th week (0-based), rolling over after fiscal week 52."""
    year, week = FIRST_FISCAL_WEEK
    total = (week - 1) + index
    return (year + total // FISCAL_WEEKS_PER_YEAR) * 100 + total % FISCAL_WEEKS_PER_YEAR + 1


def _store_table(stores: int, rng: np.random.Generator) -> list[dict]:
    table = []
    for i in range(1, stores + 1):
        fmt = FORMATS[(i - 1) % len(FORMATS)]
        first, last, shift_at, shift = 0, None, None, 1.0
        if stores >= LATE_OPENING[0] and i == LATE_OPENING[0]:
            first = LATE_OPENING[1]
        if stores >= CLOSURE[0] and i == CLOSURE[0]:
            last = CLOSURE[1]
        if stores >= LEVEL_SHIFT[0] and i == LEVEL_SHIFT[0]:
            shift_at, shift = LEVEL_SHIFT[1], LEVEL_SHIFT[2]
        table.append(
            {
                "code": f"STORE{i:05d}",
                "format": fmt,
                "region": REGIONS[(i - 1) % len(REGIONS)],
                "size": float(rng.uniform(0.7, 1.3)),
                "phase": float(rng.uniform(0, 2 * np.pi)),
                "trend": float(rng.normal(0.0005, 0.001)),
                "first": first,
                "last": last,
                "shift_at": shift_at,
                "shift": shift,
            }
        )
    return table


def _baskets_in_week(store: dict, week: int, rng: np.random.Generator) -> int:
    if week < store["first"] or (store["last"] is not None and week > store["last"]):
        return 0
    seasonal = 1 + 0.25 * np.sin(2 * np.pi * week / FISCAL_WEEKS_PER_YEAR + store["phase"])
    level = BASE_BASKETS[store["format"]] * store["size"] * seasonal * (1 + store["trend"] * week)
    if store["shift_at"] is not None and week >= store["shift_at"]:
        level *= store["shift"]
    return max(1, int(round(level * rng.lognormal(0, 0.06))))


def _week_frame(
    week: int,
    stores: list[dict],
    categories: int,
    customers: np.ndarray,
    segments: np.ndarray,
    rng: np.random.Generator,
) -> pl.DataFrame | None:
    counts = np.array([_baskets_in_week(s, week, rng) for s in stores])
    n_baskets = int(counts.sum())
    if not n_baskets:
        return None
    start = FIRST_MONDAY + timedelta(days=7 * week)
    store_idx = np.repeat(np.arange(len(stores)), counts)
    lines = rng.integers(1, 5, n_baskets)
    day = rng.integers(0, 7, n_baskets)
    hour = rng.integers(8, 22, n_baskets)
    active = np.flatnonzero((customers[:, 0] <= week) & (customers[:, 1] >= week))
    loyal = (rng.random(n_baskets) < LOYALTY_SHARE) & (len(active) > 0)
    customer = np.where(loyal, rng.choice(active, n_baskets) if len(active) else 0, -1)

    line_basket = np.repeat(np.arange(n_baskets), lines)
    n_lines = len(line_basket)
    n_products = categories * PRODUCTS_PER_CATEGORY
    product = rng.integers(0, n_products, n_lines)
    category = product // PRODUCTS_PER_CATEGORY
    quantity = rng.integers(1, 6, n_lines)
    price = 1.0 + 0.5 * (product % 4) + 0.2 * category
    spend = np.round(quantity * price * rng.uniform(0.9, 1.1, n_lines), 2)

    def per_line(values: np.ndarray) -> np.ndarray:
        return values[line_basket]

    cust_line = per_line(customer)
    has_customer = cust_line >= 0
    cust_code = np.where(has_customer, np.char.add("CUST", np.char.zfill(
        np.maximum(cust_line, 0).astype(str), 5)), "")  # fmt: skip
    seg = segments[np.maximum(cust_line, 0)]
    sizes = rng.choice(["S", "M", "L"], n_baskets, p=[0.3, 0.4, 0.3])
    types = rng.choice(["Small Shop", "Top Up", "Full Shop"], n_baskets)
    missions = rng.choice(["Fresh", "Grocery", "Mixed", "Nonfood"], n_baskets)
    sens = rng.choice(["LA", "MM", "UM"], n_baskets)
    day_line = per_line(day)
    shop_date = np.datetime64(start) + day_line.astype("timedelta64[D]")
    fmt = np.array([s["format"] for s in stores])
    region = np.array([s["region"] for s in stores])
    code = np.array([s["code"] for s in stores])
    store_line = per_line(store_idx)
    prod_code = np.char.add("PRD", np.char.zfill(product.astype(str), 4))
    cat_code = np.char.add("G", np.char.zfill(category.astype(str), 2))

    data = {
        "shop_week": np.full(n_lines, fiscal_week(week)),
        "shop_date": shop_date,
        "shop_weekday": (day_line + 1) % 7 + 1,
        "shop_hour": per_line(hour),
        "quantity": quantity,
        "spend": spend,
        "prod_code": prod_code,
        "prod_code_10": np.char.add("CL", np.char.zfill(product.astype(str), 4)),
        "prod_code_20": np.char.add("DEP", np.char.zfill((product // 2).astype(str), 3)),
        "prod_code_30": cat_code,
        "prod_code_40": np.where(category % 2 == 0, "D1", "D2"),
        "cust_code": cust_code,
        "seg_1": seg[:, 0],
        "seg_2": seg[:, 1],
        "basket_id": fiscal_week(week) * 1_000_000 + line_basket.astype(np.int64),
        "basket_size": per_line(sizes),
        "basket_price_sensitivity": per_line(sens),
        "basket_type": per_line(types),
        "basket_dominant_mission": per_line(missions),
        "store_code": code[store_line],
        "store_format": fmt[store_line],
        "store_region": region[store_line],
    }
    frame = pl.DataFrame({k: pl.Series(k, v) for k, v in data.items()})
    frame = frame.with_columns(
        pl.col("spend").cast(pl.Decimal(12, 2)),
        pl.when(pl.col("cust_code") == "")
        .then(None)
        .otherwise(pl.col("cust_code"))
        .alias("cust_code"),
        pl.when(pl.col("cust_code") == "").then(None).otherwise(pl.col("seg_1")).alias("seg_1"),
        pl.when(pl.col("cust_code") == "").then(None).otherwise(pl.col("seg_2")).alias("seg_2"),
    )
    frame = frame.cast(TRANSACTION_SCHEMA)
    return frame.with_columns(
        pl.col("cust_code").is_not_null().alias("is_loyalty"),
        pl.lit(start).alias("week_start"),
        pl.lit(start + timedelta(days=6)).alias("week_end"),
    )


def write_synthetic_silver(
    root: Path,
    source: str = SOURCE,
    stores: int = 40,
    weeks: int = 117,
    categories: int = 6,
    seed: int = 7,
) -> SyntheticSummary:
    """Write one Silver partition per week under `<root>/silver/source=<source>/`."""
    if stores < 1 or weeks < 1 or categories < 1:
        raise ValueError("stores, weeks and categories must all be at least 1")
    rng = np.random.default_rng(seed)
    table = _store_table(stores, rng)
    first = rng.integers(0, max(1, weeks // 2), CUSTOMER_POOL)
    last = first + rng.integers(weeks // 4, weeks, CUSTOMER_POOL)
    customers = np.column_stack([first, last])
    segments = np.array(
        [[f"S{rng.integers(1, 4)}", f"T{rng.integers(1, 3)}"] for _ in range(CUSTOMER_POOL)]
    )
    rows = baskets = 0
    for week in range(weeks):
        frame = _week_frame(week, table, categories, customers, segments, rng)
        if frame is None:
            continue
        target = (
            root
            / "silver"
            / f"source={source}"
            / f"shop_week={fiscal_week(week)}"
            / "part-0.parquet"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)
        rows += frame.height
        baskets += frame["basket_id"].n_unique()
    return SyntheticSummary(rows=rows, baskets=baskets, weeks=weeks, stores=stores)
