"""Deterministic synthetic Silver data for warehouse SQL tests.

Structure is chosen so edge cases exist: a store that opens mid-way (S3), a store that closes
(S4), customers who churn or start late, non-loyalty baskets, and a product tree with two
divisions. Windows in tests use window_weeks=2 and year_weeks=4 over 8 contiguous weeks.
"""

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl

from demandpulse.ingest.schema import TRANSACTION_SCHEMA

WEEKS = [200607 + i for i in range(8)]
FIRST_MONDAY = date(2006, 4, 10)
SOURCE = "fixture"

STORES = {
    # store: (format, region, first_week_idx, last_week_idx)  (idx is 0-based)
    "STORE01": ("LS", "E01", 0, 7),
    "STORE02": ("LS", "E02", 0, 7),
    "STORE03": ("SS", "E01", 4, 7),  # opens in the second half
    "STORE04": ("MS", "W01", 0, 3),  # closes after week idx 3
}
# product -> (class L10, subcategory L20, category L30, division L40)
PRODUCTS = {
    f"PRD{i:02d}": (
        f"CL{i:02d}",
        f"DEP{(i - 1) // 2 + 1}",
        f"G{(i - 1) // 4 + 1}",
        "D1" if i <= 8 else "D2",
    )
    for i in range(1, 13)
}
PRICE = {p: 1.0 + 0.5 * (i % 4) for i, p in enumerate(PRODUCTS)}
BASKETS_PER_STORE_WEEK = 24
N_CUSTOMERS = 40


def build_silver_frame(seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    products = list(PRODUCTS)
    # Customers: (first_week_idx, last_week_idx) so cohorts and churn exist.
    customers = {
        f"CUST{i:03d}": (int(rng.integers(0, 5)), int(rng.integers(3, 8)))
        for i in range(N_CUSTOMERS)
    }
    cust_seg = {
        c: (f"S{int(rng.integers(1, 4))}", f"T{int(rng.integers(1, 3))}") for c in customers
    }

    rows: list[dict] = []
    for widx, week in enumerate(WEEKS):
        week_start = FIRST_MONDAY + timedelta(days=7 * widx)
        for store, (fmt, region, first, last) in STORES.items():
            if not first <= widx <= last:
                continue
            active = [c for c, (a, b) in customers.items() if a <= widx <= b]
            for n in range(BASKETS_PER_STORE_WEEK):
                basket_id = week * 100000 + int(store[-2:]) * 1000 + n
                loyalty = rng.random() < 0.7 and bool(active)
                cust = str(rng.choice(active)) if loyalty else None
                shop_date = week_start + timedelta(days=int(rng.integers(0, 7)))
                size = str(rng.choice(["S", "M", "L"], p=[0.3, 0.4, 0.3]))
                btype = str(rng.choice(["Small Shop", "Top Up", "Full Shop"]))
                mission = str(rng.choice(["Fresh", "Grocery", "Mixed", "Nonfood"]))
                sens = str(rng.choice(["LA", "MM", "UM"]))
                chosen = rng.choice(products, size=int(rng.integers(1, 5)), replace=False)
                for prod in chosen:
                    qty = int(rng.integers(1, 6))
                    spend = round(qty * PRICE[str(prod)] * float(rng.uniform(0.9, 1.1)), 2)
                    l10, l20, l30, l40 = PRODUCTS[str(prod)]
                    rows.append(
                        {
                            "shop_week": week,
                            "shop_date": shop_date,
                            "shop_weekday": (shop_date.weekday() + 1) % 7 + 1,
                            "shop_hour": int(rng.integers(8, 22)),
                            "quantity": qty,
                            "spend": Decimal(f"{spend:.2f}"),
                            "prod_code": str(prod),
                            "prod_code_10": l10,
                            "prod_code_20": l20,
                            "prod_code_30": l30,
                            "prod_code_40": l40,
                            "cust_code": cust,
                            "seg_1": cust_seg[cust][0] if cust else None,
                            "seg_2": cust_seg[cust][1] if cust else None,
                            "basket_id": basket_id,
                            "basket_size": size,
                            "basket_price_sensitivity": sens,
                            "basket_type": btype,
                            "basket_dominant_mission": mission,
                            "store_code": store,
                            "store_format": fmt,
                            "store_region": region,
                            "is_loyalty": cust is not None,
                            "week_start": week_start,
                            "week_end": week_start + timedelta(days=6),
                        }
                    )
    schema = {
        **TRANSACTION_SCHEMA,
        "is_loyalty": pl.Boolean(),
        "week_start": pl.Date(),
        "week_end": pl.Date(),
    }
    return pl.DataFrame(rows, schema=schema)


def write_silver_lake(root: Path, frame: pl.DataFrame, source: str = SOURCE) -> None:
    for week, part in frame.group_by("shop_week"):
        target = root / "silver" / f"source={source}" / f"shop_week={week[0]}" / "part-0.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(target)
