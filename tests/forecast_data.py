"""Synthetic weekly store x category panels for forecasting tests."""

from datetime import date, timedelta

import numpy as np
import polars as pl

FIRST_MONDAY = date(2006, 4, 10)
CATEGORIES = ["G1", "G2", "G3", "G4"]
STORES = [("S1", "LS", "E01"), ("S2", "SS", "W01"), ("S3", "LS", "E01")]


def make_seasonal_panel(
    n_weeks: int = 130, scales: tuple[float, ...] = (1.0, 100.0), seed: int = 7
) -> pl.DataFrame:
    """Yearly seasonality (phase differs by category), a mild trend and multiplicative noise.

    Store sizes cycle through `scales`, so series differ by up to 100x.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s_index, (store, fmt, region) in enumerate(STORES):
        for c_index, category in enumerate(CATEGORIES):
            level = 50.0 * scales[s_index % len(scales)] * (1 + c_index)
            phase = c_index * 1.3
            for week in range(1, n_weeks + 1):
                seasonal = 1 + 0.5 * np.sin(2 * np.pi * week / 52 + phase)
                revenue = level * seasonal * (1 + 0.002 * week) * rng.lognormal(0, 0.05)
                start = FIRST_MONDAY + timedelta(days=7 * (week - 1))
                rows.append(
                    (
                        store,
                        category,
                        fmt,
                        region,
                        week,
                        200600 + week,
                        (week - 1) % 52 + 1,
                        start,
                        revenue,
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
            "shop_week": pl.Int64,
            "fiscal_week": pl.Int64,
            "week_start": pl.Date,
            "revenue": pl.Float64,
        },
        orient="row",
    )
