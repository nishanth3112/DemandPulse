"""Weekly store x category revenue panel for forecasting, with leakage-safe eligibility.

The panel is dense: every category has a row for every week a store was trading, with zero
revenue when nothing sold. Weeks before a store opened are absent (not zero), so a new store
is never mistaken for a collapsing one. All eligibility rules for a forecast origin look only
at weeks up to and including that origin.
"""

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from demandpulse.warehouse.build import GoldConfig
from demandpulse.warehouse.checks import open_gold

logger = logging.getLogger(__name__)

SERIES_KEY = ["store_code", "prod_code_30"]
HORIZON = 13
YEAR_WEEKS = 52
# Three rolling origins, one quarter apart. Test windows are the 13 weeks after each origin
# (79-91, 92-104, 105-117 of the 117 data weeks).
ORIGINS = (78, 91, 104)
# One year of history plus the 13 weeks the year-over-year growth feature looks back over. Using
# the same threshold for every model keeps the evaluation set identical across models.
MIN_HISTORY_WEEKS = YEAR_WEEKS + 13
# A series must have sold in at least half of the last 52 weeks to be forecastable.
MIN_NONZERO_WEEKS = 26
SUBSET_SEED = 20061

PANEL_SQL = """
WITH grid AS (
    SELECT s.store_code, s.store_format, s.store_region, c.prod_code_30,
           d.week_idx, d.shop_week, d.fiscal_week, d.week_start
    FROM dim_store s
    CROSS JOIN (SELECT DISTINCT prod_code_30 FROM dim_product) c
    JOIN dim_date d ON d.week_idx BETWEEN s.first_week_idx AND s.last_week_idx
)
SELECT
    g.store_code, g.prod_code_30, g.store_format, g.store_region,
    g.week_idx, g.shop_week, g.fiscal_week, g.week_start,
    CAST(coalesce(f.revenue, 0) AS DOUBLE) AS revenue,
    CAST(coalesce(f.units, 0) AS DOUBLE) AS units,
    CAST(coalesce(f.baskets, 0) AS DOUBLE) AS baskets
FROM grid g
LEFT JOIN fct_store_category_week f
    ON f.shop_week = g.shop_week AND f.store_code = g.store_code
   AND f.prod_code_30 = g.prod_code_30
ORDER BY g.store_code, g.prod_code_30, g.week_idx
"""


def build_panel(config: GoldConfig) -> pl.DataFrame:
    """Dense weekly revenue panel from the published Gold marts."""
    con = open_gold(config)
    try:
        frame = con.execute(PANEL_SQL).pl()
    finally:
        con.close()
    logger.info(
        "panel: %d rows, %d series, %d weeks",
        frame.height,
        frame.select(SERIES_KEY).unique().height,
        frame["week_idx"].n_unique(),
    )
    return frame


def history(panel: pl.DataFrame, origin: int) -> pl.DataFrame:
    """Rows observable at `origin`: nothing after it."""
    return panel.filter(pl.col("week_idx") <= origin)


def eligible_series(
    panel: pl.DataFrame,
    origin: int,
    horizon: int = HORIZON,
    min_history: int = MIN_HISTORY_WEEKS,
    min_nonzero: int = MIN_NONZERO_WEEKS,
    year_weeks: int = YEAR_WEEKS,
) -> pl.DataFrame:
    """Series that can be forecast at `origin`, decided from data up to the origin only.

    The one look-ahead is that the series must still be trading through the test window, which
    is what makes it evaluable; it never depends on test-window revenue values.
    """
    past = (
        history(panel, origin)
        .group_by(SERIES_KEY)
        .agg(
            pl.len().alias("history_weeks"),
            ((pl.col("revenue") > 0) & (pl.col("week_idx") > origin - year_weeks))
            .sum()
            .alias("nonzero_weeks_last_year"),
            pl.col("revenue")
            .filter(pl.col("week_idx") > origin - year_weeks)
            .sum()
            .alias("revenue_last_year"),
        )
    )
    future = (
        panel.filter((pl.col("week_idx") > origin) & (pl.col("week_idx") <= origin + horizon))
        .group_by(SERIES_KEY)
        .agg(pl.len().alias("test_weeks"))
    )
    return (
        past.join(future, on=SERIES_KEY)
        .filter(
            (pl.col("history_weeks") >= min_history)
            & (pl.col("nonzero_weeks_last_year") >= min_nonzero)
            & (pl.col("test_weeks") == horizon)
        )
        .select(*SERIES_KEY, "history_weeks", "nonzero_weeks_last_year", "revenue_last_year")
    )


@dataclass(frozen=True)
class Subset:
    series: pl.DataFrame  # SERIES_KEY + stratum
    seed: int


def representative_subset(
    panel: pl.DataFrame,
    origins: tuple[int, ...] = ORIGINS,
    per_stratum: int = 20,
    seed: int = SUBSET_SEED,
    **eligibility,
) -> Subset:
    """Stratified random sample used for the expensive per-series models.

    Series must be eligible at every origin so all models are compared on identical series at
    identical cutoffs. Strata are store format x revenue tier (terciles of last-year revenue at
    the final origin), so small and large series, and every store size, are represented.
    """
    eligible = None
    for origin in origins:
        current = eligible_series(panel, origin, **eligibility).select(SERIES_KEY)
        eligible = current if eligible is None else eligible.join(current, on=SERIES_KEY)
    assert eligible is not None
    if not eligible.height:
        raise ValueError(
            f"No series are eligible at every origin in {origins}; the panel needs at least "
            f"{max(origins) + HORIZON} weeks and series with {MIN_HISTORY_WEEKS}+ weeks of history"
        )

    last = eligible_series(panel, origins[-1], **eligibility).select(
        *SERIES_KEY, "revenue_last_year"
    )
    formats = panel.select("store_code", "store_format").unique(subset="store_code")
    base = eligible.join(last, on=SERIES_KEY).join(formats, on="store_code")
    cuts = base["revenue_last_year"].quantile(1 / 3), base["revenue_last_year"].quantile(2 / 3)
    base = base.with_columns(
        revenue_tier=pl.when(pl.col("revenue_last_year") <= cuts[0])
        .then(pl.lit("small"))
        .when(pl.col("revenue_last_year") <= cuts[1])
        .then(pl.lit("medium"))
        .otherwise(pl.lit("large"))
    ).with_columns(stratum=pl.col("store_format") + "|" + pl.col("revenue_tier"))

    rng = np.random.default_rng(seed)
    picked: list[pl.DataFrame] = []
    # Sorted so the draw does not depend on row order or dict ordering.
    for stratum in sorted(base["stratum"].unique().to_list()):
        members = base.filter(pl.col("stratum") == stratum).sort(SERIES_KEY)
        take = min(per_stratum, members.height)
        index = np.sort(rng.choice(members.height, size=take, replace=False))
        picked.append(members[index.tolist()])
    chosen = pl.concat(picked).select(*SERIES_KEY, "stratum").sort(SERIES_KEY)
    return Subset(series=chosen, seed=seed)
