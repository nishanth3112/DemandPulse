"""Leakage-safe feature engineering for the global LightGBM forecaster.

Everything is computed from a (series x week) revenue matrix. `make_features(matrix, t, h)`
slices the matrix to the columns observable at origin `t` before touching it, so reading a
value after the origin raises IndexError instead of silently leaking. Targets for training rows
are taken separately by the caller from weeks the training origin is allowed to know.

Revenue is divided by each series' mean over the 52 weeks before the origin, so one model can
learn from stores and categories whose revenue differs by orders of magnitude.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from demandpulse.forecasting.panel import SERIES_KEY, YEAR_WEEKS

SCALE_EPS = 1e-6
SHORT_WINDOW = 4
MEDIUM_WINDOW = 13
LONG_WINDOW = 26
MAX_YOY_LOG_RATIO = 3.0

NUMERIC_FEATURES_V1 = [
    "horizon",
    "fiscal_week_target",
    "fiscal_week_sin",
    "fiscal_week_cos",
    "fiscal_week_origin",
    "log_scale",
    "lag0",
    "lag1",
    "lag2",
    "lag3",
    "mean4",
    "mean13",
    "mean26",
    "std13",
    "nonzero13",
    "nonzero52",
    "year_ago",
    "year_ago_3wk",
    "yoy_log_ratio13",
    "category_seasonal_index",
    "network_seasonal_index",
    "store_log_scale",
]
# v2 adds seasonally adjusted momentum (recent revenue against the same weeks a year earlier) at
# series, category and network level, and medians. It was designed after v1 under-forecast at the
# final origin. On the two origins not used to design it (85 and 98) it was not better than v1
# (18.9% against 18.6% WAPE), so v1 is the adopted feature set and v2 is kept as an ablation.
NUMERIC_FEATURES_V2_EXTRA = [
    "median4",
    "median13",
    "yoy_log_ratio4",
    "category_yoy_log_ratio4",
    "category_yoy_log_ratio13",
    "network_yoy_log_ratio4",
    "network_yoy_log_ratio13",
]
CATEGORICAL_FEATURES = ["store_format", "store_region", "category"]
FEATURE_NAMES = [*NUMERIC_FEATURES_V1, *NUMERIC_FEATURES_V2_EXTRA, *CATEGORICAL_FEATURES]
FEATURE_SETS = {
    "v1": [*NUMERIC_FEATURES_V1, *CATEGORICAL_FEATURES],
    "v2": FEATURE_NAMES,
}
# Origins need a year of history plus the 13 weeks that the year-over-year ratio looks back over.
MIN_ORIGIN = YEAR_WEEKS + MEDIUM_WINDOW


@dataclass
class SeriesMatrix:
    keys: list[tuple[str, str]]
    weeks: np.ndarray  # week_idx for each column, 1..T
    values: np.ndarray  # (series, week) revenue, NaN before a store opened
    fiscal_week: dict[int, int]
    store_format: np.ndarray  # integer codes, one per series
    store_region: np.ndarray
    category: np.ndarray

    @property
    def n_series(self) -> int:
        return len(self.keys)


@dataclass(frozen=True)
class Encoders:
    """Stable string -> integer code maps shared by training and prediction."""

    store_format: dict[str, int]
    store_region: dict[str, int]
    category: dict[str, int]

    @classmethod
    def from_panel(cls, panel: pl.DataFrame) -> "Encoders":
        def codes(column: str) -> dict[str, int]:
            return {v: i for i, v in enumerate(sorted(panel[column].unique().to_list()))}

        return cls(codes("store_format"), codes("store_region"), codes("prod_code_30"))


def build_matrix(
    frame: pl.DataFrame, encoders: Encoders, last_week: int, calendar: dict[int, int]
) -> SeriesMatrix:
    """Pivot panel rows (all with week_idx <= last_week) to a series x week matrix."""
    if frame.height and frame["week_idx"].max() > last_week:
        raise ValueError("frame contains weeks after last_week")
    series = frame.select([*SERIES_KEY, "store_format", "store_region"]).unique(subset=SERIES_KEY)
    series = series.sort(SERIES_KEY)
    keys = [tuple(row) for row in series.select(SERIES_KEY).rows()]
    position = {key: i for i, key in enumerate(keys)}
    values = np.full((len(keys), last_week), np.nan)
    rows = frame.select(*SERIES_KEY, "week_idx", "revenue").iter_rows()
    for store, category, week, revenue in rows:
        values[position[(store, category)], week - 1] = revenue
    return SeriesMatrix(
        keys=keys,
        weeks=np.arange(1, last_week + 1),
        values=values,
        fiscal_week=calendar,
        store_format=np.array([encoders.store_format[v] for v in series["store_format"]]),
        store_region=np.array([encoders.store_region[v] for v in series["store_region"]]),
        category=np.array([encoders.category[k[1]] for k in keys]),
    )


def _group_totals(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    totals = np.zeros((n_groups, values.shape[1]))
    np.add.at(totals, groups, np.nan_to_num(values))
    return totals


def make_features(
    matrix: SeriesMatrix, t: int, h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features for forecasting week t+h from origin t.

    Returns (X, valid, scale). `valid` marks series with complete windows and positive scale.
    Only columns [0, t) of the matrix are read.
    """
    if not 1 <= h < YEAR_WEEKS:
        raise ValueError("horizon must be between 1 and 51 so the year-ago week is observed")
    visible = matrix.values[:, :t]  # weeks 1..t; the structural leak guard
    if t < MIN_ORIGIN:
        raise ValueError(f"origin {t} is before the minimum {MIN_ORIGIN}")

    last_year = visible[:, t - YEAR_WEEKS : t]
    scale = last_year.mean(axis=1)
    safe = np.where(scale > SCALE_EPS, scale, 1.0)
    complete = ~np.isnan(visible[:, t - YEAR_WEEKS - MEDIUM_WINDOW : t]).any(axis=1)
    valid = complete & (scale > SCALE_EPS)

    def lag(k: int) -> np.ndarray:
        return visible[:, t - 1 - k] / safe

    recent13 = visible[:, t - MEDIUM_WINDOW : t]
    prior13 = visible[:, t - MEDIUM_WINDOW - YEAR_WEEKS : t - YEAR_WEEKS]
    ratio = (recent13.sum(axis=1) + 1.0) / (prior13.sum(axis=1) + 1.0)
    yoy = np.clip(np.log(ratio), -MAX_YOY_LOG_RATIO, MAX_YOY_LOG_RATIO)

    target_col = t + h - YEAR_WEEKS - 1  # column of the same week one year before the target
    year_ago = visible[:, target_col] / safe
    year_ago_3 = visible[:, target_col - 1 : target_col + 2].mean(axis=1) / safe

    n_cat = int(matrix.category.max()) + 1
    cat_totals = _group_totals(visible, matrix.category, n_cat)
    cat_base = cat_totals[:, t - YEAR_WEEKS : t].mean(axis=1)
    cat_index = np.where(
        cat_base > 0, cat_totals[:, target_col] / np.where(cat_base > 0, cat_base, 1), 1.0
    )
    net_totals = np.nan_to_num(visible).sum(axis=0)
    net_base = net_totals[t - YEAR_WEEKS : t].mean()
    net_index = net_totals[target_col] / net_base if net_base > 0 else 1.0

    recent4 = visible[:, t - SHORT_WINDOW : t]
    prior4 = visible[:, t - SHORT_WINDOW - YEAR_WEEKS : t - YEAR_WEEKS]
    yoy4 = _log_ratio(recent4.sum(axis=1), prior4.sum(axis=1))
    cat_yoy4 = _log_ratio(
        cat_totals[:, t - SHORT_WINDOW : t].sum(axis=1),
        cat_totals[:, t - SHORT_WINDOW - YEAR_WEEKS : t - YEAR_WEEKS].sum(axis=1),
    )
    cat_yoy13 = _log_ratio(
        cat_totals[:, t - MEDIUM_WINDOW : t].sum(axis=1),
        cat_totals[:, t - MEDIUM_WINDOW - YEAR_WEEKS : t - YEAR_WEEKS].sum(axis=1),
    )
    net_yoy4 = _log_ratio(
        net_totals[t - SHORT_WINDOW : t].sum(),
        net_totals[t - SHORT_WINDOW - YEAR_WEEKS : t - YEAR_WEEKS].sum(),
    )
    net_yoy13 = _log_ratio(
        net_totals[t - MEDIUM_WINDOW : t].sum(),
        net_totals[t - MEDIUM_WINDOW - YEAR_WEEKS : t - YEAR_WEEKS].sum(),
    )

    store_scale = _store_scales(matrix, visible, t)
    fiscal_target = matrix.fiscal_week.get(t + h)
    fiscal_origin = matrix.fiscal_week[t]
    if fiscal_target is None:
        raise ValueError(f"calendar has no fiscal week for week_idx {t + h}")
    angle = 2 * np.pi * fiscal_target / YEAR_WEEKS

    n = matrix.n_series
    columns = [
        np.full(n, h),
        np.full(n, fiscal_target),
        np.full(n, np.sin(angle)),
        np.full(n, np.cos(angle)),
        np.full(n, fiscal_origin),
        np.log(safe),
        lag(0),
        lag(1),
        lag(2),
        lag(3),
        visible[:, t - SHORT_WINDOW : t].mean(axis=1) / safe,
        recent13.mean(axis=1) / safe,
        visible[:, t - LONG_WINDOW : t].mean(axis=1) / safe,
        recent13.std(axis=1) / safe,
        (recent13 > 0).mean(axis=1),
        (last_year > 0).mean(axis=1),
        year_ago,
        year_ago_3,
        yoy,
        cat_index[matrix.category],
        np.full(n, net_index),
        np.log(store_scale + 1.0),
        np.median(recent4, axis=1) / safe,
        np.median(recent13, axis=1) / safe,
        yoy4,
        cat_yoy4[matrix.category],
        cat_yoy13[matrix.category],
        np.full(n, net_yoy4),
        np.full(n, net_yoy13),
        matrix.store_format,
        matrix.store_region,
        matrix.category,
    ]
    return np.column_stack(columns).astype(np.float32), valid, scale


def _log_ratio(recent: np.ndarray | float, prior: np.ndarray | float) -> np.ndarray | float:
    """log((recent + 1) / (prior + 1)), clipped; the +1 keeps all-zero windows finite."""
    return np.clip(np.log((recent + 1.0) / (prior + 1.0)), -MAX_YOY_LOG_RATIO, MAX_YOY_LOG_RATIO)


def _store_scales(matrix: SeriesMatrix, visible: np.ndarray, t: int) -> np.ndarray:
    """Mean weekly revenue of each series' store over the last year (all of its categories)."""
    stores = np.array([k[0] for k in matrix.keys])
    unique, inverse = np.unique(stores, return_inverse=True)
    totals = np.zeros(len(unique))
    np.add.at(totals, inverse, np.nan_to_num(visible[:, t - YEAR_WEEKS : t]).sum(axis=1))
    return (totals / YEAR_WEEKS)[inverse]
