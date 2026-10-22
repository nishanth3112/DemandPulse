"""Forecast accuracy metrics.

Primary metric is WAPE (sum of absolute errors over sum of actuals): unlike MAPE or sMAPE it is
defined when individual weeks sell nothing, and it weights errors by revenue, which is what a
revenue-risk process cares about. Bias is the signed counterpart (positive = over-forecast).
MASE scales each series' error by how well a one-year seasonal naive forecast does in-sample,
so series of very different size are comparable; values below 1 beat that naive benchmark.
"""

import numpy as np
import polars as pl

from demandpulse.forecasting.panel import SERIES_KEY, YEAR_WEEKS, history

ACTUAL = "y"
PREDICTION = "y_hat"
GROUP_COLUMNS = ["model", "origin"]


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.abs(actual).sum())
    return float(np.abs(actual - predicted).sum() / total) if total else float("nan")


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.abs(actual).sum())
    return float((predicted - actual).sum() / total) if total else float("nan")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def seasonal_scale(values: np.ndarray, season: int = YEAR_WEEKS) -> float:
    """Mean absolute in-sample error of the seasonal naive forecast, the MASE denominator."""
    if len(values) > season:
        return float(np.mean(np.abs(values[season:] - values[:-season])))
    return float("nan")


def mase_scales(panel: pl.DataFrame, origin: int, season: int = YEAR_WEEKS) -> pl.DataFrame:
    """Per-series MASE denominator computed from training data only (weeks up to the origin)."""
    rows = []
    for key, group in (
        history(panel, origin).sort("week_idx").group_by(SERIES_KEY, maintain_order=True)
    ):
        rows.append((*key, origin, seasonal_scale(group["revenue"].to_numpy(), season)))
    return pl.DataFrame(
        rows,
        schema={
            "store_code": pl.String,
            "prod_code_30": pl.String,
            "origin": pl.Int64,
            "scale": pl.Float64,
        },
        orient="row",
    )


def aggregate_level(forecasts: pl.DataFrame, level: list[str]) -> pl.DataFrame:
    """Sum actuals and forecasts to a coarser level (e.g. store) before measuring error.

    Errors partly cancel when series are summed, so accuracy at store or network level is
    higher than at series level. Reporting both shows which decisions a model can support.
    """
    keys = [*GROUP_COLUMNS, *level, "horizon", "week_idx"]
    return forecasts.group_by(keys).agg(pl.col(ACTUAL).sum(), pl.col(PREDICTION).sum())


def summarise(forecasts: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """WAPE, bias, RMSE and row count per group."""
    error = pl.col(ACTUAL) - pl.col(PREDICTION)
    return (
        forecasts.group_by(by)
        .agg(
            pl.len().alias("n"),
            (error.abs().sum() / pl.col(ACTUAL).abs().sum()).alias("wape"),
            ((pl.col(PREDICTION) - pl.col(ACTUAL)).sum() / pl.col(ACTUAL).abs().sum()).alias(
                "bias"
            ),
            (error**2).mean().sqrt().alias("rmse"),
            pl.col(ACTUAL).sum().alias("actual_total"),
        )
        .sort(by)
    )


def series_mase(forecasts: pl.DataFrame, scales: pl.DataFrame) -> pl.DataFrame:
    """Per-series MASE for every model and origin, with series that have no scale dropped."""
    per_series = forecasts.group_by([*GROUP_COLUMNS, *SERIES_KEY]).agg(
        (pl.col(ACTUAL) - pl.col(PREDICTION)).abs().mean().alias("mae")
    )
    joined = per_series.join(scales, on=[*SERIES_KEY, "origin"], how="left")
    return joined.filter(pl.col("scale") > 0).with_columns(
        (pl.col("mae") / pl.col("scale")).alias("mase")
    )


def mase_summary(forecasts: pl.DataFrame, scales: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (
        series_mase(forecasts, scales)
        .group_by(by)
        .agg(pl.col("mase").mean().alias("mase_mean"), pl.col("mase").median().alias("mase_median"))
        .sort(by)
    )
