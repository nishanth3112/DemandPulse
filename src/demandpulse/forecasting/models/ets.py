"""Exponential smoothing per series: damped-trend ETS and additive Holt-Winters.

Both fit each series independently with statsmodels, in parallel across processes. A series whose
fit raises is forecast with its trailing 13-week mean instead; this is logged and counted in
`fallbacks` so it is never silent, and reported alongside the accuracy results.
"""

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import polars as pl
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from demandpulse.forecasting.backtest import FORECAST_COLUMNS, ForecastError
from demandpulse.forecasting.panel import SERIES_KEY, YEAR_WEEKS

logger = logging.getLogger(__name__)

FALLBACK_WINDOW = 13
CHUNK_SERIES = 25


@dataclass(frozen=True)
class EtsSpec:
    seasonal: bool
    season: int
    damped_trend: bool = True


def _forecast_one(values: np.ndarray, horizon: int, spec: EtsSpec) -> tuple[np.ndarray, bool]:
    """Forecast one series; return (forecast, used_fallback)."""
    try:
        model = ExponentialSmoothing(
            values,
            trend="add",
            damped_trend=spec.damped_trend,
            seasonal="add" if spec.seasonal else None,
            seasonal_periods=spec.season if spec.seasonal else None,
            initialization_method="estimated",
        )
        with warnings.catch_warnings():
            # Convergence warnings are routine for noisy retail series; hard failures are not.
            warnings.simplefilter("ignore")
            fit = model.fit(optimized=True)
        prediction = np.asarray(fit.forecast(horizon), dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("non-finite forecast")
        return prediction, False
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as err:
        logger.warning("ETS fit failed (%s); using trailing mean", err)
        return np.full(horizon, float(np.mean(values[-FALLBACK_WINDOW:]))), True


def _forecast_chunk(args: tuple[list[np.ndarray], int, EtsSpec]) -> list[tuple[np.ndarray, bool]]:
    chunk, horizon, spec = args
    return [_forecast_one(values, horizon, spec) for values in chunk]


class ExponentialSmoothingForecaster:
    def __init__(
        self, name: str, spec: EtsSpec, workers: int = 1, chunk_series: int = CHUNK_SERIES
    ) -> None:
        self.name = name
        self.spec = spec
        self.workers = workers
        self.chunk_series = chunk_series
        self.fallbacks = 0
        self.fits = 0

    def forecast(
        self, train: pl.DataFrame, future: pl.DataFrame, origin: int, horizon: int
    ) -> pl.DataFrame:
        ordered = train.sort([*SERIES_KEY, "week_idx"])
        grouped = ordered.group_by(SERIES_KEY, maintain_order=True).agg(pl.col("revenue"))
        keys = grouped.select(SERIES_KEY).rows()
        arrays = [np.asarray(v, dtype=float) for v in grouped["revenue"].to_list()]
        needed = 2 * self.spec.season if self.spec.seasonal else 3
        short = [k for k, a in zip(keys, arrays, strict=True) if len(a) < needed]
        if short:
            raise ForecastError(
                f"{self.name}: {len(short)} series have fewer than {needed} weeks of history "
                f"(seasonal fits need two full seasons)"
            )

        chunks = [
            (arrays[i : i + self.chunk_series], horizon, self.spec)
            for i in range(0, len(arrays), self.chunk_series)
        ]
        if self.workers > 1 and len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                results = [r for part in pool.map(_forecast_chunk, chunks) for r in part]
        else:
            results = [r for chunk in chunks for r in _forecast_chunk(chunk)]

        self.fits += len(results)
        self.fallbacks += sum(used for _, used in results)
        rows = [
            (*key, h + 1, origin + h + 1, float(values[h]))
            for key, (values, _) in zip(keys, results, strict=True)
            for h in range(horizon)
        ]
        return pl.DataFrame(
            rows,
            schema={
                "store_code": pl.String,
                "prod_code_30": pl.String,
                "horizon": pl.Int64,
                "week_idx": pl.Int64,
                "y_hat": pl.Float64,
            },
            orient="row",
        ).select(FORECAST_COLUMNS)


def ets_damped(
    workers: int = 1, chunk_series: int = CHUNK_SERIES
) -> ExponentialSmoothingForecaster:
    """ETS(A,Ad,N): additive damped trend, no seasonality. Needs little history."""
    spec = EtsSpec(False, YEAR_WEEKS)
    return ExponentialSmoothingForecaster("ets_damped", spec, workers, chunk_series)


def holt_winters(
    season: int = YEAR_WEEKS, workers: int = 1, chunk_series: int = CHUNK_SERIES
) -> ExponentialSmoothingForecaster:
    """Additive damped-trend Holt-Winters with a one-year seasonal cycle (needs 2 full years)."""
    return ExponentialSmoothingForecaster(
        "holt_winters", EtsSpec(True, season), workers, chunk_series
    )
