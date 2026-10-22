"""Global LightGBM forecaster: one model for every store x category series.

Direct multi-horizon: a row is (series, origin t, horizon h) with features from weeks <= t and
target revenue at week t+h, scaled by the series' trailing-year mean. The L1 objective with sample
weights equal to that scale makes the training loss the revenue-weighted absolute error, which is
the numerator of WAPE. At each backtest origin the model is fit only on rows whose targets are
observed by that origin, after an inner validation on the final 13 weeks picks the tree count.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from demandpulse.forecasting.backtest import FORECAST_COLUMNS, ForecastError
from demandpulse.forecasting.features import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FEATURE_SETS,
    MIN_ORIGIN,
    Encoders,
    SeriesMatrix,
    build_matrix,
    make_features,
)
from demandpulse.forecasting.panel import SERIES_KEY, history

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "verbosity": -1,
    "force_row_wise": True,
    "deterministic": True,
}
FALLBACK_ROUNDS = 200  # when there is too little history for an inner validation split
REFIT_ROUND_FACTOR = 1.1


class LightGBMGlobal:
    def __init__(
        self,
        panel: pl.DataFrame,
        train_series: Callable[[int], pl.DataFrame],
        *,
        params: dict[str, Any] | None = None,
        origin_stride: int = 2,
        max_rows: int = 8_000_000,
        max_rounds: int = 600,
        early_stopping_rounds: int = 30,
        num_threads: int = 8,
        seed: int = 20061,
        feature_set: str = "v2",
    ) -> None:
        # The panel includes weeks after every origin, so nothing here may read it directly:
        # every access goes through history(panel, origin).
        if feature_set not in FEATURE_SETS:
            raise KeyError(f"Unknown feature set {feature_set!r}; use one of {list(FEATURE_SETS)}")
        self.name = "lightgbm" if feature_set == "v2" else f"lightgbm_{feature_set}"
        self.feature_names = FEATURE_SETS[feature_set]
        self._columns = [FEATURE_NAMES.index(name) for name in self.feature_names]
        self._categorical = [n for n in CATEGORICAL_FEATURES if n in self.feature_names]
        self._panel = panel
        self._train_series = train_series
        self.params = {
            **DEFAULT_PARAMS,
            **(params or {}),
            "num_threads": num_threads,
            "seed": seed,
        }
        self.origin_stride = origin_stride
        self.max_rows = max_rows
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.encoders = Encoders.from_panel(panel)
        self._calendar = dict(
            panel.select("week_idx", "fiscal_week").unique().sort("week_idx").iter_rows()
        )
        self._models: dict[int, lgb.Booster] = {}
        self.fit_info: dict[int, dict[str, float]] = {}

    def _matrix(self, origin: int) -> SeriesMatrix:
        series = self._train_series(origin).select(SERIES_KEY)
        visible = history(self._panel, origin).join(series, on=SERIES_KEY)
        return build_matrix(visible, self.encoders, origin, self._calendar)

    def training_frame(
        self, matrix: SeriesMatrix, last_target: int, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Training rows whose targets are all observed by week `last_target`.

        Returns X, y (scaled), weights (scale), and the origin t and horizon h of each row.
        """
        xs, ys, ws, ts, hs = [], [], [], [], []
        for t in range(last_target - 1, MIN_ORIGIN - 1, -self.origin_stride):
            for h in range(1, horizon + 1):
                if t + h > last_target:
                    continue
                x, valid, scale = make_features(matrix, t, h)
                target = matrix.values[:, t + h - 1]
                keep = valid & ~np.isnan(target)
                if not keep.any():
                    continue
                xs.append(x[keep][:, self._columns])
                ys.append((target[keep] / scale[keep]).astype(np.float32))
                ws.append(scale[keep].astype(np.float32))
                ts.append(np.full(int(keep.sum()), t, dtype=np.int16))
                hs.append(np.full(int(keep.sum()), h, dtype=np.int16))
        if not xs:
            empty = np.empty((0, len(self._columns)), dtype=np.float32)
            return empty, np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0), np.empty(0)
        X, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        t_arr, h_arr = np.concatenate(ts), np.concatenate(hs)
        if len(y) > self.max_rows:
            rng = np.random.default_rng(self.seed)
            pick = np.sort(rng.choice(len(y), size=self.max_rows, replace=False))
            X, y, w, t_arr, h_arr = X[pick], y[pick], w[pick], t_arr[pick], h_arr[pick]
        return X, y, w, t_arr, h_arr

    def _dataset(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> lgb.Dataset:
        return lgb.Dataset(
            X,
            label=y,
            weight=w,
            feature_name=self.feature_names,
            categorical_feature=self._categorical,
            free_raw_data=False,
        )

    def _fit(self, origin: int, horizon: int) -> lgb.Booster:
        if origin in self._models:
            return self._models[origin]
        started = time.monotonic()
        matrix = self._matrix(origin)

        rounds = FALLBACK_ROUNDS
        inner_wape = float("nan")
        inner_cut = origin - horizon
        X_in, y_in, w_in, _, _ = self.training_frame(matrix, inner_cut, horizon)
        if len(y_in):
            # Validate on the last `horizon` weeks before the origin, which the inner fit never saw.
            X_val, y_val, w_val = self._validation_rows(matrix, inner_cut, horizon)
            if len(y_val):
                booster = lgb.train(
                    {**self.params, "metric": "l1"},
                    self._dataset(X_in, y_in, w_in),
                    num_boost_round=self.max_rounds,
                    valid_sets=[self._dataset(X_val, y_val, w_val)],
                    callbacks=[
                        lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    ],
                )
                rounds = max(50, int(booster.best_iteration * REFIT_ROUND_FACTOR))
                pred = booster.predict(X_val, num_iteration=booster.best_iteration)
                inner_wape = float(
                    np.sum(w_val * np.abs(y_val - pred)) / np.sum(w_val * np.abs(y_val))
                )
            del X_in, y_in, w_in

        X, y, w, _, _ = self.training_frame(matrix, origin, horizon)
        if not len(y):
            raise ForecastError(f"{self.name}: no training rows available at origin {origin}")
        model = lgb.train(self.params, self._dataset(X, y, w), num_boost_round=rounds)
        self._models[origin] = model
        self.fit_info[origin] = {
            "training_rows": float(len(y)),
            "rounds": float(rounds),
            "inner_validation_wape": inner_wape,
            "fit_seconds": round(time.monotonic() - started, 1),
        }
        logger.info("lightgbm origin %d fit: %s", origin, self.fit_info[origin])
        return model

    def _validation_rows(
        self, matrix: SeriesMatrix, t: int, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows for the single origin t, all horizons, with targets in (t, t + horizon]."""
        xs, ys, ws = [], [], []
        for h in range(1, horizon + 1):
            x, valid, scale = make_features(matrix, t, h)
            target = matrix.values[:, t + h - 1]
            keep = valid & ~np.isnan(target)
            xs.append(x[keep][:, self._columns])
            ys.append((target[keep] / scale[keep]).astype(np.float32))
            ws.append(scale[keep].astype(np.float32))
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)

    def forecast(
        self, train: pl.DataFrame, future: pl.DataFrame, origin: int, horizon: int
    ) -> pl.DataFrame:
        model = self._fit(origin, horizon)
        matrix = self._matrix(origin)
        wanted = {(s, c) for s, c in train.select(SERIES_KEY).unique().iter_rows()}
        position = [i for i, key in enumerate(matrix.keys) if key in wanted]
        if len(position) != len(wanted):
            raise ForecastError(f"{self.name}: some requested series are not in the training panel")

        rows: list[tuple[str, str, int, int, float]] = []
        for h in range(1, horizon + 1):
            x, valid, scale = make_features(matrix, origin, h)
            if not valid[position].all():
                raise ForecastError(f"{self.name}: series with incomplete history at {origin}")
            prediction = model.predict(x[position][:, self._columns]) * scale[position]
            for index, value in zip(position, prediction, strict=True):
                store, category = matrix.keys[index]
                rows.append((store, category, h, origin + h, float(value)))
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
