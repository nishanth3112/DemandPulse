"""Calibrated classifier for "will this entity fall short of target over the next 13 weeks?".

Rows are entity windows. The label is the realised outcome (actual below target); features come
from data known at the origin only: the forecast gap to target, size, own recent growth and
volatility. A window's label is only known 13 weeks after its origin, so the three roles must be
ordered in time or labels leak:

    train origins    t  with  t + 13 <= calibration origin
    calibration      c  with  c + 13 <= test origin     (isotonic regression on its predictions)
    test origin      the only window the result is scored on

With the dataset's five origins this leaves one train origin, one calibration origin and one
test origin, so the result is a single-window check, not evidence that the classifier beats the
forecast-based probability. Hyperparameters are fixed in advance and never tuned on the test
window.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from demandpulse.forecasting.panel import HORIZON
from demandpulse.risk.calibration import brier_score, reliability_table
from demandpulse.risk.drivers import entity_growth, entity_volatility
from demandpulse.risk.ranking import add_scores

FEATURES = ["relative_gap", "log_target", "growth_13", "growth_26", "volatility"]
SEED = 20061
GROWTH_LONG_WEEKS = 26
PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


@dataclass(frozen=True)
class ClassifierResult:
    metrics: dict[str, float]
    reliability: pl.DataFrame
    importance: pl.DataFrame
    predictions: pl.DataFrame  # test-origin rows with `p_raw` and `p_classifier` (calibrated)


def feature_table(windows: pl.DataFrame, panel: pl.DataFrame, level: list[str]) -> pl.DataFrame:
    """Window rows with the classifier features, each computed from history up to its origin.

    `windows` must hold one model and one target definition. Missing growth or volatility stays
    null and is handed to LightGBM, which treats it as missing.
    """
    pieces = []
    for origin in sorted(windows["origin"].unique().to_list()):
        short = entity_growth(panel, level, origin).rename({"growth": "growth_13"})
        long = entity_growth(panel, level, origin, weeks=GROWTH_LONG_WEEKS).rename(
            {"growth": "growth_26"}
        )
        vol = entity_volatility(panel, level, origin)
        part = windows.filter(pl.col("origin") == origin)
        for extra in (short, long, vol):
            part = part.join(extra, on=level, how="left")
        pieces.append(part)
    return add_scores(pl.concat(pieces)).with_columns(
        pl.when(pl.col("target") > 0)
        .then((pl.col("target") + 1).log())
        .otherwise(None)
        .alias("log_target")
    )


def _matrix(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(FEATURES).fill_null(np.nan).cast(pl.Float64).to_numpy()


def _labels(frame: pl.DataFrame) -> np.ndarray:
    return frame["realised_miss"].to_numpy().astype(int)


def evaluate_classifier(
    features: pl.DataFrame,
    train_origins: list[int],
    calibration_origin: int,
    test_origin: int,
    horizon: int = HORIZON,
    seed: int = SEED,
) -> ClassifierResult:
    """Train, calibrate and score on a strictly ordered train / calibration / test split."""
    if not train_origins:
        raise ValueError("Need at least one train origin")
    if any(t + horizon > calibration_origin for t in train_origins):
        raise ValueError("A train window must finish before the calibration origin")
    if calibration_origin + horizon > test_origin:
        raise ValueError("The calibration window must finish before the test origin")

    train = features.filter(pl.col("origin").is_in(train_origins))
    calibration = features.filter(pl.col("origin") == calibration_origin)
    test = features.filter(pl.col("origin") == test_origin)
    for name, part in (("train", train), ("calibration", calibration), ("test", test)):
        if not part.height:
            raise ValueError(f"No {name} rows")
    if len(set(_labels(train))) < 2 or len(set(_labels(calibration))) < 2:
        raise ValueError("Train and calibration labels must contain both outcomes")

    model = lgb.LGBMClassifier(
        **PARAMS, random_state=seed, n_jobs=1, verbose=-1, deterministic=True, force_row_wise=True
    )
    model.fit(_matrix(train), _labels(train))
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(model.predict_proba(_matrix(calibration))[:, 1], _labels(calibration))

    raw = model.predict_proba(_matrix(test))[:, 1]
    calibrated = calibrator.predict(raw)
    labels = _labels(test)
    predictions = test.with_columns(pl.Series("p_raw", raw), pl.Series("p_classifier", calibrated))

    constant = float(_labels(calibration).mean())
    metrics: dict[str, float] = {
        "n_train": float(train.height),
        "n_calibration": float(calibration.height),
        "n_test": float(test.height),
        "base_rate_test": float(labels.mean()),
        "brier_constant": brier_score(np.full(len(labels), constant), labels),
        "brier_raw": brier_score(raw, labels),
        "brier_calibrated": brier_score(calibrated, labels),
    }
    if "p_shortfall" in test.columns:
        comparable = predictions.filter(pl.col("p_shortfall").is_not_null())
        if comparable.height:
            y = _labels(comparable)
            metrics["n_comparable"] = float(comparable.height)
            metrics["brier_forecast_probability"] = brier_score(
                comparable["p_shortfall"].to_numpy(), y
            )
            metrics["brier_calibrated_comparable"] = brier_score(
                comparable["p_classifier"].to_numpy(), y
            )
    importance = pl.DataFrame(
        {
            "feature": FEATURES,
            "gain": model.booster_.feature_importance(importance_type="gain").astype(float),
        }
    ).sort("gain", descending=True)
    return ClassifierResult(
        metrics=metrics,
        reliability=reliability_table(calibrated, labels),
        importance=importance,
        predictions=predictions,
    )
