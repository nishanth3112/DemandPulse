"""Scoring of probabilistic predictions: Brier score and reliability (calibration) table."""

import numpy as np
import polars as pl


def brier_score(prob: np.ndarray, label: np.ndarray) -> float:
    """Mean squared error of the predicted probability against the 0/1 outcome (lower is better)."""
    return float(np.mean((np.asarray(prob, dtype=float) - np.asarray(label, dtype=float)) ** 2))


def reliability_table(prob: np.ndarray, label: np.ndarray, bins: int = 10) -> pl.DataFrame:
    """Predicted against observed shortfall rate in equal-width probability bins.

    Only non-empty bins are returned. A calibrated predictor has `mean_predicted` close to
    `observed_rate` in every bin; small bins are noisy, so read `n` alongside the rates.
    """
    prob = np.asarray(prob, dtype=float)
    label = np.asarray(label, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Probability exactly 1.0 belongs to the last bin, not a bin of its own.
    index = np.minimum(np.digitize(prob, edges[1:-1], right=False), bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            rows.append(
                (
                    float(edges[b]),
                    float(edges[b + 1]),
                    int(mask.sum()),
                    float(prob[mask].mean()),
                    float(label[mask].mean()),
                )
            )
    return pl.DataFrame(
        rows,
        schema={
            "bin_low": pl.Float64,
            "bin_high": pl.Float64,
            "n": pl.Int64,
            "mean_predicted": pl.Float64,
            "observed_rate": pl.Float64,
        },
        orient="row",
    )
