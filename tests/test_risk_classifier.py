import numpy as np
import polars as pl
import pytest

from demandpulse.risk.classifier import FEATURES, evaluate_classifier, feature_table

LEVEL = ["store_code"]


def synthetic(origins=(78, 91, 104), n=400, seed=3, signal=True) -> pl.DataFrame:
    """Entities whose shortfall is driven by relative_gap (plus noise) when `signal` is set."""
    rng = np.random.default_rng(seed)
    frames = []
    for origin in origins:
        gap = rng.normal(0, 0.05, n)
        noise = rng.normal(0, 0.03, n)
        miss = (gap + noise > 0) if signal else rng.random(n) < 0.5
        frames.append(
            pl.DataFrame(
                {
                    "origin": origin,
                    "store_code": [f"S{i}" for i in range(n)],
                    "relative_gap": gap,
                    "log_target": rng.normal(10, 1, n),
                    "growth_13": rng.normal(0, 0.05, n),
                    "growth_26": rng.normal(0, 0.05, n),
                    "volatility": rng.uniform(0.05, 0.3, n),
                    "realised_miss": miss,
                    "realised_shortfall": np.where(miss, 5.0, -5.0),
                    "p_shortfall": rng.uniform(0, 1, n),
                }
            ).with_columns(pl.col("origin").cast(pl.Int64))
        )
    return pl.concat(frames)


def test_calibrated_predictions_beat_a_constant_when_the_features_carry_signal():
    result = evaluate_classifier(synthetic(), [78], 91, 104)
    assert result.metrics["brier_calibrated"] < result.metrics["brier_constant"]
    probabilities = result.predictions["p_classifier"]
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
    assert set(result.predictions["origin"]) == {104}


def test_features_without_signal_do_not_beat_a_constant_by_much():
    result = evaluate_classifier(synthetic(signal=False), [78], 91, 104)
    assert result.metrics["brier_calibrated"] >= result.metrics["brier_constant"] - 0.02


def test_the_result_is_reproducible():
    one = evaluate_classifier(synthetic(), [78], 91, 104)
    two = evaluate_classifier(synthetic(), [78], 91, 104)
    assert one.metrics == two.metrics
    assert one.predictions["p_classifier"].to_list() == two.predictions["p_classifier"].to_list()


def test_reports_importance_reliability_and_the_forecast_probability_comparison():
    result = evaluate_classifier(synthetic(), [78], 91, 104)
    assert set(result.importance["feature"]) == set(FEATURES)
    assert result.importance["feature"][0] == "relative_gap"
    assert result.reliability["n"].sum() == 400
    assert "brier_forecast_probability" in result.metrics
    assert result.metrics["n_comparable"] == 400


def test_a_train_window_that_has_not_finished_by_calibration_is_rejected():
    with pytest.raises(ValueError, match="train window"):
        evaluate_classifier(synthetic((85, 91, 104)), [85], 91, 104)


def test_a_calibration_window_that_has_not_finished_by_test_is_rejected():
    with pytest.raises(ValueError, match="calibration window"):
        evaluate_classifier(synthetic((78, 98, 104)), [78], 98, 104)


def test_missing_rows_for_a_role_are_rejected():
    with pytest.raises(ValueError, match="No test rows"):
        evaluate_classifier(synthetic((78, 91)), [78], 91, 104)


def test_a_single_outcome_in_training_is_rejected():
    frame = synthetic().with_columns(
        pl.when(pl.col("origin") == 78)
        .then(False)
        .otherwise(pl.col("realised_miss"))
        .alias("realised_miss")
    )
    with pytest.raises(ValueError, match="both outcomes"):
        evaluate_classifier(frame, [78], 91, 104)


def window_rows(origin: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model": "m",
            "origin": [origin],
            "target_def": "flat",
            "store_code": ["S1"],
            "target": [1300.0],
            "forecast": [1200.0],
            "actual": [1250.0],
            "last_year": [1300.0],
            "forecast_shortfall": [100.0],
            "realised_shortfall": [50.0],
            "forecast_miss": [True],
            "realised_miss": [True],
        }
    ).with_columns(pl.col("origin").cast(pl.Int64))


def history(late_boost: float = 0.0) -> pl.DataFrame:
    rows = [
        ("S1", "C1", w, (100.0 if w < 78 else 110.0) + (late_boost if w > 90 else 0.0))
        for w in range(1, 121)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "store_code": pl.String,
            "prod_code_30": pl.String,
            "week_idx": pl.Int64,
            "revenue": pl.Float64,
        },
        orient="row",
    )


def test_feature_table_builds_every_feature_from_history_up_to_the_origin():
    table = feature_table(window_rows(90), history(), LEVEL)
    row = table.row(0, named=True)
    assert row["relative_gap"] == pytest.approx(100 / 1300)
    assert row["growth_13"] == pytest.approx(0.1)
    assert row["log_target"] == pytest.approx(np.log(1301.0))
    assert row["volatility"] is not None
    assert all(f in table.columns for f in FEATURES)


def test_feature_table_ignores_revenue_after_the_origin():
    base = feature_table(window_rows(90), history(), LEVEL).select(FEATURES)
    shocked = feature_table(window_rows(90), history(late_boost=500.0), LEVEL).select(FEATURES)
    # Aggregation order can differ in the last bit; a leak would move these by orders of magnitude.
    assert base.row(0) == pytest.approx(shocked.row(0), rel=1e-9)
