import numpy as np
import polars as pl
import pytest

from demandpulse.risk.calibration import brier_score, reliability_table
from demandpulse.risk.uncertainty import shortfall_risk

SCHEMA = {
    "model": pl.String,
    "origin": pl.Int64,
    "store_code": pl.String,
    "target_def": pl.String,
    "actual": pl.Float64,
    "forecast": pl.Float64,
    "target": pl.Float64,
}
# 100 stores whose actual/forecast - 1 is spread evenly from -20% to +20%.
ERRORS = np.linspace(-0.2, 0.2, 100)


def pool_rows(origin: int, model: str = "m") -> list[tuple]:
    return [
        (model, origin, f"S{i}", "flat", 100.0 * (1 + e), 100.0, 100.0)
        for i, e in enumerate(ERRORS)
    ]


def entity(origin: int, forecast=100.0, target=100.0, actual=100.0, model="m") -> tuple:
    return (model, origin, "X", "flat", actual, forecast, target)


def windows(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


def risk_of(frame: pl.DataFrame, origin: int, model: str = "m", **kwargs) -> dict:
    out = shortfall_risk(frame, **kwargs)
    return out.filter(
        (pl.col("origin") == origin) & (pl.col("store_code") == "X") & (pl.col("model") == model)
    ).row(0, named=True)


def test_probability_is_the_share_of_past_errors_below_the_needed_change():
    frame = windows(pool_rows(78) + [entity(91, forecast=100.0, target=100.0)])
    row = risk_of(frame, 91)
    # Forecast equals target, so a shortfall happens when the past error was negative: 50 of 100.
    assert row["p_shortfall"] == pytest.approx(0.5)
    assert row["pool_size"] == 100


def test_revenue_at_risk_is_the_expected_shortfall_in_revenue():
    frame = windows(pool_rows(78) + [entity(91, forecast=200.0, target=200.0)])
    expected = float(np.mean(np.maximum(-ERRORS, 0)) * 200.0)
    assert risk_of(frame, 91)["revenue_at_risk"] == pytest.approx(expected)


def test_a_target_far_below_the_forecast_has_no_risk():
    frame = windows(pool_rows(78) + [entity(91, forecast=100.0, target=50.0)])
    row = risk_of(frame, 91)
    assert row["p_shortfall"] == 0.0
    assert row["revenue_at_risk"] == 0.0


def test_only_windows_finished_by_the_origin_are_used():
    # Origin 85 is only 7 weeks after origin 78, so the 78 window had not ended yet.
    frame = windows(pool_rows(78) + [entity(85), entity(91)])
    assert risk_of(frame, 85)["p_shortfall"] is None
    assert risk_of(frame, 91)["p_shortfall"] is not None


def test_the_first_origin_has_no_history_to_learn_from():
    frame = windows(pool_rows(78) + [entity(91)])
    out = shortfall_risk(frame)
    assert out.filter(pl.col("origin") == 78)["p_shortfall"].null_count() == 100


def test_outcomes_at_the_scored_origin_never_change_its_probability():
    base = windows(pool_rows(78) + pool_rows(91) + [entity(91, actual=90.0)])
    shocked = windows(pool_rows(78) + pool_rows(91) + [entity(91, actual=10_000.0)])
    # Pool rows for origin 91 differ nothing; only the scored entity's own actual changes.
    assert risk_of(base, 91)["p_shortfall"] == risk_of(shocked, 91)["p_shortfall"]


def test_each_model_learns_only_from_its_own_errors():
    biased = [
        ("b", 78, f"S{i}", "flat", 80.0, 100.0, 100.0) for i in range(100)
    ]  # always 20% under
    frame = windows(pool_rows(78) + biased + [entity(91), entity(91, model="b")])
    assert risk_of(frame, 91)["p_shortfall"] == pytest.approx(0.5)
    assert risk_of(frame, 91, model="b")["p_shortfall"] == 1.0


def test_small_pools_give_no_estimate():
    frame = windows(pool_rows(78)[:10] + [entity(91)])
    assert risk_of(frame, 91)["p_shortfall"] is None


def test_non_positive_forecasts_give_no_estimate():
    frame = windows(pool_rows(78) + [entity(91, forecast=0.0)])
    assert risk_of(frame, 91)["p_shortfall"] is None


def test_brier_score_is_mean_squared_probability_error():
    assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)


def test_reliability_table_compares_predicted_and_observed_rates_per_bin():
    prob = np.array([0.05, 0.05, 0.95, 0.95])
    label = np.array([0, 0, 1, 0])
    table = reliability_table(prob, label, bins=2)
    assert table["n"].to_list() == [2, 2]
    assert table["mean_predicted"].to_list() == pytest.approx([0.05, 0.95])
    assert table["observed_rate"].to_list() == pytest.approx([0.0, 0.5])


def test_reliability_table_puts_probability_one_in_the_last_bin():
    table = reliability_table(np.array([1.0]), np.array([1]), bins=4)
    assert table["bin_high"].to_list() == [1.0]
