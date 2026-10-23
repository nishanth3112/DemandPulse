import polars as pl
import pytest

from demandpulse.risk.recommend import evidence_text, recommendations


def risk(n=5) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "store_code": [f"S{i}" for i in range(n)],
            "target": [1000.0] * n,
            "forecast": [900.0 + 10 * i for i in range(n)],
            "forecast_shortfall": [100.0 - 10 * i for i in range(n)],
            "p_shortfall": [0.9 - 0.1 * i for i in range(n)],
            "revenue_at_risk": [50.0 - 10 * i for i in range(n)],
        }
    )


def drivers() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "store_code": ["S0", "S1"],
            "revenue_growth": [-0.08, 0.02],
            "traffic_change": [-0.10, 0.03],
            "basket_size_change": [0.01, -0.02],
            "price_mix_change": [0.01, 0.01],
            "dominant_driver": ["traffic", "basket_size"],
        }
    )


def test_stores_are_ranked_by_revenue_at_risk_and_limited_to_top_k():
    out = recommendations(risk(), drivers(), top_k=3)
    assert out["store_code"].to_list() == ["S0", "S1", "S2"]
    assert out["rank"].to_list() == [1, 2, 3]


def test_ties_break_by_store_code():
    frame = risk(3).with_columns(pl.lit(10.0).alias("revenue_at_risk"))
    assert recommendations(frame, drivers())["store_code"].to_list() == ["S0", "S1", "S2"]


def test_evidence_states_only_measured_numbers():
    text = recommendations(risk(), drivers())["evidence"][0]
    assert "S0: forecast is 100 (10.0%) below target" in text
    assert "probability of a shortfall 90%" in text
    assert "last 13 weeks -8.0% on a year earlier" in text
    assert "weakest driver baskets per week (-10.0%)" in text


def test_stores_without_a_driver_split_say_so():
    out = recommendations(risk(), drivers())
    assert "driver split unavailable" in out.filter(pl.col("store_code") == "S4")["evidence"][0]


def test_forecast_above_target_is_described_as_above():
    row = risk().row(0, named=True) | {"forecast_shortfall": -50.0}
    assert "5.0%) above target" in evidence_text(row)


def test_no_driver_below_last_year_is_stated():
    row = risk().row(0, named=True) | {"dominant_driver": "none"}
    assert "no driver is below last year" in evidence_text(row)


def test_missing_probability_estimates_are_an_error_not_an_empty_list():
    frame = risk().with_columns(pl.lit(None, dtype=pl.Float64).alias("revenue_at_risk"))
    with pytest.raises(ValueError, match="No revenue-at-risk"):
        recommendations(frame, drivers())
