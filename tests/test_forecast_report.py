import polars as pl
import pytest

from demandpulse.forecasting.report import (
    aggregated_leaderboard,
    by_attribute,
    format_table,
    horizon_buckets,
    leaderboard,
    paired_vs_baseline,
)

SCHEMA = {
    "model": pl.String,
    "origin": pl.Int64,
    "store_code": pl.String,
    "prod_code_30": pl.String,
    "horizon": pl.Int64,
    "week_idx": pl.Int64,
    "y": pl.Float64,
    "y_hat": pl.Float64,
}


def forecasts(per_model: dict[str, float], n_series: int = 30, horizons: int = 13):
    """Series i has actual 100 + i every week; each model forecasts actual + its own error."""
    rows = []
    for model, error in per_model.items():
        for i in range(n_series):
            actual = 100.0 + i
            for h in range(1, horizons + 1):
                # a little per-series variation so bootstrap resamples differ
                wobble = (i % 5) * 0.2
                rows.append(
                    (model, 10, f"S{i % 3}", f"C{i}", h, 10 + h, actual, actual + error + wobble)
                )
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


def scales(n_series: int = 30):
    return pl.DataFrame(
        {
            "store_code": [f"S{i % 3}" for i in range(n_series)],
            "prod_code_30": [f"C{i}" for i in range(n_series)],
            "origin": [10] * n_series,
            "scale": [5.0] * n_series,
        }
    )


def test_leaderboard_orders_models_by_error_and_counts_series():
    table = leaderboard(forecasts({"good": 1.0, "bad": 8.0}), scales())
    assert table["model"].to_list() == ["good", "bad"]
    assert table["series"].to_list() == [30, 30]
    assert table["wape"][0] < table["wape"][1]
    assert table["mase_median"][0] < table["mase_median"][1]


def test_leaderboard_per_origin_keeps_origins_apart():
    frame = pl.concat(
        [
            forecasts({"m": 1.0}),
            forecasts({"m": 3.0}).with_columns(origin=pl.lit(20, dtype=pl.Int64)),
        ]
    )
    table = leaderboard(
        frame, scales().with_columns(origin=pl.lit(10, dtype=pl.Int64)), ["model", "origin"]
    )
    assert table["origin"].to_list() == [10, 20]
    assert table["wape"][0] < table["wape"][1]


def test_aggregated_leaderboard_lets_offsetting_errors_cancel():
    frame = pl.concat(
        [
            forecasts({"m": 5.0}).filter(pl.col("prod_code_30") == "C0"),
            forecasts({"m": -5.0}).filter(pl.col("prod_code_30") == "C3"),
        ]
    ).with_columns(store_code=pl.lit("S0"))
    series_level = leaderboard(frame, scales())
    store_level = aggregated_leaderboard(frame, ["store_code"], ["model"])
    assert store_level["wape"][0] < series_level["wape"][0]


def test_horizon_buckets_split_into_three_ranges():
    table = horizon_buckets(forecasts({"m": 2.0}))
    assert sorted(table["horizon_bucket"].to_list()) == ["weeks 1-4", "weeks 5-8", "weeks 9-13"]


def test_by_attribute_joins_store_attributes():
    attrs = pl.DataFrame({"store_code": ["S0", "S1", "S2"], "store_format": ["LS", "SS", "LS"]})
    table = by_attribute(forecasts({"m": 2.0}), attrs, "store_format")
    assert set(table["store_format"]) == {"LS", "SS"}


def test_a_uniformly_better_model_is_distinguishable_from_the_baseline():
    result = paired_vs_baseline(forecasts({"base": 10.0, "better": 2.0}), "base")
    row = result.row(0, named=True)
    assert row["model"] == "better"
    assert row["improvement"] > 0.5
    assert row["ci_low"] > 0
    assert row["win_rate"] == 1.0
    assert row["distinguishable"] is True
    assert row["ci_low"] <= row["improvement"] <= row["ci_high"]


def test_an_identical_model_is_not_distinguishable():
    result = paired_vs_baseline(forecasts({"base": 4.0, "twin": 4.0}), "base")
    row = result.row(0, named=True)
    assert row["improvement"] == pytest.approx(0.0)
    assert row["distinguishable"] is False
    assert row["win_rate"] == 0.0


def test_a_worse_model_has_negative_improvement():
    row = paired_vs_baseline(forecasts({"base": 2.0, "worse": 9.0}), "base").row(0, named=True)
    assert row["improvement"] < 0
    assert row["ci_high"] < 0
    assert row["distinguishable"] is True


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    frame = forecasts({"base": 6.0, "other": 3.0})
    assert paired_vs_baseline(frame, "base", seed=1).equals(
        paired_vs_baseline(frame, "base", seed=1)
    )
    assert (
        paired_vs_baseline(frame, "base", seed=1)["ci_low"][0]
        != paired_vs_baseline(frame, "base", seed=2)["ci_low"][0]
    )


def test_markdown_table_formats_percentages_and_counts():
    table = leaderboard(forecasts({"m": 2.0}), scales())
    text = format_table(table.select("model", "series", "n", "wape"))
    assert text.splitlines()[0] == "| model | series | n | wape |"
    assert "30" in text
    assert "%" in text
