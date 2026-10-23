import polars as pl
import pytest

from demandpulse.risk.ranking import (
    OWN_TREND,
    RANDOM,
    add_scores,
    own_trend_windows,
    ranking_metrics,
    summarise_ranking,
    with_random_baseline,
)

LEVEL = ["store_code"]


def windows(shortfalls: dict[str, tuple[float, float]], model="m", origin=91) -> pl.DataFrame:
    """entity -> (forecast shortfall, realised shortfall) against a target of 100."""
    rows = [
        (model, origin, "flat", e, 100.0, 100.0 - f, 100.0 - r, 100.0, f, r, f > 0, r > 0)
        for e, (f, r) in shortfalls.items()
    ]
    return pl.DataFrame(
        rows,
        schema={
            "model": pl.String,
            "origin": pl.Int64,
            "target_def": pl.String,
            "store_code": pl.String,
            "target": pl.Float64,
            "forecast": pl.Float64,
            "actual": pl.Float64,
            "last_year": pl.Float64,
            "forecast_shortfall": pl.Float64,
            "realised_shortfall": pl.Float64,
            "forecast_miss": pl.Boolean,
            "realised_miss": pl.Boolean,
        },
        orient="row",
    )


def ten_entities(order_matches: bool) -> pl.DataFrame:
    """Entities E0..E9; E0-E1 truly fell short. The score ranks them first or last."""
    data = {}
    for i in range(10):
        truly_short = i < 2
        score = (10 - i) if order_matches else i
        data[f"E{i}"] = (float(score), 5.0 if truly_short else -5.0)
    return windows(data)


def metric(frame, score="forecast_gap", fraction=0.2, **kw):
    out = ranking_metrics(add_scores(frame), LEVEL, scores=(score,), k_fractions=(fraction,), **kw)
    return out.row(0, named=True)


def test_a_perfect_ranking_has_full_precision_recall_and_capture():
    row = metric(ten_entities(True))
    assert row["k"] == 2
    assert row["n"] == 10
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["capture"] == 1.0
    assert row["base_rate"] == pytest.approx(0.2)
    assert row["lift"] == pytest.approx(5.0)


def test_a_reversed_ranking_catches_nothing():
    row = metric(ten_entities(False))
    assert row["precision"] == 0.0
    assert row["recall"] == 0.0
    assert row["capture"] == 0.0


def test_capture_weights_by_revenue_not_by_count():
    data = {f"E{i}": (float(10 - i), -5.0) for i in range(10)}
    data["E0"] = (10.0, 1.0)  # top-ranked, small real miss
    data["E9"] = (0.0, 99.0)  # ranked last, huge real miss
    row = metric(windows(data), fraction=0.1)
    assert row["precision"] == 1.0
    assert row["capture"] == pytest.approx(1 / 100)


def test_tied_scores_return_no_metrics_instead_of_an_arbitrary_order():
    data = {f"E{i}": (0.0, 5.0 if i < 2 else -5.0) for i in range(10)}
    row = metric(windows(data))
    assert row["tied"] is True
    assert row["precision"] is None
    assert row["lift"] is None


def test_k_is_a_fraction_of_the_entities_rounded_up():
    assert metric(ten_entities(True), fraction=0.05)["k"] == 1
    assert metric(ten_entities(True), fraction=0.11)["k"] == 2


def test_ties_between_neighbours_break_by_entity_code_for_reproducibility():
    data = {f"E{i}": (1.0, 5.0 if i == 0 else -5.0) for i in range(10)}
    data["E5"] = (2.0, -5.0)
    first = metric(windows(data), fraction=0.1)
    again = metric(windows(dict(reversed(list(data.items())))), fraction=0.1)
    assert first == again


def test_scores_that_are_missing_for_a_group_are_skipped():
    frame = ten_entities(True).with_columns(pl.lit(None, dtype=pl.Float64).alias("p_shortfall"))
    out = ranking_metrics(add_scores(frame), LEVEL, scores=("forecast_gap", "p_shortfall"))
    assert set(out["score"]) == {"forecast_gap"}


def test_relative_gap_divides_the_shortfall_by_the_target():
    scored = add_scores(windows({"A": (10.0, 0.0)}))
    assert scored["relative_gap"][0] == pytest.approx(0.1)


def test_random_baseline_has_exact_expected_values():
    metrics = ranking_metrics(add_scores(ten_entities(True)), LEVEL, scores=("forecast_gap",))
    random = with_random_baseline(metrics).filter(pl.col("model") == RANDOM)
    row = random.filter(pl.col("k_fraction") == 0.2).row(0, named=True)
    assert row["precision"] == pytest.approx(0.2)
    assert row["recall"] == pytest.approx(0.2)
    assert row["lift"] == 1.0
    assert row["capture"] == pytest.approx(0.2)


def test_summary_averages_over_origins():
    both = pl.concat(
        [ten_entities(True), ten_entities(False).with_columns(origin=pl.lit(98, dtype=pl.Int64))]
    )
    metrics = ranking_metrics(add_scores(both), LEVEL, scores=("forecast_gap",), k_fractions=(0.2,))
    summary = summarise_ranking(metrics)
    assert summary["precision"][0] == pytest.approx(0.5)
    assert summary["origins"][0] == 2
    assert summarise_ranking(metrics, origins=[91])["precision"][0] == 1.0


def panel_for(growth_by_store: dict[str, float]) -> pl.DataFrame:
    rows = [
        (s, "C1", w, 100.0 * (1 + g) if w >= 78 else 100.0)
        for s, g in growth_by_store.items()
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


def test_own_trend_forecasts_last_year_grown_by_the_entitys_own_recent_growth():
    frame = windows({"S1": (0.0, 0.0), "S2": (0.0, 0.0)}, origin=90)
    out = own_trend_windows(frame, panel_for({"S1": 0.2, "S2": -0.1}), LEVEL).sort("store_code")
    assert set(out["model"]) == {OWN_TREND}
    assert out["forecast"].to_list() == pytest.approx([120.0, 90.0])
    assert out["forecast_shortfall"].to_list() == pytest.approx([-20.0, 10.0])
    assert out["forecast_miss"].to_list() == [False, True]
    assert out["actual"].to_list() == frame.sort("store_code")["actual"].to_list()


def test_own_trend_assumes_last_year_when_growth_is_unknown():
    frame = windows({"S9": (0.0, 0.0)}, origin=90)
    out = own_trend_windows(frame, panel_for({"S1": 0.2}), LEVEL)
    assert out["forecast"][0] == 100.0
