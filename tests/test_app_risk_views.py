import polars as pl
import pytest

from demandpulse.app import risk_charts, risk_views
from demandpulse.app.snapshot import Snapshot

TARGET = "flat_yoy"


@pytest.fixture(scope="module")
def snap(app_snapshot_dir):
    return Snapshot(app_snapshot_dir)


@pytest.fixture(scope="module")
def forecast_tables(snap):
    return {
        name: snap.table(name) for name in snap.manifest["tables"] if name.startswith("forecast_")
    }


def stores_at_risk(snap, origin=104, **kwargs):
    return risk_views.risk_table(
        snap.table("risk_windows_store"),
        snap.table("dim_store"),
        "ensemble",
        origin,
        TARGET,
        **kwargs,
    )


def test_leaderboard_is_one_run_best_wape_first(forecast_tables):
    board = risk_views.model_leaderboard(forecast_tables["forecast_full_pooled"], "main")
    assert board["model"].to_list() == ["ensemble", "lightgbm"]
    assert board["wape"].is_sorted()


def test_accuracy_by_level_has_every_level_for_every_model(forecast_tables):
    table = risk_views.accuracy_by_level(forecast_tables, "main")
    assert set(table["level"]) == {"Series (store x category)", "Store", "Category", "Network"}
    assert table.group_by("level").len()["len"].to_list() == [2, 2, 2, 2]


def test_network_forecast_has_actual_and_one_line_per_model(snap):
    weeks = snap.table("forecast_network_weeks")
    lines = risk_views.network_forecast(weeks, "main", 104, ["ensemble", "lightgbm"])
    assert set(lines["series"]) == {
        "Actual",
        risk_views.MODEL_LABELS["ensemble"],
        risk_views.MODEL_LABELS["lightgbm"],
    }
    assert lines.filter(pl.col("series") == "Actual").height == 13


def test_network_forecast_for_an_origin_not_in_the_run_is_an_error(snap):
    with pytest.raises(ValueError, match="No forecasts"):
        risk_views.network_forecast(snap.table("forecast_network_weeks"), "main", 85, ["ensemble"])


def test_risk_origins_lists_what_the_runs_covered(snap):
    origins = risk_views.risk_origins(snap.table("risk_windows_store"), "ensemble")
    assert origins == [78, 85, 91, 98, 104]


def test_risk_table_is_riskiest_first_and_filters_stores(snap):
    table = stores_at_risk(snap)
    assert table.height == 45
    assert table["revenue_at_risk"].drop_nulls().is_sorted(descending=True)
    east = stores_at_risk(snap, regions=["E01"])
    assert set(east["store_region"]) == {"E01"}
    assert 0 < east.height < 45
    assert stores_at_risk(snap, formats=["XX"]).height == 0


def test_origins_without_estimates_fall_back_to_the_forecast_gap(snap):
    table = stores_at_risk(snap, origin=78)
    assert table["revenue_at_risk"].null_count() == table.height
    assert table["forecast_shortfall"].is_sorted(descending=True)


def test_headline_counts_forecast_misses_risk_and_hindsight(snap):
    table = stores_at_risk(snap)
    headline = risk_views.risk_headline(table)
    assert headline["stores"] == 45
    assert headline["forecast_below_target"] == int((table["forecast_shortfall"] > 0).sum())
    assert headline["revenue_at_risk"] == pytest.approx(table["revenue_at_risk"].sum())
    assert 0 <= headline["missed_in_hindsight"] <= 1
    empty = risk_views.risk_headline(table.head(0))
    assert empty["stores"] == 0
    assert empty["revenue_at_risk"] is None


def test_ranking_view_hides_scores_that_cannot_rank_and_sorts_by_precision(snap):
    view = risk_views.ranking_view(snap.table("risk_ranking_summary_store"), TARGET, 0.1)
    assert view["precision"].is_sorted(descending=True)
    assert view["precision"].null_count() == 0
    assert {"ensemble", "own_trend", "random"} <= set(view["model"])


def test_classifier_verdict_is_computed_from_the_numbers(snap):
    metrics = risk_views.classifier_metrics(snap.json("risk_classifier_store"), TARGET)
    verdict = risk_views.classifier_verdict(metrics)
    beat = metrics["brier_calibrated_comparable"] < metrics["brier_forecast_probability"]
    assert ("did not beat" not in verdict) == beat


def test_verdict_wording_for_both_outcomes():
    worse = {"brier_calibrated_comparable": 0.30, "brier_forecast_probability": 0.25}
    better = {"brier_calibrated_comparable": 0.20, "brier_forecast_probability": 0.25}
    assert "did not beat" in risk_views.classifier_verdict(worse)
    assert "beat the forecast probability 0.250" in risk_views.classifier_verdict(better)


def test_a_skipped_classifier_is_reported_not_hidden():
    with pytest.raises(ValueError, match="skipped"):
        risk_views.classifier_metrics({"skipped": "not enough origins"}, TARGET)


def test_custom_scenario_totals_gap_before_and_after():
    stores = pl.DataFrame(
        {"store_code": ["A", "B"], "forecast": [100.0, 100.0], "target": [110.0, 90.0]}
    )
    rows, totals = risk_views.custom_scenario(stores, 0.05, 0.0, 0.0)
    assert totals["gap_before"] == pytest.approx(10.0)
    assert totals["gap_after"] == pytest.approx(5.0)
    assert totals["gap_closed_share"] == pytest.approx(0.5)
    assert totals["stores_meeting_target"] == 1
    assert rows.height == 2


def test_custom_scenario_with_no_gap_does_not_divide_by_zero():
    stores = pl.DataFrame({"store_code": ["A"], "forecast": [100.0], "target": [90.0]})
    _, totals = risk_views.custom_scenario(stores, 0.0, 0.0, 0.0)
    assert totals["gap_closed_share"] == 0.0


def test_risk_and_forecast_charts_build(snap, forecast_tables):
    table = stores_at_risk(snap)
    reliability = pl.DataFrame(snap.json("risk_classifier_store")[TARGET]["reliability"])
    rows, _ = risk_views.custom_scenario(
        table.select("store_code", "forecast", "target"), 0.02, 0, 0
    )
    lines = risk_views.network_forecast(
        snap.table("forecast_network_weeks"), "main", 104, ["ensemble"]
    )
    figures = [
        risk_charts.forecast_chart(lines),
        risk_charts.accuracy_chart(risk_views.accuracy_by_level(forecast_tables, "main")),
        risk_charts.at_risk_chart(table),
        risk_charts.at_risk_chart(stores_at_risk(snap, origin=78)),
        risk_charts.ranking_chart(
            risk_views.ranking_view(snap.table("risk_ranking_summary_store"), TARGET, 0.1)
        ),
        risk_charts.reliability_chart(reliability),
        risk_charts.scenario_chart(rows),
    ]
    assert all(len(fig.data) >= 1 for fig in figures)
