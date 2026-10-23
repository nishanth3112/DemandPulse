import json

import polars as pl
import pytest

from demandpulse.risk.runner import (
    LEVELS,
    choose_split,
    compute_risk,
    load_forecasts,
    load_store_weeks,
    render_report,
    run_risk,
    save_results,
)
from demandpulse.storage import LocalStore
from demandpulse.warehouse.build import GoldConfig
from risk_data import (
    CATEGORIES,
    N_STORES,
    ORIGINS,
    make_forecasts,
    make_panel,
    save_forecasts,
    store_weeks,
)


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    return make_panel()


@pytest.fixture(scope="module")
def results(panel):
    return compute_risk(make_forecasts(panel), panel, store_weeks(panel), runs=["main"], top_k=5)


def test_choose_split_orders_train_calibration_and_test_in_time():
    assert choose_split([78, 85, 91, 98, 104]) == ([78], 91, 104)
    assert choose_split([78, 91, 104]) == ([78], 91, 104)


def test_choose_split_is_none_when_labels_cannot_be_known_in_time():
    assert choose_split([78, 85]) is None
    assert choose_split([78, 91]) is None


def test_every_level_gets_windows_scores_and_ranking(results):
    for name in LEVELS:
        windows = results.windows[name]
        assert {"p_shortfall", "revenue_at_risk", "forecast_gap", "relative_gap"} <= set(
            windows.columns
        )
        assert set(windows["model"]) == {"ensemble", "lightgbm", "own_trend"}
        assert set(windows["target_def"]) == {"flat_yoy", "trailing_growth"}
        assert set(results.ranking[name]["model"]) >= {"ensemble", "own_trend", "random"}


def test_probability_estimates_exist_only_where_earlier_windows_had_finished(results):
    assert results.pool_origins == [91, 98, 104]
    early = results.windows["store"].filter(pl.col("origin").is_in([78, 85]))
    assert early["p_shortfall"].null_count() == early.height


def test_summary_averages_only_over_origins_with_estimates(results):
    summary = results.ranking_summary["store"]
    ensemble = summary.filter(
        (pl.col("model") == "ensemble") & (pl.col("score") == "revenue_at_risk")
    )
    assert ensemble["origins"].max() == 3


def test_the_classifier_runs_on_the_latest_ordered_split(results):
    outcome = results.classifier["store_category"]
    assert outcome["split"] == {"train": [78], "calibration": 91, "test": 104}
    assert outcome["flat_yoy"]["result"].metrics["n_test"] == N_STORES * len(CATEGORIES)


def test_recommendations_and_scenarios_cover_both_target_definitions(results):
    recs = results.recommendations
    assert set(recs["target_def"]) == {"flat_yoy", "trailing_growth"}
    assert recs.filter(pl.col("target_def") == "flat_yoy").height == 5
    assert recs["store_region"].null_count() == 0
    assert recs["evidence"].str.contains("probability of a shortfall").all()
    assert set(results.scenarios["scenario"]) == {
        "traffic_+2%",
        "basket_size_+2%",
        "price_mix_+1%",
        "all_three",
    }


def test_the_primary_model_must_exist(panel):
    with pytest.raises(ValueError, match="Primary model"):
        compute_risk(make_forecasts(panel), panel, None, ["main"], primary="nope")


def test_without_store_weeks_recommendations_say_the_driver_split_is_unavailable(panel):
    out = compute_risk(make_forecasts(panel), panel, None, ["main"], top_k=3)
    assert out.recommendations["evidence"].str.contains("driver split unavailable").all()


def test_report_labels_targets_as_hypothetical_and_lists_each_section(results):
    text = render_report(results)
    for heading in (
        "# Revenue risk report",
        "## Target growth assumptions",
        "## Store level",
        "## Store x category level",
        "### Classifier (store)",
        "## Stores to look at first",
        "## Scenario arithmetic (hypothetical",
    ):
        assert heading in text
    assert "hypothetical" in text
    assert "single test window" in text


def test_save_results_writes_every_output(results, tmp_path):
    store = LocalStore(tmp_path / "lake")
    text = save_results(store, results)
    keys = set(store.list_keys("risk/"))
    expected = {
        "risk/growth.parquet",
        "risk/windows_store.parquet",
        "risk/windows_store_category.parquet",
        "risk/ranking_store.parquet",
        "risk/ranking_store_category.parquet",
        "risk/ranking_summary_store.parquet",
        "risk/ranking_summary_store_category.parquet",
        "risk/classifier_store.json",
        "risk/classifier_store_category.json",
        "risk/recommendations.parquet",
        "risk/scenarios.parquet",
        "risk/report.md",
        "risk/summary.json",
    }
    assert expected <= keys
    assert store.read_bytes("risk/report.md").decode() == text
    category_windows = pl.read_parquet(tmp_path / "lake/risk/windows_store_category.parquet")
    assert set(category_windows["model"]) == {"ensemble"}
    payload = json.loads(store.read_bytes("risk/classifier_store.json"))
    assert payload["split"]["test"] == 104
    assert "brier_calibrated" in payload["flat_yoy"]["metrics"]


def test_forecasts_from_two_runs_are_combined(panel, tmp_path):
    store = LocalStore(tmp_path / "lake")
    forecasts = make_forecasts(panel)
    save_forecasts(store, forecasts, "main", [78, 91, 104])
    save_forecasts(store, forecasts, "robustness", [85, 98])
    combined = load_forecasts(store, ["main", "robustness"])
    assert sorted(combined["origin"].unique().to_list()) == list(ORIGINS)
    assert set(combined["model"]) == {"ensemble", "lightgbm"}


def test_overlapping_runs_are_rejected(panel, tmp_path):
    store = LocalStore(tmp_path / "lake")
    forecasts = make_forecasts(panel)
    save_forecasts(store, forecasts, "a", [78, 91])
    save_forecasts(store, forecasts, "b", [91, 104])
    with pytest.raises(ValueError, match="more than once"):
        load_forecasts(store, ["a", "b"])


def test_models_covering_different_origins_are_rejected(panel, tmp_path):
    store = LocalStore(tmp_path / "lake")
    forecasts = make_forecasts(panel)
    save_forecasts(store, forecasts.filter(pl.col("model") == "ensemble"), "a", [78, 91, 104])
    save_forecasts(store, forecasts.filter(pl.col("model") == "lightgbm"), "a", [78, 91])
    with pytest.raises(ValueError, match="different origins"):
        load_forecasts(store, ["a"])


def test_missing_forecasts_are_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No forecasts"):
        load_forecasts(LocalStore(tmp_path / "lake"), ["main"])


def test_store_weeks_come_from_gold_with_true_baskets(tmp_path):
    lake = tmp_path / "lake"
    (lake / "gold").mkdir(parents=True)
    pl.DataFrame(
        {
            "shop_week": [200607, 200608],
            "store_code": ["S1", "S1"],
            "revenue": [10.5, 20.25],
            "baskets": [3, 4],
            "units": [7, 8],
        }
    ).write_parquet(lake / "gold/fct_store_week.parquet")
    pl.DataFrame({"shop_week": [200607, 200608], "week_idx": [1, 2]}).write_parquet(
        lake / "gold/dim_date.parquet"
    )
    config = GoldConfig(lake_root=str(lake), work_dir=tmp_path / "work", threads=1)
    out = load_store_weeks(config)
    assert out.columns == ["store_code", "week_idx", "revenue", "baskets", "units"]
    assert out["revenue"].to_list() == [10.5, 20.25]
    assert out["baskets"].to_list() == [3.0, 4.0]


def test_run_risk_reads_the_lake_and_writes_the_report(panel, tmp_path):
    store = LocalStore(tmp_path / "lake")
    forecasts = make_forecasts(panel)
    save_forecasts(store, forecasts, "main", [78, 91, 104])
    save_forecasts(store, forecasts, "robustness", [85, 98])
    text, out = run_risk(store, panel, store_weeks(panel), ["main", "robustness"], top_k=3)
    assert store.exists("risk/report.md")
    assert "Revenue risk report" in text
    assert out.origins == list(ORIGINS)
