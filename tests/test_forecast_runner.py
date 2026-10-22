import json

import polars as pl
import pytest

from demandpulse.forecasting.backtest import ForecastError, run_backtest
from demandpulse.forecasting.models.baselines import SeasonalNaive
from demandpulse.forecasting.panel import (
    ORIGINS,
    SERIES_KEY,
    Subset,
    eligible_series,
    representative_subset,
)
from demandpulse.forecasting.runner import (
    MODEL_ORIGINS,
    combine_run,
    load_inputs,
    load_run,
    make_model,
    render_report,
    run_model,
    save_inputs,
    save_run,
    summary_tables,
    write_report,
)
from demandpulse.storage import LocalStore
from forecast_data import make_seasonal_panel

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def panel():
    return make_seasonal_panel()


@pytest.fixture(scope="module")
def subset(panel):
    return representative_subset(panel, per_stratum=2)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "lake")


def test_inputs_round_trip_through_the_store(store, panel, subset):
    save_inputs(store, panel, subset)
    loaded_panel, loaded_subset = load_inputs(store)
    assert loaded_panel.height == panel.height
    assert loaded_subset.series.equals(subset.series)
    assert loaded_subset.seed == subset.seed


def test_unknown_model_names_are_rejected(panel):
    with pytest.raises(KeyError, match="Unknown model"):
        make_model("prophet_of_doom", panel, 1, {})


def test_cheap_models_run_on_every_eligible_series(panel, subset):
    run = run_model("seasonal_naive", panel, subset, origins=(104,))
    assert run.scope == "full"
    expected = eligible_series(panel, 104).height
    assert run.info["series_per_origin"] == {"104": expected}
    assert run.result.forecasts.height == expected * 13


def test_expensive_models_run_only_on_the_subset(panel, subset):
    run = run_model("ets_damped", panel, subset, origins=(104,), workers=1)
    assert run.scope == "subset"
    assert run.info["series_per_origin"] == {"104": subset.series.height}
    assert run.info["ets_fits"] == subset.series.height
    assert run.info["ets_fallbacks"] == 0
    keys = run.result.forecasts.select(SERIES_KEY).unique()
    assert keys.join(subset.series, on=SERIES_KEY).height == subset.series.height


def test_holt_winters_is_limited_to_origins_with_two_full_seasons(panel, subset):
    assert MODEL_ORIGINS["holt_winters"] == (104,)
    run = run_model("holt_winters", panel, subset, workers=1)
    assert run.info["origins"] == [104]


def test_lightgbm_records_its_fit_details(panel, subset):
    run = run_model("lightgbm", panel, subset, origins=(104,), workers=1)
    assert run.scope == "full"
    assert run.info["fit_info"]["104"]["training_rows"] > 0


def test_runs_are_saved_and_reloaded_with_a_single_schema(store, panel, subset):
    for name in ("seasonal_naive", "trailing_mean_13"):
        save_run(store, "t", run_model(name, panel, subset, origins=(104,)))
    forecasts, scales, meta = load_run(store, "t")
    assert set(forecasts["model"]) == {"seasonal_naive", "trailing_mean_13"}
    assert set(meta) == {"seasonal_naive", "trailing_mean_13"}
    assert scales.height > 0


def test_loading_a_run_that_does_not_exist_is_an_error(store):
    with pytest.raises(FileNotFoundError, match="forecast backtest"):
        load_run(store, "nope")


@pytest.fixture
def finished_run(store, panel, subset):
    for name in ("seasonal_naive", "trailing_mean_13", "ets_damped", "lightgbm"):
        save_run(store, "r", run_model(name, panel, subset, workers=1))
    return load_run(store, "r")


def test_summary_tables_compare_models_on_identical_series(finished_run, panel, subset):
    forecasts, scales, _ = finished_run
    tables = summary_tables(forecasts, scales, panel, subset)
    common = tables["subset_final_origin"]
    assert set(common["model"]) == {"seasonal_naive", "trailing_mean_13", "ets_damped", "lightgbm"}
    assert set(common["series"]) == {subset.series.height}  # identical series for every model
    assert set(common["n"]) == {subset.series.height * 13}


def test_full_panel_tables_cover_only_full_scope_models(finished_run, panel, subset):
    forecasts, scales, _ = finished_run
    tables = summary_tables(forecasts, scales, panel, subset)
    assert "ets_damped" not in set(tables["full_pooled"]["model"])
    assert {"full_store_level", "full_category_level", "full_network_level"} <= set(tables)


def test_network_accuracy_is_at_least_as_good_as_series_accuracy(finished_run, panel, subset):
    forecasts, scales, _ = finished_run
    tables = summary_tables(forecasts, scales, panel, subset)
    series = dict(zip(tables["full_pooled"]["model"], tables["full_pooled"]["wape"], strict=True))
    network = dict(
        zip(
            tables["full_network_level"]["model"], tables["full_network_level"]["wape"], strict=True
        )
    )
    for model, value in network.items():
        assert value <= series[model] + 1e-12


def test_report_text_is_generated_from_the_run(finished_run, panel, subset):
    forecasts, scales, meta = finished_run
    tables = summary_tables(forecasts, scales, panel, subset)
    text = render_report("r", tables, meta, panel, subset)
    assert text.startswith("# Forecast model comparison: run `r`")
    assert f"{subset.series.height} series" in text
    assert "Improvement over seasonal naive" in text
    assert "seed" in text
    for section in ("Subset, final origin", "Full panel, by forecast horizon", "Run details"):
        assert section in text


def test_write_report_saves_markdown_and_json_summary(store, finished_run, panel, subset):
    text, tables = write_report(store, "r", panel, subset)
    assert store.read_bytes("forecast/runs/r/report.md").decode() == text
    summary = json.loads(store.read_bytes("forecast/runs/r/summary.json"))
    assert set(summary) == set(tables)


def test_origins_in_the_report_match_the_shared_backtest(finished_run):
    forecasts, _, _ = finished_run
    assert set(forecasts.filter(pl.col("model") == "lightgbm")["origin"]) == set(ORIGINS)


def test_runner_results_equal_a_direct_backtest(panel, subset):
    run = run_model("seasonal_naive", panel, subset, origins=(104,))
    direct = run_backtest(panel, SeasonalNaive(), (104,))
    order = ["origin", *SERIES_KEY, "horizon"]
    assert run.result.forecasts.sort(order).equals(direct.forecasts.sort(order))


def test_combine_is_the_exact_equal_weight_mean_of_its_components(store, panel, subset):
    for name in ("seasonal_naive", "trailing_mean_13"):
        save_run(store, "c", run_model(name, panel, subset, origins=(91, 104)))

    combined = combine_run(
        store, "c", "ensemble", ["seasonal_naive", "trailing_mean_13"], "post hoc"
    )

    keys = ["origin", *SERIES_KEY, "horizon", "week_idx"]
    forecasts, _, _ = load_run(store, "c")
    parts = {
        m: forecasts.filter(pl.col("model") == m).sort(keys)["y_hat"].to_numpy()
        for m in ("seasonal_naive", "trailing_mean_13")
    }
    got = combined.result.forecasts.sort(keys)["y_hat"].to_numpy()
    assert got == pytest.approx((parts["seasonal_naive"] + parts["trailing_mean_13"]) / 2)
    assert combined.info["components"] == ["seasonal_naive", "trailing_mean_13"]
    assert combined.info["method"] == "equal-weight mean of component forecasts"
    assert combined.info["note"] == "post hoc"
    assert combined.scope == "full"


def test_combine_needs_two_components_that_exist(store, panel, subset):
    save_run(store, "c", run_model("seasonal_naive", panel, subset, origins=(104,)))
    with pytest.raises(ValueError, match="at least two"):
        combine_run(store, "c", "x", ["seasonal_naive"])
    with pytest.raises(FileNotFoundError, match="ghost"):
        combine_run(store, "c", "x", ["seasonal_naive", "ghost"])


def test_combine_refuses_components_that_cover_different_origins(store, panel, subset):
    save_run(store, "c", run_model("seasonal_naive", panel, subset, origins=(91, 104)))
    save_run(store, "c", run_model("trailing_mean_13", panel, subset, origins=(104,)))
    with pytest.raises(ForecastError, match="identical series, origins and weeks"):
        combine_run(store, "c", "x", ["seasonal_naive", "trailing_mean_13"])


def test_combined_model_appears_in_the_full_panel_report(store, panel, subset):
    for name in ("seasonal_naive", "trailing_mean_13"):
        save_run(store, "c", run_model(name, panel, subset))
    save_run(
        store, "c", combine_run(store, "c", "ensemble", ["seasonal_naive", "trailing_mean_13"])
    )
    text, tables = write_report(store, "c", panel, subset)
    assert "ensemble" in set(tables["full_pooled"]["model"])
    assert "equal-weight mean of component forecasts of seasonal_naive" in text


def test_reports_can_be_built_for_other_origins(store, panel, subset):
    for name in ("seasonal_naive", "trailing_mean_13"):
        save_run(store, "r2", run_model(name, panel, subset, origins=(85, 98)))
    text, tables = write_report(store, "r2", panel, subset, origins=(85, 98))
    assert set(tables["subset_by_origin"]["origin"]) == {85, 98}
    assert "weeks 86-98, 99-111" in text


def late_store_panel_and_subset(panel):
    """Series S3/G4 has 84 weeks of history at origin 104: enough to forecast, too few for HW."""
    late = panel.filter(
        ~(
            (pl.col("store_code") == "S3")
            & (pl.col("prod_code_30") == "G4")
            & (pl.col("week_idx") <= 20)
        )
    )
    keys = pl.DataFrame(
        {"store_code": ["S1", "S1", "S2", "S3"], "prod_code_30": ["G1", "G2", "G1", "G4"]}
    ).with_columns(stratum=pl.lit("LS|medium"))
    return late, Subset(series=keys, seed=1)


def test_holt_winters_skips_series_without_two_years_and_records_it(panel):
    late, subset = late_store_panel_and_subset(panel)
    run = run_model("holt_winters", late, subset, workers=1)
    assert run.info["series_per_origin"] == {"104": 3}
    assert run.info["subset_series_excluded"] == 1
    assert ("S3", "G4") not in set(run.result.forecasts.select(SERIES_KEY).unique().rows())


def test_report_compares_holt_winters_on_the_series_it_covers(store, panel):
    late, subset = late_store_panel_and_subset(panel)
    for name in ("seasonal_naive", "trailing_mean_13", "ets_damped", "holt_winters"):
        save_run(store, "h", run_model(name, late, subset, workers=1))
    forecasts, scales, meta = load_run(store, "h")

    tables = summary_tables(forecasts, scales, late, subset, meta)

    assert "holt_winters" not in set(tables["subset_final_origin"]["model"])
    assert set(tables["subset_final_origin"]["series"]) == {4}
    like_for_like = tables["subset_final_origin_holt_winters_series"]
    assert "holt_winters" in set(like_for_like["model"])
    assert set(like_for_like["series"]) == {3}  # every model restricted to HW's 3 series
    text = render_report("h", tables, meta, late, subset)
    assert "the 3 series holt_winters could forecast" in text
    assert "1 subset series excluded" in text
