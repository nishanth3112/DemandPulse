import json

import polars as pl
import pytest

from demandpulse.forecasting.backtest import run_backtest
from demandpulse.forecasting.models.baselines import SeasonalNaive
from demandpulse.forecasting.panel import (
    ORIGINS,
    SERIES_KEY,
    eligible_series,
    representative_subset,
)
from demandpulse.forecasting.runner import (
    MODEL_ORIGINS,
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
