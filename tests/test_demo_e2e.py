import json

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from demandpulse import cli
from demandpulse.app.snapshot import APP_DATA_ENV, Snapshot
from demandpulse.demo import ENSEMBLE, MIN_WEEKS, RUN, render_timings, run_demo
from demandpulse.synthetic import SOURCE
from demandpulse.warehouse.build import GoldConfig
from demandpulse.warehouse.checks import run_checks


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return run_demo(tmp_path_factory.mktemp("demo"), stores=40, weeks=MIN_WEEKS, threads=2)


def test_every_stage_ran_and_was_timed(demo):
    assert list(demo.seconds) == ["synthetic", "gold", "panel_and_forecast", "risk", "export"]
    assert all(value >= 0 for value in demo.seconds.values())
    assert demo.total_seconds == pytest.approx(sum(demo.seconds.values()))


def test_gold_passes_all_of_the_projects_validation_checks(demo):
    config = GoldConfig(
        lake_root=str(demo.root),
        source_glob=SOURCE,
        work_dir=demo.root / "work",
        threads=2,
        memory_limit="2GB",
    )
    outcomes = run_checks(config)
    failed = [o.name for o in outcomes if not o.passed]
    assert not failed
    assert len(outcomes) == 24


def test_gold_totals_reconcile_with_the_synthetic_silver(demo):
    manifest = json.loads((demo.root / "gold/_manifest.json").read_text())
    assert manifest["silver_rows"] == demo.data.rows
    assert manifest["silver_weeks"] == demo.data.weeks


def test_forecasts_include_the_models_and_their_average(demo):
    forecasts = demo.root / f"forecast/runs/{RUN}/forecasts"
    names = {p.stem for p in forecasts.iterdir()}
    assert names == {"seasonal_naive", "trailing_mean_13", "lightgbm", ENSEMBLE}
    ensemble = pl.read_parquet(forecasts / f"{ENSEMBLE}.parquet")
    assert sorted(ensemble["origin"].unique().to_list()) == [78, 91, 104]
    assert ensemble["y_hat"].null_count() == 0


def test_the_risk_run_wrote_its_report_and_tables(demo):
    risk = demo.root / "risk"
    for name in ("report.md", "windows_store.parquet", "recommendations.parquet"):
        assert (risk / name).exists(), name
    windows = pl.read_parquet(risk / "windows_store.parquet")
    estimated = windows.filter(pl.col("p_shortfall").is_not_null())
    assert sorted(estimated["origin"].unique().to_list()) == [91, 104]


def test_the_snapshot_opens_and_has_no_basket_fact(demo):
    snapshot = Snapshot(demo.app_dir)
    assert snapshot.has("risk_windows_store")
    assert not snapshot.has("fct_basket")


def test_every_dashboard_page_renders_on_the_demo_snapshot(demo, monkeypatch):
    monkeypatch.setenv(APP_DATA_ENV, str(demo.app_dir))
    for name in ("overview", "stores", "forecasting", "risk", "customers"):
        at = AppTest.from_string(
            f"from demandpulse.app import pages\npages.{name}()", default_timeout=120
        ).run()
        assert not at.exception, f"{name}: {[e.value for e in at.exception]}"


def test_the_drilldown_handles_a_store_that_closed_and_has_no_recent_sales(demo, monkeypatch):
    monkeypatch.setenv(APP_DATA_ENV, str(demo.app_dir))
    at = AppTest.from_string(
        "from demandpulse.app import pages\npages.stores()", default_timeout=120
    ).run()
    store = next(s for s in at.selectbox if s.label == "Store")
    assert "STORE00006" in store.options
    store.set_value("STORE00006").run()
    assert not at.exception, [e.value for e in at.exception]
    assert {m.label: m.value for m in at.metric}["Average basket"] == "n/a"


def test_the_cli_can_repeat_the_risk_and_export_steps_on_the_demo_lake(demo, tmp_path, capsys):
    base = ["--local-root", str(demo.root)]
    assert cli.main([*base, "risk", "run", "--forecast-run", RUN, "--top-k", "5"]) == 0
    assert "# Revenue risk report" in capsys.readouterr().out
    out = tmp_path / "snapshot"
    assert cli.main([*base, "app", "export", "--out", str(out)]) == 0
    assert Snapshot(out).has("store_week")


def test_timings_render_as_a_markdown_table(demo):
    text = render_timings(demo)
    assert "| stage | seconds |" in text
    assert "| total |" in text
    assert "40 stores, 117 weeks" in text


def test_a_run_too_short_for_the_forecast_origins_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least 117 weeks"):
        run_demo(tmp_path, weeks=100)
