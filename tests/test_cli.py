import polars as pl
import pytest

from demandpulse import cli
from demandpulse.ingest.bronze import bronze_key
from demandpulse.ingest.silver import silver_key


@pytest.fixture
def sources_file(make_zip, tmp_path):
    source = make_zip()
    path = tmp_path / "sources.yaml"
    path.write_text(f"sources:\n  - name: {source.name}\n    kind: sample\n    url: {source.url}\n")
    return path, source


def run(args, sources_path, lake):
    return cli.main(["--sources-file", str(sources_path), "--local-root", str(lake), *args])


def test_run_builds_every_layer_and_reports_quality(sources_file, tmp_path, capsys):
    path, source = sources_file
    lake = tmp_path / "lake"

    assert run(["run", source.name], path, lake) == 0

    assert (lake / bronze_key(source.name, 200607)).exists()
    assert pl.read_parquet(lake / silver_key(source.name, 200609)).height == 3
    assert f"# Data quality report: {source.name}" in capsys.readouterr().out


def test_stages_can_run_separately_and_resume(sources_file, tmp_path):
    path, source = sources_file
    lake = tmp_path / "lake"

    assert run(["ingest", source.name, "--max-weeks", "1"], path, lake) == 0
    assert run(["ingest", source.name], path, lake) == 0
    assert run(["silver", source.name, "--weeks", "200607"], path, lake) == 0
    assert (lake / silver_key(source.name, 200607)).exists()
    assert not (lake / silver_key(source.name, 200609)).exists()


def test_failed_quality_gate_returns_nonzero(sources_file, tmp_path):
    path, source = sources_file
    # Quality on an empty lake has no data, which must fail rather than pass vacuously.
    assert run(["quality", source.name], path, tmp_path / "empty-lake") == 1


def test_sources_command_lists_registry(sources_file, tmp_path, capsys):
    path, source = sources_file
    assert run(["sources"], path, tmp_path / "x") == 0
    assert source.name in capsys.readouterr().out


def test_unknown_source_is_a_clear_error(sources_file, tmp_path):
    path, _ = sources_file
    with pytest.raises(KeyError, match="Unknown source"):
        run(["ingest", "nope"], path, tmp_path / "lake")


def test_known_issue_sources_are_refused_unless_overridden(tmp_path, capsys):
    path = tmp_path / "sources.yaml"
    path.write_text(
        "sources:\n  - name: bad\n    kind: sample\n    url: file:///nowhere.zip\n"
        '    known_issue: "ids are corrupted"\n'
    )
    assert run(["ingest", "bad"], path, tmp_path / "lake") == 2
    assert "ids are corrupted" in capsys.readouterr().err


def test_gold_build_validates_and_exports_marts(silver_lake, tmp_path, capsys):
    args = [
        "--local-root", str(silver_lake), "gold", "build",
        "--source-glob", "fixture", "--window-weeks", "2", "--year-weeks", "4",
        "--work-dir", str(tmp_path / "work"), "--threads", "2", "--memory-limit", "1GB",
    ]  # fmt: skip
    assert cli.main(args) == 0

    out = capsys.readouterr().out
    assert "checks passed" in out
    assert "FAIL" not in out
    assert (silver_lake / "gold" / "mart_store_scorecard.parquet").exists()


def test_gold_validate_fails_when_nothing_was_built(silver_lake, tmp_path):
    args = [
        "--local-root", str(silver_lake), "gold", "validate",
        "--source-glob", "fixture", "--work-dir", str(tmp_path / "work"),
    ]  # fmt: skip
    with pytest.raises(FileNotFoundError, match="build Gold first"):
        cli.main(args)


def test_forecast_backtest_and_report_run_from_saved_inputs(tmp_path, capsys):
    from demandpulse.forecasting.panel import representative_subset
    from demandpulse.forecasting.runner import save_inputs
    from demandpulse.storage import LocalStore
    from forecast_data import make_seasonal_panel

    lake = tmp_path / "lake"
    panel = make_seasonal_panel()
    save_inputs(LocalStore(lake), panel, representative_subset(panel, per_stratum=2))
    base = ["--local-root", str(lake), "forecast"]

    assert cli.main([*base, "backtest", "seasonal_naive", "trailing_mean_13", "--run", "t"]) == 0
    assert cli.main([*base, "report", "--run", "t"]) == 0

    out = capsys.readouterr().out
    assert "# Forecast model comparison: run `t`" in out
    assert (lake / "forecast" / "runs" / "t" / "report.md").exists()
    assert (lake / "forecast" / "runs" / "t" / "forecasts" / "seasonal_naive.parquet").exists()


def test_forecast_prepare_explains_when_the_data_is_too_short(silver_lake, tmp_path):
    gold_args = [
        "--local-root", str(silver_lake), "gold", "build",
        "--source-glob", "fixture", "--window-weeks", "2", "--year-weeks", "4",
        "--work-dir", str(tmp_path / "work"), "--threads", "2", "--memory-limit", "1GB",
    ]  # fmt: skip
    assert cli.main(gold_args) == 0
    prepare = [
        "--local-root", str(silver_lake), "forecast", "prepare",
        "--source-glob", "fixture", "--work-dir", str(tmp_path / "work"),
    ]  # fmt: skip
    with pytest.raises(ValueError, match="No series are eligible"):
        cli.main(prepare)


def test_risk_run_scores_saved_forecasts_and_writes_the_report(tmp_path, capsys):
    from demandpulse.forecasting.panel import Subset
    from demandpulse.forecasting.runner import save_inputs
    from demandpulse.storage import LocalStore
    from risk_data import make_forecasts, make_panel, save_forecasts, store_weeks

    frame = make_panel()
    lake = tmp_path / "lake"
    store = LocalStore(lake)
    series = frame.select("store_code", "prod_code_30").unique().with_columns(stratum=pl.lit("x"))
    save_inputs(store, frame, Subset(series=series, seed=1))
    forecasts = make_forecasts(frame)
    save_forecasts(store, forecasts, "main", [78, 91, 104])
    save_forecasts(store, forecasts, "robustness", [85, 98])
    weeks = store_weeks(frame)
    (lake / "gold").mkdir()
    weeks.select(
        pl.col("week_idx").alias("shop_week"), "store_code", "revenue", "baskets", "units"
    ).write_parquet(lake / "gold/fct_store_week.parquet")
    weeks.select(pl.col("week_idx").alias("shop_week"), "week_idx").unique().write_parquet(
        lake / "gold/dim_date.parquet"
    )

    code = cli.main(
        [
            "--local-root",
            str(lake),
            "risk",
            "run",
            "--top-k",
            "3",
            "--work-dir",
            str(tmp_path / "w"),
        ]
    )

    assert code == 0
    assert "# Revenue risk report" in capsys.readouterr().out
    assert store.exists("risk/report.md")


def test_app_export_writes_a_snapshot_the_dashboard_can_open(tmp_path):
    from app_data import write_lake
    from demandpulse.app.snapshot import Snapshot

    lake = tmp_path / "lake"
    write_lake(lake)
    out = tmp_path / "snapshot"

    assert cli.main(["--local-root", str(lake), "app", "export", "--out", str(out)]) == 0

    assert Snapshot(out).has("store_week")
