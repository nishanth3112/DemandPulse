import json

import polars as pl
import pytest

from demandpulse.app.export import GOLD_TABLES, RISK_JSON, RISK_TABLES, export_snapshot
from demandpulse.app.snapshot import MANIFEST, Snapshot
from demandpulse.storage import LocalStore


def test_snapshot_contains_every_gold_mart_and_risk_output(app_snapshot_dir):
    snapshot = Snapshot(app_snapshot_dir)
    for name in GOLD_TABLES:
        assert snapshot.has(name), name
    for name in RISK_TABLES:
        assert snapshot.has(f"risk_{name}"), name
    for name in RISK_JSON:
        assert isinstance(snapshot.json(f"risk_{name}"), dict)
    assert snapshot.has("risk_recommendations")
    assert snapshot.has("risk_scenarios")


def test_the_basket_fact_and_category_week_fact_are_never_exported(app_snapshot_dir):
    names = set(Snapshot(app_snapshot_dir).manifest["tables"])
    assert not names & {"fct_basket", "fct_store_category_week"}
    assert not [p for p in app_snapshot_dir.iterdir() if "basket" in p.name and "mix" not in p.name]


def test_store_week_is_joined_to_dates_and_store_attributes(app_snapshot_dir):
    weeks = Snapshot(app_snapshot_dir).table("store_week")
    assert {"week_idx", "week_start", "store_format", "store_region", "revenue"} <= set(
        weeks.columns
    )
    assert weeks["week_idx"].min() == 1
    assert weeks.schema["revenue"] == pl.Float64


def test_money_columns_are_floats_not_decimals(app_snapshot_dir):
    snapshot = Snapshot(app_snapshot_dir)
    for name in snapshot.manifest["tables"]:
        assert not any(isinstance(t, pl.Decimal) for t in snapshot.table(name).dtypes), name


def test_network_forecast_weeks_cover_models_runs_and_origins(app_snapshot_dir):
    weeks = Snapshot(app_snapshot_dir).table("forecast_network_weeks")
    assert set(weeks["model"]) == {"ensemble", "lightgbm"}
    assert set(weeks["run"]) == {"main", "robustness"}
    assert sorted(weeks["origin"].unique().to_list()) == [78, 85, 91, 98, 104]
    assert weeks["week_start"].null_count() == 0


def test_forecast_summaries_from_both_runs_are_stacked_with_a_run_column(app_snapshot_dir):
    pooled = Snapshot(app_snapshot_dir).table("forecast_full_pooled")
    assert set(pooled["run"]) == {"main", "robustness"}
    assert {"model", "wape", "bias"} <= set(pooled.columns)


def test_manifest_records_row_counts_and_gold_build(app_snapshot_dir):
    manifest = Snapshot(app_snapshot_dir).manifest
    assert manifest["tables"]["dim_store"] == 45
    assert manifest["gold"]["silver_weeks"] == 117
    assert "created_at" in manifest


def test_opening_a_folder_without_a_manifest_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="app export"):
        Snapshot(tmp_path)


def test_unknown_tables_and_documents_are_errors(app_snapshot_dir):
    snapshot = Snapshot(app_snapshot_dir)
    with pytest.raises(KeyError, match="no table"):
        snapshot.table("nope")
    with pytest.raises(KeyError, match="no document"):
        snapshot.json("nope")


def test_export_fails_clearly_when_an_earlier_phase_has_not_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="gold/dim_store"):
        export_snapshot(LocalStore(tmp_path / "empty"), tmp_path / "out")


def test_a_failed_export_does_not_leave_a_usable_snapshot(app_snapshot_dir, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / MANIFEST).write_text(json.dumps({"tables": {}}))
    with pytest.raises(FileNotFoundError):
        export_snapshot(LocalStore(tmp_path / "empty"), out)
    assert not (out / MANIFEST).exists()
    with pytest.raises(FileNotFoundError, match="app export"):
        Snapshot(out)
