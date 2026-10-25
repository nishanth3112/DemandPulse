import json
from pathlib import Path

import polars as pl
import pytest

from demandpulse import cli, tableau
from demandpulse.app.risk_views import MODEL_LABELS
from demandpulse.app.views import COMPARISON_LABELS


@pytest.fixture(scope="module")
def exported(app_snapshot_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("tableau")
    manifest = tableau.export_tableau(app_snapshot_dir, out)
    return out, manifest


def read(exported, name: str) -> pl.DataFrame:
    return pl.read_csv(exported[0] / f"{name}.csv", try_parse_dates=True)


def test_every_table_is_written_with_the_manifest_row_counts(exported):
    out, manifest = exported
    assert {"dim_store", "fact_forecast_store_week", "fact_revenue_bridge"} <= set(
        manifest["tables"]
    )
    for name, rows in manifest["tables"].items():
        assert rows > 0, name
        assert read(exported, name).height == rows, name
    assert json.loads((out / tableau.MANIFEST).read_text()) == manifest


def test_data_dictionary_lists_every_column_of_every_table(exported):
    out, manifest = exported
    dictionary = pl.read_csv(out / tableau.DICTIONARY)
    for name in manifest["tables"]:
        columns = dictionary.filter(pl.col("table") == name)["column"].to_list()
        assert columns == read(exported, name).columns, name
        assert dictionary.filter(pl.col("table") == name)["rows"].unique().to_list() == [
            manifest["tables"][name]
        ]


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("dim_store", ["store_code"]),
        ("dim_date", ["week_idx"]),
        ("dim_model", ["model"]),
        ("fact_store_week", ["store_code", "week_idx"]),
        ("fact_region_benchmark", ["comparison", "store_region"]),
        ("fact_store_scorecard", ["comparison", "store_code"]),
        ("fact_forecast_accuracy", ["run", "model", "slice", "slice_value"]),
        ("fact_forecast_network_week", ["run", "model", "origin", "week_idx"]),
        ("fact_forecast_store_week", ["run", "model", "origin", "week_idx", "store_code"]),
        ("fact_risk_ranking", ["grain", "model", "score", "target_def", "k_fraction"]),
        ("fact_trading_pattern", ["store_format", "shop_weekday", "shop_hour"]),
        ("fact_category_benchmark", ["comparison", "prod_code_30"]),
        ("fact_region_category_mix", ["comparison", "store_region", "prod_code_30"]),
        ("fact_forecast_vs_baseline", ["run", "model", "baseline"]),
        ("fact_risk_store_window", ["model", "origin", "store_code", "target_def"]),
        ("fact_risk_recommendations", ["model", "origin", "store_code", "target_def"]),
        ("fact_risk_scenarios", ["store_code", "target_def", "scenario"]),
        ("fact_customer_cohort", ["cohort_period", "periods_since_first"]),
        ("fact_rfm_segment", ["rfm_segment"]),
        ("fact_revenue_bridge", ["comparison", "level", "store_region", "store_code", "step"]),
    ],
)
def test_each_file_has_one_row_per_key(exported, name, key):
    table = read(exported, name)
    assert table.select(key).is_duplicated().sum() == 0, name


def test_forecast_facts_join_to_the_dimensions(exported):
    weeks = read(exported, "fact_forecast_store_week")
    assert set(weeks["store_code"]) <= set(read(exported, "dim_store")["store_code"])
    assert set(weeks["week_idx"]) <= set(read(exported, "dim_date")["week_idx"])
    assert set(weeks["model"]) <= set(read(exported, "dim_model")["model"])
    assert weeks["store_region"].null_count() == 0
    assert weeks["week_start"].null_count() == 0


def test_labels_replace_raw_codes(exported):
    models = read(exported, "dim_model")
    for row in models.iter_rows(named=True):
        assert row["model_label"] == MODEL_LABELS.get(row["model"], row["model"])
    labels = set(read(exported, "dim_comparison")["comparison_label"])
    assert labels <= set(COMPARISON_LABELS.values())
    assert read(exported, "fact_risk_store_window")["target_label"].null_count() == 0


def test_accuracy_stacks_four_slices_and_fills_origin_only_for_origins(exported):
    accuracy = read(exported, "fact_forecast_accuracy")
    assert set(accuracy["slice"]) == {"level", "origin", "horizon", "format"}
    levels = accuracy.filter(pl.col("slice") == "level")["slice_value"].unique()
    assert set(levels) == {"Series (store x category)", "Store", "Category", "Network"}
    assert accuracy.filter(pl.col("slice") != "origin")["origin"].null_count() == (
        accuracy.filter(pl.col("slice") != "origin").height
    )
    assert accuracy.filter(pl.col("slice") == "origin")["origin"].null_count() == 0


def test_bridge_has_the_eight_steps_in_order_for_every_entity(exported):
    bridge = read(exported, "fact_revenue_bridge")
    assert set(bridge["level"]) == {"region", "store"}
    per_entity = bridge.group_by("comparison", "level", "store_region", "store_code").agg(
        pl.col("step_order").sort().alias("order")
    )
    assert all(order == list(range(1, 9)) for order in per_entity["order"].to_list())
    assert bridge.filter(pl.col("level") == "region")["store_code"].null_count() == (
        bridge.filter(pl.col("level") == "region").height
    )


class _Stub:
    def __init__(self, region, store):
        self._tables = {
            "mart_revenue_variance_region": region,
            "mart_revenue_variance_store": store,
        }

    def table(self, name):
        return self._tables[name]


def test_bridge_steps_add_up_when_the_effects_add_up():
    effects = {
        "trading_weeks_effect": 1.0, "traffic_effect": -4.0, "basket_size_effect": 2.0,
        "price_mix_effect": 0.5, "new_store_effect": 3.0, "closed_store_effect": -1.5,
    }  # fmt: skip
    region = pl.DataFrame(
        {"comparison": "c", "store_region": "ALL", "base_revenue": 100.0, "current_revenue": 101.0}
        | effects
    )
    store = region.with_columns(
        pl.lit("S1").alias("store_code"), pl.lit("continuing").alias("store_status")
    )
    bridge = tableau._bridge(_Stub(region, store))
    for level in ("region", "store"):
        mine = bridge.filter(pl.col("level") == level)
        base = mine.filter(pl.col("step") == "Base period")["value"][0]
        current = mine.filter(pl.col("step") == "Current period")["value"][0]
        change = mine.filter(pl.col("step_kind") == "change")["value"].sum()
        assert base + change == pytest.approx(current)


def test_nothing_customer_level_or_row_level_is_exported(exported):
    out, manifest = exported
    assert not any("basket" in name and name != "fact_basket_mix" for name in manifest["tables"])
    columns = pl.read_csv(out / tableau.DICTIONARY)["column"].unique().to_list()
    assert not {"cust_code", "basket_id", "prod_code"} & set(columns)


def test_cli_exports_from_a_snapshot_folder(app_snapshot_dir, tmp_path):
    out = tmp_path / "csv"
    assert (
        cli.main(["tableau", "export", "--snapshot", str(app_snapshot_dir), "--out", str(out)]) == 0
    )
    assert (out / "dim_store.csv").is_file()
    assert (out / tableau.MANIFEST).is_file()


def test_a_missing_snapshot_points_at_the_export_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="demandpulse app export"):
        tableau.export_tableau(tmp_path / "none", tmp_path / "out")


def test_export_reads_only_the_snapshot_not_the_lake():
    text = Path(tableau.__file__).read_text()
    assert "boto3" not in text
    assert "demandpulse.storage" not in text
