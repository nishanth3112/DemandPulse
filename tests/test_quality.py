import io
import zipfile
from decimal import Decimal

import polars as pl
import pytest

from demandpulse.ingest.bronze import bronze_key, ingest_source
from demandpulse.ingest.schema import normalize_calendar, normalize_transactions
from demandpulse.ingest.silver import (
    SilverManifest,
    build_silver,
    silver_key,
    silver_manifest_key,
    transform,
)
from demandpulse.quality import run_quality_checks, week_violations
from demandpulse.storage import LocalStore


@pytest.fixture
def clean_week(week_csv, calendar_csv):
    bronze = normalize_transactions(week_csv(200607, "20060412").encode())
    silver, _ = transform(bronze, normalize_calendar(calendar_csv.encode()))
    return silver


def nonzero(violations):
    return {k: v for k, v in violations.items() if v}


def test_clean_week_has_no_violations(clean_week):
    assert nonzero(week_violations(clean_week)) == {}


@pytest.mark.parametrize(
    ("mutation", "expected_key"),
    [
        (pl.col("quantity") * 0, "nonpositive_quantity"),
        (pl.col("spend") * -1, "negative_spend"),
    ],
)
def test_numeric_rules_flag_bad_values(clean_week, mutation, expected_key):
    column = "quantity" if expected_key == "nonpositive_quantity" else "spend"
    bad = clean_week.with_columns(mutation.alias(column))
    assert week_violations(bad)[expected_key] == 3


def test_zero_spend_is_counted_separately(clean_week):
    bad = clean_week.with_columns(spend=pl.col("spend") * 0)
    assert week_violations(bad)["zero_spend"] == 3
    assert week_violations(bad)["negative_spend"] == 0


def test_unexpected_category_values_are_flagged(clean_week):
    bad = clean_week.with_columns(basket_dominant_mission=pl.lit("Non Food"))
    assert week_violations(bad)["unexpected_basket_dominant_mission"] == 3


def test_out_of_range_weekday_and_hour(clean_week):
    bad = clean_week.with_columns(
        shop_weekday=pl.lit(9, dtype=pl.Int8), shop_hour=pl.lit(24, dtype=pl.Int8)
    )
    violations = week_violations(bad)
    assert violations["weekday_out_of_range"] == 3
    assert violations["hour_out_of_range"] == 3


def test_date_outside_calendar_week_is_flagged(clean_week):
    bad = clean_week.with_columns(shop_date=pl.date(2007, 1, 1))
    assert week_violations(bad)["date_outside_week"] == 3


def test_missing_calendar_week_counts_as_outside(clean_week):
    bad = clean_week.with_columns(week_start=pl.lit(None, dtype=pl.Date))
    assert week_violations(bad)["date_outside_week"] == 3


def test_basket_split_across_stores_is_flagged(clean_week):
    stores = ["STORE00001", "STORE00009", "STORE00002"]
    bad = clean_week.with_columns(store_code=pl.Series(stores))
    assert week_violations(bad)["baskets_with_multiple_stores"] == 1


def test_duplicate_basket_product_pairs_are_counted(clean_week):
    bad = clean_week.with_columns(prod_code=pl.lit("PRD1"))
    assert week_violations(bad)["duplicate_basket_product"] == 1


def test_null_key_values_are_counted(clean_week):
    bad = clean_week.with_columns(prod_code=pl.lit(None, dtype=pl.String))
    assert week_violations(bad)["null_key_values"] == 3


@pytest.fixture
def lake(make_zip, tmp_path):
    store = LocalStore(tmp_path / "lake")
    source = make_zip()
    ingest_source(source, store, sleep=lambda _: None)
    build_silver(source.name, store)
    return store, source


def statuses(report):
    return {c.name: c.status for c in report.checks}


def test_clean_pipeline_output_passes_every_check(lake):
    store, source = lake
    report = run_quality_checks(source.name, store)

    failing = {k: v for k, v in statuses(report).items() if v == "fail"}
    assert failing == {}
    assert report.ok
    assert report.weeks_checked == 2
    assert report.rows == 6
    assert report.baskets == 4
    assert str(report.spend_total) == "19.00"
    assert report.loyalty_row_share == pytest.approx(4 / 6)
    # Week 200608 sits between the two data weeks in the calendar but has no data.
    assert statuses(report)["calendar_gaps"] == "warn"
    assert store.exists(f"quality/{source.name}/report.json")
    assert "| check | status |" in store.read_bytes(f"quality/{source.name}/report.md").decode()


def test_corrupted_silver_spend_is_caught_by_reconciliation(lake):
    store, source = lake
    key = silver_key(source.name, 200607)
    frame = pl.read_parquet(io.BytesIO(store.read_bytes(key)))
    tampered = frame.with_columns(spend=pl.col("spend") + 1)
    buffer = io.BytesIO()
    tampered.write_parquet(buffer)
    store.write_bytes(key, buffer.getvalue())

    report = run_quality_checks(source.name, store)

    assert statuses(report)["silver_spend_vs_bronze"] == "fail"
    assert not report.ok


def test_dropped_silver_rows_are_caught(lake):
    store, source = lake
    key = silver_key(source.name, 200609)
    frame = pl.read_parquet(io.BytesIO(store.read_bytes(key)))
    buffer = io.BytesIO()
    frame.head(2).write_parquet(buffer)
    store.write_bytes(key, buffer.getvalue())

    assert statuses(run_quality_checks(source.name, store))["silver_rows_vs_bronze"] == "fail"


def test_tampered_bronze_file_is_caught_against_its_manifest(lake):
    store, source = lake
    key = bronze_key(source.name, 200607)
    frame = pl.read_parquet(io.BytesIO(store.read_bytes(key)))
    buffer = io.BytesIO()
    frame.head(1).write_parquet(buffer)
    store.write_bytes(key, buffer.getvalue())

    assert statuses(run_quality_checks(source.name, store))["bronze_file_vs_manifest"] == "fail"


def test_week_without_silver_is_reported(make_zip, tmp_path):
    store = LocalStore(tmp_path / "lake")
    source = make_zip()
    ingest_source(source, store, sleep=lambda _: None)
    build_silver(source.name, store, weeks={200607})

    assert statuses(run_quality_checks(source.name, store))["silver_missing"] == "fail"


def test_empty_source_fails(tmp_path):
    report = run_quality_checks("ghost", LocalStore(tmp_path / "lake"))
    assert not report.ok
    assert statuses(report)["has_data"] == "fail"


@pytest.fixture
def lake_with_duplicates(make_zip, week_csv, tmp_path):
    """Week 200609 contains one exact duplicate of its first data row."""
    store = LocalStore(tmp_path / "lake")
    source = make_zip(weeks={200607: "20060412"})
    week_text = week_csv(200609, "20060428")
    first_row = week_text.split("\n")[1]
    with zipfile.ZipFile(tmp_path / "fixture-sample.zip", "a") as zf:
        zf.writestr("p/transactions_200609.csv", week_text + first_row + "\n")
    ingest_source(source, store, sleep=lambda _: None)
    build_silver(source.name, store)
    return store, source


def test_dropped_duplicates_are_reconciled_on_money_and_units(lake_with_duplicates):
    store, source = lake_with_duplicates
    manifest = SilverManifest.model_validate_json(
        store.read_bytes(silver_manifest_key(source.name, 200609))
    )
    assert manifest.duplicates_dropped == 1
    assert manifest.duplicate_spend_dropped == Decimal("3.50")
    assert manifest.duplicate_quantity_dropped == 2

    report = run_quality_checks(source.name, store)

    assert statuses(report)["silver_rows_vs_bronze"] == "pass"
    assert statuses(report)["silver_spend_vs_bronze"] == "pass"
    assert statuses(report)["silver_quantity_vs_bronze"] == "pass"


def test_tampering_is_still_caught_in_a_week_with_duplicates(lake_with_duplicates):
    store, source = lake_with_duplicates
    key = silver_key(source.name, 200609)
    frame = pl.read_parquet(io.BytesIO(store.read_bytes(key)))
    buffer = io.BytesIO()
    frame.with_columns(spend=pl.col("spend") + 1).write_parquet(buffer)
    store.write_bytes(key, buffer.getvalue())

    assert statuses(run_quality_checks(source.name, store))["silver_spend_vs_bronze"] == "fail"
