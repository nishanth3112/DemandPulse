import io
from datetime import date
from decimal import Decimal

import polars as pl
import pytest

from demandpulse.ingest.bronze import bronze_key, ingest_source, manifest_key
from demandpulse.ingest.schema import normalize_calendar, normalize_transactions
from demandpulse.ingest.silver import (
    SilverManifest,
    build_silver,
    silver_key,
    silver_manifest_key,
    transform,
)
from demandpulse.storage import LocalStore


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "lake")


@pytest.fixture
def ingested(make_zip, store):
    source = make_zip()
    ingest_source(source, store, sleep=lambda _: None)
    return source


def silver_frame(store, source, week):
    return pl.read_parquet(io.BytesIO(store.read_bytes(silver_key(source.name, week))))


def test_blank_customer_fields_become_null_and_loyalty_is_flagged(week_csv, calendar_csv):
    bronze = normalize_transactions(week_csv(200607, "20060412").encode())
    silver, dropped = transform(bronze, normalize_calendar(calendar_csv.encode()))

    assert dropped == 0
    assert silver["is_loyalty"].to_list() == [True, True, False]
    assert silver["cust_code"].to_list() == ["CUST1", "CUST1", None]
    assert silver["seg_1"].to_list() == ["CT", "CT", None]
    # XX is dunnhumby's "unclassified" and stays as data.
    assert silver["basket_dominant_mission"].to_list() == ["Fresh", "Fresh", "Nonfood"]


def test_week_dates_are_joined_from_calendar(week_csv, calendar_csv):
    bronze = normalize_transactions(week_csv(200607, "20060412").encode())
    silver, _ = transform(bronze, normalize_calendar(calendar_csv.encode()))
    assert silver["week_start"].unique().to_list() == [date(2006, 4, 10)]
    assert silver["week_end"].unique().to_list() == [date(2006, 4, 16)]


def test_exact_duplicates_are_dropped_and_counted(week_csv, calendar_csv):
    text = week_csv(200607, "20060412")
    doubled = text + text.split("\n", 1)[1]
    bronze = normalize_transactions(doubled.encode())
    silver, dropped = transform(bronze, normalize_calendar(calendar_csv.encode()))
    assert bronze.height == 6
    assert silver.height == 3
    assert dropped == 3


def test_build_writes_weeks_and_manifests(ingested, store):
    report = build_silver(ingested.name, store)

    assert report.ok
    assert report.processed == [200607, 200609]
    frame = silver_frame(store, ingested, 200607)
    manifest = SilverManifest.model_validate_json(
        store.read_bytes(silver_manifest_key(ingested.name, 200607))
    )
    assert frame.height == manifest.rows_out == manifest.rows_in == 3
    assert manifest.spend_sum == Decimal("9.50")
    assert manifest.duplicates_dropped == 0


def test_second_build_skips_current_weeks(ingested, store):
    build_silver(ingested.name, store)
    again = build_silver(ingested.name, store)
    assert again.skipped == [200607, 200609]
    assert again.processed == []


def test_reingested_bronze_triggers_silver_rebuild(make_zip, store):
    source = make_zip()
    ingest_source(source, store, sleep=lambda _: None)
    build_silver(source.name, store)

    store.write_bytes(
        bronze_key(source.name, 200607), store.read_bytes(bronze_key(source.name, 200607))
    )
    (store.root / manifest_key(source.name, 200607)).unlink()
    ingest_source(source, store, weeks={200607}, sleep=lambda _: None)

    report = build_silver(source.name, store)
    assert report.processed == [200607]
    assert report.skipped == [200609]


def test_weeks_filter(ingested, store):
    report = build_silver(ingested.name, store, weeks={200609})
    assert report.processed == [200609]
    assert not store.exists(silver_key(ingested.name, 200607))


def test_missing_calendar_is_an_error(store):
    with pytest.raises(FileNotFoundError, match="Bronze ingestion"):
        build_silver("nothing-here", store)
