import io
import zipfile
from contextlib import contextmanager
from decimal import Decimal

import polars as pl
import pytest

from demandpulse.ingest.bronze import (
    WeekManifest,
    bronze_key,
    calendar_key,
    ingest_source,
    manifest_key,
)
from demandpulse.ingest.remote_zip import open_archive
from demandpulse.storage import LocalStore


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "lake")


def no_sleep(_seconds: float) -> None:
    return None


def test_ingests_every_week_and_the_calendar(make_zip, store):
    source = make_zip()
    report = ingest_source(source, store, sleep=no_sleep)

    assert report.ok
    assert report.processed == [200607, 200609]
    assert store.exists(calendar_key(source.name))
    for week in (200607, 200609):
        assert store.exists(bronze_key(source.name, week))
        assert store.exists(manifest_key(source.name, week))


def test_parquet_content_and_manifest_agree(make_zip, store):
    source = make_zip()
    ingest_source(source, store, sleep=no_sleep)

    frame = pl.read_parquet(io.BytesIO(store.read_bytes(bronze_key(source.name, 200607))))
    manifest = WeekManifest.model_validate_json(store.read_bytes(manifest_key(source.name, 200607)))

    assert frame.height == manifest.rows == 3
    assert manifest.baskets == 2
    assert manifest.quantity_sum == 6
    assert manifest.spend_sum == Decimal("9.50")
    assert frame["spend"].sum() == manifest.spend_sum
    assert manifest.source_url == source.url


def test_second_run_skips_finished_weeks_without_reading_them(make_zip, store):
    source = make_zip()
    ingest_source(source, store, sleep=no_sleep)

    reads: list[str] = []

    @contextmanager
    def counting_opener(url):
        with open_archive(url) as archive:
            original = archive.read

            def spy(name, *args, **kwargs):
                reads.append(name)
                return original(name, *args, **kwargs)

            archive.read = spy  # type: ignore[method-assign]
            yield archive

    report = ingest_source(source, store, opener=counting_opener, sleep=no_sleep)

    assert report.skipped == [200607, 200609]
    assert report.processed == []
    assert not any("transactions_" in name for name in reads)


def test_interrupted_run_resumes_where_it_stopped(make_zip, store):
    source = make_zip()
    first = ingest_source(source, store, max_weeks=1, sleep=no_sleep)
    assert first.processed == [200607]

    second = ingest_source(source, store, sleep=no_sleep)
    assert second.skipped == [200607]
    assert second.processed == [200609]


def test_changed_source_data_is_reingested(make_zip, store):
    source = make_zip()
    ingest_source(source, store, sleep=no_sleep)

    changed = make_zip(weeks={200607: "20060413", 200609: "20060428"})
    report = ingest_source(changed, store, sleep=no_sleep)

    assert report.processed == [200607]
    assert report.skipped == [200609]


def test_missing_manifest_means_week_is_redone(make_zip, store):
    source = make_zip()
    ingest_source(source, store, sleep=no_sleep)
    (store.root / manifest_key(source.name, 200609)).unlink()

    report = ingest_source(source, store, sleep=no_sleep)
    assert report.processed == [200609]


def test_bad_week_is_reported_and_does_not_stop_others(make_zip, store, tmp_path):
    source = make_zip()
    path = tmp_path / "fixture-sample.zip"
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("dunnhumby_x/transactions_200611.csv", "NOT,A,VALID,FILE\n1,2,3,4\n")

    report = ingest_source(source, store, sleep=no_sleep)

    assert report.processed == [200607, 200609]
    assert list(report.failed) == [200611]
    assert "SchemaError" in report.failed[200611]
    assert not report.ok
    assert not store.exists(manifest_key(source.name, 200611))


def test_week_filter_limits_work(make_zip, store):
    report = ingest_source(make_zip(), store, weeks={200609}, sleep=no_sleep)
    assert report.processed == [200609]


def test_file_containing_a_different_week_is_rejected(make_zip, store, tmp_path, week_csv):
    source = make_zip(weeks={200607: "20060412"})
    path = tmp_path / "fixture-sample.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("p/transactions_200607.csv", week_csv(200610, "20060501"))

    report = ingest_source(source, store, sleep=no_sleep)
    assert "shop_week values" in report.failed[200607]


def test_transient_read_errors_are_retried(make_zip, store):
    source = make_zip(weeks={200607: "20060412"})
    attempts = {"n": 0}
    delays: list[float] = []

    @contextmanager
    def flaky_opener(url):
        with open_archive(url) as archive:
            original = archive.read

            def flaky(name, *args, **kwargs):
                if "transactions_" in name:
                    attempts["n"] += 1
                    if attempts["n"] < 3:
                        raise ConnectionResetError("boom")
                return original(name, *args, **kwargs)

            archive.read = flaky  # type: ignore[method-assign]
            yield archive

    report = ingest_source(source, store, opener=flaky_opener, sleep=delays.append)

    assert report.processed == [200607]
    assert attempts["n"] == 3
    assert delays == [2.0, 4.0]


def test_persistent_read_errors_are_reported_not_hidden(make_zip, store):
    source = make_zip(weeks={200607: "20060412"})

    @contextmanager
    def dead_opener(url):
        with open_archive(url) as archive:
            original = archive.read

            def dead(name, *args, **kwargs):
                if "transactions_" in name:
                    raise ConnectionResetError("always")
                return original(name, *args, **kwargs)

            archive.read = dead  # type: ignore[method-assign]
            yield archive

    report = ingest_source(source, store, opener=dead_opener, sleep=no_sleep)
    assert "ConnectionResetError" in report.failed[200607]


def test_calendar_read_is_retried_too(make_zip, store):
    source = make_zip(weeks={200607: "20060412"})
    attempts = {"n": 0}

    @contextmanager
    def flaky_opener(url):
        with open_archive(url) as archive:
            original = archive.read

            def flaky(name, *args, **kwargs):
                if name.endswith("time.csv"):
                    attempts["n"] += 1
                    if attempts["n"] == 1:
                        raise ConnectionResetError("boom")
                return original(name, *args, **kwargs)

            archive.read = flaky  # type: ignore[method-assign]
            yield archive

    report = ingest_source(source, store, opener=flaky_opener, sleep=no_sleep)

    assert report.ok
    assert attempts["n"] == 2
    assert store.exists(calendar_key(source.name))
