"""Stream source zips into Bronze Parquet, one weekly file at a time.

Layout (under the configured storage root):
    bronze/source=<name>/shop_week=<YYYYWW>/part-0.parquet
    bronze/source=<name>/calendar.parquet
    bronze/_manifests/<name>/transactions_<YYYYWW>.json   (written last; marks the week done)

Sources are kept apart because samples and full parts overlap on weeks but not on rows.
A week is skipped when its manifest records the same CRC as the zip member, so interrupted
or repeated runs resume without re-downloading finished weeks.
"""

import io
import logging
import time
import zipfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from decimal import Decimal

import aiohttp
import polars as pl
from pydantic import BaseModel

from demandpulse.ingest.remote_zip import Member, list_members, open_archive, read_member
from demandpulse.ingest.schema import normalize_calendar, normalize_transactions
from demandpulse.ingest.sources import Source
from demandpulse.storage import Store

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 2.0
PARQUET_COMPRESSION = "zstd"
RETRYABLE = (OSError, TimeoutError, aiohttp.ClientError, zipfile.BadZipFile)


def bronze_key(source: str, week: int) -> str:
    return f"bronze/source={source}/shop_week={week}/part-0.parquet"


def calendar_key(source: str) -> str:
    return f"bronze/source={source}/calendar.parquet"


def manifest_key(source: str, week: int) -> str:
    return f"bronze/_manifests/{source}/transactions_{week}.json"


class WeekManifest(BaseModel):
    source: str
    source_url: str
    member: str
    shop_week: int
    csv_crc: int
    csv_bytes: int
    parquet_key: str
    parquet_bytes: int
    rows: int
    baskets: int
    quantity_sum: int
    spend_sum: Decimal
    ingested_at: datetime


class IngestReport(BaseModel):
    source: str
    processed: list[int] = []
    skipped: list[int] = []
    failed: dict[int, str] = {}

    @property
    def ok(self) -> bool:
        return not self.failed


def _with_retries[T](fn: Callable[[], T], what: str, sleep: Callable[[float], None]) -> T:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except RETRYABLE as err:
            if attempt == MAX_ATTEMPTS:
                raise
            delay = RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1)
            logger.warning(
                "%s failed (%s); retry %d/%d in %.0fs", what, err, attempt, MAX_ATTEMPTS, delay
            )
            sleep(delay)
    raise AssertionError("unreachable")


def _already_ingested(store: Store, source: Source, member: Member) -> bool:
    assert member.shop_week is not None
    key = manifest_key(source.name, member.shop_week)
    if not store.exists(key):
        return False
    manifest = WeekManifest.model_validate_json(store.read_bytes(key))
    return manifest.csv_crc == member.crc and store.exists(manifest.parquet_key)


def _parquet_bytes(frame: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.write_parquet(buffer, compression=PARQUET_COMPRESSION)
    return buffer.getvalue()


def _ingest_week(
    archive: zipfile.ZipFile,
    member: Member,
    source: Source,
    store: Store,
    sleep: Callable[[float], None],
) -> WeekManifest:
    assert member.shop_week is not None
    raw = _with_retries(lambda: read_member(archive, member), member.name, sleep)
    frame = normalize_transactions(raw)

    weeks_in_file = frame["shop_week"].unique().to_list()
    if weeks_in_file != [member.shop_week]:
        raise ValueError(f"{member.name}: contains shop_week values {weeks_in_file}")

    payload = _parquet_bytes(frame)
    key = bronze_key(source.name, member.shop_week)
    store.write_bytes(key, payload)

    spend_sum = frame["spend"].sum()
    manifest = WeekManifest(
        source=source.name,
        source_url=source.url,
        member=member.name,
        shop_week=member.shop_week,
        csv_crc=member.crc,
        csv_bytes=member.size,
        parquet_key=key,
        parquet_bytes=len(payload),
        rows=frame.height,
        baskets=frame["basket_id"].n_unique(),
        quantity_sum=int(frame["quantity"].sum()),
        spend_sum=Decimal(spend_sum) if spend_sum is not None else Decimal(0),
        ingested_at=datetime.now(UTC),
    )
    # Written last: a crash before this line means the week is simply redone next run.
    store.write_bytes(
        manifest_key(source.name, member.shop_week), manifest.model_dump_json().encode()
    )
    return manifest


def _ingest_calendar(
    archive: zipfile.ZipFile,
    member: Member,
    source: Source,
    store: Store,
    sleep: Callable[[float], None],
) -> None:
    raw = _with_retries(lambda: read_member(archive, member), member.name, sleep)
    frame = normalize_calendar(raw)
    store.write_bytes(calendar_key(source.name), _parquet_bytes(frame))


def ingest_source(
    source: Source,
    store: Store,
    *,
    weeks: set[int] | None = None,
    max_weeks: int | None = None,
    opener: Callable[[str], AbstractContextManager[zipfile.ZipFile]] = open_archive,
    sleep: Callable[[float], None] = time.sleep,
) -> IngestReport:
    report = IngestReport(source=source.name)
    with opener(source.url) as archive:
        members = list_members(archive)
        for member in (m for m in members if m.kind == "calendar"):
            _ingest_calendar(archive, member, source, store, sleep)

        todo = [m for m in members if m.kind == "transactions"]
        if weeks is not None:
            todo = [m for m in todo if m.shop_week in weeks]

        for member in todo:
            week = member.shop_week
            assert week is not None
            if _already_ingested(store, source, member):
                report.skipped.append(week)
                continue
            if max_weeks is not None and len(report.processed) >= max_weeks:
                break
            started = time.monotonic()
            try:
                manifest = _ingest_week(archive, member, source, store, sleep)
            except (ValueError, pl.exceptions.PolarsError, *RETRYABLE) as err:
                # Record and continue: one bad week must not abort the other 100+.
                logger.error("%s week %d failed: %s", source.name, week, err)
                report.failed[week] = f"{type(err).__name__}: {err}"
                continue
            report.processed.append(week)
            logger.info(
                "%s week %d: %d rows, %.1f MB parquet, %.1fs",
                source.name,
                week,
                manifest.rows,
                manifest.parquet_bytes / 1e6,
                time.monotonic() - started,
            )
    return report
