"""Bronze -> Silver: cleaned, analysis-ready weekly Parquet.

Silver changes, all lossless or counted:
  * blank cust_code / seg_1 / seg_2 become null and `is_loyalty` is derived (no-card baskets
    have no customer);
  * exact duplicate rows are dropped and counted in the manifest, never silently;
  * week start/end dates are joined from the calendar.
`XX` placeholders in basket attributes are kept as-is: they are dunnhumby's "unclassified".

Layout mirrors Bronze: silver/source=<name>/shop_week=<YYYYWW>/part-0.parquet plus
silver/_manifests/<name>/<YYYYWW>.json. A Silver week is rebuilt when its Bronze week was
re-ingested (the Bronze manifest timestamp is recorded in the Silver manifest).
"""

import io
import logging
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
from pydantic import BaseModel

from demandpulse.ingest.bronze import WeekManifest, bronze_key, calendar_key
from demandpulse.storage import Store

logger = logging.getLogger(__name__)

NULLABLE_BLANK_COLUMNS = ("cust_code", "seg_1", "seg_2")


def silver_key(source: str, week: int) -> str:
    return f"silver/source={source}/shop_week={week}/part-0.parquet"


def silver_manifest_key(source: str, week: int) -> str:
    return f"silver/_manifests/{source}/{week}.json"


def bronze_manifest_prefix(source: str) -> str:
    return f"bronze/_manifests/{source}/"


class SilverManifest(BaseModel):
    source: str
    shop_week: int
    parquet_key: str
    bronze_ingested_at: datetime
    rows_in: int
    rows_out: int
    duplicates_dropped: int
    duplicate_quantity_dropped: int = 0
    duplicate_spend_dropped: Decimal = Decimal(0)
    quantity_sum: int
    spend_sum: Decimal
    built_at: datetime


class SilverReport(BaseModel):
    source: str
    processed: list[int] = []
    skipped: list[int] = []
    failed: dict[int, str] = {}

    @property
    def ok(self) -> bool:
        return not self.failed


def _decimal(value: object) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal(0)


def _blank_to_null(column: str) -> pl.Expr:
    stripped = pl.col(column).str.strip_chars()
    return pl.when(stripped == "").then(None).otherwise(stripped).alias(column)


def transform(bronze: pl.DataFrame, calendar: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Return the Silver frame and the number of exact duplicate rows removed."""
    deduped = bronze.unique(maintain_order=True)
    dropped = bronze.height - deduped.height
    weeks = calendar.select(
        "shop_week",
        pl.col("date_from").alias("week_start"),
        pl.col("date_to").alias("week_end"),
    )
    silver = (
        deduped.with_columns([_blank_to_null(c) for c in NULLABLE_BLANK_COLUMNS])
        .with_columns(is_loyalty=pl.col("cust_code").is_not_null())
        .join(weeks, on="shop_week", how="left")
    )
    return silver, dropped


def _read_parquet(store: Store, key: str) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(store.read_bytes(key)))


def _is_current(store: Store, source: str, week: int, bronze: WeekManifest) -> bool:
    key = silver_manifest_key(source, week)
    if not store.exists(key):
        return False
    manifest = SilverManifest.model_validate_json(store.read_bytes(key))
    return manifest.bronze_ingested_at == bronze.ingested_at and store.exists(manifest.parquet_key)


def build_silver(source: str, store: Store, *, weeks: set[int] | None = None) -> SilverReport:
    report = SilverReport(source=source)
    if not store.exists(calendar_key(source)):
        raise FileNotFoundError(f"No Bronze calendar for {source}; run Bronze ingestion first")
    calendar = _read_parquet(store, calendar_key(source))

    manifests = [
        WeekManifest.model_validate_json(store.read_bytes(key))
        for key in store.list_keys(bronze_manifest_prefix(source))
    ]
    for bronze in sorted(manifests, key=lambda m: m.shop_week):
        week = bronze.shop_week
        if weeks is not None and week not in weeks:
            continue
        if _is_current(store, source, week, bronze):
            report.skipped.append(week)
            continue
        try:
            frame = _read_parquet(store, bronze_key(source, week))
            silver, dropped = transform(frame, calendar)
            buffer = io.BytesIO()
            silver.write_parquet(buffer, compression="zstd")
            key = silver_key(source, week)
            store.write_bytes(key, buffer.getvalue())
            spend = silver["spend"].sum()
            bronze_spend = frame["spend"].sum()
            manifest = SilverManifest(
                source=source,
                shop_week=week,
                parquet_key=key,
                bronze_ingested_at=bronze.ingested_at,
                rows_in=frame.height,
                rows_out=silver.height,
                duplicates_dropped=dropped,
                duplicate_quantity_dropped=int(frame["quantity"].sum() - silver["quantity"].sum()),
                duplicate_spend_dropped=_decimal(bronze_spend) - _decimal(spend),
                quantity_sum=int(silver["quantity"].sum()),
                spend_sum=_decimal(spend),
                built_at=datetime.now(UTC),
            )
            store.write_bytes(
                silver_manifest_key(source, week), manifest.model_dump_json().encode()
            )
        except (OSError, ValueError, pl.exceptions.PolarsError) as err:
            logger.error("%s week %d Silver build failed: %s", source, week, err)
            report.failed[week] = f"{type(err).__name__}: {err}"
            continue
        if dropped:
            logger.warning("%s week %d: dropped %d exact duplicate rows", source, week, dropped)
        report.processed.append(week)
    return report
