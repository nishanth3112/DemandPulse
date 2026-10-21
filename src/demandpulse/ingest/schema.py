"""Canonical Bronze schema and CSV normalization for dunnhumby transaction files."""

import io

import polars as pl

TRANSACTION_SCHEMA: dict[str, pl.DataType] = {
    "shop_week": pl.Int32(),
    "shop_date": pl.Date(),
    "shop_weekday": pl.Int8(),
    "shop_hour": pl.Int8(),
    "quantity": pl.Int32(),
    # Exact decimals so Bronze/Silver reconciliation does not drift on float rounding.
    "spend": pl.Decimal(12, 2),
    "prod_code": pl.String(),
    "prod_code_10": pl.String(),
    "prod_code_20": pl.String(),
    "prod_code_30": pl.String(),
    "prod_code_40": pl.String(),
    "cust_code": pl.String(),
    "seg_1": pl.String(),
    "seg_2": pl.String(),
    "basket_id": pl.Int64(),
    "basket_size": pl.String(),
    "basket_price_sensitivity": pl.String(),
    "basket_type": pl.String(),
    "basket_dominant_mission": pl.String(),
    "store_code": pl.String(),
    "store_format": pl.String(),
    "store_region": pl.String(),
}

CALENDAR_SCHEMA: dict[str, pl.DataType] = {
    "shop_week": pl.Int32(),
    "date_from": pl.Date(),
    "date_to": pl.Date(),
}

DATE_FORMAT = "%Y%m%d"
DATE_COLUMNS = {"shop_date", "date_from", "date_to"}


class SchemaError(ValueError):
    pass


def _read_all_strings(csv_bytes: bytes) -> pl.DataFrame:
    # Everything is read as text first so a bad value fails a strict cast with a clear
    # message instead of being silently inferred as a different type.
    frame = pl.read_csv(io.BytesIO(csv_bytes), infer_schema_length=0)
    return frame.rename(str.lower)


def _conform(frame: pl.DataFrame, schema: dict[str, pl.DataType], what: str) -> pl.DataFrame:
    missing = [c for c in schema if c not in frame.columns]
    unexpected = [c for c in frame.columns if c not in schema]
    if missing or unexpected:
        raise SchemaError(f"{what}: missing columns {missing}, unexpected columns {unexpected}")
    casts = [
        pl.col(name).str.strptime(pl.Date, DATE_FORMAT, strict=True)
        if name in DATE_COLUMNS
        else pl.col(name).cast(dtype, strict=True)
        for name, dtype in schema.items()
    ]
    return frame.select(casts)


def normalize_transactions(csv_bytes: bytes) -> pl.DataFrame:
    return _conform(_read_all_strings(csv_bytes), TRANSACTION_SCHEMA, "transactions")


def normalize_calendar(csv_bytes: bytes) -> pl.DataFrame:
    return _conform(_read_all_strings(csv_bytes), CALENDAR_SCHEMA, "time")
