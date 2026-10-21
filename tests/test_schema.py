from datetime import date
from decimal import Decimal

import polars as pl
import pytest

from demandpulse.ingest.schema import (
    TRANSACTION_SCHEMA,
    SchemaError,
    normalize_calendar,
    normalize_transactions,
)

# Headers copied from the real files: UPPERCASE except seg_1 / seg_2.
HEADER = (
    "SHOP_WEEK,SHOP_DATE,SHOP_WEEKDAY,SHOP_HOUR,QUANTITY,SPEND,PROD_CODE,PROD_CODE_10,"
    "PROD_CODE_20,PROD_CODE_30,PROD_CODE_40,CUST_CODE,seg_1,seg_2,BASKET_ID,BASKET_SIZE,"
    "BASKET_PRICE_SENSITIVITY,BASKET_TYPE,BASKET_DOMINANT_MISSION,STORE_CODE,STORE_FORMAT,"
    "STORE_REGION"
)
LOYALTY_ROW = (
    "200609,20060428,6,8,4,6.16,PRD0901015,CL00201,DEP00067,G00021,D00005,CUST0000307323,"
    "CT,CZ,994100300352033,L,MM,Top Up,Fresh,STORE00001,LS,E02"
)
ANONYMOUS_ROW = (
    "200609,20060429,7,21,1,0.97,PRD0900830,CL00043,DEP00011,G00004,D00002,,,,"
    "994100300352999,S,XX,XX,XX,STORE00002,SS,W01"
)


def csv_of(*rows: str, header: str = HEADER) -> bytes:
    return ("\n".join([header, *rows]) + "\n").encode()


def test_normalizes_names_and_types():
    frame = normalize_transactions(csv_of(LOYALTY_ROW))
    assert frame.columns == list(TRANSACTION_SCHEMA)
    assert dict(frame.schema) == TRANSACTION_SCHEMA
    row = frame.row(0, named=True)
    assert row["shop_week"] == 200609
    assert row["shop_date"] == date(2006, 4, 28)
    assert row["spend"] == Decimal("6.16")
    assert row["basket_id"] == 994100300352033


def test_blank_customer_fields_are_preserved_as_raw_values_in_bronze():
    frame = normalize_transactions(csv_of(ANONYMOUS_ROW))
    row = frame.row(0, named=True)
    # Bronze stays faithful to the source; blank -> null happens in Silver.
    assert row["cust_code"] in ("", None)
    assert row["store_code"] == "STORE00002"


def test_missing_column_raises_schema_error():
    header = HEADER.replace(",SPEND", "")
    with pytest.raises(SchemaError, match="missing columns"):
        normalize_transactions(csv_of(header=header))


def test_unexpected_column_raises_schema_error():
    with pytest.raises(SchemaError, match="unexpected columns"):
        normalize_transactions(csv_of(LOYALTY_ROW + ",extra", header=HEADER + ",EXTRA"))


def test_non_numeric_value_fails_loudly():
    bad = LOYALTY_ROW.replace(",6.16,", ",abc,")
    with pytest.raises(pl.exceptions.PolarsError):
        normalize_transactions(csv_of(bad))


def test_invalid_date_fails_loudly():
    bad = LOYALTY_ROW.replace("20060428", "20061340")
    with pytest.raises(pl.exceptions.PolarsError):
        normalize_transactions(csv_of(bad))


def test_calendar_is_typed():
    frame = normalize_calendar(b"shop_week,date_from,date_to\n200607,20060410,20060416\n")
    assert frame.row(0, named=True) == {
        "shop_week": 200607,
        "date_from": date(2006, 4, 10),
        "date_to": date(2006, 4, 16),
    }
