import zipfile
from pathlib import Path

import pytest

from demandpulse.ingest.sources import Source

HEADER = (
    "SHOP_WEEK,SHOP_DATE,SHOP_WEEKDAY,SHOP_HOUR,QUANTITY,SPEND,PROD_CODE,PROD_CODE_10,"
    "PROD_CODE_20,PROD_CODE_30,PROD_CODE_40,CUST_CODE,seg_1,seg_2,BASKET_ID,BASKET_SIZE,"
    "BASKET_PRICE_SENSITIVITY,BASKET_TYPE,BASKET_DOMINANT_MISSION,STORE_CODE,STORE_FORMAT,"
    "STORE_REGION"
)
CALENDAR = (
    "shop_week,date_from,date_to\n"
    "200607,20060410,20060416\n"
    "200608,20060417,20060423\n"
    "200609,20060424,20060430\n"
)


def transactions_csv(week: int, day: str) -> str:
    """Two baskets: one loyalty (2 lines), one anonymous (1 line)."""
    rows = [
        f"{week},{day},6,8,2,3.50,PRD1,CL1,DEP1,G1,D1,CUST1,CT,CZ,{week}001,L,MM,Top Up,Fresh,"
        "STORE00001,LS,E02",
        f"{week},{day},6,8,1,1.25,PRD2,CL2,DEP1,G1,D1,CUST1,CT,CZ,{week}001,L,MM,Top Up,Fresh,"
        "STORE00001,LS,E02",
        f"{week},{day},7,21,3,4.75,PRD1,CL1,DEP1,G1,D1,,,,{week}002,S,UM,Small Shop,Nonfood,"
        "STORE00002,SS,W01",
    ]
    return "\n".join([HEADER, *rows]) + "\n"


@pytest.fixture
def make_zip(tmp_path):
    """Build a dunnhumby-shaped zip and return a Source pointing at it via file:// URL."""

    def _make(
        name: str = "fixture-sample",
        weeks: dict[int, str] | None = None,
        calendar: str | None = CALENDAR,
    ) -> Source:
        weeks = weeks or {200607: "20060412", 200609: "20060428"}
        path: Path = tmp_path / f"{name}.zip"
        prefix = f"dunnhumby_{name}"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            if calendar is not None:
                zf.writestr(f"{prefix}/time.csv", calendar)
            for week, day in weeks.items():
                zf.writestr(f"{prefix}/transactions_{week}.csv", transactions_csv(week, day))
        return Source(name=name, kind="sample", url=path.as_uri())

    return _make


@pytest.fixture
def week_csv():
    return transactions_csv


@pytest.fixture
def calendar_csv():
    return CALENDAR
