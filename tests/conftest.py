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


@pytest.fixture(scope="session")
def silver_frame():
    from silver_factory import build_silver_frame

    return build_silver_frame()


@pytest.fixture
def silver_lake(tmp_path, silver_frame):
    """A local lake holding the synthetic Silver data; returns its root directory."""
    from silver_factory import write_silver_lake

    root = tmp_path / "lake"
    write_silver_lake(root, silver_frame)
    return root


class GoldHandle:
    """Query the exported Gold marts of a built fixture lake with plain SQL."""

    def __init__(self, root):
        import duckdb

        self.root = root
        self.con = duckdb.connect()
        gold = root / "gold"
        for item in sorted(gold.iterdir()):
            if item.suffix == ".parquet":
                self.con.execute(
                    f"CREATE VIEW {item.stem} AS SELECT * FROM read_parquet('{item.as_posix()}')"
                )
            elif item.is_dir():
                pattern = f"{item.as_posix()}/**/*.parquet"
                self.con.execute(
                    f"CREATE VIEW {item.name} AS "
                    f"SELECT * FROM read_parquet('{pattern}', hive_partitioning=true)"
                )

    def sql(self, query: str):
        return self.con.execute(query).fetchall()

    def one(self, query: str):
        return self.con.execute(query).fetchone()[0]


@pytest.fixture(scope="session")
def gold(tmp_path_factory, silver_frame):
    """Build every Gold mart once over the synthetic Silver data (window 2, year 4)."""
    from demandpulse.warehouse.build import GoldConfig, build_gold
    from silver_factory import write_silver_lake

    root = tmp_path_factory.mktemp("gold_lake")
    write_silver_lake(root, silver_frame)
    config = GoldConfig(
        lake_root=str(root),
        source_glob="fixture",
        window_weeks=2,
        year_weeks=4,
        work_dir=tmp_path_factory.mktemp("gold_work"),
        threads=2,
        memory_limit="1GB",
    )
    build_gold(config)
    handle = GoldHandle(root)
    handle.config = config
    return handle


@pytest.fixture(scope="session")
def forecast_panel(gold):
    from demandpulse.forecasting.panel import build_panel

    return build_panel(gold.config)


@pytest.fixture(scope="session")
def app_snapshot_dir(tmp_path_factory):
    """Dashboard snapshot exported from a synthetic lake (real risk pipeline, fake Gold marts)."""
    from app_data import write_lake
    from demandpulse.app.export import export_snapshot

    store = write_lake(tmp_path_factory.mktemp("app_lake"))
    out = tmp_path_factory.mktemp("app_snapshot")
    export_snapshot(store, out)
    return out
