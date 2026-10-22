"""Build the Gold layer: SQL marts computed by DuckDB over Silver Parquet.

Each step is a SQL file in `sql/` that creates tables in a scratch DuckDB database. Selected
tables are then exported to `<lake>/gold/<table>.parquet` (or a partitioned directory for the
largest fact). `<lake>` is an S3 URL or a local directory; DuckDB reads and writes both, so the
same SQL runs against real data and against small test fixtures.

SQL files use `$name` placeholders (string.Template) for parameters, never user input.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from string import Template

import duckdb

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).parent / "sql"
DEFAULT_WORK_DIR = Path("data/work")


@dataclass(frozen=True)
class Export:
    table: str
    partition_by: str | None = None


@dataclass(frozen=True)
class Step:
    name: str
    sql_file: str
    exports: tuple[Export, ...] = ()
    # When set, sql_file only creates the (empty) tables and this file is run once per week
    # with $week and $silver_week filled in, keeping memory bounded on the full dataset.
    weekly_sql_file: str | None = None


# Always run first: every other step and the manifest read the `silver` view.
SILVER_VIEW = Step("silver_view", "00_silver_view.sql")

# Order matters: later steps read tables created by earlier ones.
STEPS: tuple[Step, ...] = (
    Step("fct_basket", "10_fct_basket.sql", (Export("fct_basket", partition_by="shop_week"),)),
    Step(
        "dimensions",
        "11_dimensions.sql",
        (
            Export("dim_date"),
            Export("dim_comparison"),
            Export("dim_comparison_week"),
            Export("dim_store"),
            Export("dim_customer"),
            Export("dim_product"),
        ),
    ),
    Step(
        "store_week_and_mix",
        "20_store_week_and_basket_marts.sql",
        (Export("fct_store_week"), Export("mart_basket_mix"), Export("mart_trading_pattern")),
    ),
    Step(
        "store_category_week",
        "30_store_category_week_init.sql",
        (Export("fct_store_category_week"),),
        weekly_sql_file="30_store_category_week_weekly.sql",
    ),
    Step(
        "category_and_region",
        "31_category_and_region_marts.sql",
        (
            Export("mart_category_benchmark"),
            Export("mart_region_benchmark"),
            Export("mart_region_category_mix"),
        ),
    ),
    Step("store_scorecard", "40_store_scorecard.sql", (Export("mart_store_scorecard"),)),
    Step(
        "customer_marts",
        "50_customer_marts.sql",
        (
            Export("mart_customer_rfm"),
            Export("mart_rfm_segment_summary"),
            Export("mart_customer_cohort"),
        ),
    ),
)


@dataclass(frozen=True)
class GoldConfig:
    lake_root: str
    source_glob: str = "full-part-*"
    window_weeks: int = 13
    year_weeks: int = 52
    aws_profile: str | None = None
    aws_region: str = "us-east-1"
    work_dir: Path = DEFAULT_WORK_DIR
    memory_limit: str = "8GB"
    threads: int = 8
    # Skip a step's SQL when its tables already exist in the scratch database. Only safe when
    # Silver has not changed since they were built; exports are always redone.
    resume: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_*.\-]+", self.source_glob):
            raise ValueError(f"Unsafe source glob: {self.source_glob!r}")
        if self.window_weeks < 1 or self.year_weeks < self.window_weeks:
            raise ValueError("Need 1 <= window_weeks <= year_weeks")

    @property
    def silver_pattern(self) -> str:
        return f"{self.lake_root}/silver/source={self.source_glob}/shop_week=*/part-0.parquet"

    def silver_week_pattern(self, week: int) -> str:
        source = f"{self.lake_root}/silver/source={self.source_glob}"
        return f"{source}/shop_week={int(week)}/part-0.parquet"

    @property
    def gold_prefix(self) -> str:
        return f"{self.lake_root}/gold"

    @property
    def uses_s3(self) -> bool:
        return self.lake_root.startswith("s3://")

    def template_values(self) -> dict[str, str | int]:
        return {
            "silver_pattern": self.silver_pattern,
            "window_weeks": self.window_weeks,
            "year_weeks": self.year_weeks,
        }


@dataclass
class GoldResult:
    tables: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)


def connect(config: GoldConfig) -> duckdb.DuckDBPyConnection:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.work_dir / "gold.duckdb"))
    con.execute(f"SET threads={int(config.threads)}")
    con.execute(f"SET memory_limit='{config.memory_limit}'")
    con.execute(f"SET temp_directory='{(config.work_dir / 'spill').as_posix()}'")
    # Row order does not matter for aggregates and unordered scans use less memory.
    con.execute("SET preserve_insertion_order=false")
    if config.uses_s3:
        con.execute("INSTALL httpfs; LOAD httpfs; INSTALL aws; LOAD aws")
        profile = f", PROFILE '{config.aws_profile}'" if config.aws_profile else ""
        con.execute(
            "CREATE OR REPLACE SECRET lake (TYPE s3, PROVIDER credential_chain, "
            f"CHAIN 'config;env', REGION '{config.aws_region}'{profile})"
        )
    return con


def _statements(sql: str) -> list[str]:
    # Comments are stripped first so a ';' inside one cannot split a statement. SQL files never
    # contain '--' or ';' inside string literals, so plain text handling is safe.
    without_comments = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in without_comments.split(";") if s.strip()]


def run_sql_file(con: duckdb.DuckDBPyConnection, config: GoldConfig, sql_file: str) -> None:
    text = Template((SQL_DIR / sql_file).read_text()).substitute(config.template_values())
    for statement in _statements(text):
        con.execute(statement)


def run_weekly_sql(con: duckdb.DuckDBPyConnection, config: GoldConfig, sql_file: str) -> None:
    template = Template((SQL_DIR / sql_file).read_text())
    weeks = [row[0] for row in con.execute("SELECT shop_week FROM dim_date ORDER BY 1").fetchall()]
    for week in weeks:
        values = {
            **config.template_values(),
            "week": int(week),
            "silver_week": config.silver_week_pattern(week),
        }
        for statement in _statements(template.substitute(values)):
            con.execute(statement)
    logger.info("weekly SQL %s ran for %d weeks", sql_file, len(weeks))


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    found = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    return bool(found and found[0])


def export_table(con: duckdb.DuckDBPyConnection, config: GoldConfig, export: Export) -> int:
    rows = con.execute(f"SELECT count(*) FROM {export.table}").fetchone()
    count = int(rows[0]) if rows else 0
    if not config.uses_s3:
        Path(config.gold_prefix).mkdir(parents=True, exist_ok=True)
    options = "FORMAT parquet, COMPRESSION zstd"
    if not export.partition_by:
        target = f"{config.gold_prefix}/{export.table}.parquet"
        con.execute(f"COPY {export.table} TO '{target}' ({options})")
        return count

    # One partition at a time: DuckDB's own PARTITION_BY keeps a writer open per partition
    # and ran out of memory on 117 weeks of the full dataset. The layout is the same hive
    # style (<table>/<col>=<value>/data_0.parquet) with the column kept only in the path.
    column = export.partition_by
    values = con.execute(f"SELECT DISTINCT {column} FROM {export.table} ORDER BY 1").fetchall()
    for (value,) in values:
        if not isinstance(value, int):
            raise TypeError(f"Partition column {column} must be integer, got {value!r}")
        target = f"{config.gold_prefix}/{export.table}/{column}={value}/data_0.parquet"
        if not config.uses_s3:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        con.execute(
            f"COPY (SELECT * EXCLUDE ({column}) FROM {export.table} WHERE {column} = {value}) "
            f"TO '{target}' ({options})"
        )
    return count


def build_gold(
    config: GoldConfig,
    steps: tuple[Step, ...] = STEPS,
    only: set[str] | None = None,
) -> GoldResult:
    con = connect(config)
    result = GoldResult()
    try:
        for step in (SILVER_VIEW, *steps):
            if only is not None and step.name not in only and step is not SILVER_VIEW:
                continue
            started = time.monotonic()
            can_resume = (
                config.resume
                and step is not SILVER_VIEW
                and step.exports
                and all(_table_exists(con, e.table) for e in step.exports)
            )
            if can_resume:
                logger.info("resume: reusing existing tables for step %s", step.name)
            else:
                run_sql_file(con, config, step.sql_file)
                if step.weekly_sql_file:
                    run_weekly_sql(con, config, step.weekly_sql_file)
            for export in step.exports:
                result.tables[export.table] = export_table(con, config, export)
            result.seconds[step.name] = round(time.monotonic() - started, 1)
            logger.info("gold step %s done in %.1fs", step.name, result.seconds[step.name])
        _write_manifest(con, config, result)
    finally:
        con.close()
    return result


def _write_manifest(con: duckdb.DuckDBPyConnection, config: GoldConfig, result: GoldResult) -> None:
    silver = con.execute(
        "SELECT count(*), sum(spend), min(shop_week), max(shop_week), count(DISTINCT shop_week) "
        "FROM silver"
    ).fetchone()
    manifest = {
        "built_at": datetime.now(UTC).isoformat(),
        "source_glob": config.source_glob,
        "window_weeks": config.window_weeks,
        "year_weeks": config.year_weeks,
        "silver_rows": silver[0] if silver else None,
        "silver_spend": str(silver[1]) if silver else None,
        "silver_first_week": silver[2] if silver else None,
        "silver_last_week": silver[3] if silver else None,
        "silver_weeks": silver[4] if silver else None,
        "tables": result.tables,
        "seconds": result.seconds,
    }
    target = f"{config.gold_prefix}/_manifest.json"
    con.execute("CREATE OR REPLACE TEMP TABLE _manifest AS SELECT ? AS doc", [json.dumps(manifest)])
    if not config.uses_s3:
        Path(config.gold_prefix).mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT doc FROM _manifest) TO '{target}' (FORMAT csv, HEADER false, QUOTE '')"
    )
