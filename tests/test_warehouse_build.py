import json

import duckdb
import pytest

from demandpulse.warehouse.build import (
    STEPS,
    Export,
    GoldConfig,
    Step,
    build_gold,
    connect,
    run_sql_file,
)


@pytest.fixture
def config(silver_lake, tmp_path):
    return GoldConfig(
        lake_root=str(silver_lake),
        source_glob="fixture",
        window_weeks=2,
        year_weeks=4,
        work_dir=tmp_path / "work",
        threads=2,
        memory_limit="1GB",
    )


def test_silver_view_reads_every_fixture_row(config, silver_frame):
    con = connect(config)
    run_sql_file(con, config, "00_silver_view.sql")
    rows, spend = con.execute("SELECT count(*), sum(spend) FROM silver").fetchone()
    assert rows == silver_frame.height
    assert spend == silver_frame["spend"].sum()


def test_build_writes_manifest_with_silver_totals(config, silver_frame):
    result = build_gold(config)

    manifest = json.loads((config.work_dir.parent / "lake" / "gold" / "_manifest.json").read_text())
    assert manifest["silver_rows"] == silver_frame.height
    assert manifest["silver_weeks"] == 8
    assert manifest["silver_first_week"] == 200607
    assert manifest["window_weeks"] == 2
    assert manifest["tables"] == result.tables
    assert result.tables["fct_basket"] == silver_frame["basket_id"].n_unique()


def test_exports_land_as_parquet(config, tmp_path):
    export_sql = tmp_path / "x.sql"
    export_sql.write_text("CREATE OR REPLACE TABLE t AS SELECT 1 AS a UNION ALL SELECT 2")

    result = build_gold(config, steps=(Step("t", str(export_sql), (Export("t"),)),))

    gold_file = config.work_dir.parent / "lake" / "gold" / "t.parquet"
    assert result.tables == {"t": 2}
    assert duckdb.sql(f"SELECT count(*) FROM '{gold_file}'").fetchone() == (2,)


def test_partitioned_export_writes_one_directory_per_value(config, tmp_path):
    export_sql = tmp_path / "p.sql"
    export_sql.write_text(
        "CREATE OR REPLACE TABLE p AS SELECT 1 AS k, 'a' AS v UNION ALL SELECT 2, 'b'"
    )

    build_gold(config, steps=(Step("p", str(export_sql), (Export("p", partition_by="k"),)),))

    gold_dir = config.work_dir.parent / "lake" / "gold" / "p"
    assert sorted(d.name for d in gold_dir.iterdir()) == ["k=1", "k=2"]


def test_only_filter_skips_other_steps(config, tmp_path):
    for name in ("a", "b"):
        (tmp_path / f"{name}.sql").write_text(f"CREATE OR REPLACE TABLE {name} AS SELECT 1 AS x")
    steps = tuple(Step(n, str(tmp_path / f"{n}.sql"), (Export(n),)) for n in ("a", "b"))

    result = build_gold(config, steps=steps, only={"b"})

    assert list(result.tables) == ["b"]


@pytest.mark.parametrize("glob", ["a;drop table x", "a b", "../x", "a'b"])
def test_unsafe_source_globs_are_rejected(tmp_path, glob):
    with pytest.raises(ValueError, match="Unsafe source glob"):
        GoldConfig(lake_root=str(tmp_path), source_glob=glob)


def test_window_must_fit_in_year(tmp_path):
    with pytest.raises(ValueError, match="window_weeks"):
        GoldConfig(lake_root=str(tmp_path), window_weeks=10, year_weeks=4)


def test_statement_splitter_ignores_semicolons_in_comments():
    from demandpulse.warehouse.build import _statements

    sql = "-- a note; with a semicolon\nCREATE TABLE a AS SELECT 1; -- trailing; note\nSELECT 2"
    assert _statements(sql) == ["CREATE TABLE a AS SELECT 1", "SELECT 2"]


def test_weekly_step_runs_once_per_week_against_that_weeks_silver_only(
    config, tmp_path, silver_frame
):
    init = tmp_path / "init.sql"
    init.write_text("CREATE OR REPLACE TABLE wk (shop_week INTEGER, n BIGINT)")
    weekly = tmp_path / "weekly.sql"
    weekly.write_text("INSERT INTO wk SELECT $week, count(*) FROM read_parquet('$silver_week')")
    step = Step("wk", str(init), (Export("wk"),), weekly_sql_file=str(weekly))

    result = build_gold(config, steps=(*STEPS[:2], step))

    gold_file = config.work_dir.parent / "lake" / "gold" / "wk.parquet"
    rows = dict(duckdb.sql(f"SELECT shop_week, n FROM '{gold_file}'").fetchall())
    expected = dict(silver_frame.group_by("shop_week").len().iter_rows())
    assert rows == expected
    assert result.tables["wk"] == 8


def test_resume_reuses_existing_tables_instead_of_recomputing(silver_lake, tmp_path):
    sql = tmp_path / "r.sql"
    sql.write_text("CREATE OR REPLACE TABLE r AS SELECT random() AS v")
    step = Step("r", str(sql), (Export("r"),))

    def build(resume: bool) -> float:
        cfg = GoldConfig(
            lake_root=str(silver_lake),
            source_glob="fixture",
            window_weeks=2,
            year_weeks=4,
            work_dir=tmp_path / "work",
            threads=2,
            memory_limit="1GB",
            resume=resume,
        )
        build_gold(cfg, steps=(step,))
        return duckdb.sql(f"SELECT v FROM '{silver_lake}/gold/r.parquet'").fetchone()[0]

    first = build(resume=False)
    assert build(resume=True) == first
    assert build(resume=False) != first
