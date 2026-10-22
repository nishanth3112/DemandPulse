import os

import duckdb
import polars as pl
import pytest

from demandpulse.warehouse.build import GoldConfig, build_gold
from demandpulse.warehouse.checks import CHECKS, Check, open_gold, read_manifest, run_checks
from silver_factory import write_silver_lake


@pytest.fixture
def built(silver_lake, tmp_path):
    config = GoldConfig(
        lake_root=str(silver_lake),
        source_glob="fixture",
        window_weeks=2,
        year_weeks=4,
        work_dir=tmp_path / "work",
        threads=2,
        memory_limit="1GB",
    )
    build_gold(config)
    return config


def failed(outcomes):
    return {o.name for o in outcomes if not o.passed}


def test_every_check_passes_on_a_clean_build(gold):
    outcomes = run_checks(gold.config)
    assert len(outcomes) == len(CHECKS)
    assert failed(outcomes) == set()


def test_check_names_are_unique():
    names = [c.name for c in CHECKS]
    assert len(names) == len(set(names))


def test_manifest_is_readable_through_duckdb(built):
    con = open_gold(built)
    manifest = read_manifest(con, built)
    assert manifest["silver_weeks"] == 8
    assert manifest["window_weeks"] == 2


def test_tampered_store_week_revenue_is_caught(built):
    path = f"{built.gold_prefix}/fct_store_week.parquet"
    duckdb.execute(
        f"COPY (SELECT * REPLACE (revenue + 1 AS revenue) FROM '{path}') "
        f"TO '{path}.new' (FORMAT parquet)"
    )
    os.replace(f"{path}.new", path)

    assert "store_week_revenue_equals_silver_spend" in failed(run_checks(built))


def test_missing_basket_rows_are_caught_by_several_checks(built):
    partition = next(
        (built.work_dir.parent / "lake" / "gold" / "fct_basket").glob("*/data_0.parquet")
    )
    duckdb.execute(
        f"COPY (SELECT * FROM '{partition}' LIMIT 5) TO '{partition}.new' (FORMAT parquet)"
    )
    os.replace(f"{partition}.new", partition)

    caught = failed(run_checks(built))
    assert "basket_lines_equal_silver_rows" in caught
    assert "basket_revenue_equals_silver_spend" in caught
    assert "store_week_baskets_equal_basket_count" in caught


def test_a_check_that_cannot_run_counts_as_a_failure(built):
    broken = (Check("broken", "SELECT * FROM table_that_does_not_exist"),)
    [outcome] = run_checks(built, broken)
    assert not outcome.passed
    assert outcome.violations == -1
    assert "table_that_does_not_exist" in outcome.error


def test_inconsistent_basket_attributes_are_detected(silver_frame, tmp_path):
    # Change the hour on one line of a multi-line basket so min() and max() disagree.
    indexed = silver_frame.with_row_index()
    sizes = indexed.group_by("basket_id").len().filter(pl.col("len") > 1).sort("basket_id")
    basket = sizes["basket_id"][0]
    target_line = indexed.filter(pl.col("basket_id") == basket)["index"][0]
    mutated = indexed.with_columns(
        shop_hour=pl.when(pl.col("index") == target_line)
        .then(((pl.col("shop_hour") + 1) % 24).cast(pl.Int8))
        .otherwise(pl.col("shop_hour"))
    ).drop("index")
    root = tmp_path / "lake"
    write_silver_lake(root, mutated)
    config = GoldConfig(
        lake_root=str(root),
        source_glob="fixture",
        window_weeks=2,
        year_weeks=4,
        work_dir=tmp_path / "work",
        threads=2,
        memory_limit="1GB",
    )
    build_gold(config)

    outcomes = {o.name: o for o in run_checks(config)}

    assert not outcomes["baskets_have_constant_attributes"].passed
    assert outcomes["baskets_have_constant_attributes"].violations == 1
