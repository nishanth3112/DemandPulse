from datetime import date
from decimal import Decimal

import polars as pl


def test_basket_fact_has_one_row_per_basket_and_reconciles_to_silver(gold, silver_frame):
    assert gold.one("SELECT count(*) FROM fct_basket") == silver_frame["basket_id"].n_unique()
    assert gold.one("SELECT count(DISTINCT basket_id) FROM fct_basket") == gold.one(
        "SELECT count(*) FROM fct_basket"
    )
    assert gold.one("SELECT sum(revenue) FROM fct_basket") == silver_frame["spend"].sum()
    assert gold.one("SELECT sum(units) FROM fct_basket") == silver_frame["quantity"].sum()
    assert gold.one("SELECT sum(lines) FROM fct_basket") == silver_frame.height


def test_basket_loyalty_flag_matches_customer_presence(gold):
    assert (
        gold.one("SELECT count(*) FROM fct_basket WHERE is_loyalty <> (cust_code IS NOT NULL)") == 0
    )


def test_dim_date_is_a_contiguous_week_index(gold):
    rows = gold.sql("SELECT shop_week, week_idx, period_idx, fiscal_week FROM dim_date ORDER BY 1")
    assert [r[1] for r in rows] == list(range(1, 9))
    assert [r[2] for r in rows] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert rows[0][0] == 200607
    assert rows[0][3] == 7
    assert gold.one("SELECT week_start FROM dim_date WHERE week_idx = 1") == date(2006, 4, 10)


def test_comparison_windows_follow_window_and_year_settings(gold):
    rows = gold.sql(
        "SELECT comparison, current_from, current_to, base_from, base_to "
        "FROM dim_comparison ORDER BY 1"
    )
    assert rows == [
        ("window_vs_prior", 7, 8, 5, 6),
        ("window_vs_year_ago", 7, 8, 3, 4),
        ("year_vs_prior_year", 5, 8, 1, 4),
    ]


def test_store_lifecycle_detects_openings_and_closures(gold):
    lifecycle = dict(gold.sql("SELECT store_code, lifecycle FROM dim_store"))
    assert lifecycle == {
        "STORE01": "open_throughout",
        "STORE02": "open_throughout",
        "STORE03": "opened_in_period",
        "STORE04": "closed_in_period",
    }
    assert gold.one("SELECT weeks_active FROM dim_store WHERE store_code = 'STORE03'") == 4


def test_customer_dimension_matches_loyalty_customers(gold, silver_frame):
    loyal = silver_frame.filter(pl.col("cust_code").is_not_null())
    assert gold.one("SELECT count(*) FROM dim_customer") == loyal["cust_code"].n_unique()
    first = loyal.group_by("cust_code").agg(pl.col("shop_week").min())
    expected = dict(zip(first["cust_code"], first["shop_week"], strict=True))
    actual = dict(gold.sql("SELECT cust_code, first_week FROM dim_customer"))
    assert actual == expected


def test_product_dimension_has_a_single_hierarchy_per_product(gold, silver_frame):
    assert gold.one("SELECT count(*) FROM dim_product") == silver_frame["prod_code"].n_unique()
    assert gold.one("SELECT prod_code_30 FROM dim_product WHERE prod_code = 'PRD05'") == "G2"
    assert gold.one("SELECT sum(revenue) FROM fct_basket") > Decimal(0)
