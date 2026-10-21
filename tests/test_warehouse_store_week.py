import polars as pl
import pytest


@pytest.fixture(scope="module")
def store_week_expected(silver_frame):
    baskets = silver_frame.group_by("basket_id").agg(
        pl.col("shop_week").first(),
        pl.col("store_code").first(),
        pl.col("cust_code").first(),
        pl.col("spend").sum().alias("revenue"),
        pl.col("quantity").sum().alias("units"),
        pl.len().alias("lines"),
    )
    return baskets.group_by("shop_week", "store_code").agg(
        pl.len().alias("baskets"),
        pl.col("revenue").sum(),
        pl.col("units").sum(),
        pl.col("lines").sum(),
        pl.col("cust_code").n_unique().alias("customers_incl_null"),
        pl.col("cust_code").null_count().alias("anonymous_baskets"),
    )


def test_store_week_grain_is_unique(gold):
    total = gold.one("SELECT count(*) FROM fct_store_week")
    distinct = gold.one(
        "SELECT count(*) FROM (SELECT DISTINCT shop_week, store_code FROM fct_store_week)"
    )
    assert total == distinct


def test_store_week_matches_an_independent_polars_aggregation(gold, store_week_expected):
    actual = {
        (week, store): (baskets, revenue, units, lines)
        for week, store, baskets, revenue, units, lines in gold.sql(
            "SELECT shop_week, store_code, baskets, revenue, units, lines FROM fct_store_week"
        )
    }
    expected = {
        (r["shop_week"], r["store_code"]): (r["baskets"], r["revenue"], r["units"], r["lines"])
        for r in store_week_expected.iter_rows(named=True)
    }
    assert actual == expected


def test_distinct_customers_exclude_anonymous_baskets(gold, store_week_expected):
    actual = {
        (w, s): c
        for w, s, c in gold.sql("SELECT shop_week, store_code, customers FROM fct_store_week")
    }
    for row in store_week_expected.iter_rows(named=True):
        has_anonymous = row["anonymous_baskets"] > 0
        expected = row["customers_incl_null"] - (1 if has_anonymous else 0)
        assert actual[(row["shop_week"], row["store_code"])] == expected


def test_ratio_kpis_follow_their_definitions(gold):
    bad = gold.one(
        """
        SELECT count(*) FROM fct_store_week
        WHERE abs(avg_basket_value - revenue / baskets) > 1e-9
           OR abs(units_per_basket - units * 1.0 / baskets) > 1e-9
           OR abs(lines_per_basket - lines * 1.0 / baskets) > 1e-9
           OR abs(avg_unit_price - revenue / units) > 1e-9
           OR abs(loyalty_basket_share - loyalty_baskets * 1.0 / baskets) > 1e-9
        """
    )
    assert bad == 0


def test_stores_only_appear_in_weeks_they_traded(gold):
    assert (
        gold.one("SELECT min(shop_week) FROM fct_store_week WHERE store_code = 'STORE03'") == 200611
    )
    assert (
        gold.one("SELECT max(shop_week) FROM fct_store_week WHERE store_code = 'STORE04'") == 200610
    )


def test_basket_mix_reconciles_to_baskets(gold):
    assert gold.one("SELECT sum(baskets) FROM mart_basket_mix") == gold.one(
        "SELECT count(*) FROM fct_basket"
    )
    assert gold.one("SELECT sum(revenue) FROM mart_basket_mix") == gold.one(
        "SELECT sum(revenue) FROM fct_basket"
    )
    keys = "shop_week, store_format, basket_type, basket_size, basket_dominant_mission, "
    keys += "basket_price_sensitivity, is_loyalty"
    assert gold.one("SELECT count(*) FROM mart_basket_mix") == gold.one(
        f"SELECT count(*) FROM (SELECT DISTINCT {keys} FROM mart_basket_mix)"
    )


def test_trading_pattern_shares_sum_to_one_per_format(gold, silver_frame):
    sums = gold.sql(
        "SELECT store_format, sum(revenue_share_of_format) FROM mart_trading_pattern GROUP BY 1"
    )
    assert {fmt for fmt, _ in sums} == set(silver_frame["store_format"].unique())
    assert all(total == pytest.approx(1.0) for _, total in sums)
    assert gold.one("SELECT sum(baskets) FROM mart_trading_pattern") == gold.one(
        "SELECT count(*) FROM fct_basket"
    )
