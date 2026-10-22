import polars as pl
import pytest

WINDOWS = {
    "window_vs_prior": ({7, 8}, {5, 6}),
    "window_vs_year_ago": ({7, 8}, {3, 4}),
    "year_vs_prior_year": ({5, 6, 7, 8}, {1, 2, 3, 4}),
}


def window_revenue(frame: pl.DataFrame, idxs: set[int], **filters) -> float:
    selected = frame.filter((pl.col("shop_week") - 200607 + 1).is_in(idxs))
    for column, value in filters.items():
        selected = selected.filter(pl.col(column) == value)
    return float(selected["spend"].sum())


def test_store_category_week_matches_independent_aggregation(gold, silver_frame):
    expected = silver_frame.group_by("shop_week", "store_code", "prod_code_30").agg(
        pl.col("spend").sum().alias("revenue"),
        pl.col("quantity").sum().alias("units"),
        pl.len().alias("lines"),
        pl.col("basket_id").n_unique().alias("baskets"),
    )
    expected_rows = {
        (r["shop_week"], r["store_code"], r["prod_code_30"]): (
            r["revenue"],
            r["units"],
            r["lines"],
            r["baskets"],
        )
        for r in expected.iter_rows(named=True)
    }
    actual_rows = {
        (w, s, c): (rev, units, lines, baskets)
        for w, s, c, rev, units, lines, baskets in gold.sql(
            "SELECT shop_week, store_code, prod_code_30, revenue, units, lines, baskets "
            "FROM fct_store_category_week"
        )
    }
    assert actual_rows == expected_rows


def test_category_revenue_reconciles_to_store_week_revenue(gold):
    assert gold.one("SELECT sum(revenue) FROM fct_store_category_week") == gold.one(
        "SELECT sum(revenue) FROM fct_store_week"
    )


@pytest.mark.parametrize("comparison", list(WINDOWS))
def test_category_shares_sum_to_one_in_both_windows(gold, comparison):
    cur, base = gold.sql(
        f"SELECT sum(current_share), sum(base_share) FROM mart_category_benchmark "
        f"WHERE comparison = '{comparison}'"
    )[0]
    assert cur == pytest.approx(1.0)
    assert base == pytest.approx(1.0)


@pytest.mark.parametrize("comparison", list(WINDOWS))
def test_category_growth_matches_independent_calculation(gold, silver_frame, comparison):
    cur_idx, base_idx = WINDOWS[comparison]
    actual = {
        cat: (cur, base, growth)
        for cat, cur, base, growth in gold.sql(
            f"SELECT prod_code_30, current_revenue, base_revenue, revenue_growth "
            f"FROM mart_category_benchmark WHERE comparison = '{comparison}'"
        )
    }
    for category in silver_frame["prod_code_30"].unique():
        cur = window_revenue(silver_frame, cur_idx, prod_code_30=category)
        base = window_revenue(silver_frame, base_idx, prod_code_30=category)
        assert actual[category][0] == pytest.approx(cur)
        assert actual[category][1] == pytest.approx(base)
        assert actual[category][2] == pytest.approx(cur / base - 1)


def test_category_penetration_is_a_share_of_all_baskets(gold):
    assert gold.one("SELECT max(current_penetration) FROM mart_category_benchmark") <= 1.0
    assert gold.one("SELECT min(current_penetration) FROM mart_category_benchmark") > 0


@pytest.mark.parametrize("comparison", list(WINDOWS))
def test_regions_add_up_to_the_national_row(gold, comparison):
    national = gold.sql(
        f"SELECT current_revenue, base_revenue, current_baskets, stores_current "
        f"FROM mart_region_benchmark WHERE comparison = '{comparison}' AND store_region = 'ALL'"
    )[0]
    regional = gold.sql(
        f"SELECT sum(current_revenue), sum(base_revenue), sum(current_baskets), "
        f"sum(stores_current) FROM mart_region_benchmark "
        f"WHERE comparison = '{comparison}' AND store_region <> 'ALL'"
    )[0]
    assert regional == pytest.approx(national)


def test_region_revenue_matches_independent_calculation(gold, silver_frame):
    for comparison, (cur_idx, base_idx) in WINDOWS.items():
        for region in ("E01", "E02"):
            row = gold.sql(
                f"SELECT current_revenue, base_revenue FROM mart_region_benchmark "
                f"WHERE comparison = '{comparison}' AND store_region = '{region}'"
            )
            cur = window_revenue(silver_frame, cur_idx, store_region=region)
            base = window_revenue(silver_frame, base_idx, store_region=region)
            assert row[0] == (pytest.approx(cur), pytest.approx(base))


def test_like_for_like_excludes_openings_and_closures(gold):
    def region(comparison, name):
        return gold.sql(
            f"SELECT stores_current, stores_base, lfl_stores FROM mart_region_benchmark "
            f"WHERE comparison = '{comparison}' AND store_region = '{name}'"
        )[0]

    # STORE03 (E01) opened at week idx 5: comparable against idx 5-6 but not against idx 3-4.
    assert region("window_vs_prior", "E01") == (2, 2, 2)
    assert region("window_vs_year_ago", "E01") == (2, 1, 1)
    # STORE04 (W01) closed after idx 4: only a base-window store against the year-ago window.
    assert region("window_vs_year_ago", "W01") == (0, 1, 0)


def test_lfl_growth_uses_only_continuously_trading_stores(gold, silver_frame):
    cur_idx, base_idx = WINDOWS["window_vs_year_ago"]
    lfl_cur = window_revenue(silver_frame, cur_idx, store_code="STORE01") + window_revenue(
        silver_frame, cur_idx, store_code="STORE02"
    )
    lfl_base = window_revenue(silver_frame, base_idx, store_code="STORE01") + window_revenue(
        silver_frame, base_idx, store_code="STORE02"
    )
    growth = gold.one(
        "SELECT lfl_revenue_growth FROM mart_region_benchmark "
        "WHERE comparison = 'window_vs_year_ago' AND store_region = 'ALL'"
    )
    assert growth == pytest.approx(lfl_cur / lfl_base - 1)


def test_closed_region_shows_full_decline_not_missing_data(gold):
    growth = gold.one(
        "SELECT revenue_growth FROM mart_region_benchmark "
        "WHERE comparison = 'window_vs_year_ago' AND store_region = 'W01'"
    )
    assert growth == pytest.approx(-1.0)


@pytest.mark.parametrize("comparison", list(WINDOWS))
def test_region_mix_shares_sum_to_one_and_index_matches_definition(gold, comparison):
    sums = gold.sql(
        f"SELECT store_region, sum(region_share) FROM mart_region_category_mix "
        f"WHERE comparison = '{comparison}' GROUP BY 1"
    )
    assert all(total == pytest.approx(1.0) for _, total in sums)
    bad = gold.one(
        f"SELECT count(*) FROM mart_region_category_mix WHERE comparison = '{comparison}' "
        f"AND abs(mix_index - 100 * region_share / national_share) > 1e-9"
    )
    assert bad == 0


@pytest.mark.parametrize("comparison", list(WINDOWS))
def test_category_contributions_add_up_to_national_growth(gold, silver_frame, comparison):
    cur_idx, base_idx = WINDOWS[comparison]
    total = gold.one(
        f"SELECT sum(contribution_to_growth_pp) FROM mart_category_benchmark "
        f"WHERE comparison = '{comparison}'"
    )
    cur = window_revenue(silver_frame, cur_idx)
    base = window_revenue(silver_frame, base_idx)
    assert total == pytest.approx((cur / base - 1) * 100)
