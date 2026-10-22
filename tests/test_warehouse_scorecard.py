import polars as pl
import pytest

WINDOWS = {
    "window_vs_prior": ({7, 8}, {5, 6}),
    "window_vs_year_ago": ({7, 8}, {3, 4}),
    "year_vs_prior_year": ({5, 6, 7, 8}, {1, 2, 3, 4}),
}


def revenue(frame: pl.DataFrame, idxs: set[int], store: str) -> float:
    rows = frame.filter(
        (pl.col("shop_week") - 200607 + 1).is_in(idxs) & (pl.col("store_code") == store)
    )
    return float(rows["spend"].sum())


def card(gold, comparison, store, columns):
    return gold.sql(
        f"SELECT {columns} FROM mart_store_scorecard "
        f"WHERE comparison = '{comparison}' AND store_code = '{store}'"
    )[0]


def test_scorecard_grain_is_unique(gold):
    total = gold.one("SELECT count(*) FROM mart_store_scorecard")
    keys = gold.one(
        "SELECT count(*) FROM (SELECT DISTINCT comparison, store_code FROM mart_store_scorecard)"
    )
    assert total == keys


@pytest.mark.parametrize("comparison", list(WINDOWS))
def test_store_revenue_sums_to_the_national_figure(gold, comparison):
    stores = gold.sql(
        f"SELECT sum(current_revenue), sum(base_revenue) FROM mart_store_scorecard "
        f"WHERE comparison = '{comparison}'"
    )[0]
    national = gold.sql(
        f"SELECT current_revenue, base_revenue FROM mart_region_benchmark "
        f"WHERE comparison = '{comparison}' AND store_region = 'ALL'"
    )[0]
    assert stores == pytest.approx(national)


@pytest.mark.parametrize("comparison", list(WINDOWS))
@pytest.mark.parametrize("store", ["STORE01", "STORE02"])
def test_growth_matches_independent_calculation_for_continuous_stores(
    gold, silver_frame, comparison, store
):
    cur_idx, base_idx = WINDOWS[comparison]
    current = revenue(silver_frame, cur_idx, store)
    base = revenue(silver_frame, base_idx, store)
    growth, is_lfl = card(gold, comparison, store, "revenue_growth, is_lfl")
    assert growth == pytest.approx(current / base - 1)
    assert is_lfl is True


def test_new_store_has_no_growth_and_is_not_like_for_like(gold):
    growth, is_lfl, base_rev, lifecycle = card(
        gold, "window_vs_year_ago", "STORE03", "revenue_growth, is_lfl, base_revenue, lifecycle"
    )
    assert growth is None
    assert base_rev == 0
    assert is_lfl is False
    assert lifecycle == "opened_in_period"


def test_closed_store_shows_full_decline_and_is_not_ranked(gold):
    growth, is_lfl, rank_value, per_week = card(
        gold,
        "window_vs_year_ago",
        "STORE04",
        "revenue_growth, is_lfl, revenue_rank_in_region, current_revenue_per_week",
    )
    assert growth == pytest.approx(-1.0)
    assert is_lfl is False
    assert per_week is None
    assert rank_value is None  # a store that is not trading in the window is not ranked


def test_revenue_per_week_normalises_partial_windows(gold, silver_frame):
    # STORE03 traded weeks idx 5-8: in the 4-week year window it has 4 weeks, in the prior
    # comparison's base (idx 1-4) none.
    per_week = card(gold, "year_vs_prior_year", "STORE03", "current_revenue_per_week")[0]
    assert per_week == pytest.approx(revenue(silver_frame, {5, 6, 7, 8}, "STORE03") / 4)


def test_format_peer_index_is_relative_to_the_peer_median(gold, silver_frame):
    cur_idx, _ = WINDOWS["window_vs_prior"]
    rev = {s: revenue(silver_frame, cur_idx, s) / 2 for s in ("STORE01", "STORE02")}
    median = (rev["STORE01"] + rev["STORE02"]) / 2
    for store, value in rev.items():
        index = card(gold, "window_vs_prior", store, "revenue_index_vs_format")[0]
        assert index == pytest.approx(100 * value / median)


def test_rank_in_region_orders_by_revenue_per_week(gold):
    rows = gold.sql(
        "SELECT store_code, current_revenue_per_week, revenue_rank_in_region "
        "FROM mart_store_scorecard "
        "WHERE comparison = 'window_vs_prior' AND store_region = 'E01'"
    )
    ranked = sorted(rows, key=lambda r: r[1], reverse=True)
    assert [r[2] for r in ranked] == [1, 2]
    assert (
        gold.one(
            "SELECT max(trading_stores_in_region) FROM mart_store_scorecard "
            "WHERE comparison = 'window_vs_prior' AND store_region = 'E01'"
        )
        == 2
    )


def test_growth_gap_and_zscore_use_like_for_like_stores_only(gold):
    only_lfl = gold.one(
        "SELECT count(*) FROM mart_store_scorecard "
        "WHERE (lfl_growth_gap_pp IS NOT NULL OR lfl_growth_zscore IS NOT NULL) AND NOT is_lfl"
    )
    assert only_lfl == 0
    gaps = [
        r[0]
        for r in gold.sql(
            "SELECT lfl_growth_gap_pp FROM mart_store_scorecard "
            "WHERE comparison = 'window_vs_prior' AND store_format = 'LS'"
        )
    ]
    # Two LS peers: each sits the same distance from their median, on opposite sides.
    assert sum(gaps) == pytest.approx(0.0)
    zscores = [
        r[0]
        for r in gold.sql(
            "SELECT lfl_growth_zscore FROM mart_store_scorecard "
            "WHERE comparison = 'window_vs_prior' AND store_format = 'LS'"
        )
    ]
    assert sum(zscores) == pytest.approx(0.0)
