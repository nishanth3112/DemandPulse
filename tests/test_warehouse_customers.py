import math

import polars as pl
import pytest

WINDOW_IDX = {5, 6, 7, 8}  # year_weeks = 4 in the fixture build
LAST_IDX = 8


def loyalty_in_window(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(
        pl.col("cust_code").is_not_null() & (pl.col("shop_week") - 200607 + 1).is_in(WINDOW_IDX)
    )


@pytest.fixture(scope="module")
def expected_rfm(silver_frame):
    activity = (
        loyalty_in_window(silver_frame)
        .group_by("cust_code")
        .agg(
            pl.col("basket_id").n_unique().alias("baskets"),
            pl.col("spend").sum().cast(pl.Float64).alias("revenue"),
            (LAST_IDX - (pl.col("shop_week").max() - 200607 + 1)).alias("recency_weeks"),
        )
    )
    n = activity.height

    def score(column: str, descending: bool = False) -> pl.Expr:
        # ceil(5 * cume_dist): cume_dist = (number of rows ordered at or before) / n
        rank = pl.col(column).rank(method="max", descending=descending)
        return (5.0 * rank / n).ceil().cast(pl.Int64)

    return activity.with_columns(
        r_score=score("recency_weeks", descending=True),
        f_score=score("baskets"),
        m_score=score("revenue"),
    )


def test_rfm_covers_exactly_the_customers_active_in_the_window(gold, expected_rfm):
    actual = {r[0] for r in gold.sql("SELECT cust_code FROM mart_customer_rfm")}
    assert actual == set(expected_rfm["cust_code"])
    assert gold.one("SELECT count(*) FROM mart_customer_rfm") == len(actual)


def test_rfm_measures_match_independent_calculation(gold, expected_rfm):
    actual = {
        c: (b, rev, rec)
        for c, b, rev, rec in gold.sql(
            "SELECT cust_code, baskets, revenue, recency_weeks FROM mart_customer_rfm"
        )
    }
    for row in expected_rfm.iter_rows(named=True):
        baskets, revenue, recency = actual[row["cust_code"]]
        assert baskets == row["baskets"]
        assert revenue == pytest.approx(row["revenue"])
        assert recency == row["recency_weeks"]


def test_rfm_scores_match_the_cume_dist_definition(gold, expected_rfm):
    actual = {
        c: (r, f, m)
        for c, r, f, m in gold.sql(
            "SELECT cust_code, r_score, f_score, m_score FROM mart_customer_rfm"
        )
    }
    for row in expected_rfm.iter_rows(named=True):
        assert actual[row["cust_code"]] == (row["r_score"], row["f_score"], row["m_score"])


def test_rfm_scores_are_in_range_and_every_customer_has_a_segment(gold):
    assert gold.one("SELECT min(least(r_score, f_score, m_score)) FROM mart_customer_rfm") >= 1
    assert gold.one("SELECT max(greatest(r_score, f_score, m_score)) FROM mart_customer_rfm") <= 5
    assert gold.one("SELECT count(*) FROM mart_customer_rfm WHERE rfm_segment IS NULL") == 0
    assert gold.one("SELECT min(recency_weeks) FROM mart_customer_rfm") >= 0


def test_rfm_revenue_reconciles_to_loyalty_revenue_in_window(gold, silver_frame):
    expected = float(loyalty_in_window(silver_frame)["spend"].sum())
    assert gold.one("SELECT sum(revenue) FROM mart_customer_rfm") == pytest.approx(expected)


def test_segment_summary_adds_up(gold):
    assert gold.one("SELECT sum(customers) FROM mart_rfm_segment_summary") == gold.one(
        "SELECT count(*) FROM mart_customer_rfm"
    )
    assert gold.one("SELECT sum(customer_share) FROM mart_rfm_segment_summary") == pytest.approx(
        1.0
    )
    assert gold.one("SELECT sum(revenue_share) FROM mart_rfm_segment_summary") == pytest.approx(1.0)
    assert gold.one("SELECT sum(revenue) FROM mart_rfm_segment_summary") == pytest.approx(
        gold.one("SELECT sum(revenue) FROM mart_customer_rfm")
    )


def test_segment_rules_assign_champions_and_hibernating_consistently(gold):
    wrong = gold.one(
        """
        SELECT count(*) FROM mart_customer_rfm
        WHERE (rfm_segment = 'Champions' AND NOT (r_score >= 4 AND f_score >= 4 AND m_score >= 4))
           OR (rfm_segment = 'Hibernating' AND NOT (r_score <= 2 AND f_score <= 2))
        """
    )
    assert wrong == 0


@pytest.fixture(scope="module")
def expected_cohorts(silver_frame):
    loyal = silver_frame.filter(pl.col("cust_code").is_not_null()).with_columns(
        period=((pl.col("shop_week") - 200607) // 4 + 1)
    )
    first = loyal.group_by("cust_code").agg(pl.col("period").min().alias("cohort"))
    joined = loyal.join(first, on="cust_code")
    active = joined.group_by("cohort", "period").agg(pl.col("cust_code").n_unique().alias("active"))
    size = first.group_by("cohort").agg(pl.len().alias("size"))
    return active.join(size, on="cohort")


def test_cohort_cells_match_independent_calculation(gold, expected_cohorts):
    actual = {
        (c, p): (active, size)
        for c, p, active, size in gold.sql(
            "SELECT cohort_period, period_idx, active_customers, cohort_customers "
            "FROM mart_customer_cohort"
        )
    }
    expected = {
        (r["cohort"], r["period"]): (r["active"], r["size"])
        for r in expected_cohorts.iter_rows(named=True)
    }
    assert actual == expected


def test_every_cohort_is_fully_active_in_its_first_period(gold):
    rates = gold.sql(
        "SELECT retention_rate FROM mart_customer_cohort WHERE periods_since_first = 0"
    )
    assert rates
    assert all(math.isclose(r[0], 1.0) for r in rates)


def test_retention_never_exceeds_one_and_periods_never_precede_the_cohort(gold):
    assert gold.one("SELECT max(retention_rate) FROM mart_customer_cohort") <= 1.0
    assert gold.one("SELECT min(periods_since_first) FROM mart_customer_cohort") == 0


def test_cohort_sizes_add_up_to_all_loyalty_customers(gold, silver_frame):
    total = gold.one(
        "SELECT sum(cohort_customers) FROM "
        "(SELECT DISTINCT cohort_period, cohort_customers FROM mart_customer_cohort)"
    )
    assert total == silver_frame.filter(pl.col("cust_code").is_not_null())["cust_code"].n_unique()


def test_cohort_revenue_reconciles_to_all_loyalty_revenue(gold, silver_frame):
    expected = silver_frame.filter(pl.col("cust_code").is_not_null())["spend"].sum()
    assert gold.one("SELECT sum(revenue) FROM mart_customer_cohort") == expected


def test_only_the_first_cohort_is_flagged_as_baseline(gold):
    flagged = {
        r[0]
        for r in gold.sql(
            "SELECT DISTINCT cohort_period FROM mart_customer_cohort WHERE is_baseline_cohort"
        )
    }
    assert flagged == {1}
