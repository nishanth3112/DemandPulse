import polars as pl
import pytest

from demandpulse.forecasting.panel import (
    SERIES_KEY,
    build_panel,
    eligible_series,
    history,
    representative_subset,
)
from demandpulse.warehouse.build import GoldConfig, build_gold
from silver_factory import write_silver_lake

# The 8-week fixture cannot satisfy production thresholds, so tests use proportionally small ones.
SMALL = {"horizon": 2, "min_history": 4, "min_nonzero": 2, "year_weeks": 4}


@pytest.fixture(scope="module")
def panel(gold):
    return build_panel(gold.config)


def test_panel_is_dense_over_each_stores_trading_span(panel):
    per_store = panel.group_by("store_code").agg(
        pl.col("week_idx").n_unique().alias("weeks"),
        pl.col("prod_code_30").n_unique().alias("categories"),
        pl.len().alias("rows"),
    )
    assert (per_store["rows"] == per_store["weeks"] * per_store["categories"]).all()
    assert per_store["categories"].unique().to_list() == [3]


def test_stores_have_no_rows_outside_their_trading_weeks(panel):
    spans = panel.group_by("store_code").agg(
        pl.col("week_idx").min().alias("first"), pl.col("week_idx").max().alias("last")
    )
    by_store = {r["store_code"]: (r["first"], r["last"]) for r in spans.iter_rows(named=True)}
    assert by_store["STORE01"] == (1, 8)
    assert by_store["STORE03"] == (5, 8)  # opened mid-way: earlier weeks absent, not zero
    assert by_store["STORE04"] == (1, 4)


def test_panel_revenue_reconciles_to_gold(panel, gold):
    assert panel["revenue"].sum() == pytest.approx(
        float(gold.one("SELECT sum(revenue) FROM fct_store_week"))
    )


def test_missing_sales_become_explicit_zero_rows(silver_frame, tmp_path):
    # Remove every G3 line for STORE01 in week 200610 (week_idx 4) to create a real gap.
    gap = silver_frame.filter(
        ~(
            (pl.col("store_code") == "STORE01")
            & (pl.col("shop_week") == 200610)
            & (pl.col("prod_code_30") == "G3")
        )
    )
    root = tmp_path / "lake"
    write_silver_lake(root, gap)
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

    gapped = build_panel(config)

    cell = gapped.filter(
        (pl.col("store_code") == "STORE01")
        & (pl.col("prod_code_30") == "G3")
        & (pl.col("week_idx") == 4)
    )
    assert cell.height == 1
    assert cell["revenue"][0] == 0.0
    assert gapped.height == 72  # the grid is unchanged: the gap is filled, not dropped
    assert (gapped["revenue"] > 0).sum() == 71


def test_history_never_contains_later_weeks(panel):
    assert history(panel, 5)["week_idx"].max() == 5
    assert history(panel, 1)["week_idx"].max() == 1


def test_eligibility_requires_enough_history_and_a_full_test_window(panel):
    at_6 = eligible_series(panel, 6, **SMALL)
    # STORE01/02 have 6 weeks of history and both test weeks; STORE03 only 2 weeks of history;
    # STORE04 stopped trading at week 4 so it has no test window.
    assert set(at_6["store_code"]) == {"STORE01", "STORE02"}
    assert at_6["history_weeks"].unique().to_list() == [6]


def test_eligibility_uses_only_the_past_for_the_activity_rule(panel):
    at_6 = eligible_series(panel, 6, **SMALL)
    assert at_6["nonzero_weeks_last_year"].min() >= SMALL["min_nonzero"]
    stricter = eligible_series(panel, 6, **{**SMALL, "min_nonzero": 5})
    assert stricter.height < at_6.height


def test_eligibility_ignores_test_window_values(panel):
    baseline = eligible_series(panel, 6, **SMALL)
    wiped = panel.with_columns(
        revenue=pl.when(pl.col("week_idx") > 6).then(0.0).otherwise(pl.col("revenue"))
    )
    assert eligible_series(wiped, 6, **SMALL).sort(SERIES_KEY).equals(baseline.sort(SERIES_KEY))


def test_subset_is_deterministic_and_stratified(panel):
    options = {"origins": (5, 6), "per_stratum": 1, **SMALL}
    first = representative_subset(panel, **options)
    second = representative_subset(panel, **options)
    assert first.series.equals(second.series)
    assert first.series["stratum"].n_unique() == first.series.height  # one per stratum
    assert set(first.series["stratum"].str.split("|").list.get(0)) <= {"LS", "SS", "MS"}


def test_subset_only_contains_series_eligible_at_every_origin(panel):
    options = {"origins": (5, 6), "per_stratum": 5, **SMALL}
    subset = representative_subset(panel, **options)
    for origin in (5, 6):
        eligible = eligible_series(panel, origin, **SMALL).select(SERIES_KEY)
        assert (
            subset.series.select(SERIES_KEY).join(eligible, on=SERIES_KEY).height
            == subset.series.height
        )


def test_subset_changes_with_the_seed(panel):
    options = {"origins": (5, 6), "per_stratum": 2, **SMALL}
    seeds = {
        tuple(
            map(
                tuple,
                representative_subset(panel, seed=s, **options).series.select(SERIES_KEY).rows(),
            )
        )
        for s in range(8)
    }
    assert len(seeds) > 1
