import polars as pl
import pytest

from demandpulse.app import charts, views
from demandpulse.app.snapshot import Snapshot

WINDOW = "window_vs_year_ago"


@pytest.fixture(scope="module")
def snap(app_snapshot_dir):
    return Snapshot(app_snapshot_dir)


def test_network_kpis_read_the_all_row_of_the_requested_window(snap):
    kpis = views.network_kpis(snap.table("mart_region_benchmark"), WINDOW)
    row = snap.table("mart_region_benchmark").filter(
        (pl.col("comparison") == WINDOW) & (pl.col("store_region") == "ALL")
    )
    assert kpis["revenue"] == row["current_revenue"][0]
    assert kpis["avg_basket_value_growth"] == pytest.approx(
        row["current_avg_basket_value"][0] / row["base_avg_basket_value"][0] - 1
    )


def test_network_kpis_reject_a_window_without_a_network_row(snap):
    with pytest.raises(ValueError, match="network row"):
        views.network_kpis(snap.table("mart_region_benchmark"), "nope")


def test_weekly_trend_sums_stores_and_adds_a_13_week_average(snap):
    weekly = views.weekly_trend(snap.table("store_week"))
    assert weekly.height == 117
    assert weekly["rolling_13"].null_count() == 12
    one_week = snap.table("store_week").filter(pl.col("week_idx") == 50)
    assert weekly.filter(pl.col("week_idx") == 50)["revenue"][0] == pytest.approx(
        one_week["revenue"].sum()
    )


def test_weekly_trend_filters_by_region_and_format(snap):
    weeks = snap.table("store_week")
    region = views.weekly_trend(weeks, regions=["E01"])
    fmt = views.weekly_trend(weeks, formats=["LS"])
    both = views.weekly_trend(weeks, regions=["E01"], formats=["LS"])
    assert region["revenue"].sum() < weeks["revenue"].sum()
    assert both["revenue"].sum() <= min(region["revenue"].sum(), fmt["revenue"].sum())


def test_revenue_bridge_runs_from_base_to_current_and_shows_any_gap(snap):
    bridge = views.revenue_bridge(snap.table("mart_revenue_variance_region"), WINDOW)
    assert bridge["step"][0] == "Base period"
    assert bridge["step"][-1] == "Current period"
    assert "Unreconciled" in bridge["step"].to_list()  # fixture effects do not add up


def test_revenue_bridge_has_no_gap_row_when_effects_add_up():
    variance = pl.DataFrame(
        {
            "comparison": "c", "store_region": "ALL", "base_revenue": 100.0,
            "current_revenue": 110.0, "trading_weeks_effect": 0.0, "traffic_effect": 6.0,
            "basket_size_effect": 3.0, "price_mix_effect": 1.0, "new_store_effect": 2.0,
            "closed_store_effect": -2.0,
        }
    )  # fmt: skip
    bridge = views.revenue_bridge(variance, "c")
    assert "Unreconciled" not in bridge["step"].to_list()
    assert bridge.filter(pl.col("kind") == "delta")["value"].sum() == pytest.approx(10.0)


def test_category_contribution_is_sorted_and_not_duplicated(snap):
    out = views.category_contribution(snap.table("mart_category_benchmark"), WINDOW, top=5)
    assert out["contribution_to_growth_pp"].is_sorted()
    assert out["prod_code_30"].n_unique() == out.height


def test_mission_mix_shares_sum_to_one(snap):
    mix = views.mission_mix(snap.table("mart_basket_mix"))
    assert mix["revenue_share"].sum() == pytest.approx(1.0)
    assert views.mission_mix(snap.table("mart_basket_mix"), ["LS"])["revenue_share"].sum() == (
        pytest.approx(1.0)
    )


def test_trading_matrix_has_a_row_per_weekday_and_a_column_per_hour(snap):
    matrix = views.trading_matrix(snap.table("mart_trading_pattern"), "LS")
    assert matrix.height == 7
    assert matrix.columns[1:] == [str(h) for h in range(6, 22)]


def test_region_table_excludes_the_network_row_and_orders_by_lfl_growth(snap):
    table = views.region_table(snap.table("mart_region_benchmark"), WINDOW)
    assert "ALL" not in table["store_region"].to_list()
    assert table["lfl_revenue_growth"].is_sorted(descending=True)


def test_region_mix_matrix_is_regions_by_categories(snap):
    matrix = views.region_mix_matrix(snap.table("mart_region_category_mix"), WINDOW)
    assert matrix.columns == ["store_region", "C1", "C2"]


def test_scorecard_filters_combine(snap):
    card = snap.table("mart_store_scorecard")
    everything = views.scorecard(card, WINDOW)
    assert everything.height == 45
    assert everything["revenue_growth"].is_sorted()
    assert views.scorecard(card, WINDOW, regions=["E01"]).height < 45
    assert views.scorecard(card, WINDOW, lfl_only=True).height == 44
    one = views.scorecard(card, WINDOW, search="s07")
    assert one["store_code"].to_list() == ["S07"]
    assert views.scorecard(card, WINDOW, formats=["XX"]).height == 0


def test_store_detail_joins_profile_drivers_and_history(snap):
    detail = views.store_detail(
        snap.table("mart_store_scorecard"),
        snap.table("mart_revenue_variance_store"),
        snap.table("store_week"),
        "S03",
        WINDOW,
    )
    assert detail.profile["store_code"] == "S03"
    assert detail.drivers.height == 6
    assert detail.weekly.height == 117


def test_store_detail_for_an_unknown_store_is_an_error(snap):
    with pytest.raises(KeyError, match="No scorecard"):
        views.store_detail(
            snap.table("mart_store_scorecard"),
            snap.table("mart_revenue_variance_store"),
            snap.table("store_week"),
            "NOPE",
            WINDOW,
        )


def test_loyalty_coverage_is_a_share_of_revenue_and_baskets(snap):
    coverage = views.loyalty_coverage(snap.table("store_week"))
    assert coverage["revenue_share"] == pytest.approx(0.8)
    assert 0 < coverage["basket_share"] <= 1


def test_segments_follow_the_customer_journey_order(snap):
    table = views.segment_table(snap.table("mart_rfm_segment_summary"))
    assert table["rfm_segment"].to_list() == list(views.SEGMENT_ORDER)


def test_retention_matrix_orders_periods_numerically(snap):
    matrix = views.retention_matrix(snap.table("mart_customer_cohort"))
    assert matrix.columns == ["cohort_period", "0", "1", "2", "3"]
    assert matrix["0"].to_list() == [1.0, 1.0, 1.0]


def test_every_chart_builds_from_the_views(snap):
    weeks = views.weekly_trend(snap.table("store_week"))
    bridge = views.revenue_bridge(snap.table("mart_revenue_variance_region"), WINDOW)
    contribution = views.category_contribution(snap.table("mart_category_benchmark"), WINDOW)
    regions = views.region_table(snap.table("mart_region_benchmark"), WINDOW)
    card = views.scorecard(snap.table("mart_store_scorecard"), WINDOW)
    detail = views.store_detail(
        snap.table("mart_store_scorecard"),
        snap.table("mart_revenue_variance_store"),
        snap.table("store_week"),
        "S01",
        WINDOW,
    )
    figures = [
        charts.trend_chart(weeks),
        charts.trend_chart(weeks, "baskets", "Baskets"),
        charts.bridge_chart(bridge),
        charts.contribution_chart(contribution),
        charts.growth_bars(regions, "store_region", "lfl_revenue_growth", "LFL growth"),
        charts.store_scatter(card),
        charts.driver_chart(detail.drivers),
        charts.segment_chart(views.segment_table(snap.table("mart_rfm_segment_summary"))),
        charts.heatmap(
            views.trading_matrix(snap.table("mart_trading_pattern"), "LS"), "Hour", "Weekday"
        ),
        charts.heatmap(
            views.region_mix_matrix(snap.table("mart_region_category_mix"), WINDOW),
            "Category",
            "Region",
            centered=True,
        ),
        charts.heatmap(
            views.retention_matrix(snap.table("mart_customer_cohort")),
            "Periods since first purchase",
            "Cohort",
            percent=True,
        ),
    ]
    assert all(len(fig.data) >= 1 for fig in figures)


def test_bridge_chart_marks_totals_as_absolute(snap):
    bridge = views.revenue_bridge(snap.table("mart_revenue_variance_region"), WINDOW)
    measures = charts.bridge_chart(bridge).data[0].measure
    assert measures[0] == "absolute"
    assert measures[-1] == "absolute"
    assert set(measures[1:-1]) == {"relative"}
