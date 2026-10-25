import polars as pl
import pytest

from demandpulse.app import forecast_charts, forecast_views
from demandpulse.app.snapshot import Snapshot

RUN, ORIGIN, STORE = "main", 104, "S03"


@pytest.fixture(scope="module")
def snap(app_snapshot_dir):
    return Snapshot(app_snapshot_dir)


@pytest.fixture(scope="module")
def weeks(snap):
    return snap.table("forecast_store_weeks")


def test_store_options_need_forecasts_and_respect_filters(snap, weeks):
    stores = snap.table("dim_store")
    have = set(weeks["store_code"])
    everything = forecast_views.store_options(stores, have, [], [])
    assert everything == sorted(have)
    east = forecast_views.store_options(stores, have, ["E01"], [])
    assert 0 < len(east) < len(everything)
    assert forecast_views.store_options(stores, have, [], ["XX"]) == []
    assert forecast_views.store_options(stores, {"S00"}, [], []) == ["S00"]


def test_series_has_history_to_the_origin_then_actual_and_forecasts(snap, weeks):
    series = forecast_views.store_series(
        weeks, snap.table("store_week"), RUN, ORIGIN, STORE, ["ensemble", "lightgbm"]
    )
    actual = series.filter(pl.col("series") == "Actual")
    assert actual["week_start"].is_unique().all()
    forecast = series.filter(pl.col("series") != "Actual")
    assert forecast.height == 2 * 13
    first_forecast_week = forecast["week_start"].min()
    history = actual.filter(pl.col("week_start") < first_forecast_week)
    assert history.height == forecast_views.HISTORY_WEEKS


def test_series_rejects_unknown_origin_and_store(snap, weeks):
    with pytest.raises(ValueError, match="origin"):
        forecast_views.store_series(weeks, snap.table("store_week"), RUN, 1, STORE, ["ensemble"])
    with pytest.raises(ValueError, match="store"):
        forecast_views.store_series(
            weeks, snap.table("store_week"), RUN, ORIGIN, "nope", ["ensemble"]
        )


def test_accuracy_matches_a_hand_computation(weeks):
    table = forecast_views.store_accuracy(weeks, RUN, ORIGIN, STORE, ["lightgbm"])
    rows = weeks.filter(
        (pl.col("run") == RUN)
        & (pl.col("origin") == ORIGIN)
        & (pl.col("store_code") == STORE)
        & (pl.col("model") == "lightgbm")
    )
    actual, forecast = rows["actual"].to_list(), rows["forecast"].to_list()
    wape = sum(abs(a - f) for a, f in zip(actual, forecast, strict=True)) / sum(actual)
    assert table["wape"][0] == pytest.approx(wape)
    assert table["bias"][0] == pytest.approx((sum(forecast) - sum(actual)) / sum(actual))
    assert table["model"][0] == "LightGBM"


def test_a_store_with_no_revenue_has_undefined_error_not_a_crash():
    weeks = pl.DataFrame(
        {
            "run": "main", "model": "ensemble", "origin": 1, "week_idx": [2, 3],
            "store_code": "Z", "actual": [0.0, 0.0], "forecast": [5.0, 6.0],
        }
    )  # fmt: skip
    table = forecast_views.store_accuracy(weeks, "main", 1, "Z", ["ensemble"])
    assert table["wape"][0] is None
    assert table["bias"][0] is None
    assert forecast_views.wape_by_store(weeks, "main", 1, "ensemble").height == 0
    assert (
        forecast_views.store_rank(forecast_views.wape_by_store(weeks, "main", 1, "ensemble"), "Z")
        is None
    )


def test_rank_is_one_for_the_most_accurate_store(weeks):
    by_store = forecast_views.wape_by_store(weeks, RUN, ORIGIN, "ensemble")
    best = by_store["store_code"][0]
    rank = forecast_views.store_rank(by_store, best)
    assert rank["rank"] == 1
    assert rank["of"] == by_store.height
    assert rank["median_wape"] == pytest.approx(by_store["wape"].median())


def test_charts_build(snap, weeks):
    series = forecast_views.store_series(
        weeks, snap.table("store_week"), RUN, ORIGIN, STORE, ["ensemble"]
    )
    start = snap.table("dim_date").filter(pl.col("week_idx") == ORIGIN)["week_start"][0]
    assert len(forecast_charts.store_series_chart(series, start).data) == 2
    by_store = forecast_views.wape_by_store(weeks, RUN, ORIGIN, "ensemble")
    assert len(forecast_charts.wape_distribution_chart(by_store, 0.1).data) == 1
    assert len(forecast_charts.wape_distribution_chart(by_store, None).layout.shapes) == 0
