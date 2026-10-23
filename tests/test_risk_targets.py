import polars as pl
import pytest

from demandpulse.risk.targets import (
    FLAT_YOY,
    TRAILING_GROWTH,
    TREND_WEEKS,
    network_growth,
    target_growth,
    window_table,
)

PANEL_SCHEMA = {
    "store_code": pl.String,
    "prod_code_30": pl.String,
    "week_idx": pl.Int64,
    "revenue": pl.Float64,
}
FORECAST_SCHEMA = {
    "model": pl.String,
    "origin": pl.Int64,
    "store_code": pl.String,
    "prod_code_30": pl.String,
    "horizon": pl.Int64,
    "week_idx": pl.Int64,
    "y": pl.Float64,
    "y_hat": pl.Float64,
}


def panel(revenue_by_week, stores=("S1",), categories=("C1",)) -> pl.DataFrame:
    rows = [
        (store, category, week, revenue_by_week(store, category, week))
        for store in stores
        for category in categories
        for week in range(1, 121)
    ]
    return pl.DataFrame(rows, schema=PANEL_SCHEMA, orient="row")


def test_network_growth_compares_last_13_weeks_with_the_same_weeks_a_year_earlier():
    # 100 a week before week 78, 110 from then on: the 13 weeks up to origin 90 are all 110.
    frame = panel(lambda s, c, w: 110.0 if w >= 78 else 100.0)
    assert network_growth(frame, origin=90) == pytest.approx(0.10)


def test_network_growth_ignores_revenue_after_the_origin():
    base = panel(lambda s, c, w: 100.0)
    shocked = panel(lambda s, c, w: 1000.0 if w > 90 else 100.0)
    assert network_growth(base, origin=90) == network_growth(shocked, origin=90)


def test_network_growth_is_like_for_like_and_skips_series_missing_a_year_ago():
    # S2 only opened at week 70, so it has no year-ago weeks and must not inflate growth.
    frame = panel(lambda s, c, w: 100.0, stores=("S1", "S2")).filter(
        ~((pl.col("store_code") == "S2") & (pl.col("week_idx") < 70))
    )
    assert network_growth(frame, origin=90) == pytest.approx(0.0)


def test_network_growth_needs_a_full_year_plus_the_trend_window():
    frame = panel(lambda s, c, w: 100.0).filter(pl.col("week_idx") > 80)
    with pytest.raises(ValueError, match="history"):
        network_growth(frame, origin=100)


def test_target_growth_defines_flat_and_trailing():
    frame = panel(lambda s, c, w: 110.0 if w >= 78 else 100.0)
    assert target_growth(frame, origin=90) == {
        FLAT_YOY: 0.0,
        TRAILING_GROWTH: pytest.approx(0.10),
    }


def forecasts(origin=90, y=100.0, y_hat=90.0, stores=("S1",), categories=("C1",)) -> pl.DataFrame:
    rows = [
        ("m", origin, s, c, h, origin + h, y, y_hat)
        for s in stores
        for c in categories
        for h in range(1, TREND_WEEKS + 1)
    ]
    return pl.DataFrame(rows, schema=FORECAST_SCHEMA, orient="row")


def test_window_table_sums_13_weeks_and_builds_each_target_from_last_year():
    frame = panel(lambda s, c, w: 100.0)
    table = window_table(forecasts(), frame, ["store_code"], {FLAT_YOY: 0.0, TRAILING_GROWTH: 0.1})
    flat = table.filter(pl.col("target_def") == FLAT_YOY).row(0, named=True)
    assert flat["actual"] == pytest.approx(1300.0)
    assert flat["forecast"] == pytest.approx(1170.0)
    assert flat["target"] == pytest.approx(1300.0)  # 13 weeks x 100 last year x 1.0
    trailing = table.filter(pl.col("target_def") == TRAILING_GROWTH).row(0, named=True)
    assert trailing["target"] == pytest.approx(1430.0)


def test_window_table_marks_shortfalls_from_forecast_and_from_actuals():
    frame = panel(lambda s, c, w: 100.0)
    table = window_table(forecasts(y=1.0 * 95, y_hat=105), frame, ["store_code"], {FLAT_YOY: 0.0})
    row = table.row(0, named=True)
    assert row["forecast_shortfall"] == pytest.approx(1300 - 13 * 105)  # negative: forecast above
    assert row["realised_shortfall"] == pytest.approx(1300 - 13 * 95)
    assert row["realised_miss"] is True
    assert row["forecast_miss"] is False


def test_window_table_aggregates_categories_to_the_store_level():
    frame = panel(lambda s, c, w: 100.0, categories=("C1", "C2"))
    table = window_table(forecasts(categories=("C1", "C2")), frame, ["store_code"], {FLAT_YOY: 0.0})
    assert table.height == 1
    assert table["target"][0] == pytest.approx(2600.0)


def test_window_table_keeps_the_category_when_asked():
    frame = panel(lambda s, c, w: 100.0, categories=("C1", "C2"))
    table = window_table(
        forecasts(categories=("C1", "C2")), frame, ["store_code", "prod_code_30"], {FLAT_YOY: 0.0}
    )
    assert table.height == 2


def test_window_table_fails_when_last_year_revenue_is_missing():
    frame = panel(lambda s, c, w: 100.0).filter(pl.col("week_idx") > 60)
    with pytest.raises(ValueError, match="last-year"):
        window_table(forecasts(), frame, ["store_code"], {FLAT_YOY: 0.0})
