import polars as pl
import pytest

from demandpulse.risk.drivers import driver_changes, entity_growth, entity_volatility

PANEL = {
    "store_code": pl.String,
    "prod_code_30": pl.String,
    "week_idx": pl.Int64,
    "revenue": pl.Float64,
}


def panel(fn, stores=("S1",), cats=("C1",), weeks=range(1, 121)) -> pl.DataFrame:
    rows = [(s, c, w, fn(s, c, w)) for s in stores for c in cats for w in weeks]
    return pl.DataFrame(rows, schema=PANEL, orient="row")


def test_entity_growth_compares_the_entity_with_its_own_year_ago_weeks():
    frame = panel(lambda s, c, w: {"S1": 120.0, "S2": 80.0}[s] if w > 77 else 100.0, ("S1", "S2"))
    out = entity_growth(frame, ["store_code"], origin=90).sort("store_code")
    assert out["growth"].to_list() == pytest.approx([0.2, -0.2])


def test_entity_growth_ignores_weeks_after_the_origin():
    base = panel(lambda s, c, w: 100.0)
    later = panel(lambda s, c, w: 500.0 if w > 90 else 100.0)
    one = entity_growth(base, ["store_code"], origin=90)["growth"][0]
    assert one == entity_growth(later, ["store_code"], origin=90)["growth"][0]


def test_entity_growth_is_null_without_a_full_year_ago_window():
    frame = panel(lambda s, c, w: 100.0, weeks=range(70, 121))
    assert entity_growth(frame, ["store_code"], origin=90)["growth"][0] is None


def test_entity_growth_aggregates_categories_to_the_store():
    frame = panel(lambda s, c, w: 100.0 if w < 78 else 110.0, cats=("C1", "C2"))
    out = entity_growth(frame, ["store_code"], origin=90)
    assert out["growth"][0] == pytest.approx(0.1)


def test_entity_volatility_is_zero_for_constant_revenue_and_positive_otherwise():
    flat = panel(lambda s, c, w: 100.0)
    bumpy = panel(lambda s, c, w: 100.0 + 50.0 * (w % 2))
    assert entity_volatility(flat, ["store_code"], origin=90)["volatility"][0] == 0.0
    assert entity_volatility(bumpy, ["store_code"], origin=90)["volatility"][0] > 0.1


def store_weeks(baskets, units_per_basket, price, stores=("S1",)) -> pl.DataFrame:
    rows = []
    for s in stores:
        for w in range(1, 121):
            b = baskets(s, w)
            u = b * units_per_basket(s, w)
            rows.append((s, w, u * price(s, w), b, u))
    return pl.DataFrame(
        rows,
        schema={
            "store_code": pl.String,
            "week_idx": pl.Int64,
            "revenue": pl.Float64,
            "baskets": pl.Float64,
            "units": pl.Float64,
        },
        orient="row",
    )


def test_driver_changes_split_revenue_growth_into_traffic_basket_size_and_price():
    recent = lambda w: w > 77  # noqa: E731
    frame = store_weeks(
        baskets=lambda s, w: 1100.0 if recent(w) else 1000.0,
        units_per_basket=lambda s, w: 9.0 if recent(w) else 10.0,
        price=lambda s, w: 2.0 if recent(w) else 1.0,
    )
    row = driver_changes(frame, origin=90).row(0, named=True)
    assert row["traffic_change"] == pytest.approx(0.10)
    assert row["basket_size_change"] == pytest.approx(-0.10)
    assert row["price_mix_change"] == pytest.approx(1.0)
    growth = (
        (1 + row["traffic_change"])
        * (1 + row["basket_size_change"])
        * (1 + row["price_mix_change"])
    )
    assert 1 + row["revenue_growth"] == pytest.approx(growth)
    assert row["dominant_driver"] == "basket_size"


def test_driver_changes_name_traffic_and_price_mix_when_they_fall_most():
    def frame(basket_drop, price_drop):
        return store_weeks(
            baskets=lambda s, w: 1000.0 * (1 - basket_drop if w > 77 else 1),
            units_per_basket=lambda s, w: 10.0,
            price=lambda s, w: 1.0 * (1 - price_drop if w > 77 else 1),
        )

    assert driver_changes(frame(0.2, 0.0), 90)["dominant_driver"][0] == "traffic"
    assert driver_changes(frame(0.0, 0.2), 90)["dominant_driver"][0] == "price_mix"


def test_driver_changes_report_no_driver_when_nothing_fell():
    frame = store_weeks(
        baskets=lambda s, w: 1000.0, units_per_basket=lambda s, w: 10.0, price=lambda s, w: 1.0
    )
    assert driver_changes(frame, 90)["dominant_driver"][0] == "none"


def test_driver_changes_skip_stores_without_a_year_ago_window():
    frame = store_weeks(lambda s, w: 1000.0, lambda s, w: 10.0, lambda s, w: 1.0).filter(
        pl.col("week_idx") > 60
    )
    assert driver_changes(frame, 90).height == 0
