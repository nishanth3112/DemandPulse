import math

import polars as pl
import pytest

from demandpulse.warehouse.build import GoldConfig, build_gold
from silver_factory import write_silver_lake

WINDOWS = {
    "window_vs_prior": ({7, 8}, {5, 6}),
    "window_vs_year_ago": ({7, 8}, {3, 4}),
    "year_vs_prior_year": ({5, 6, 7, 8}, {1, 2, 3, 4}),
}
EFFECTS = [
    "trading_weeks_effect",
    "traffic_effect",
    "basket_size_effect",
    "price_mix_effect",
    "new_store_effect",
    "closed_store_effect",
]


def factors(frame: pl.DataFrame, idxs: set[int], store: str) -> dict[str, float]:
    rows = frame.filter(
        (pl.col("shop_week") - 200607 + 1).is_in(idxs) & (pl.col("store_code") == store)
    )
    return {
        "weeks": rows["shop_week"].n_unique(),
        "revenue": float(rows["spend"].sum()),
        "baskets": rows["basket_id"].n_unique(),
        "units": float(rows["quantity"].sum()),
    }


def lmdi(base: dict[str, float], cur: dict[str, float]) -> dict[str, float]:
    r0, r1 = base["revenue"], cur["revenue"]
    log_mean = r0 if r0 == r1 else (r1 - r0) / math.log(r1 / r0)

    def effect(x1: float, x0: float) -> float:
        return log_mean * math.log(x1 / x0)

    return {
        "trading_weeks_effect": effect(cur["weeks"], base["weeks"]),
        "traffic_effect": effect(cur["baskets"] / cur["weeks"], base["baskets"] / base["weeks"]),
        "basket_size_effect": effect(
            cur["units"] / cur["baskets"], base["units"] / base["baskets"]
        ),
        "price_mix_effect": effect(r1 / cur["units"], r0 / base["units"]),
    }


def store_row(gold, comparison, store):
    cols = "store_status, revenue_change, " + ", ".join(EFFECTS)
    row = gold.sql(
        f"SELECT {cols} FROM mart_revenue_variance_store "
        f"WHERE comparison = '{comparison}' AND store_code = '{store}'"
    )[0]
    return row[0], row[1], dict(zip(EFFECTS, row[2:], strict=True))


def test_variance_grain_is_unique(gold):
    total = gold.one("SELECT count(*) FROM mart_revenue_variance_store")
    keys = gold.one(
        "SELECT count(*) FROM "
        "(SELECT DISTINCT comparison, store_code FROM mart_revenue_variance_store)"
    )
    assert total == keys


def test_effects_sum_exactly_to_the_revenue_change_for_every_store(gold):
    residual = gold.one(
        f"""
        SELECT max(abs(revenue_change - ({" + ".join(EFFECTS)})))
        FROM mart_revenue_variance_store
        """
    )
    assert residual < 1e-6


@pytest.mark.parametrize("comparison", list(WINDOWS))
@pytest.mark.parametrize("store", ["STORE01", "STORE02"])
def test_continuing_store_effects_match_independent_lmdi(gold, silver_frame, comparison, store):
    cur_idx, base_idx = WINDOWS[comparison]
    expected = lmdi(factors(silver_frame, base_idx, store), factors(silver_frame, cur_idx, store))
    status, _, actual = store_row(gold, comparison, store)
    assert status == "continuing"
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, abs=1e-6)
    assert actual["trading_weeks_effect"] == pytest.approx(0.0, abs=1e-9)  # same weeks traded


def test_new_store_contributes_its_whole_revenue(gold, silver_frame):
    status, change, effects = store_row(gold, "window_vs_year_ago", "STORE03")
    cur = factors(silver_frame, {7, 8}, "STORE03")["revenue"]
    assert status == "new"
    assert change == pytest.approx(cur)
    assert effects["new_store_effect"] == pytest.approx(cur)
    assert effects["traffic_effect"] == 0


def test_closed_store_removes_its_base_revenue(gold, silver_frame):
    status, change, effects = store_row(gold, "window_vs_year_ago", "STORE04")
    base = factors(silver_frame, {3, 4}, "STORE04")["revenue"]
    assert status == "closed"
    assert change == pytest.approx(-base)
    assert effects["closed_store_effect"] == pytest.approx(-base)


@pytest.mark.parametrize("comparison", list(WINDOWS))
def test_national_bridge_closes_from_base_to_current(gold, comparison):
    base, current, *effects = gold.sql(
        f"SELECT base_revenue, current_revenue, {', '.join(EFFECTS)} "
        f"FROM mart_revenue_variance_region "
        f"WHERE comparison = '{comparison}' AND store_region = 'ALL'"
    )[0]
    assert base + sum(effects) == pytest.approx(current)


@pytest.mark.parametrize("comparison", list(WINDOWS))
def test_regions_sum_to_the_national_bridge(gold, comparison):
    columns = ", ".join(f"sum({c})" for c in ["base_revenue", "current_revenue", *EFFECTS])
    regional = gold.sql(
        f"SELECT {columns} FROM mart_revenue_variance_region "
        f"WHERE comparison = '{comparison}' AND store_region <> 'ALL'"
    )[0]
    columns = ", ".join(["base_revenue", "current_revenue", *EFFECTS])
    national = gold.sql(
        f"SELECT {columns} FROM mart_revenue_variance_region "
        f"WHERE comparison = '{comparison}' AND store_region = 'ALL'"
    )[0]
    assert regional == pytest.approx(national)


def test_variance_base_and_current_match_the_store_scorecard(gold):
    mismatches = gold.one(
        """
        SELECT count(*) FROM mart_revenue_variance_store v
        JOIN mart_store_scorecard s USING (comparison, store_code)
        WHERE abs(v.current_revenue - s.current_revenue) > 1e-6
           OR abs(v.base_revenue - s.base_revenue) > 1e-6
        """
    )
    assert mismatches == 0


def test_part_window_trading_gets_a_trading_weeks_effect(silver_frame, tmp_path):
    # STORE02 skips week idx 3, so against the year-ago window (idx 3-4) it traded one week
    # in the base window and two in the current one: a continuing store that is not LFL.
    gap = silver_frame.filter(
        ~((pl.col("store_code") == "STORE02") & (pl.col("shop_week") == 200609))
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

    import duckdb

    row = duckdb.sql(
        f"SELECT {', '.join(EFFECTS)}, revenue_change "
        f"FROM '{root}/gold/mart_revenue_variance_store.parquet' "
        "WHERE comparison = 'window_vs_year_ago' AND store_code = 'STORE02'"
    ).fetchone()
    base = factors(gap, {3, 4}, "STORE02")
    assert base["weeks"] == 1
    expected = lmdi(base, factors(gap, {7, 8}, "STORE02"))
    actual = dict(zip(EFFECTS, row[:-1], strict=True))
    assert actual["trading_weeks_effect"] == pytest.approx(
        expected["trading_weeks_effect"], abs=1e-6
    )
    assert actual["trading_weeks_effect"] > 0
    assert sum(actual.values()) == pytest.approx(row[-1])
