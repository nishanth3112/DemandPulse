import polars as pl
import pytest

from demandpulse.ingest.schema import TRANSACTION_SCHEMA
from demandpulse.synthetic import (
    CLOSURE,
    LATE_OPENING,
    LEVEL_SHIFT,
    fiscal_week,
    write_synthetic_silver,
)


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    summary = write_synthetic_silver(root, stores=10, weeks=117, seed=3)
    frame = pl.read_parquet(root / "silver/source=synthetic/*/part-0.parquet")
    return root, summary, frame


def test_fiscal_calendar_matches_the_real_dataset_span():
    codes = [fiscal_week(i) for i in range(117)]
    assert codes[0] == 200607
    assert codes[-1] == 200819
    assert 200652 in codes
    assert 200653 not in codes
    assert codes[46] == 200701
    assert len(set(codes)) == 117


def test_summary_counts_match_what_was_written(lake):
    _, summary, frame = lake
    assert summary.rows == frame.height
    assert summary.baskets == frame["basket_id"].n_unique()
    assert summary.weeks == 117
    assert summary.stores == 10


def test_every_week_is_written_as_its_own_silver_partition(lake):
    root, _, _ = lake
    partitions = sorted((root / "silver/source=synthetic").iterdir())
    assert len(partitions) == 117
    assert partitions[0].name == "shop_week=200607"


def test_columns_and_types_are_the_silver_schema(lake):
    _, _, frame = lake
    expected = {
        **TRANSACTION_SCHEMA,
        "is_loyalty": pl.Boolean(),
        "week_start": pl.Date(),
        "week_end": pl.Date(),
    }
    assert dict(frame.schema) == expected


def test_money_has_two_decimals_and_is_positive(lake):
    _, _, frame = lake
    assert frame["spend"].min() > 0
    assert frame["spend"].dtype == pl.Decimal(12, 2)


def test_loyalty_flag_matches_customer_presence(lake):
    _, _, frame = lake
    assert (frame["is_loyalty"] == frame["cust_code"].is_not_null()).all()
    assert (frame["seg_1"].is_null() == frame["cust_code"].is_null()).all()
    share = frame["is_loyalty"].mean()
    assert 0.6 < share < 0.85


def test_basket_ids_belong_to_one_store_and_week(lake):
    _, _, frame = lake
    per_basket = frame.group_by("basket_id").agg(
        pl.col("store_code").n_unique().alias("stores"), pl.col("shop_week").n_unique().alias("w")
    )
    assert per_basket["stores"].max() == 1
    assert per_basket["w"].max() == 1


def test_dates_fall_inside_their_week(lake):
    _, _, frame = lake
    inside = (frame["shop_date"] >= frame["week_start"]) & (frame["shop_date"] <= frame["week_end"])
    assert inside.all()


def test_one_store_opens_late_one_closes_and_one_shifts_level(lake):
    _, _, frame = lake
    first_idx = {c: i for i, c in enumerate(fiscal_week(i) for i in range(117))}
    by_store = (
        frame.group_by("store_code")
        .agg(pl.col("shop_week").min().alias("first"), pl.col("shop_week").max().alias("last"))
        .to_dicts()
    )
    info = {r["store_code"]: (first_idx[r["first"]], first_idx[r["last"]]) for r in by_store}
    assert info[f"STORE{LATE_OPENING[0]:05d}"][0] == LATE_OPENING[1]
    assert info[f"STORE{CLOSURE[0]:05d}"][1] == CLOSURE[1]
    assert info["STORE00001"] == (0, 116)

    shifted = f"STORE{LEVEL_SHIFT[0]:05d}"
    weekly = (
        frame.filter(pl.col("store_code") == shifted)
        .group_by("shop_week")
        .agg(pl.col("basket_id").n_unique().alias("baskets"))
    )
    codes = {fiscal_week(i): i for i in range(117)}
    before = weekly.filter(pl.col("shop_week").map_elements(codes.get, return_dtype=pl.Int64) < 85)
    after = weekly.filter(pl.col("shop_week").map_elements(codes.get, return_dtype=pl.Int64) >= 92)
    assert after["baskets"].mean() < 0.75 * before["baskets"].mean()


def test_same_seed_gives_identical_data_and_a_different_seed_does_not(tmp_path):
    a = write_synthetic_silver(tmp_path / "a", stores=6, weeks=20, seed=1)
    b = write_synthetic_silver(tmp_path / "b", stores=6, weeks=20, seed=1)
    c = write_synthetic_silver(tmp_path / "c", stores=6, weeks=20, seed=2)
    path = "silver/source=synthetic/shop_week=200610/part-0.parquet"
    assert pl.read_parquet(tmp_path / "a" / path).equals(pl.read_parquet(tmp_path / "b" / path))
    assert a == b
    assert a.rows != c.rows


@pytest.mark.parametrize("kwargs", [{"stores": 0}, {"weeks": 0}, {"categories": 0}])
def test_empty_sizes_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError, match="at least 1"):
        write_synthetic_silver(tmp_path, **kwargs)
