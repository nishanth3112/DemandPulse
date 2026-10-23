import polars as pl
import pytest

from demandpulse.risk.scenario import SCENARIOS, required_uplift, scenario_table


def frame(forecast=100.0, target=110.0) -> pl.DataFrame:
    return pl.DataFrame({"store_code": ["S1"], "forecast": [forecast], "target": [target]})


def test_required_uplift_is_the_increase_that_reaches_target():
    assert required_uplift(frame())["required_uplift"][0] == pytest.approx(0.10)


def test_required_uplift_is_zero_when_forecast_already_meets_target():
    assert required_uplift(frame(forecast=120.0))["required_uplift"][0] == 0.0


def test_required_uplift_is_null_without_a_positive_forecast():
    assert required_uplift(frame(forecast=0.0))["required_uplift"][0] is None


def test_scenario_multiplies_the_three_drivers():
    table = scenario_table(frame(), {"x": (0.10, -0.05, 0.02)})
    assert table["scenario_forecast"][0] == pytest.approx(100.0 * 1.10 * 0.95 * 1.02)
    assert table["traffic_change"][0] == 0.10


def test_scenario_reports_remaining_gap_and_share_closed():
    table = scenario_table(frame(), {"traffic": (0.05, 0.0, 0.0)})
    row = table.row(0, named=True)
    assert row["gap_before"] == pytest.approx(10.0)
    assert row["gap_after"] == pytest.approx(5.0)
    assert row["gap_closed_share"] == pytest.approx(0.5)
    assert row["meets_target"] is False


def test_a_scenario_that_overshoots_closes_the_whole_gap_and_no_more():
    row = scenario_table(frame(), {"big": (0.5, 0.0, 0.0)}).row(0, named=True)
    assert row["gap_after"] == 0.0
    assert row["gap_closed_share"] == 1.0
    assert row["meets_target"] is True


def test_no_gap_means_no_share_to_close():
    row = scenario_table(frame(forecast=120.0), {"x": (0.0, 0.0, 0.0)}).row(0, named=True)
    assert row["gap_before"] == 0.0
    assert row["gap_closed_share"] is None
    assert row["meets_target"] is True


def test_default_scenarios_produce_one_row_each_and_keep_other_columns():
    table = scenario_table(frame())
    assert table["scenario"].to_list() == list(SCENARIOS)
    assert set(table["store_code"]) == {"S1"}


def test_a_driver_change_of_minus_100_percent_is_rejected():
    with pytest.raises(ValueError, match="-100%"):
        scenario_table(frame(), {"bad": (-1.0, 0.0, 0.0)})
