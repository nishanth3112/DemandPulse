import numpy as np
import polars as pl
import pytest

from demandpulse.forecasting.backtest import ForecastError, run_backtest
from demandpulse.forecasting.features import (
    FEATURE_NAMES,
    MIN_ORIGIN,
    Encoders,
    build_matrix,
    make_features,
)
from demandpulse.forecasting.metrics import summarise
from demandpulse.forecasting.models.baselines import TrailingMean
from demandpulse.forecasting.models.lgbm import LightGBMGlobal
from demandpulse.forecasting.panel import SERIES_KEY, eligible_series
from forecast_data import make_seasonal_panel

WEEKS = 130
ORIGIN = 104
HORIZON = 13
FAST = {"min_data_in_leaf": 5, "num_leaves": 15, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def panel():
    return make_seasonal_panel()


def all_series(panel):
    keys = panel.select(SERIES_KEY).unique()
    return lambda _origin: keys


def model(panel, **overrides):
    options = {
        "params": FAST,
        "origin_stride": 3,
        "max_rounds": 60,
        "early_stopping_rounds": 10,
        "num_threads": 1,
    }
    return LightGBMGlobal(panel, all_series(panel), **{**options, **overrides})


def matrix_at(panel, origin):
    encoders = Encoders.from_panel(panel)
    calendar = dict(panel.select("week_idx", "fiscal_week").unique().iter_rows())
    visible = panel.filter(pl.col("week_idx") <= origin)
    return build_matrix(visible, encoders, origin, calendar), encoders


def test_features_have_the_documented_shape(panel):
    matrix, _ = matrix_at(panel, ORIGIN)
    x, valid, scale = make_features(matrix, ORIGIN, 5)
    assert x.shape == (matrix.n_series, len(FEATURE_NAMES))
    assert valid.all()
    assert np.isfinite(x).all()
    assert (scale > 0).all()


def test_features_match_hand_computed_values(panel):
    matrix, _ = matrix_at(panel, ORIGIN)
    x, _, scale = make_features(matrix, ORIGIN, 5)
    first = matrix.keys[0]
    series = (
        panel.filter((pl.col("store_code") == first[0]) & (pl.col("prod_code_30") == first[1]))
        .sort("week_idx")["revenue"]
        .to_numpy()
    )
    expected_scale = series[ORIGIN - 52 : ORIGIN].mean()
    names = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert scale[0] == pytest.approx(expected_scale)
    assert x[0, names["lag0"]] == pytest.approx(series[ORIGIN - 1] / expected_scale, rel=1e-5)
    assert x[0, names["lag3"]] == pytest.approx(series[ORIGIN - 4] / expected_scale, rel=1e-5)
    # target week is 109, so the same week a year earlier is week 57 (index 56)
    assert x[0, names["year_ago"]] == pytest.approx(series[56] / expected_scale, rel=1e-5)
    assert x[0, names["mean13"]] == pytest.approx(
        series[ORIGIN - 13 : ORIGIN].mean() / expected_scale, rel=1e-5
    )
    assert x[0, names["horizon"]] == 5


def test_features_cannot_see_weeks_after_the_origin(panel):
    encoders = Encoders.from_panel(panel)
    calendar = dict(panel.select("week_idx", "fiscal_week").unique().iter_rows())
    full = build_matrix(panel, encoders, WEEKS, calendar)
    baseline, _, _ = make_features(full, ORIGIN, 7)
    full.values[:, ORIGIN:] = 9999.0  # corrupt every future week
    corrupted, _, _ = make_features(full, ORIGIN, 7)
    assert np.array_equal(baseline, corrupted)


def test_matrix_refuses_data_beyond_its_last_week(panel):
    encoders = Encoders.from_panel(panel)
    calendar = dict(panel.select("week_idx", "fiscal_week").unique().iter_rows())
    with pytest.raises(ValueError, match="after last_week"):
        build_matrix(panel, encoders, ORIGIN, calendar)


def test_origins_before_the_minimum_are_rejected(panel):
    matrix, _ = matrix_at(panel, MIN_ORIGIN - 1)
    with pytest.raises(ValueError, match="before the minimum"):
        make_features(matrix, MIN_ORIGIN - 1, 1)


def test_training_rows_only_use_targets_observed_by_the_cutoff(panel):
    lgbm = model(panel)
    matrix, _ = matrix_at(panel, ORIGIN)
    for cutoff in (ORIGIN, ORIGIN - HORIZON):
        _, _, _, t, h = lgbm.training_frame(matrix, cutoff, HORIZON)
        assert len(t)
        assert (t.astype(int) + h.astype(int)).max() <= cutoff
        assert t.min() >= MIN_ORIGIN


def test_changing_future_revenue_cannot_change_any_forecast(panel):
    series = all_series(panel)
    baseline = run_backtest(panel, model(panel), (ORIGIN,), HORIZON, series)
    tampered_panel = panel.with_columns(
        revenue=pl.when(pl.col("week_idx") > ORIGIN)
        .then(pl.col("revenue") * 1000 + 3)
        .otherwise(pl.col("revenue"))
    )
    tampered = run_backtest(tampered_panel, model(tampered_panel), (ORIGIN,), HORIZON, series)

    keys = ["origin", *SERIES_KEY, "horizon"]
    joined = baseline.forecasts.join(tampered.forecasts, on=keys, suffix="_t")
    assert joined.height == baseline.forecasts.height
    assert joined["y_hat"].to_list() == pytest.approx(joined["y_hat_t"].to_list(), rel=1e-9)
    assert (joined["y"] != joined["y_t"]).any()


def test_it_beats_a_trailing_mean_on_seasonal_data(panel):
    series = all_series(panel)
    lgbm = run_backtest(panel, model(panel), (ORIGIN,), HORIZON, series).forecasts
    naive = run_backtest(panel, TrailingMean(13), (ORIGIN,), HORIZON, series).forecasts
    wape_lgbm = summarise(lgbm, ["model"])["wape"][0]
    wape_naive = summarise(naive, ["model"])["wape"][0]
    assert wape_lgbm < wape_naive
    assert wape_lgbm < 0.15


def test_forecasts_scale_with_the_size_of_the_series(panel):
    result = run_backtest(panel, model(panel), (ORIGIN,), HORIZON, all_series(panel)).forecasts
    by_store = result.group_by("store_code").agg(pl.col("y").sum(), pl.col("y_hat").sum())
    for row in by_store.iter_rows(named=True):
        assert row["y_hat"] == pytest.approx(row["y"], rel=0.25)  # stores differ 100x in size


def test_each_origin_is_fit_once_and_recorded(panel):
    lgbm = model(panel)
    run_backtest(panel, lgbm, (ORIGIN,), HORIZON, all_series(panel))
    run_backtest(panel, lgbm, (ORIGIN,), HORIZON, all_series(panel))
    assert list(lgbm.fit_info) == [ORIGIN]
    info = lgbm.fit_info[ORIGIN]
    assert info["training_rows"] > 0
    assert info["rounds"] >= 50
    assert 0 <= info["inner_validation_wape"] < 1


def test_series_with_short_history_are_refused_not_guessed(panel):
    short = panel.filter(~((pl.col("store_code") == "S1") & (pl.col("week_idx") <= 60)))
    lgbm = LightGBMGlobal(
        short,
        lambda _o: short.select(SERIES_KEY).unique(),
        params=FAST,
        origin_stride=3,
        max_rounds=30,
        num_threads=1,
    )
    with pytest.raises(ForecastError, match="incomplete history"):
        run_backtest(short, lgbm, (ORIGIN,), HORIZON, lambda _o: short.select(SERIES_KEY).unique())


def test_eligibility_matches_the_models_history_requirement(panel):
    eligible = eligible_series(panel, ORIGIN, horizon=HORIZON)
    assert eligible.height == panel.select(SERIES_KEY).unique().height
    assert eligible["history_weeks"].min() >= MIN_ORIGIN


def test_feature_sets_select_columns_and_name_the_model(panel):
    v1 = model(panel)
    v2 = model(panel, feature_set="v2")
    assert (v1.name, v2.name) == ("lightgbm", "lightgbm_v2")
    assert len(v1.feature_names) < len(v2.feature_names)
    assert set(v1.feature_names) < set(v2.feature_names)
    with pytest.raises(KeyError, match="Unknown feature set"):
        model(panel, feature_set="v9")


def test_momentum_features_match_hand_computed_values(panel):
    matrix, _ = matrix_at(panel, ORIGIN)
    x, _, scale = make_features(matrix, ORIGIN, 5)
    names = {name: i for i, name in enumerate(FEATURE_NAMES)}
    first = matrix.keys[0]
    series = (
        panel.filter((pl.col("store_code") == first[0]) & (pl.col("prod_code_30") == first[1]))
        .sort("week_idx")["revenue"]
        .to_numpy()
    )
    recent = series[ORIGIN - 4 : ORIGIN]
    year_ago = series[ORIGIN - 4 - 52 : ORIGIN - 52]
    expected = np.log((recent.sum() + 1) / (year_ago.sum() + 1))
    assert x[0, names["yoy_log_ratio4"]] == pytest.approx(expected, rel=1e-4)
    assert x[0, names["median4"]] == pytest.approx(np.median(recent) / scale[0], rel=1e-5)


def test_network_momentum_is_shared_by_every_series(panel):
    matrix, _ = matrix_at(panel, ORIGIN)
    x, _, _ = make_features(matrix, ORIGIN, 5)
    names = {name: i for i, name in enumerate(FEATURE_NAMES)}
    column = x[:, names["network_yoy_log_ratio13"]]
    assert np.ptp(column) == 0
    total = (
        panel.filter(pl.col("week_idx") <= ORIGIN)
        .group_by("week_idx")
        .agg(pl.col("revenue").sum())
        .sort("week_idx")["revenue"]
        .to_numpy()
    )
    expected = np.log(
        (total[ORIGIN - 13 : ORIGIN].sum() + 1) / (total[ORIGIN - 13 - 52 : ORIGIN - 52].sum() + 1)
    )
    assert column[0] == pytest.approx(expected, rel=1e-4)
