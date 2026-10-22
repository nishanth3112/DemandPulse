import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from demandpulse.forecasting.backtest import ForecastError, run_backtest
from demandpulse.forecasting.models.isolated import (
    SubprocessForecaster,
    env_python,
    greykite,
    neuralprophet,
)
from demandpulse.forecasting.panel import SERIES_KEY

ORIGIN = 91
HORIZON = 13
FIRST_MONDAY = date(2006, 4, 10)

ECHO_WORKER = """
import json, sys
import pandas as pd

train = pd.read_parquet(sys.argv[1])
request = json.load(open(sys.argv[2]))
assert train["week_idx"].max() <= request["origin"], "worker was given future weeks"
last = train.sort_values("week_idx").groupby(["store_code", "prod_code_30"])["revenue"].last()
rows = []
factor = request["options"].get("factor", 1.0)
for (store, category), value in last.items():
    for h in range(1, request["horizon"] + 1):
        rows.append((store, category, h, request["origin"] + h, value * factor))
columns = ["store_code", "prod_code_30", "horizon", "week_idx", "y_hat"]
pd.DataFrame(rows, columns=columns).to_parquet(sys.argv[3])
"""


def make_panel(n_series: int = 4, n_weeks: int = 117) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for s in range(n_series):
        level = 100.0 * (s + 1)
        for week in range(1, n_weeks + 1):
            seasonal = 1 + 0.4 * np.sin(2 * np.pi * week / 52 + s)
            revenue = level * seasonal * rng.lognormal(0, 0.05)
            start = FIRST_MONDAY + timedelta(days=7 * (week - 1))
            rows.append(
                (
                    f"S{s}",
                    "G1",
                    "LS",
                    "E01",
                    week,
                    200600 + week,
                    (week - 1) % 52 + 1,
                    start,
                    revenue,
                )
            )
    return pl.DataFrame(
        rows,
        schema={
            "store_code": pl.String,
            "prod_code_30": pl.String,
            "store_format": pl.String,
            "store_region": pl.String,
            "week_idx": pl.Int64,
            "shop_week": pl.Int64,
            "fiscal_week": pl.Int64,
            "week_start": pl.Date,
            "revenue": pl.Float64,
        },
        orient="row",
    )


def fake(tmp_path: Path, script: str, **kwargs) -> SubprocessForecaster:
    worker = tmp_path / "worker.py"
    worker.write_text(script)
    return SubprocessForecaster("fake", Path(sys.executable), worker, **kwargs)


def keys(panel):
    series = panel.select(SERIES_KEY).unique()
    return lambda _origin: series


def test_subprocess_round_trip_returns_valid_forecasts(tmp_path):
    panel = make_panel()
    model = fake(tmp_path, ECHO_WORKER, options={"factor": 2.0})
    result = run_backtest(panel, model, (ORIGIN,), HORIZON, keys(panel))
    assert result.forecasts.height == 4 * HORIZON
    last = panel.filter(pl.col("week_idx") == ORIGIN).select(*SERIES_KEY, "revenue")
    joined = result.forecasts.filter(pl.col("horizon") == 1).join(last, on=SERIES_KEY)
    assert (joined["y_hat"] == joined["revenue"] * 2.0).all()  # options reach the worker


def test_worker_only_receives_weeks_up_to_the_origin(tmp_path):
    # ECHO_WORKER asserts this inside the subprocess; a violation would fail the run.
    panel = make_panel()
    model = fake(tmp_path, ECHO_WORKER)
    run_backtest(panel, model, (78, ORIGIN), HORIZON, keys(panel))


def test_failing_worker_raises_with_its_error_output(tmp_path):
    model = fake(tmp_path, "import sys\nsys.stderr.write('model exploded')\nsys.exit(3)\n")
    panel = make_panel()
    with pytest.raises(ForecastError, match=r"exited 3[\s\S]*model exploded"):
        run_backtest(panel, model, (ORIGIN,), HORIZON, keys(panel))


def test_worker_that_writes_nothing_is_an_error(tmp_path):
    model = fake(tmp_path, "pass\n")
    panel = make_panel()
    with pytest.raises(ForecastError, match="no forecast file"):
        run_backtest(panel, model, (ORIGIN,), HORIZON, keys(panel))


def test_missing_environment_explains_how_to_create_it(tmp_path):
    missing = tmp_path / "envs" / "ghost" / ".venv" / "bin" / "python"
    model = SubprocessForecaster("ghost", missing, tmp_path / "worker.py")
    panel = make_panel()
    with pytest.raises(ForecastError, match="uv sync --project envs/ghost"):
        run_backtest(panel, model, (ORIGIN,), HORIZON, keys(panel))


def test_incomplete_worker_output_is_rejected_by_the_harness(tmp_path):
    script = ECHO_WORKER.replace('range(1, request["horizon"] + 1)', "range(1, 3)")
    model = fake(tmp_path, script)
    panel = make_panel()
    with pytest.raises(ForecastError, match="expected"):
        run_backtest(panel, model, (ORIGIN,), HORIZON, keys(panel))


def test_request_file_is_valid_json_with_the_documented_fields(tmp_path):
    seen_path = tmp_path / "seen.json"
    script = (
        f"import shutil, sys\nshutil.copy(sys.argv[2], {str(seen_path)!r})\nraise SystemExit(5)\n"
    )
    model = fake(tmp_path, script, options={"epochs": 3})
    panel = make_panel()
    with pytest.raises(ForecastError):
        run_backtest(panel, model, (ORIGIN,), HORIZON, keys(panel))
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen == {"origin": ORIGIN, "horizon": HORIZON, "options": {"epochs": 3}}


def require_env(name: str) -> None:
    if not env_python(name).exists():
        pytest.skip(f"environment envs/{name} not created (uv sync --project envs/{name})")


@pytest.mark.integration
def test_real_neuralprophet_worker_produces_valid_forecasts():
    require_env("neuralprophet")
    panel = make_panel(n_series=3)
    model = neuralprophet({"epochs": 20})
    result = run_backtest(panel, model, (104,), HORIZON, keys(panel))
    assert result.forecasts.height == 3 * HORIZON
    assert result.forecasts["y_hat"].min() >= 0


@pytest.mark.integration
def test_real_greykite_worker_produces_valid_forecasts():
    require_env("greykite")
    panel = make_panel(n_series=3)
    model = greykite({"workers": 2})
    result = run_backtest(panel, model, (104,), HORIZON, keys(panel))
    assert result.forecasts.height == 3 * HORIZON
    assert result.forecasts["y_hat"].min() >= 0
