"""Greykite (Silverkite) worker: one model per series, fitted in parallel processes.

Standalone script run inside envs/greykite. Usage:
    worker.py TRAIN.parquet REQUEST.json OUT.parquet
Any series that fails to fit makes the whole worker fail, so a partial result never looks complete.
"""

import os

# One fit already runs in its own process; extra BLAS threads per process oversubscribe the machine
# (measured: 4 workers used 4.4 cores' worth of CPU and ran several times slower). Set before numpy.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULTS = {"yearly_order": 6, "n_changepoints": 5, "workers": 4}


def fit_one(args):
    key, dates, values, horizon, options = args
    warnings.filterwarnings("ignore")
    from greykite.framework.templates.autogen.forecast_config import (
        ForecastConfig,
        MetadataParam,
        ModelComponentsParam,
    )
    from greykite.framework.templates.forecaster import Forecaster
    from greykite.framework.templates.model_templates import ModelTemplateEnum

    frame = pd.DataFrame({"ts": pd.to_datetime(dates), "y": values})
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=horizon,
        coverage=None,
        metadata_param=MetadataParam(time_col="ts", value_col="y", freq="W-MON"),
        model_components_param=ModelComponentsParam(
            seasonality={
                "yearly_seasonality": int(options["yearly_order"]),
                "quarterly_seasonality": 0,
                "monthly_seasonality": 0,
                "weekly_seasonality": 0,
                "daily_seasonality": 0,
            },
            changepoints={
                "changepoints_dict": {
                    "method": "uniform",
                    "n_changepoints": int(options["n_changepoints"]),
                }
            },
        ),
    )
    try:
        result = Forecaster().run_forecast_config(df=frame, config=config)
        forecast = result.forecast.df["forecast"].to_numpy(dtype=float)[-horizon:]
    except Exception as err:  # noqa: BLE001 - reported to the parent, which fails the run
        return key, None, f"{type(err).__name__}: {err}"
    return key, forecast, None


def main(train_path: str, request_path: str, out_path: str) -> None:
    with open(request_path) as handle:
        request = json.load(handle)
    horizon = int(request["horizon"])
    origin = int(request["origin"])
    options = {**DEFAULTS, **request.get("options", {})}

    frame = pd.read_parquet(train_path).sort_values(["store_code", "prod_code_30", "week_idx"])
    tasks = []
    for (store, category), group in frame.groupby(["store_code", "prod_code_30"], sort=True):
        tasks.append(
            (
                (store, category),
                group["week_start"].astype(str).tolist(),
                group["revenue"].to_numpy(),
                horizon,
                options,
            )
        )

    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=int(options["workers"])) as pool:
        for key, forecast, error in pool.map(fit_one, tasks, chunksize=4):
            if error:
                failures.append(f"{key}: {error}")
                continue
            for h, value in enumerate(forecast, start=1):
                rows.append((key[0], key[1], h, origin + h, float(value)))
    if failures:
        raise SystemExit(f"{len(failures)} series failed, first: {failures[0]}")
    result = pd.DataFrame(
        rows, columns=["store_code", "prod_code_30", "horizon", "week_idx", "y_hat"]
    )
    if not np.isfinite(result["y_hat"]).all():
        raise SystemExit("non-finite Greykite forecasts")
    result.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
