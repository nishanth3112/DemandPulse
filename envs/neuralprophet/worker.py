"""NeuralProphet worker: one joint model with local (per-series) trend and seasonality.

Standalone script run inside envs/neuralprophet. Usage:
    worker.py TRAIN.parquet REQUEST.json OUT.parquet
Training data holds revenue only up to the forecast origin; nothing after it is ever passed in.
"""

import json
import logging
import sys
import warnings

import numpy as np
import pandas as pd

DEFAULTS = {
    "epochs": 60,
    "batch_size": 128,
    "learning_rate": 0.1,
    "yearly_seasonality": 6,
    "n_changepoints": 5,
    "seed": 20061,
}


def main(train_path: str, request_path: str, out_path: str) -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("NP").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    with open(request_path) as handle:
        request = json.load(handle)
    horizon = int(request["horizon"])
    options = {**DEFAULTS, **request.get("options", {})}

    from neuralprophet import NeuralProphet, set_random_seed

    set_random_seed(int(options["seed"]))
    frame = pd.read_parquet(train_path)
    frame["ID"] = frame["store_code"] + "|" + frame["prod_code_30"]
    frame["ds"] = pd.to_datetime(frame["week_start"])
    frame = frame.rename(columns={"revenue": "y"})[["ID", "ds", "y"]].sort_values(["ID", "ds"])
    last_ds = frame["ds"].max()

    model = NeuralProphet(
        n_forecasts=1,
        n_lags=0,
        epochs=int(options["epochs"]),
        batch_size=int(options["batch_size"]),
        learning_rate=float(options["learning_rate"]),
        yearly_seasonality=int(options["yearly_seasonality"]),
        weekly_seasonality=False,
        daily_seasonality=False,
        n_changepoints=int(options["n_changepoints"]),
        trend_global_local="local",
        season_global_local="local",
        accelerator="cpu",
    )
    model.fit(frame, freq="W-MON", progress=None)
    future = model.make_future_dataframe(frame, periods=horizon, n_historic_predictions=False)
    predicted = model.predict(future)

    predicted["horizon"] = ((predicted["ds"] - last_ds).dt.days // 7).astype(int)
    predicted = predicted[(predicted["horizon"] >= 1) & (predicted["horizon"] <= horizon)]
    split = predicted["ID"].str.split("|", n=1, expand=True)
    result = pd.DataFrame(
        {
            "store_code": split[0].to_numpy(),
            "prod_code_30": split[1].to_numpy(),
            "horizon": predicted["horizon"].to_numpy(dtype="int64"),
            "week_idx": (int(request["origin"]) + predicted["horizon"]).to_numpy(dtype="int64"),
            "y_hat": predicted["yhat1"].to_numpy(dtype="float64"),
        }
    )
    if not np.isfinite(result["y_hat"]).all():
        raise SystemExit("non-finite NeuralProphet forecasts")
    result.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
