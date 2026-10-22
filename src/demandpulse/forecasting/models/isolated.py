"""Run a forecaster that lives in its own Python environment, exchanging Parquet files.

NeuralProphet and Greykite cannot share the main environment (they need older pandas and numpy),
so each has its own locked uv project under `envs/`. This class writes the training panel and a
request to a temporary directory, runs the environment's worker script as a subprocess, and reads
the forecasts back. A worker that fails raises ForecastError with its stderr; nothing is swallowed.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from demandpulse.forecasting.backtest import FORECAST_COLUMNS, ForecastError
from demandpulse.forecasting.panel import SERIES_KEY

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
STDERR_TAIL_CHARS = 2000
WORKER_INPUT_COLUMNS = [*SERIES_KEY, "week_idx", "week_start", "revenue"]


def env_python(env_name: str) -> Path:
    return REPO_ROOT / "envs" / env_name / ".venv" / "bin" / "python"


class SubprocessForecaster:
    def __init__(
        self,
        name: str,
        python: Path,
        worker: Path,
        options: dict[str, object] | None = None,
        timeout_seconds: int = 7200,
    ) -> None:
        self.name = name
        self.python = Path(python)
        self.worker = Path(worker)
        self.options = options or {}
        self.timeout_seconds = timeout_seconds

    def forecast(
        self, train: pl.DataFrame, future: pl.DataFrame, origin: int, horizon: int
    ) -> pl.DataFrame:
        if not self.python.exists():
            env = self.python.parents[2].name
            raise ForecastError(
                f"{self.name}: environment not found at {self.python}. "
                f"Create it with: uv sync --project envs/{env}"
            )
        with tempfile.TemporaryDirectory(prefix=f"dp_{self.name}_") as tmp:
            directory = Path(tmp)
            train.select(WORKER_INPUT_COLUMNS).write_parquet(directory / "train.parquet")
            request = {"origin": origin, "horizon": horizon, "options": self.options}
            (directory / "request.json").write_text(json.dumps(request))
            command = [
                str(self.python),
                "-I",
                str(self.worker),
                str(directory / "train.parquet"),
                str(directory / "request.json"),
                str(directory / "forecast.parquet"),
            ]
            logger.info("%s: running worker for origin %d", self.name, origin)
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=self.timeout_seconds, check=False
            )
            if done.returncode != 0:
                tail = (done.stderr or done.stdout)[-STDERR_TAIL_CHARS:]
                raise ForecastError(f"{self.name}: worker exited {done.returncode}:\n{tail}")
            output = directory / "forecast.parquet"
            if not output.exists():
                raise ForecastError(f"{self.name}: worker produced no forecast file")
            return pl.read_parquet(output).select(FORECAST_COLUMNS)


def neuralprophet(options: dict[str, object] | None = None) -> SubprocessForecaster:
    return SubprocessForecaster(
        "neuralprophet",
        env_python("neuralprophet"),
        REPO_ROOT / "envs" / "neuralprophet" / "worker.py",
        options,
    )


def greykite(options: dict[str, object] | None = None) -> SubprocessForecaster:
    return SubprocessForecaster(
        "greykite",
        env_python("greykite"),
        REPO_ROOT / "envs" / "greykite" / "worker.py",
        options,
    )
