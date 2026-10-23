"""Run the whole pipeline end to end on synthetic data, with a timing for every stage.

Silver (synthetic) -> Gold marts -> forecasting panel -> three forecasters and their average ->
revenue-risk run -> dashboard snapshot. It backs `demandpulse demo` (a no-AWS tour of the
project), `demandpulse benchmark` (stage timings) and the end-to-end test. The data is invented;
nothing here describes the real dataset.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from demandpulse.app.export import export_snapshot
from demandpulse.forecasting.panel import HORIZON, ORIGINS, build_panel
from demandpulse.forecasting.runner import (
    combine_run,
    prepare_subset,
    run_model,
    save_inputs,
    save_run,
    write_report,
)
from demandpulse.risk.runner import load_store_weeks, run_risk
from demandpulse.storage import LocalStore
from demandpulse.synthetic import SOURCE, SyntheticSummary, write_synthetic_silver
from demandpulse.warehouse.build import GoldConfig, build_gold

logger = logging.getLogger(__name__)

RUN = "main"
MODELS = ("seasonal_naive", "trailing_mean_13", "lightgbm")
ENSEMBLE = "ensemble"
MIN_WEEKS = max(ORIGINS) + HORIZON

T = TypeVar("T")


@dataclass
class DemoResult:
    root: Path
    app_dir: Path
    data: SyntheticSummary
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds.values())


def run_demo(
    root: Path,
    stores: int = 40,
    weeks: int = MIN_WEEKS,
    seed: int = 7,
    threads: int = 4,
    memory_limit: str = "2GB",
) -> DemoResult:
    """Build everything under `root`; the dashboard snapshot lands in `<root>/app`."""
    if weeks < MIN_WEEKS:
        raise ValueError(
            f"The forecast origins {ORIGINS} with a {HORIZON}-week horizon need at least "
            f"{MIN_WEEKS} weeks; got {weeks}"
        )
    root = Path(root)
    store = LocalStore(root)
    config = GoldConfig(
        lake_root=str(root),
        source_glob=SOURCE,
        work_dir=root / "work",
        threads=threads,
        memory_limit=memory_limit,
    )
    app_dir = root / "app"
    seconds: dict[str, float] = {}

    def stage(name: str, work: Callable[[], T]) -> T:
        started = time.monotonic()
        result = work()
        seconds[name] = round(time.monotonic() - started, 2)
        logger.info("demo stage %s done in %.1fs", name, seconds[name])
        return result

    data = stage(
        "synthetic",
        lambda: write_synthetic_silver(root, SOURCE, stores=stores, weeks=weeks, seed=seed),
    )
    stage("gold", lambda: build_gold(config))

    def forecast() -> tuple:
        panel = build_panel(config)
        subset = prepare_subset(panel)
        save_inputs(store, panel, subset)
        for name in MODELS:
            save_run(store, RUN, run_model(name, panel, subset, workers=threads))
        save_run(store, RUN, combine_run(store, RUN, ENSEMBLE, list(MODELS)))
        write_report(store, RUN, panel, subset)
        return panel, subset

    panel, _ = stage("panel_and_forecast", forecast)
    store_weeks = load_store_weeks(config)
    stage("risk", lambda: run_risk(store, panel, store_weeks, runs=[RUN]))
    stage("export", lambda: export_snapshot(store, app_dir))
    return DemoResult(root=root, app_dir=app_dir, data=data, seconds=seconds)


def render_timings(result: DemoResult) -> str:
    """Markdown table of the stage timings and the size of the synthetic data behind them."""
    lines = [
        f"Synthetic data: {result.data.stores} stores, {result.data.weeks} weeks, "
        f"{result.data.rows:,} basket lines, {result.data.baskets:,} baskets.",
        "",
        "| stage | seconds |",
        "|---|---|",
        *(f"| {name} | {value:.1f} |" for name, value in result.seconds.items()),
        f"| total | {result.total_seconds:.1f} |",
    ]
    return "\n".join(lines)


def render_benchmark(results: list[DemoResult]) -> str:
    """One table, a column per data size, so how each stage scales with stores is visible."""
    header = ["stage", *(f"{r.data.stores} stores" for r in results)]
    stages = list(results[0].seconds)
    rows = [
        ["basket lines", *(f"{r.data.rows:,}" for r in results)],
        *([name, *(f"{r.seconds[name]:.1f}s" for r in results)] for name in stages),
        ["total", *(f"{r.total_seconds:.1f}s" for r in results)],
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)
