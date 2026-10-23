"""Revenue-risk pipeline: targets, shortfall probabilities, ranking backtest, classifier, report.

Forecasts come from the forecasting runs in the lake. Everything is a historical backtest on the
synthetic dataset: the targets are hypothetical, the "future" windows have already happened, and
no result here describes live operations or realised financial gain.
"""

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from demandpulse.forecasting.panel import HORIZON
from demandpulse.forecasting.report import format_table
from demandpulse.risk.calibration import brier_score, reliability_table
from demandpulse.risk.classifier import ClassifierResult, evaluate_classifier, feature_table
from demandpulse.risk.drivers import driver_changes
from demandpulse.risk.ranking import (
    RANDOM,
    add_scores,
    own_trend_windows,
    ranking_metrics,
    summarise_ranking,
    with_random_baseline,
)
from demandpulse.risk.recommend import recommendations
from demandpulse.risk.scenario import SCENARIOS, scenario_table
from demandpulse.risk.targets import target_growth, window_table
from demandpulse.risk.uncertainty import MIN_POOL, shortfall_risk
from demandpulse.storage import Store
from demandpulse.warehouse.build import GoldConfig, connect

logger = logging.getLogger(__name__)

RISK_PREFIX = "risk"
PRIMARY_MODEL = "ensemble"
MODELS = ("ensemble", "lightgbm", "seasonal_naive", "trailing_mean_13")
LEVELS = {"store": ["store_code"], "store_category": ["store_code", "prod_code_30"]}
TOP_K = 20
REPORT_K = 0.10
FORECAST_KEYS = ["model", "origin", "store_code", "prod_code_30", "week_idx"]
STORE_WEEK_SQL = """
SELECT f.store_code, d.week_idx,
       CAST(f.revenue AS DOUBLE) AS revenue,
       CAST(f.baskets AS DOUBLE) AS baskets,
       CAST(f.units AS DOUBLE) AS units
FROM read_parquet('{path}/fct_store_week.parquet') f
JOIN read_parquet('{path}/dim_date.parquet') d USING (shop_week)
ORDER BY f.store_code, d.week_idx
"""


@dataclass
class RiskResults:
    primary: str
    runs: list[str]
    origins: list[int]
    pool_origins: list[int]
    growth: pl.DataFrame
    windows: dict[str, pl.DataFrame]
    ranking: dict[str, pl.DataFrame]
    ranking_summary: dict[str, pl.DataFrame]
    classifier: dict[str, dict[str, Any]]
    recommendations: pl.DataFrame
    scenarios: pl.DataFrame
    notes: list[str] = field(default_factory=list)


def load_forecasts(store: Store, runs: list[str], models: tuple[str, ...] = MODELS) -> pl.DataFrame:
    """Forecasts of `models` from every run, which must not overlap on model, origin and week."""
    frames = []
    for run in runs:
        for model in models:
            key = f"forecast/runs/{run}/forecasts/{model}.parquet"
            if store.exists(key):
                frames.append(pl.read_parquet(io.BytesIO(store.read_bytes(key))))
    if not frames:
        raise FileNotFoundError(f"No forecasts for {list(models)} in runs {runs}")
    forecasts = pl.concat(frames)
    if forecasts.select(FORECAST_KEYS).is_duplicated().any():
        raise ValueError(f"Runs {runs} contain the same model and origin more than once")
    counts = forecasts.group_by("model").agg(pl.col("origin").n_unique())
    if counts["origin"].n_unique() != 1:
        raise ValueError(f"Models cover different origins: {counts.to_dicts()}")
    return forecasts


def load_store_weeks(config: GoldConfig) -> pl.DataFrame:
    """Store-week revenue, baskets and units from Gold; baskets are true store baskets."""
    con = connect(config, database=":memory:")
    try:
        return con.execute(STORE_WEEK_SQL.format(path=config.gold_prefix)).pl()
    finally:
        con.close()


def choose_split(origins: list[int], horizon: int = HORIZON) -> tuple[list[int], int, int] | None:
    """Latest (train origins, calibration origin, test origin) whose labels are known in time."""
    test = max(origins)
    calibration = [o for o in origins if o + horizon <= test]
    if not calibration:
        return None
    calibrate = max(calibration)
    train = [o for o in origins if o + horizon <= calibrate]
    return (train, calibrate, test) if train else None


def build_windows(
    forecasts: pl.DataFrame, panel: pl.DataFrame, level: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Window tables for every model plus the own-trend baseline, with risk estimates and scores."""
    growth_rows, pieces = [], []
    for origin in sorted(forecasts["origin"].unique().to_list()):
        growth = target_growth(panel, origin)
        growth_rows.extend((origin, name, rate) for name, rate in growth.items())
        pieces.append(
            window_table(forecasts.filter(pl.col("origin") == origin), panel, level, growth)
        )
    raw = pl.concat(pieces)
    raw = pl.concat([raw, own_trend_windows(raw, panel, level).select(raw.columns)])
    growth_table = pl.DataFrame(
        growth_rows,
        schema={"origin": pl.Int64, "target_def": pl.String, "growth": pl.Float64},
        orient="row",
    )
    return add_scores(shortfall_risk(raw)), growth_table


def _classifier(
    windows: pl.DataFrame,
    panel: pl.DataFrame,
    level: list[str],
    primary: str,
    origins: list[int],
) -> dict[str, Any]:
    split = choose_split(origins)
    if split is None:
        return {"skipped": "not enough origins for train, calibration and test windows in order"}
    train, calibration, test = split
    outcome: dict[str, Any] = {"split": {"train": train, "calibration": calibration, "test": test}}
    for target_def in sorted(windows["target_def"].unique().to_list()):
        rows = windows.filter((pl.col("model") == primary) & (pl.col("target_def") == target_def))
        features = feature_table(rows, panel, level)
        result = evaluate_classifier(features, train, calibration, test)
        ranking = ranking_metrics(
            result.predictions, level, scores=("p_classifier", "p_shortfall", "relative_gap")
        )
        outcome[target_def] = {"result": result, "ranking": ranking}
    return outcome


def compute_risk(
    forecasts: pl.DataFrame,
    panel: pl.DataFrame,
    store_weeks: pl.DataFrame | None,
    runs: list[str],
    primary: str = PRIMARY_MODEL,
    top_k: int = TOP_K,
) -> RiskResults:
    if primary not in set(forecasts["model"]):
        raise ValueError(f"Primary model {primary!r} is not in the forecasts")
    origins = sorted(forecasts["origin"].unique().to_list())
    windows, ranking, summaries, classifier = {}, {}, {}, {}
    growth = pl.DataFrame()
    pool_origins: list[int] = []
    for name, level in LEVELS.items():
        logger.info("risk: %s level", name)
        windows[name], growth = build_windows(forecasts, panel, level)
        pool_origins = sorted(
            windows[name].filter(pl.col("pool_size") >= MIN_POOL)["origin"].unique().to_list()
        )
        ranking[name] = with_random_baseline(ranking_metrics(windows[name], level))
        summaries[name] = summarise_ranking(ranking[name], pool_origins or None)
        classifier[name] = _classifier(windows[name], panel, level, primary, origins)

    notes: list[str] = []
    recs, scenarios = pl.DataFrame(), pl.DataFrame()
    final = max(origins)
    if final in pool_origins:
        attributes = panel.select("store_code", "store_format", "store_region").unique("store_code")
        drivers = (
            driver_changes(store_weeks, final)
            if store_weeks is not None
            else pl.DataFrame(schema={"store_code": pl.String})
        )
        rec_frames, scenario_frames = [], []
        at_final = windows["store"].filter(
            (pl.col("model") == primary) & (pl.col("origin") == final)
        )
        for target_def in sorted(at_final["target_def"].unique().to_list()):
            risk = at_final.filter(pl.col("target_def") == target_def).join(
                attributes, on="store_code", how="left"
            )
            top = recommendations(risk, drivers, top_k).with_columns(
                pl.lit(target_def).alias("target_def")
            )
            rec_frames.append(top)
            scenario_frames.append(
                scenario_table(top.select("store_code", "target_def", "forecast", "target"))
            )
        recs = pl.concat(rec_frames, how="diagonal")
        scenarios = pl.concat(scenario_frames)
    else:
        notes.append("No probability estimates at the final origin, so no recommendations.")
    return RiskResults(
        primary=primary,
        runs=runs,
        origins=origins,
        pool_origins=pool_origins,
        growth=growth,
        windows=windows,
        ranking=ranking,
        ranking_summary=summaries,
        classifier=classifier,
        recommendations=recs,
        scenarios=scenarios,
        notes=notes,
    )


def _pct(frame: pl.DataFrame, *columns: str) -> str:
    return format_table(frame, tuple(columns))


def _ranking_rows(summary: pl.DataFrame) -> pl.DataFrame:
    return (
        summary.filter(pl.col("k_fraction") == REPORT_K)
        .select(
            "target_def",
            "model",
            "score",
            "origins",
            "base_rate",
            "precision",
            "recall",
            "lift",
            "capture",
        )
        .sort(["target_def", "precision"], descending=[False, True], nulls_last=True)
    )


def render_report(results: RiskResults) -> str:
    primary = results.primary
    windows_text = ", ".join(f"{o + 1}-{o + HORIZON}" for o in results.origins)
    lines = [
        "# Revenue risk report",
        "",
        f"Forecast runs: {', '.join(f'`{r}`' for r in results.runs)}. Primary model: `{primary}`. "
        f"Origins: {', '.join(map(str, results.origins))} (13-week windows {windows_text} of the "
        "117 data weeks).",
        "",
        "**How to read this.** The data has no budgets, so every target is hypothetical: last "
        "year's revenue for the same weeks, either unchanged (`flat_yoy`) or grown by the "
        "network's like-for-like growth over the 13 weeks before the origin "
        "(`trailing_growth`). All windows are historical and already observed, so results are "
        "backtest measurements on synthetic data. They are not live results, and none of them is "
        "a claim about profit, margin or promotion effects. The scenario section is arithmetic "
        "on forecasts, not a causal estimate.",
        "",
        f"Estimated shortfall probabilities use only forecast errors from windows that had "
        f"finished by each origin, so they exist for origins {results.pool_origins or 'none'}; "
        "the earlier origins have no history to learn from. The `ensemble` was chosen after "
        "seeing the forecast comparison and is not independently validated.",
        "",
        "## Target growth assumptions (observed at each origin)",
        "",
        _pct(results.growth, "growth"),
        "",
    ]
    for level_name, level_label in (("store", "store"), ("store_category", "store x category")):
        windows = results.windows[level_name]
        base = (
            windows.filter(pl.col("model") == primary)
            .group_by(["target_def", "origin"])
            .agg(
                pl.len().alias("entities"),
                pl.col("realised_miss").mean().alias("share_missed_target"),
                pl.col("forecast_miss").mean().alias("share_forecast_below_target"),
            )
            .sort(["target_def", "origin"])
        )
        lines.extend(
            [
                f"## {level_label.capitalize()} level",
                "",
                "### Realised outcomes (observed in the backtest)",
                "",
                _pct(base, "share_missed_target", "share_forecast_below_target"),
                "",
                f"### Probability calibration, `{primary}`, origins with an estimate",
                "",
            ]
        )
        scored = windows.filter((pl.col("model") == primary) & pl.col("p_shortfall").is_not_null())
        if scored.height:
            for target_def in sorted(scored["target_def"].unique().to_list()):
                part = scored.filter(pl.col("target_def") == target_def)
                prob, label = part["p_shortfall"].to_numpy(), part["realised_miss"].to_numpy()
                lines.extend(
                    [
                        f"`{target_def}`: Brier score {brier_score(prob, label):.3f} against "
                        f"{brier_score(np.full(len(label), label.mean()), label):.3f} for always "
                        "predicting the pooled miss rate (lower is better; the pooled rate "
                        "uses the answers, so it is an optimistic reference).",
                        "",
                        _pct(
                            reliability_table(prob, label),
                            "bin_low",
                            "bin_high",
                            "mean_predicted",
                            "observed_rate",
                        ),
                        "",
                    ]
                )
        lines.extend(
            [
                f"### Ranking at the top {REPORT_K:.0%} (mean over origins "
                f"{results.pool_origins or 'with data'})",
                "",
                "precision = share of the top group that really missed target; recall = share of "
                "all misses caught; lift = precision over the overall miss rate; capture = share "
                "of total revenue shortfall in the top group. `random` has exact expected values. "
                "`own_trend` forecasts last year grown by the entity's own recent growth. A score "
                "with every entity tied has no row values.",
                "",
                _pct(
                    _ranking_rows(results.ranking_summary[level_name]),
                    "base_rate",
                    "precision",
                    "recall",
                    "capture",
                ),
                "",
            ]
        )
        outcome = results.classifier[level_name]
        lines.extend([f"### Classifier ({level_label})", ""])
        if "skipped" in outcome:
            lines.extend([f"Skipped: {outcome['skipped']}.", ""])
            continue
        split = outcome["split"]
        lines.extend(
            [
                f"Train origins {split['train']}, calibration origin {split['calibration']}, "
                f"test origin {split['test']}. This is a single test window: it shows whether "
                "the classifier worked once, not that it is better than the forecast-based "
                "probability. Hyperparameters were fixed before the test window was scored.",
                "",
            ]
        )
        for target_def in sorted(k for k in outcome if k != "split"):
            result: ClassifierResult = outcome[target_def]["result"]
            metrics = pl.DataFrame([{"metric": k, "value": v} for k, v in result.metrics.items()])
            lines.extend(
                [
                    f"`{target_def}`",
                    "",
                    format_table(metrics, ()),
                    "",
                    f"Top-group ranking on the test window (K = {REPORT_K:.0%}):",
                    "",
                    _pct(
                        outcome[target_def]["ranking"]
                        .filter(pl.col("k_fraction") == REPORT_K)
                        .select(
                            "score", "base_rate", "precision", "recall", "lift", "capture", "tied"
                        ),
                        "base_rate",
                        "precision",
                        "recall",
                        "capture",
                    ),
                    "",
                    "Feature importance (gain):",
                    "",
                    format_table(result.importance, ()),
                    "",
                ]
            )
    lines.extend(["## Stores to look at first (final origin, store level)", ""])
    if results.recommendations.height:
        for target_def in sorted(results.recommendations["target_def"].unique().to_list()):
            part = results.recommendations.filter(pl.col("target_def") == target_def)
            lines.extend(
                [
                    f"`{target_def}`: top {min(10, part.height)} of {part.height} by expected "
                    "revenue below target.",
                    "",
                    *(f"{r['rank']}. {r['evidence']}" for r in part.head(10).to_dicts()),
                    "",
                ]
            )
        lines.extend(
            [
                "## Scenario arithmetic (hypothetical, for the stores above)",
                "",
                "Each scenario multiplies the store's forecast by (1 + traffic) x (1 + basket "
                "size) x (1 + price/mix). It shows how large a driver change would be needed, "
                "not that any action can deliver it.",
                "",
                _pct(
                    results.scenarios.group_by(["target_def", "scenario"], maintain_order=True)
                    .agg(
                        pl.len().alias("stores"),
                        pl.col("meets_target").mean().alias("share_meeting_target"),
                        pl.col("gap_closed_share").mean().alias("mean_gap_closed"),
                        pl.col("required_uplift").median().alias("median_uplift_needed"),
                    )
                    .sort(["target_def", "scenario"]),
                    "share_meeting_target",
                    "mean_gap_closed",
                    "median_uplift_needed",
                ),
                "",
                "Scenarios: "
                + "; ".join(
                    f"{name} = traffic {t:+.0%}, basket size {b:+.0%}, price/mix {p:+.0%}"
                    for name, (t, b, p) in SCENARIOS.items()
                )
                + ".",
                "",
            ]
        )
    else:
        lines.extend(["No recommendations were produced.", ""])
    for note in results.notes:
        lines.extend([f"- {note}"])
    return "\n".join(lines) + "\n"


def _put(store: Store, key: str, frame: pl.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    store.write_bytes(key, buffer.getvalue())


def save_results(store: Store, results: RiskResults) -> str:
    """Write every output under `risk/` and return the report text."""
    prefix = RISK_PREFIX
    _put(store, f"{prefix}/growth.parquet", results.growth)
    for name in LEVELS:
        windows = results.windows[name]
        # Store x category windows for every model are large; the app needs the primary model.
        kept = windows if name == "store" else windows.filter(pl.col("model") == results.primary)
        _put(store, f"{prefix}/windows_{name}.parquet", kept)
        _put(store, f"{prefix}/ranking_{name}.parquet", results.ranking[name])
        _put(store, f"{prefix}/ranking_summary_{name}.parquet", results.ranking_summary[name])
        outcome = results.classifier[name]
        payload: dict[str, Any] = {k: v for k, v in outcome.items() if k in {"split", "skipped"}}
        for target_def in (k for k in outcome if k not in {"split", "skipped"}):
            result: ClassifierResult = outcome[target_def]["result"]
            payload[target_def] = {
                "metrics": result.metrics,
                "importance": result.importance.to_dicts(),
                "reliability": result.reliability.to_dicts(),
                "ranking": outcome[target_def]["ranking"].to_dicts(),
            }
        store.write_bytes(
            f"{prefix}/classifier_{name}.json", json.dumps(payload, indent=2).encode()
        )
    if results.recommendations.height:
        _put(store, f"{prefix}/recommendations.parquet", results.recommendations)
        _put(store, f"{prefix}/scenarios.parquet", results.scenarios)
    text = render_report(results)
    store.write_bytes(f"{prefix}/report.md", text.encode())
    summary = {
        "primary": results.primary,
        "runs": results.runs,
        "origins": results.origins,
        "pool_origins": results.pool_origins,
        "random_baseline": RANDOM,
    }
    store.write_bytes(f"{prefix}/summary.json", json.dumps(summary, indent=2).encode())
    return text


def run_risk(
    store: Store,
    panel: pl.DataFrame,
    store_weeks: pl.DataFrame | None,
    runs: list[str],
    primary: str = PRIMARY_MODEL,
    top_k: int = TOP_K,
) -> tuple[str, RiskResults]:
    forecasts = load_forecasts(store, runs)
    results = compute_risk(forecasts, panel, store_weeks, runs, primary, top_k)
    return save_results(store, results), results
