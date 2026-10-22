"""Run forecasting models over the shared backtest and assemble the comparison report.

Scopes: cheap models (seasonal naive, trailing mean, global LightGBM) run on every eligible
series ("full"). Per-series statistical and neural models run on the documented representative
subset only ("subset"). The report compares every model on the subset, and the full-panel models
again on all series, so no claim extends past what was actually run.
"""

import io
import json
import logging
from dataclasses import dataclass
from typing import Any

import polars as pl

from demandpulse.forecasting.backtest import (
    BacktestResult,
    Forecaster,
    ForecastError,
    run_backtest,
)
from demandpulse.forecasting.models.baselines import SeasonalNaive, TrailingMean
from demandpulse.forecasting.models.ets import ets_damped, holt_winters
from demandpulse.forecasting.models.isolated import greykite, neuralprophet
from demandpulse.forecasting.models.lgbm import LightGBMGlobal
from demandpulse.forecasting.panel import (
    HORIZON,
    ORIGINS,
    SERIES_KEY,
    YEAR_WEEKS,
    Subset,
    eligible_series,
    representative_subset,
)
from demandpulse.forecasting.report import (
    aggregated_leaderboard,
    by_attribute,
    format_table,
    horizon_buckets,
    leaderboard,
    paired_vs_baseline,
)
from demandpulse.storage import Store

logger = logging.getLogger(__name__)

BASELINE = "seasonal_naive"
FULL_SCOPE_MODELS = ("seasonal_naive", "trailing_mean_13", "lightgbm", "lightgbm_v2")
ALL_MODELS = (
    "seasonal_naive",
    "trailing_mean_13",
    "ets_damped",
    "holt_winters",
    "lightgbm",
    "lightgbm_v2",
    "neuralprophet",
    "greykite",
)
# Holt-Winters needs two full seasons (104 weeks) of history.
MODEL_ORIGINS = {"holt_winters": (104,)}
TWO_SEASONS = 2 * YEAR_WEEKS


def run_prefix(run: str) -> str:
    return f"forecast/runs/{run}"


def _read(store: Store, key: str) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(store.read_bytes(key)))


def _write(store: Store, key: str, frame: pl.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.write_parquet(buffer, compression="zstd")
    store.write_bytes(key, buffer.getvalue())


def save_inputs(store: Store, panel: pl.DataFrame, subset: Subset) -> None:
    _write(store, "forecast/panel.parquet", panel)
    _write(store, "forecast/subset.parquet", subset.series)
    store.write_bytes("forecast/subset_meta.json", json.dumps({"seed": subset.seed}).encode())


def load_inputs(store: Store) -> tuple[pl.DataFrame, Subset]:
    panel = _read(store, "forecast/panel.parquet")
    series = _read(store, "forecast/subset.parquet")
    seed = json.loads(store.read_bytes("forecast/subset_meta.json"))["seed"]
    return panel, Subset(series=series, seed=seed)


def prepare_subset(panel: pl.DataFrame) -> Subset:
    return representative_subset(panel)


def make_model(name: str, panel: pl.DataFrame, workers: int, options: dict[str, Any]) -> Forecaster:
    def eligible(origin: int) -> pl.DataFrame:
        return eligible_series(panel, origin)

    if name == "seasonal_naive":
        return SeasonalNaive()
    if name == "trailing_mean_13":
        return TrailingMean(13)
    if name == "ets_damped":
        return ets_damped(workers=workers)
    if name == "holt_winters":
        return holt_winters(workers=workers)
    if name in ("lightgbm", "lightgbm_v2"):
        feature_set = "v1" if name == "lightgbm" else "v2"
        return LightGBMGlobal(panel, eligible, num_threads=max(workers, 1), feature_set=feature_set)
    if name == "neuralprophet":
        return neuralprophet(options)
    if name == "greykite":
        return greykite({"workers": workers, **options})
    raise KeyError(f"Unknown model {name!r}. Available: {', '.join(ALL_MODELS)}")


@dataclass
class ModelRun:
    name: str
    scope: str
    result: BacktestResult
    info: dict[str, Any]


def run_model(
    name: str,
    panel: pl.DataFrame,
    subset: Subset,
    *,
    origins: tuple[int, ...] | None = None,
    workers: int = 8,
    options: dict[str, Any] | None = None,
) -> ModelRun:
    scope = "full" if name in FULL_SCOPE_MODELS else "subset"
    chosen = tuple(origins or MODEL_ORIGINS.get(name, ORIGINS))
    model = make_model(name, panel, workers, options or {})
    if scope == "full":

        def series_for_origin(origin: int) -> pl.DataFrame:
            return eligible_series(panel, origin)
    elif name == "holt_winters":

        def series_for_origin(origin: int) -> pl.DataFrame:
            # Stores that opened later lack two full years of history, which the seasonal fit
            # requires. They are left out and the number is recorded, not silently dropped.
            enough = eligible_series(panel, origin, min_history=TWO_SEASONS).select(SERIES_KEY)
            return subset.series.select(SERIES_KEY).join(enough, on=SERIES_KEY)
    else:

        def series_for_origin(origin: int) -> pl.DataFrame:
            return subset.series.select(SERIES_KEY)

    result = run_backtest(panel, model, chosen, HORIZON, series_for_origin)
    info: dict[str, Any] = {
        "model": name,
        "scope": scope,
        "origins": list(chosen),
        "horizon": HORIZON,
        "seconds": result.seconds,
        "series_per_origin": {
            str(o): int(
                result.forecasts.filter(pl.col("origin") == o).select(SERIES_KEY).unique().height
            )
            for o in chosen
        },
    }
    covered = max(info["series_per_origin"].values())
    if scope == "subset" and covered < subset.series.height:
        info["subset_series_excluded"] = subset.series.height - covered
    if hasattr(model, "fit_info"):
        info["fit_info"] = {str(k): v for k, v in model.fit_info.items()}
    if hasattr(model, "fallbacks"):
        info["ets_fits"] = model.fits
        info["ets_fallbacks"] = model.fallbacks
    return ModelRun(name=name, scope=scope, result=result, info=info)


def save_run(store: Store, run: str, model_run: ModelRun) -> None:
    prefix = run_prefix(run)
    _write(store, f"{prefix}/forecasts/{model_run.name}.parquet", model_run.result.forecasts)
    _write(store, f"{prefix}/scales/{model_run.name}.parquet", model_run.result.scales)
    store.write_bytes(
        f"{prefix}/meta/{model_run.name}.json", json.dumps(model_run.info, indent=2).encode()
    )


def load_run(
    store: Store, run: str
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, dict[str, Any]]]:
    prefix = run_prefix(run)
    keys = store.list_keys(f"{prefix}/forecasts/")
    if not keys:
        raise FileNotFoundError(f"No forecasts under {prefix}; run `demandpulse forecast backtest`")
    forecasts, scales, meta = [], [], {}
    for key in keys:
        name = key.rsplit("/", 1)[1].removesuffix(".parquet")
        forecasts.append(_read(store, key))
        scales.append(_read(store, f"{prefix}/scales/{name}.parquet"))
        meta[name] = json.loads(store.read_bytes(f"{prefix}/meta/{name}.json"))
    return pl.concat(forecasts), pl.concat(scales).unique(), meta


def combine_run(
    store: Store, run: str, name: str, components: list[str], note: str = ""
) -> ModelRun:
    """Equal-weight average of already-computed model forecasts, saved as a new model.

    No weights are fitted. The components must cover exactly the same series, origins and weeks.
    """
    if len(components) < 2:
        raise ValueError("A combination needs at least two component models")
    prefix = run_prefix(run)
    frames, scales, scopes = [], [], []
    for component in components:
        key = f"{prefix}/forecasts/{component}.parquet"
        if not store.exists(key):
            raise FileNotFoundError(f"Component {component!r} has no forecasts in run {run!r}")
        frames.append(_read(store, key).drop("model"))
        scales.append(_read(store, f"{prefix}/scales/{component}.parquet"))
        scopes.append(json.loads(store.read_bytes(f"{prefix}/meta/{component}.json"))["scope"])

    keys = ["origin", *SERIES_KEY, "horizon", "week_idx"]
    base = frames[0].select(*keys, "y")
    averaged = pl.concat(frames).group_by(keys).agg(pl.col("y_hat").mean(), pl.len().alias("count"))
    if averaged.height != base.height or (averaged["count"] != len(components)).any():
        raise ForecastError(f"{name}: components do not cover identical series, origins and weeks")
    combined = (
        base.join(averaged.select(*keys, "y_hat"), on=keys)
        .with_columns(pl.lit(name).alias("model"))
        .select("model", "origin", *SERIES_KEY, "horizon", "week_idx", "y", "y_hat")
    )
    origins = sorted(combined["origin"].unique().to_list())
    info: dict[str, Any] = {
        "model": name,
        "scope": "subset" if "subset" in scopes else "full",
        "origins": origins,
        "horizon": HORIZON,
        "seconds": {},
        "series_per_origin": {
            str(o): int(combined.filter(pl.col("origin") == o).select(SERIES_KEY).unique().height)
            for o in origins
        },
        "components": components,
        "method": "equal-weight mean of component forecasts",
        "note": note,
    }
    result = BacktestResult(
        model=name, forecasts=combined, scales=pl.concat(scales).unique(), seconds={}
    )
    return ModelRun(name=name, scope=str(info["scope"]), result=result, info=info)


def _restrict(forecasts: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    return forecasts.join(series.select(SERIES_KEY), on=SERIES_KEY)


def summary_tables(
    forecasts: pl.DataFrame,
    scales: pl.DataFrame,
    panel: pl.DataFrame,
    subset: Subset,
    meta: dict[str, dict[str, Any]] | None = None,
    origins: tuple[int, ...] = ORIGINS,
) -> dict[str, pl.DataFrame]:
    """Every table the report shows, as frames (also used for the JSON summary)."""
    on_subset = _restrict(forecasts, subset.series)
    final_origin = max(origins)
    tables: dict[str, pl.DataFrame] = {}

    at_final = on_subset.filter(pl.col("origin") == final_origin)
    coverage = at_final.group_by("model").agg(pl.struct(SERIES_KEY).n_unique().alias("n"))
    partial = [r["model"] for r in coverage.iter_rows(named=True) if r["n"] < subset.series.height]
    # Models that could not forecast every subset series get their own like-for-like table.
    common = at_final.filter(~pl.col("model").is_in(partial))
    tables["subset_final_origin"] = leaderboard(common, scales)
    tables["subset_final_origin_vs_baseline"] = paired_vs_baseline(common, BASELINE)
    for model in partial:
        covered = at_final.filter(pl.col("model") == model).select(SERIES_KEY).unique()
        same_series = at_final.join(covered, on=SERIES_KEY)
        tables[f"subset_final_origin_{model}_series"] = leaderboard(same_series, scales)
        tables[f"subset_final_origin_{model}_series_vs_baseline"] = paired_vs_baseline(
            same_series, BASELINE
        )

    origin_sets = on_subset.group_by("model").agg(pl.col("origin").unique().sort().alias("o"))
    full_origin_models = [
        row["model"] for row in origin_sets.iter_rows(named=True) if set(row["o"]) >= set(origins)
    ]
    rolling = on_subset.filter(pl.col("model").is_in(full_origin_models))
    tables["subset_rolling"] = leaderboard(rolling, scales)
    tables["subset_rolling_vs_baseline"] = paired_vs_baseline(rolling, BASELINE)
    tables["subset_by_origin"] = leaderboard(rolling, scales, ["model", "origin"])

    present = set(forecasts["model"])
    if meta is None:
        full_models = [m for m in FULL_SCOPE_MODELS if m in present]
    else:
        full_models = [m for m in present if meta.get(m, {}).get("scope") == "full"]
    full = forecasts.filter(pl.col("model").is_in(full_models))
    if full.height:
        tables["full_by_origin"] = leaderboard(full, scales, ["model", "origin"])
        tables["full_pooled"] = leaderboard(full, scales)
        tables["full_vs_baseline"] = paired_vs_baseline(full, BASELINE)
        tables["full_store_level"] = aggregated_leaderboard(full, ["store_code"], ["model"])
        tables["full_category_level"] = aggregated_leaderboard(full, ["prod_code_30"], ["model"])
        tables["full_network_level"] = aggregated_leaderboard(full, [], ["model"])
        tables["full_by_horizon"] = horizon_buckets(full)
        attrs = panel.select("store_code", "store_format").unique("store_code")
        tables["full_by_format"] = by_attribute(full, attrs, "store_format")
    return tables


def render_report(
    run: str,
    tables: dict[str, pl.DataFrame],
    meta: dict[str, dict[str, Any]],
    panel: pl.DataFrame,
    subset: Subset,
    origins: tuple[int, ...] = ORIGINS,
) -> str:
    n_series = panel.select(SERIES_KEY).unique().height
    windows = ", ".join(f"{o + 1}-{o + HORIZON}" for o in origins)
    lines = [
        f"# Forecast model comparison: run `{run}`",
        "",
        "Weekly revenue per store x category (product hierarchy level 30). Forecast horizon "
        f"{HORIZON} weeks from rolling origins at weeks {', '.join(map(str, origins))} of the "
        f"117 data weeks (test windows weeks {windows}). "
        "Models see only revenue up to the origin. Negative forecasts are clipped to zero "
        "for every model.",
        "",
        f"Panel: {n_series:,} series. Representative subset: {subset.series.height} series, "
        f"stratified by store format x revenue tier (seed {subset.seed}).",
        "",
        "WAPE = total absolute error / total actual revenue. Bias = (forecast - actual) / actual, "
        "positive means over-forecasting. MASE below 1 beats a one-year seasonal naive in-sample.",
        "",
    ]

    def section(title: str, key: str, percent: tuple[str, ...] = ("wape", "bias")) -> None:
        if key in tables and tables[key].height:
            lines.extend([f"## {title}", "", format_table(tables[key], percent), ""])

    section("Subset, final origin (all models, identical series and weeks)", "subset_final_origin")
    section(
        "Improvement over seasonal naive, subset final origin (paired bootstrap, 95% CI)",
        "subset_final_origin_vs_baseline",
        ("improvement", "ci_low", "ci_high", "win_rate"),
    )
    for key in sorted(
        k for k in tables if k.startswith("subset_final_origin_") and k.endswith("_series")
    ):
        model = key.removeprefix("subset_final_origin_").removesuffix("_series")
        count = tables[key]["series"].max()
        section(f"Subset, final origin, the {count} series {model} could forecast", key)
        section(
            f"Improvement over seasonal naive on those {count} series",
            f"{key}_vs_baseline",
            ("improvement", "ci_low", "ci_high", "win_rate"),
        )
    section("Subset, all rolling origins pooled", "subset_rolling")
    section(
        "Improvement over seasonal naive, subset rolling origins",
        "subset_rolling_vs_baseline",
        ("improvement", "ci_low", "ci_high", "win_rate"),
    )
    section("Subset, by origin", "subset_by_origin")
    section("Full panel, by origin", "full_by_origin")
    section("Full panel, pooled over origins", "full_pooled")
    section(
        "Improvement over seasonal naive, full panel",
        "full_vs_baseline",
        ("improvement", "ci_low", "ci_high", "win_rate"),
    )
    section("Full panel, accuracy summed to store level", "full_store_level")
    section("Full panel, accuracy summed to category level", "full_category_level")
    section("Full panel, accuracy summed to the whole network", "full_network_level")
    section("Full panel, by forecast horizon", "full_by_horizon")
    section("Full panel, by store format", "full_by_format")

    lines.extend(["## Run details", ""])
    for name in sorted(meta):
        info = meta[name]
        lines.append(
            f"- **{name}** ({info['scope']} scope, origins {info['origins']}): "
            f"series per origin {info['series_per_origin']}, seconds {info['seconds']}"
        )
        if "components" in info:
            lines.append(f"  - {info['method']} of {', '.join(info['components'])}. {info['note']}")
        if "subset_series_excluded" in info:
            lines.append(
                f"  - {info['subset_series_excluded']} subset series excluded "
                "(stores opened too recently for this model's history requirement)"
            )
        if "ets_fallbacks" in info:
            fallbacks = info["ets_fallbacks"]
            lines.append(f"  - fits {info['ets_fits']}, trailing-mean fallbacks {fallbacks}")
        if "fit_info" in info:
            for origin, fit in info["fit_info"].items():
                lines.append(
                    f"  - origin {origin}: {int(fit['training_rows']):,} training rows, "
                    f"{int(fit['rounds'])} rounds, inner validation WAPE "
                    f"{fit['inner_validation_wape']:.3f}, fit {fit['fit_seconds']}s"
                )
    lines.append("")
    return "\n".join(lines)


def write_report(
    store: Store,
    run: str,
    panel: pl.DataFrame,
    subset: Subset,
    origins: tuple[int, ...] = ORIGINS,
) -> tuple[str, dict[str, pl.DataFrame]]:
    forecasts, scales, meta = load_run(store, run)
    tables = summary_tables(forecasts, scales, panel, subset, meta, origins)
    text = render_report(run, tables, meta, panel, subset, origins)
    prefix = run_prefix(run)
    store.write_bytes(f"{prefix}/report.md", text.encode())
    summary = {name: table.to_dicts() for name, table in tables.items()}
    store.write_bytes(f"{prefix}/summary.json", json.dumps(summary, indent=2, default=str).encode())
    return text, tables
