"""Rank entities by risk of missing target and score the ranking against what happened.

A ranking is only useful if the entities it puts first are the ones that really missed. For each
model, origin and target definition the entities are sorted by a risk score and the top K (a
fraction of all entities) are compared with the realised outcome:

    precision@K  share of the top K that really fell short of target
    recall@K     share of all entities that fell short that the top K caught
    lift         precision@K divided by the share of all entities that fell short (random = 1)
    capture      share of the total revenue shortfall that sits in the top K

Baselines are `own_trend` (last year's revenue grown by the entity's own recent year-over-year
growth, so no model is involved) and `random`, whose expected values are exact: precision equals
the overall miss rate, recall and capture equal K / N. A score with no variation (every entity
tied) cannot rank anything and is reported with empty metrics instead of an arbitrary order.
"""

import math

import numpy as np
import polars as pl

from demandpulse.risk.drivers import entity_growth

K_FRACTIONS = (0.05, 0.10, 0.20)
SCORES = ("forecast_gap", "relative_gap", "p_shortfall", "revenue_at_risk")
GROUP = ["model", "origin", "target_def"]
OWN_TREND = "own_trend"
RANDOM = "random"


def add_scores(windows: pl.DataFrame) -> pl.DataFrame:
    """Add the forecast-only scores: revenue gap and relative gap between target and forecast."""
    return windows.with_columns(
        pl.col("forecast_shortfall").alias("forecast_gap"),
        pl.when(pl.col("target") > 0)
        .then(pl.col("forecast_shortfall") / pl.col("target"))
        .otherwise(None)
        .alias("relative_gap"),
    )


def own_trend_windows(windows: pl.DataFrame, panel: pl.DataFrame, level: list[str]) -> pl.DataFrame:
    """Windows for the `own_trend` baseline, built from one reference model's actuals and targets.

    Entities with no measurable own growth are assumed to repeat last year, so the evaluation
    set stays identical to the model rows.
    """
    reference = windows.filter(pl.col("model") == windows["model"].min())
    pieces = []
    for origin in sorted(reference["origin"].unique().to_list()):
        growth = entity_growth(panel, level, origin).rename({"growth": "own_growth"})
        pieces.append(
            reference.filter(pl.col("origin") == origin).join(growth, on=level, how="left")
        )
    return (
        pl.concat(pieces)
        .with_columns(
            pl.lit(OWN_TREND).alias("model"),
            (pl.col("last_year") * (1 + pl.col("own_growth").fill_null(0.0))).alias("forecast"),
        )
        .with_columns((pl.col("target") - pl.col("forecast")).alias("forecast_shortfall"))
        .with_columns((pl.col("forecast_shortfall") > 0).alias("forecast_miss"))
        .drop("own_growth")
    )


def ranking_metrics(
    frame: pl.DataFrame,
    level: list[str],
    scores: tuple[str, ...] = SCORES,
    k_fractions: tuple[float, ...] = K_FRACTIONS,
) -> pl.DataFrame:
    """Precision, recall, lift and shortfall capture at each K for every score and group."""
    rows = []
    for key, part in frame.partition_by(GROUP, as_dict=True).items():
        for score in (s for s in scores if s in part.columns):
            scored = part.filter(pl.col(score).is_not_null())
            if not scored.height:
                continue
            tied = scored[score].n_unique() == 1
            ordered = scored.sort([score, *level], descending=[True, *[False] * len(level)])
            miss = ordered["realised_miss"].to_numpy().astype(float)
            shortfall = np.maximum(ordered["realised_shortfall"].to_numpy(), 0.0)
            n = len(miss)
            for fraction in k_fractions:
                k = max(1, math.ceil(fraction * n))
                total_miss, total_shortfall = miss.sum(), shortfall.sum()
                base_rate = float(miss.mean())
                precision = float(miss[:k].mean())
                rows.append(
                    (
                        *key,
                        score,
                        fraction,
                        n,
                        k,
                        base_rate,
                        None if tied else precision,
                        None if tied or not total_miss else float(miss[:k].sum() / total_miss),
                        None if tied or not base_rate else precision / base_rate,
                        None
                        if tied or not total_shortfall
                        else float(shortfall[:k].sum() / total_shortfall),
                        tied,
                    )
                )
    return pl.DataFrame(
        rows,
        schema={
            "model": pl.String,
            "origin": pl.Int64,
            "target_def": pl.String,
            "score": pl.String,
            "k_fraction": pl.Float64,
            "n": pl.Int64,
            "k": pl.Int64,
            "base_rate": pl.Float64,
            "precision": pl.Float64,
            "recall": pl.Float64,
            "lift": pl.Float64,
            "capture": pl.Float64,
            "tied": pl.Boolean,
        },
        orient="row",
    )


def with_random_baseline(metrics: pl.DataFrame) -> pl.DataFrame:
    """Append the exact expected metrics of a random ranking, per origin, definition and K."""
    first_model = metrics["model"].min()
    first_score = metrics.filter(pl.col("model") == first_model)["score"].min()
    reference = metrics.filter(
        (pl.col("model") == first_model) & (pl.col("score") == first_score)
    ).select("origin", "target_def", "k_fraction", "n", "k", "base_rate")
    random = reference.with_columns(
        pl.lit(RANDOM).alias("model"),
        pl.lit(RANDOM).alias("score"),
        pl.col("base_rate").alias("precision"),
        pl.when(pl.col("base_rate") > 0)
        .then(pl.col("k") / pl.col("n"))
        .otherwise(None)
        .alias("recall"),
        pl.when(pl.col("base_rate") > 0).then(1.0).otherwise(None).alias("lift"),
        (pl.col("k") / pl.col("n")).alias("capture"),
        pl.lit(False).alias("tied"),
    ).select(metrics.columns)
    return pl.concat([metrics, random])


def summarise_ranking(metrics: pl.DataFrame, origins: list[int] | None = None) -> pl.DataFrame:
    """Mean of each metric over origins (all with a defined value, or only `origins`)."""
    data = metrics if origins is None else metrics.filter(pl.col("origin").is_in(origins))
    return (
        data.group_by(["model", "score", "target_def", "k_fraction"])
        .agg(
            pl.col("origin").filter(pl.col("precision").is_not_null()).n_unique().alias("origins"),
            pl.col("base_rate").mean(),
            pl.col("precision").mean(),
            pl.col("recall").mean(),
            pl.col("lift").mean(),
            pl.col("capture").mean(),
        )
        .sort(["target_def", "k_fraction", "model", "score"])
    )
