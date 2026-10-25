"""Plotly figure builders for the forecasting and revenue-risk pages."""

import plotly.graph_objects as go
import polars as pl

from demandpulse.app.charts import ACCENT, DOWN, MODEL_COLORS, MUTED, pad_axis, style
from demandpulse.app.risk_views import MODEL_LABELS

LABEL_COLORS = {MODEL_LABELS[m]: c for m, c in MODEL_COLORS.items() if m in MODEL_LABELS}


def forecast_chart(lines: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name in lines["series"].unique(maintain_order=True):
        part = lines.filter(pl.col("series") == name).sort("week_start")
        actual = name == "Actual"
        fig.add_scatter(
            x=part["week_start"], y=part["y"], name=name, mode="lines",
            line={"color": "#0F172A" if actual else LABEL_COLORS.get(name, MUTED),
                  "width": 3 if actual else 2, "dash": "solid" if actual else "dot"},
        )  # fmt: skip
    fig.update_yaxes(tickformat="~s")
    return style(fig)


def accuracy_chart(accuracy: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    for model in accuracy["model"].unique(maintain_order=True):
        part = accuracy.filter(pl.col("model") == model)
        fig.add_bar(x=part["level"], y=part["wape"], name=MODEL_LABELS.get(model, model),
                    marker_color=MODEL_COLORS.get(model, MUTED),
                    text=[f"{v:.1%}" for v in part["wape"]], textposition="outside")  # fmt: skip
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="WAPE (lower is better)", tickformat=".0%", rangemode="tozero")
    return style(fig)


def at_risk_chart(table: pl.DataFrame, top: int = 15) -> go.Figure:
    head = table.head(top)
    column = "revenue_at_risk" if head["revenue_at_risk"].count() else "forecast_shortfall"
    ordered = head.sort(column)
    fig = go.Figure(
        go.Bar(
            x=ordered[column], y=ordered["store_code"], orientation="h",
            marker={"color": DOWN}, text=[f"{v:,.0f}" for v in ordered[column]],
            textposition="outside",
        )
    )  # fmt: skip
    pad_axis(fig, ordered[column])
    title = "Expected revenue below target" if column == "revenue_at_risk" else "Forecast gap"
    fig.update_xaxes(title=title, tickformat="~s")
    fig.update_yaxes(type="category")
    return style(fig, height=max(260, 26 * ordered.height + 80), legend=False)


def ranking_chart(view: pl.DataFrame, metric: str = "precision") -> go.Figure:
    ordered = view.sort(metric)
    labels = [
        m if s == m else f"{m}: {s}"
        for m, s in zip(ordered["model_label"], ordered["score_label"], strict=True)
    ]
    colours = [MUTED if model in {"random", "own_trend"} else ACCENT for model in ordered["model"]]
    fig = go.Figure(
        go.Bar(
            x=ordered[metric], y=labels, orientation="h", marker={"color": colours},
            text=[f"{v:.1%}" if metric != "lift" else f"{v:.2f}" for v in ordered[metric]],
            textposition="outside",
        )
    )  # fmt: skip
    fig.update_xaxes(tickformat=".0%" if metric != "lift" else None, title=metric.capitalize())
    fig.update_yaxes(type="category")
    return style(fig, height=max(300, 24 * ordered.height + 80), legend=False)


def reliability_chart(reliability: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                    line={"color": MUTED, "dash": "dash"})  # fmt: skip
    fig.add_scatter(
        x=reliability["mean_predicted"], y=reliability["observed_rate"], mode="lines+markers",
        name="Observed", line={"color": ACCENT},
        marker={"size": [max(6, min(18, n**0.5)) for n in reliability["n"]]},
    )  # fmt: skip
    fig.update_xaxes(title="Predicted probability", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Observed shortfall rate", tickformat=".0%", range=[0, 1])
    return style(fig)


def scenario_chart(rows: pl.DataFrame, top: int = 15) -> go.Figure:
    head = rows.sort("gap_before", descending=True).head(top).sort("gap_before")
    fig = go.Figure()
    fig.add_bar(y=head["store_code"], x=head["gap_before"], name="Gap before", orientation="h",
                marker_color=MUTED)  # fmt: skip
    fig.add_bar(y=head["store_code"], x=head["gap_after"], name="Gap after", orientation="h",
                marker_color=ACCENT)  # fmt: skip
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Revenue below target", tickformat="~s")
    fig.update_yaxes(type="category")
    return style(fig, height=max(280, 40 * head.height + 80))
