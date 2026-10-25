"""Plotly figure builders for the single-store forecast explorer."""

import plotly.graph_objects as go
import polars as pl

from demandpulse.app.charts import ACCENT, MUTED, style
from demandpulse.app.forecast_views import ACTUAL_LABEL
from demandpulse.app.risk_charts import LABEL_COLORS


def store_series_chart(series: pl.DataFrame, origin_week_start) -> go.Figure:
    """Actual and forecast lines for one store; a vertical rule marks where forecasting began."""
    fig = go.Figure()
    for name in series["series"].unique(maintain_order=True):
        part = series.filter(pl.col("series") == name).sort("week_start")
        actual = name == ACTUAL_LABEL
        fig.add_scatter(
            x=part["week_start"], y=part["y"], name=name, mode="lines",
            line={"color": "#0F172A" if actual else LABEL_COLORS.get(name, MUTED),
                  "width": 3 if actual else 2, "dash": "solid" if actual else "dot"},
        )  # fmt: skip
    fig.add_vline(x=origin_week_start, line={"color": MUTED, "dash": "dash", "width": 1})
    fig.update_yaxes(tickformat="~s")
    return style(fig)


def wape_distribution_chart(by_store: pl.DataFrame, selected_wape: float | None) -> go.Figure:
    """Histogram of store-level WAPE with the chosen store marked."""
    shown_to = max(float(by_store["wape"].quantile(0.99)) * 1.1, 0.05)
    clipped = by_store.filter(pl.col("wape") <= shown_to)
    hidden = by_store.height - clipped.height
    fig = go.Figure(
        go.Histogram(x=clipped["wape"], marker={"color": MUTED}, xbins={"start": 0, "end": shown_to,
                     "size": shown_to / 30})
    )  # fmt: skip
    fig.update_xaxes(range=[0, shown_to])
    if selected_wape is not None:
        fig.add_vline(
            x=selected_wape, line={"color": ACCENT, "width": 3},
            annotation_text="Selected store", annotation_position="top",
        )  # fmt: skip
    note = f" ({hidden} stores above {shown_to:.0%} not shown)" if hidden else ""
    fig.update_xaxes(title=f"Store WAPE (lower is better){note}", tickformat=".0%")
    fig.update_yaxes(title="Stores")
    return style(fig, height=280, legend=False)
