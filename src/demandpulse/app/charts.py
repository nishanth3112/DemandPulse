"""Plotly figure builders for the dashboard.

Plain functions from display-ready frames to figures. Colours are a small fixed set: one accent
for the thing being measured, a muted grey for context, and green/amber-red only where a sign
carries meaning (growth up or down), so colour is never the sole signal (values are labelled).
"""

import plotly.graph_objects as go
import polars as pl

ACCENT = "#0F766E"
MUTED = "#94A3B8"
UP = "#15803D"
DOWN = "#C2410C"
FONT = "Inter, system-ui, sans-serif"
SCALE = [[0.0, "#C2410C"], [0.5, "#F8FAFC"], [1.0, "#0F766E"]]
SEQUENTIAL = [[0.0, "#F8FAFC"], [1.0, "#0F766E"]]


def style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
        font={"family": FONT, "size": 13},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hoverlabel={"font_family": FONT},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)", zeroline=False)
    return fig


def trend_chart(weekly: pl.DataFrame, measure: str = "revenue", name: str = "Weekly") -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=weekly["week_start"], y=weekly[measure], name=name, mode="lines",
        line={"color": MUTED, "width": 1.5},
    )  # fmt: skip
    if "rolling_13" in weekly.columns and measure == "revenue":
        fig.add_scatter(
            x=weekly["week_start"], y=weekly["rolling_13"], name="13-week average", mode="lines",
            line={"color": ACCENT, "width": 3},
        )  # fmt: skip
    fig.update_yaxes(tickformat="~s", rangemode="tozero")
    return style(fig)


def bridge_chart(bridge: pl.DataFrame) -> go.Figure:
    measures = ["absolute" if kind == "total" else "relative" for kind in bridge["kind"]]
    fig = go.Figure(
        go.Waterfall(
            x=bridge["step"], y=bridge["value"], measure=measures,
            increasing={"marker": {"color": UP}}, decreasing={"marker": {"color": DOWN}},
            totals={"marker": {"color": ACCENT}}, connector={"line": {"color": MUTED}},
            texttemplate="%{y:,.3s}", textposition="outside",
        )
    )  # fmt: skip
    fig.update_yaxes(tickformat="~s")
    return style(fig, legend=False)


def contribution_chart(contribution: pl.DataFrame) -> go.Figure:
    values = contribution["contribution_to_growth_pp"]
    fig = go.Figure(
        go.Bar(
            x=values, y=contribution["prod_code_30"], orientation="h",
            marker={"color": [UP if v >= 0 else DOWN for v in values]},
            text=[f"{v:+.2f}" for v in values], textposition="outside",
        )
    )  # fmt: skip
    fig.update_xaxes(title="Contribution to revenue growth (percentage points)")
    fig.update_yaxes(type="category", title="Category (hierarchy level 30)")
    return style(fig, height=max(280, 24 * contribution.height + 80), legend=False)


def heatmap(
    matrix: pl.DataFrame, x_title: str, y_title: str, centered: bool = False, percent: bool = False
) -> go.Figure:
    index = matrix.columns[0]
    columns = matrix.columns[1:]
    z = matrix.select(columns).to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=z, x=columns, y=matrix[index].cast(pl.String).to_list(),
            colorscale=SCALE if centered else SEQUENTIAL, zmid=1.0 if centered else None,
            colorbar={"tickformat": ".0%" if percent else None},
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_xaxes(title=x_title, type="category")
    fig.update_yaxes(title=y_title, type="category", autorange="reversed")
    return style(fig, legend=False)


def growth_bars(table: pl.DataFrame, label: str, value: str, title: str) -> go.Figure:
    ordered = table.sort(value)
    values = ordered[value]
    fig = go.Figure(
        go.Bar(
            x=values, y=ordered[label], orientation="h",
            marker={"color": [UP if v >= 0 else DOWN for v in values]},
            text=[f"{v:+.1%}" for v in values], textposition="outside",
        )
    )  # fmt: skip
    fig.update_xaxes(title=title, tickformat=".0%")
    fig.update_yaxes(type="category")
    return style(fig, height=max(260, 28 * ordered.height + 80), legend=False)


def segment_chart(segments: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=segments["rfm_segment"], y=segments["customer_share"], name="Share of customers",
                marker_color=MUTED)  # fmt: skip
    fig.add_bar(x=segments["rfm_segment"], y=segments["revenue_share"], name="Share of revenue",
                marker_color=ACCENT)  # fmt: skip
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    return style(fig)


def store_scatter(card: pl.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=card["revenue_growth"], y=card["basket_growth"], mode="markers",
            text=card["store_code"], marker={"color": ACCENT, "size": 8, "opacity": 0.65},
            hovertemplate="%{text}<br>revenue %{x:+.1%}<br>baskets %{y:+.1%}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_xaxes(title="Revenue growth", tickformat=".0%")
    fig.update_yaxes(title="Basket-count growth", tickformat=".0%")
    return style(fig, legend=False)


def driver_chart(drivers: pl.DataFrame) -> go.Figure:
    values = drivers["effect"]
    fig = go.Figure(
        go.Bar(
            x=values, y=drivers["driver"], orientation="h",
            marker={"color": [UP if v >= 0 else DOWN for v in values]},
            text=[f"{v:+,.0f}" for v in values], textposition="outside",
        )
    )  # fmt: skip
    fig.update_xaxes(title="Revenue change explained", tickformat="~s")
    fig.update_yaxes(autorange="reversed")
    return style(fig, height=300, legend=False)
