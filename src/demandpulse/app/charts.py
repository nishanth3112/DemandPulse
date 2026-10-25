"""Plotly figure builders for the dashboard.

Plain functions from display-ready frames to figures. Colours are a small fixed set: one accent
for the thing being measured, a muted grey for context, and green/amber-red only where a sign
carries meaning (growth up or down), so colour is never the sole signal (values are labelled).
"""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

ACCENT = "#0F766E"
MUTED = "#94A3B8"
UP = "#15803D"
DOWN = "#C2410C"
FONT = "Inter, system-ui, sans-serif"
SCALE = [[0.0, "#C2410C"], [0.5, "#F8FAFC"], [1.0, "#0F766E"]]
SEQUENTIAL = [[0.0, "#F8FAFC"], [1.0, "#0F766E"]]


# Light-theme colour -> dark-theme colour. Charts are written once in light colours and swapped
# here, so every chart follows the active theme without each builder knowing about it.
DARK_SWAP = {
    ACCENT: "#2DD4BF",
    MUTED: "#64748B",
    UP: "#4ADE80",
    DOWN: "#FB923C",
    "#0F172A": "#F8FAFC",
    "#F8FAFC": "#1B2A41",
}
# Region mix index: 100 = the national share. Colours saturate 30 points either side.
MIX_CENTER, MIX_SPAN = 100.0, 30.0
LIGHT_TEXT, DARK_TEXT = "#334155", "#CBD5E1"
LIGHT_GRID, DARK_GRID = "rgba(100,116,139,0.18)", "rgba(148,163,184,0.16)"
MODEL_COLORS = {
    "ensemble": ACCENT,
    "ensemble_baselines": "#5EEAD4",
    "lightgbm": "#6366F1",
    "lightgbm_v2": "#A5B4FC",
    "seasonal_naive": "#F59E0B",
    "trailing_mean_13": "#EC4899",
}


def is_dark() -> bool:
    """True when the viewer's active Streamlit theme is dark (light outside a running app)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:  # no script run context, e.g. in unit tests
        return False


def _swap(value, mapping: dict[str, str]):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_swap(v, mapping) for v in value]
    if isinstance(value, tuple):
        return tuple(_swap(v, mapping) for v in value)
    if isinstance(value, dict):
        return {k: _swap(v, mapping) for k, v in value.items()}
    return value


def pad_axis(fig: go.Figure, values, axis: str = "x", room: float = 0.3) -> None:
    """Widen an axis past the data so outside bar labels stay inside the chart."""
    lo, hi = min(0.0, float(min(values))), max(0.0, float(max(values)))
    span = (hi - lo) or 1.0
    update = fig.update_xaxes if axis == "x" else fig.update_yaxes
    update(range=[lo - span * room if lo < 0 else 0, hi + span * room if hi > 0 else 0])


def style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    dark = is_dark()
    if dark:
        fig = go.Figure(_swap(fig.to_plotly_json(), DARK_SWAP))
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
        font={"family": FONT, "size": 13, "color": DARK_TEXT if dark else LIGHT_TEXT},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hoverlabel={"font_family": FONT},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=DARK_GRID if dark else LIGHT_GRID, zeroline=False)
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
    fig.update_yaxes(tickformat="~s", range=[0, float(bridge["value"].max()) * 1.15])
    fig.update_xaxes(tickangle=-20, automargin=True)
    return style(fig, height=400, legend=False)


def contribution_chart(contribution: pl.DataFrame) -> go.Figure:
    values = contribution["contribution_to_growth_pp"]
    fig = go.Figure(
        go.Bar(
            x=values, y=contribution["prod_code_30"], orientation="h",
            marker={"color": [UP if v >= 0 else DOWN for v in values]},
            text=[f"{v:+.2f}" for v in values], textposition="outside",
        )
    )  # fmt: skip
    pad_axis(fig, values)
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
            colorscale=SCALE if centered else SEQUENTIAL, zmid=MIX_CENTER if centered else None,
            zmin=MIX_CENTER - MIX_SPAN if centered else None,
            zmax=MIX_CENTER + MIX_SPAN if centered else None,
            colorbar={"tickformat": ".0%" if percent else None},
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
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
    pad_axis(fig, values)
    fig.update_xaxes(title=title, tickformat=".1%")
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
    lo_x, hi_x = (float(card["revenue_growth"].quantile(q)) for q in (0.01, 0.99))
    lo_y, hi_y = (float(card["basket_growth"].quantile(q)) for q in (0.01, 0.99))
    pad_x, pad_y = (hi_x - lo_x) * 0.1, (hi_y - lo_y) * 0.1
    outside = card.filter(
        ~pl.col("revenue_growth").is_between(lo_x - pad_x, hi_x + pad_x)
        | ~pl.col("basket_growth").is_between(lo_y - pad_y, hi_y + pad_y)
    ).height
    note = f" ({outside} outlier stores off-chart)" if outside else ""
    fig.update_xaxes(
        title=f"Revenue growth{note}", tickformat=".0%", range=[lo_x - pad_x, hi_x + pad_x]
    )
    fig.update_yaxes(
        title="Basket-count growth", tickformat=".0%", range=[lo_y - pad_y, hi_y + pad_y]
    )
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
    pad_axis(fig, values, room=0.5)
    fig.update_xaxes(title="Revenue change explained", tickformat="~s")
    fig.update_yaxes(autorange="reversed")
    return style(fig, height=300, legend=False)
