"""Shared Streamlit helpers: snapshot loading, the comparison picker and number formatting."""

import os
from pathlib import Path

import streamlit as st

from demandpulse.app.snapshot import APP_DATA_ENV, DEFAULT_APP_DIR, Snapshot
from demandpulse.app.views import COMPARISON_LABELS

DEFAULT_COMPARISON = "window_vs_year_ago"
DISCLAIMER = (
    "Synthetic, historical dataset. Revenue is the dataset's `spend`; it has no currency, cost, "
    "margin or promotion data, so nothing here is profit or a promotion effect."
)


@st.cache_resource(show_spinner=False)
def _open(path: str) -> Snapshot:
    return Snapshot(Path(path))


def snapshot() -> Snapshot:
    """Open the exported snapshot, or stop the page with the command that creates it."""
    path = os.environ.get(APP_DATA_ENV, str(DEFAULT_APP_DIR))
    try:
        return _open(path)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()


def comparison() -> str:
    return st.session_state.get("comparison", DEFAULT_COMPARISON)


def comparison_picker() -> None:
    st.selectbox(
        "Comparison window",
        list(COMPARISON_LABELS),
        index=list(COMPARISON_LABELS).index(DEFAULT_COMPARISON),
        format_func=COMPARISON_LABELS.get,
        key="comparison",
    )


def money(value: float | None) -> str:
    """Compact revenue: 66.7M, 412K. Missing values (a store with no sales) read n/a."""
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{value / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"{value / 1e6:,.1f}M"
    if magnitude >= 1e3:
        return f"{value / 1e3:,.0f}K"
    return f"{value:,.0f}"


def number(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{decimals}f}"


def pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


STYLE = """
<style>
:root {
  --dp-line: color-mix(in srgb, currentColor 13%, transparent);
  --dp-card: color-mix(in srgb, currentColor 4%, transparent);
  --dp-accent: #14B8A6;
  --dp-accent-2: #6366F1;
}
[data-testid="stToolbar"], .stAppDeployButton, #MainMenu { visibility: hidden; height: 0; }
.block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem; }

h1 { font-weight: 800; letter-spacing: -0.03em; font-size: 2.4rem; padding-bottom: .15rem;
     background: linear-gradient(90deg, var(--dp-accent), var(--dp-accent-2));
     -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
h2, h3 { font-weight: 700; letter-spacing: -0.015em; }
[data-testid="stHeading"] h2 { font-size: 1.35rem; margin-top: 1.6rem; }
[data-testid="stCaptionContainer"] { opacity: .78; }

[data-testid="stMetric"] { background: var(--dp-card); border: 1px solid var(--dp-line);
  border-radius: 16px; padding: 16px 18px;
  transition: transform .15s ease, border-color .15s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: var(--dp-accent); }
[data-testid="stMetric"] { min-height: 118px; }
[data-testid="stMetricLabel"] p { font-size: .85rem; font-weight: 600; opacity: .75;
  white-space: normal; overflow: visible; text-overflow: clip; }
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 750; letter-spacing: -0.02em; }

[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { background: var(--dp-card);
  border: 1px solid var(--dp-line); border-radius: 16px; padding: 8px; }
[data-testid="stDataFrame"] { padding: 4px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--dp-line); }

[data-testid="stSidebar"] { border-right: 1px solid var(--dp-line); }
section[data-testid="stSidebar"] { min-width: 330px; }
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] { border-radius: 10px;
  font-weight: 550; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background:
  color-mix(in srgb, var(--dp-accent) 18%, transparent); }
.dp-brand { display: flex; align-items: center; gap: .6rem; margin: .25rem 0 .9rem; }
.dp-logo { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
  color: white; font-weight: 800; background: linear-gradient(135deg, #14B8A6, #6366F1); }
.dp-brand b { font-size: 1.15rem; letter-spacing: -0.02em; display: block; line-height: 1.1; }
.dp-brand span { font-size: .75rem; opacity: .7; }
.dp-chip { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
  border: 1px solid var(--dp-line); background: var(--dp-card); }

button[data-baseweb="tab"] { font-weight: 600; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { border-radius: 12px; }
hr { border-color: var(--dp-line); }
</style>
"""


def inject_style() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)


def brand(snapshot_date: str) -> None:
    st.markdown(
        '<div class="dp-brand"><div class="dp-logo">D</div><div><b>DemandPulse</b>'
        "<span>Retail demand intelligence</span></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f'<span class="dp-chip">Snapshot {snapshot_date}</span>', unsafe_allow_html=True)


def footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
