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


def footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
