"""Streamlit entry point: `uv run streamlit run src/demandpulse/app/main.py`."""

import streamlit as st

from demandpulse.app import pages, ui

st.set_page_config(page_title="DemandPulse", layout="wide", initial_sidebar_state="expanded")

ui.inject_style()
navigation = st.navigation(
    [
        st.Page(pages.overview, title="Executive overview", url_path="overview", default=True),
        st.Page(pages.stores, title="Stores and regions", url_path="stores"),
        st.Page(pages.forecasting, title="Forecast accuracy", url_path="forecasting"),
        st.Page(pages.risk, title="Revenue risk", url_path="risk"),
        st.Page(pages.customers, title="Customers", url_path="customers"),
    ]
)
with st.sidebar:
    ui.brand(ui.snapshot().manifest["created_at"][:10])
    ui.comparison_picker()
navigation.run()
