"""The dashboard pages. Each reads only the exported snapshot and draws views and charts."""

import polars as pl
import streamlit as st

from demandpulse.app import (
    charts,
    forecast_charts,
    forecast_views,
    risk_charts,
    risk_views,
    ui,
    views,
)
from demandpulse.app.views import COMPARISON_LABELS

K_OPTIONS = {"Top 5%": 0.05, "Top 10%": 0.10, "Top 20%": 0.20}
SCENARIO_RANGE = (-10, 10)
HYPOTHETICAL = (
    "Targets are hypothetical: last year's revenue for the same weeks, flat or grown by recent "
    "network growth. The dataset has no budgets, and every window is historical."
)


def overview() -> None:
    snap = ui.snapshot()
    window = ui.comparison()
    st.title("Executive overview")
    manifest = snap.manifest["gold"]
    st.caption(f"{manifest.get('silver_weeks', 'n/a')} weeks of data. {COMPARISON_LABELS[window]}.")
    kpis = views.network_kpis(snap.table("mart_region_benchmark"), window)
    cols = st.columns(5)
    cols[0].metric("Revenue", ui.money(kpis["revenue"]), ui.pct(kpis["revenue_growth"], True))
    cols[1].metric(
        "Like-for-like growth",
        ui.pct(kpis["lfl_revenue_growth"], True),
        help=f"Stores trading in both windows ({kpis['lfl_stores']} of {kpis['stores']}).",
        delta=None,
    )
    cols[2].metric("Baskets", ui.money(kpis["baskets"]), ui.pct(kpis["basket_growth"], True))
    cols[3].metric(
        "Average basket value",
        f"{kpis['avg_basket_value']:.2f}",
        ui.pct(kpis["avg_basket_value_growth"], True),
    )
    cols[4].metric("Stores trading", f"{kpis['stores']:,}")

    st.subheader("Weekly revenue")
    store_week = snap.table("store_week")
    left, right, measure = st.columns([2, 2, 1])
    regions = left.multiselect("Regions", sorted(store_week["store_region"].unique()))
    formats = right.multiselect("Store formats", sorted(store_week["store_format"].unique()))
    pick = measure.radio("Measure", ["Revenue", "Baskets"], horizontal=True)
    weekly = views.weekly_trend(store_week, regions, formats)
    if weekly.height:
        st.plotly_chart(
            charts.trend_chart(weekly, pick.lower(), f"Weekly {pick.lower()}"),
            width="stretch",
        )
        st.caption("Network totals include stores that opened or closed during the period.")
    else:
        st.info("No stores match these filters.")

    bridge_col, category_col = st.columns(2)
    with bridge_col:
        st.subheader("What changed revenue")
        bridge = views.revenue_bridge(snap.table("mart_revenue_variance_region"), window)
        st.plotly_chart(charts.bridge_chart(bridge), width="stretch")
        st.caption(
            "Traffic is baskets per week, basket size is units per basket, price and mix is "
            "revenue per unit. Price and mix cannot be separated: there are no list prices."
        )
    with category_col:
        st.subheader("Category contribution")
        contribution = views.category_contribution(snap.table("mart_category_benchmark"), window)
        st.plotly_chart(charts.contribution_chart(contribution), width="stretch")
        st.caption("Categories are product hierarchy level 30; the data has no product names.")

    mix_col, pattern_col = st.columns(2)
    with mix_col:
        st.subheader("Basket missions")
        mix = views.mission_mix(snap.table("mart_basket_mix"))
        st.dataframe(
            mix.head(10),
            hide_index=True,
            width="stretch",
            column_config={
                "basket_dominant_mission": "Mission",
                "basket_type": "Basket type",
                "revenue": None,
                "revenue_share": st.column_config.ProgressColumn(
                    "Share of revenue", format="percent", min_value=0.0, max_value=1.0
                ),
            },
        )
        st.caption("From a sample of weeks in the basket-mix mart.")
    with pattern_col:
        st.subheader("Trading pattern")
        pattern = snap.table("mart_trading_pattern")
        chosen = st.selectbox("Store format", sorted(pattern["store_format"].unique()))
        st.plotly_chart(
            charts.heatmap(
                views.trading_matrix(pattern, chosen),
                "Hour of day",
                "Weekday (1 = first)",
                percent=True,
            ),
            width="stretch",
        )
    ui.footer()


def stores() -> None:
    snap = ui.snapshot()
    window = ui.comparison()
    st.title("Stores and regions")
    st.caption(COMPARISON_LABELS[window])

    regions = views.region_table(snap.table("mart_region_benchmark"), window)
    left, right = st.columns(2)
    with left:
        st.subheader("Like-for-like growth by region")
        st.plotly_chart(
            charts.growth_bars(regions, "store_region", "lfl_revenue_growth", "LFL revenue growth"),
            width="stretch",
        )
    with right:
        st.subheader("Category mix by region")
        matrix = views.region_mix_matrix(snap.table("mart_region_category_mix"), window)
        st.plotly_chart(
            charts.heatmap(matrix, "Category", "Region", centered=True), width="stretch"
        )
        st.caption(
            "100 means the region buys a category at the national share; 120 means 20% more. "
            "Colours saturate at 70 and 130."
        )

    st.subheader("Store scorecard")
    card = snap.table("mart_store_scorecard").filter(pl.col("comparison") == window)
    a, b, c, d = st.columns([2, 2, 1, 2])
    pick_regions = a.multiselect("Regions", sorted(card["store_region"].unique()), key="s_regions")
    pick_formats = b.multiselect("Formats", sorted(card["store_format"].unique()), key="s_formats")
    lfl = c.checkbox("Like-for-like only")
    search = d.text_input("Store code contains")
    table = views.scorecard(card, window, pick_regions, pick_formats, lfl, search)
    st.caption(f"{table.height:,} stores, weakest revenue growth first.")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "store_code": "Store",
            "store_region": "Region",
            "store_format": "Format",
            "lifecycle": "Lifecycle",
            "is_lfl": "Like-for-like",
            "current_revenue": st.column_config.NumberColumn("Revenue", format="compact"),
            "revenue_growth": st.column_config.NumberColumn("Revenue growth", format="percent"),
            "basket_growth": st.column_config.NumberColumn("Basket growth", format="percent"),
            "current_avg_basket_value": st.column_config.NumberColumn("Avg basket", format="%.2f"),
            "revenue_index_vs_format": st.column_config.NumberColumn(
                "Index vs format", format="%.2f", help="1.0 = average store of the same format"
            ),
            "lfl_growth_zscore": st.column_config.NumberColumn("LFL z-score", format="%.2f"),
        },
    )
    if table.height:
        st.plotly_chart(charts.store_scatter(table), width="stretch")

        st.subheader("Store drilldown")
        code = st.selectbox("Store", table["store_code"].to_list())
        detail = views.store_detail(
            snap.table("mart_store_scorecard"),
            snap.table("mart_revenue_variance_store"),
            snap.table("store_week"),
            code,
            window,
        )
        p = detail.profile
        m = st.columns(4)
        m[0].metric("Revenue", ui.money(p["current_revenue"]), ui.pct(p["revenue_growth"], True))
        m[1].metric("Baskets", ui.money(p["current_baskets"]), ui.pct(p["basket_growth"], True))
        m[2].metric("Average basket", ui.number(p["current_avg_basket_value"]))
        m[3].metric("Format / region", f"{p['store_format']} / {p['store_region']}")
        trend, drivers = st.columns(2)
        trend.plotly_chart(charts.trend_chart(detail.weekly), width="stretch")
        if detail.drivers.height:
            drivers.plotly_chart(charts.driver_chart(detail.drivers), width="stretch")
        else:
            drivers.info("No driver split for this store and window.")
    ui.footer()


def forecasting() -> None:
    snap = ui.snapshot()
    st.title("Forecast accuracy")
    st.caption(
        "Weekly revenue per store and category, 13 weeks ahead from rolling origins. Models see "
        "only data up to each origin. Ensemble = seasonal naive + 13-week trailing mean + LightGBM."
    )
    tables = {n: snap.table(n) for n in snap.manifest["tables"] if n.startswith("forecast_")}
    runs = sorted(tables["forecast_full_pooled"]["run"].unique())
    run = st.radio(
        "Forecast run",
        runs,
        horizontal=True,
        help="`main`: origins 78, 91, 104. `robustness`: origins 85 and 98.",
    )
    board = risk_views.model_leaderboard(tables["forecast_full_pooled"], run)
    best = board.row(0, named=True)
    naive = board.filter(pl.col("model") == "seasonal_naive")
    gain = 1 - best["wape"] / naive["wape"][0] if naive.height else None
    cards = st.columns(4)
    cards[0].metric("Best model", risk_views.MODEL_LABELS.get(best["model"], best["model"]))
    cards[1].metric("Its WAPE", ui.pct(best["wape"]), help="Lower is better.")
    cards[2].metric(
        "Error cut vs seasonal naive",
        ui.pct(gain),
        help="Relative reduction in WAPE against forecasting each week as last year's.",
    )
    cards[3].metric("Series scored", f"{best['series']:,}")
    st.subheader("Accuracy over every series and origin")
    st.dataframe(
        board.with_columns(pl.col("model").replace(risk_views.MODEL_LABELS)),
        hide_index=True,
        width="stretch",
        column_config={
            "model": "Model",
            "wape": st.column_config.NumberColumn("WAPE", format="percent"),
            "bias": st.column_config.NumberColumn("Bias", format="percent"),
            "mase_mean": st.column_config.NumberColumn("MASE", format="%.2f"),
            "series": st.column_config.NumberColumn("Series", format="localized"),
        },
    )
    st.caption(
        "WAPE is total absolute error over total actual revenue. Bias above zero is "
        "over-forecasting. MASE below 1 beats a one-year seasonal naive. The ensemble was "
        "chosen after seeing these results and is not independently validated."
    )
    st.subheader("Accuracy improves as forecasts are summed to coarser levels")
    st.plotly_chart(
        risk_charts.accuracy_chart(risk_views.accuracy_by_level(tables, run)), width="stretch"
    )

    st.subheader("Network forecast against actual")
    weeks = snap.table("forecast_network_weeks").filter(pl.col("run") == run)
    origins = sorted(weeks["origin"].unique().to_list())
    left, right = st.columns([1, 2])
    origin = left.selectbox("Forecast origin (week)", origins, index=len(origins) - 1)
    models = sorted(weeks["model"].unique().to_list())
    picked = right.multiselect(
        "Models",
        models,
        default=[m for m in models if m == "ensemble"] or models[:1],
        format_func=lambda m: risk_views.MODEL_LABELS.get(m, m),
    )
    if picked:
        lines = risk_views.network_forecast(
            snap.table("forecast_network_weeks"), run, origin, picked
        )
        st.plotly_chart(risk_charts.forecast_chart(lines), width="stretch")
    else:
        st.info("Pick at least one model.")
    _store_explorer(snap, run)
    ui.footer()


def _store_explorer(snap, run: str) -> None:
    st.subheader("Store explorer")
    if "forecast_store_weeks" not in snap.manifest["tables"]:
        st.info("This snapshot predates the store explorer. Re-run `demandpulse app export`.")
        return
    weeks = snap.table("forecast_store_weeks").filter(pl.col("run") == run)
    stores_dim = snap.table("dim_store")
    left, middle, right = st.columns(3)
    regions = left.multiselect("Region", sorted(stores_dim["store_region"].unique().to_list()))
    formats = middle.multiselect("Format", sorted(stores_dim["store_format"].unique().to_list()))
    options = forecast_views.store_options(
        stores_dim, set(weeks["store_code"].unique().to_list()), regions, formats
    )
    if not options:
        st.info("No stores match these filters.")
        return
    store = right.selectbox("Store", options)
    origins = sorted(weeks["origin"].unique().to_list())
    origin_col, models_col = st.columns([1, 2])
    origin = origin_col.selectbox("Origin (week)", origins, index=len(origins) - 1, key="explorer")
    models = sorted(weeks["model"].unique().to_list())
    picked = models_col.multiselect(
        "Models to compare",
        models,
        default=[m for m in models if m in ("ensemble", "seasonal_naive")],
        format_func=lambda m: risk_views.MODEL_LABELS.get(m, m),
        key="explorer_models",
    )
    if not picked:
        st.info("Pick at least one model to compare for this store.")
        return
    series = forecast_views.store_series(
        weeks, snap.table("store_week"), run, origin, store, picked
    )
    origin_start = snap.table("dim_date").filter(pl.col("week_idx") == origin)["week_start"][0]
    st.plotly_chart(forecast_charts.store_series_chart(series, origin_start), width="stretch")
    st.caption(
        "Solid line: actual weekly revenue (history, then the weeks being forecast). Dotted lines: "
        "each model's forecast from the origin, made with data up to that week only."
    )
    accuracy = forecast_views.store_accuracy(weeks, run, origin, store, picked)
    st.dataframe(
        accuracy.select("model", "wape", "bias", "actual_total", "forecast_total"),
        hide_index=True,
        width="stretch",
        column_config={
            "model": "Model",
            "wape": st.column_config.NumberColumn("WAPE", format="percent"),
            "bias": st.column_config.NumberColumn("Bias", format="percent"),
            "actual_total": st.column_config.NumberColumn("Actual revenue", format="compact"),
            "forecast_total": st.column_config.NumberColumn("Forecast revenue", format="compact"),
        },
    )
    if accuracy["wape"].null_count() == accuracy.height:
        st.warning(
            "This store had no revenue in the forecast weeks (for example a closure), so its "
            "error rates are undefined."
        )
    reference = picked[0]
    by_store = forecast_views.wape_by_store(weeks, run, origin, reference)
    rank = forecast_views.store_rank(by_store, store)
    label = risk_views.MODEL_LABELS.get(reference, reference)
    if rank:
        st.caption(
            f"For {label}, this store's WAPE of {ui.pct(rank['wape'])} ranks {rank['rank']} of "
            f"{rank['of']} stores at this origin (1 is most accurate); the median store is "
            f"{ui.pct(rank['median_wape'])}."
        )
    st.plotly_chart(
        forecast_charts.wape_distribution_chart(by_store, rank["wape"] if rank else None),
        width="stretch",
    )


def risk() -> None:
    snap = ui.snapshot()
    st.title("Revenue risk")
    st.warning(HYPOTHETICAL)
    windows, stores_dim = snap.table("risk_windows_store"), snap.table("dim_store")
    model_list = sorted(windows["model"].unique().to_list())
    c = st.columns(4)
    target = c[0].selectbox(
        "Target", list(risk_views.TARGET_LABELS), format_func=risk_views.TARGET_LABELS.get
    )
    model = c[1].selectbox(
        "Model",
        model_list,
        index=model_list.index("ensemble") if "ensemble" in model_list else 0,
        format_func=lambda m: risk_views.MODEL_LABELS.get(m, m),
    )
    origins = risk_views.risk_origins(windows, model)
    origin = c[2].selectbox("Forecast origin (week)", origins, index=len(origins) - 1)
    growth = snap.table("risk_growth").filter(
        (pl.col("origin") == origin) & (pl.col("target_def") == target)
    )["growth"][0]
    c[3].metric("Target growth assumption", ui.pct(growth, True))
    f1, f2 = st.columns(2)
    regions = f1.multiselect(
        "Regions", sorted(stores_dim["store_region"].unique()), key="r_regions"
    )
    formats = f2.multiselect(
        "Formats", sorted(stores_dim["store_format"].unique()), key="r_formats"
    )

    table = risk_views.risk_table(windows, stores_dim, model, origin, target, regions, formats)
    head = risk_views.risk_headline(table)
    m = st.columns(4)
    m[0].metric("Stores", f"{head['stores']:,}")
    m[1].metric("Forecast below target", f"{head['forecast_below_target']:,}")
    m[2].metric(
        "Expected revenue below target",
        ui.money(head["revenue_at_risk"]) if head["revenue_at_risk"] is not None else "n/a",
        help="Needs earlier finished windows, so it is missing for the first two origins.",
    )
    m[3].metric(
        "Actually missed (hindsight)",
        ui.pct(head["missed_in_hindsight"]),
        help="Backtest only: the 13 weeks after this origin have already happened.",
    )

    rank_tab, evidence_tab, classifier_tab, scenario_tab = st.tabs(
        ["At-risk stores", "How well the ranking works", "Classifier", "Scenario planner"]
    )
    with rank_tab:
        if table.height:
            st.plotly_chart(risk_charts.at_risk_chart(table), width="stretch")
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "store_code": "Store",
                    "store_region": "Region",
                    "store_format": "Format",
                    "target": st.column_config.NumberColumn("Target", format="compact"),
                    "forecast": st.column_config.NumberColumn("Forecast", format="compact"),
                    "forecast_shortfall": st.column_config.NumberColumn(
                        "Target minus forecast", format="compact"
                    ),
                    "relative_gap": st.column_config.NumberColumn("Gap / target", format="percent"),
                    "p_shortfall": st.column_config.NumberColumn(
                        "P(shortfall)",
                        format="percent",
                        help="Ranks stores well; it is not a calibrated chance.",
                    ),
                    "revenue_at_risk": st.column_config.NumberColumn("At risk", format="compact"),
                    "actual": st.column_config.NumberColumn("Actual (hindsight)", format="compact"),
                    "realised_miss": st.column_config.CheckboxColumn("Missed (hindsight)"),
                },
            )
            st.caption(
                "Large gaps are prompts to look, not findings. A store that was closed for part "
                "of its history can be forecast far too low and still beat target."
            )
            _recommendations(snap, target, origin)
        else:
            st.info("No stores match these filters.")
    with evidence_tab:
        _ranking_evidence(snap, target)
    with classifier_tab:
        _classifier(snap, target)
    with scenario_tab:
        _scenario(table, origin)
    ui.footer()


def _recommendations(snap, target: str, origin: int) -> None:
    if not snap.has("risk_recommendations"):
        return
    recs = snap.table("risk_recommendations").filter(pl.col("target_def") == target)
    if not recs.height or origin != snap.json("risk_summary")["origins"][-1]:
        return
    st.subheader("Where to look first, with the evidence")
    for row in recs.head(10).to_dicts():
        st.markdown(f"{row['rank']}. {row['evidence']}")


def _ranking_evidence(snap, target: str) -> None:
    st.markdown(
        "Each origin ranks stores by a score, and the top group is compared with what really "
        "happened. **Precision** is the share of the top group that missed target; **lift** is "
        "that over the overall miss rate (random = 1); **capture** is the share of the total "
        "revenue shortfall in the top group. Mean over origins with estimates."
    )
    a, b, c = st.columns(3)
    level = a.radio("Level", ["Store", "Store x category"], horizontal=True)
    k_label = b.radio("Group size", list(K_OPTIONS), index=1, horizontal=True)
    metric = c.radio("Metric", ["precision", "lift", "capture"], horizontal=True)
    name = (
        "risk_ranking_summary_store" if level == "Store" else "risk_ranking_summary_store_category"
    )
    view = risk_views.ranking_view(snap.table(name), target, K_OPTIONS[k_label])
    if not view.height:
        st.info("No ranking results for this selection.")
        return
    st.plotly_chart(risk_charts.ranking_chart(view, metric), width="stretch")
    st.caption(
        "Grey bars are baselines: random, and the entity's own recent growth with no model. "
        f"Overall miss rate here: {ui.pct(view['base_rate'][0])}. Three overlapping windows, "
        "so differences of a few points are not reliable."
    )
    st.dataframe(
        view.drop("model", "score"),
        hide_index=True,
        width="stretch",
        column_config={
            "base_rate": st.column_config.NumberColumn("Miss rate", format="percent"),
            "precision": st.column_config.NumberColumn("Precision", format="percent"),
            "recall": st.column_config.NumberColumn("Recall", format="percent"),
            "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
            "capture": st.column_config.NumberColumn("Capture", format="percent"),
        },
    )


def _classifier(snap, target: str) -> None:
    st.markdown(
        "An exploratory LightGBM classifier with isotonic calibration, scored on one test window "
        "after training and calibrating on earlier ones. It is compared with the forecast-based "
        "probability using the Brier score (lower is better)."
    )
    level = st.radio("Level", ["Store", "Store x category"], horizontal=True, key="clf_level")
    name = "risk_classifier_store" if level == "Store" else "risk_classifier_store_category"
    document = snap.json(name)
    try:
        metrics = risk_views.classifier_metrics(document, target)
    except ValueError as error:
        st.info(str(error))
        return
    split = document["split"]
    st.caption(
        f"Train origins {split['train']}, calibration origin {split['calibration']}, "
        f"test origin {split['test']}."
    )
    st.info(risk_views.classifier_verdict(metrics))
    st.dataframe(
        pl.DataFrame({"metric": list(metrics), "value": list(metrics.values())}),
        hide_index=True,
        width="stretch",
    )
    st.plotly_chart(
        risk_charts.reliability_chart(pl.DataFrame(document[target]["reliability"])),
        width="stretch",
    )


def _scenario(table: pl.DataFrame, origin: int) -> None:
    st.markdown(
        "What-if arithmetic on the forecasts of the riskiest stores: forecast x (1 + traffic) x "
        "(1 + basket size) x (1 + price/mix). It shows how big a driver change would have to be. "
        "It does not say any action can deliver it, or what it would cost."
    )
    if not table.height:
        st.info("No stores to apply a scenario to.")
        return
    lo, hi = SCENARIO_RANGE
    a, b, c, d = st.columns(4)
    traffic = a.slider("Traffic (baskets per week), %", lo, hi, 2) / 100
    basket = b.slider("Basket size (units per basket), %", lo, hi, 0) / 100
    price = c.slider("Price and mix (revenue per unit), %", lo, hi, 0) / 100
    top_n = d.slider("Riskiest stores", 5, min(50, table.height), min(20, table.height))
    rows, totals = risk_views.custom_scenario(
        table.head(top_n).select("store_code", "forecast", "target"), traffic, basket, price
    )
    m = st.columns(4)
    m[0].metric("Gap before", ui.money(totals["gap_before"]))
    m[1].metric("Gap after", ui.money(totals["gap_after"]))
    m[2].metric("Gap closed", ui.pct(totals["gap_closed_share"]))
    m[3].metric("Stores meeting target", f"{totals['stores_meeting_target']} of {totals['stores']}")
    st.plotly_chart(risk_charts.scenario_chart(rows), width="stretch")
    uplift = rows["required_uplift"].median()
    st.caption(
        f"Origin {origin}. Median uplift needed to reach target across these stores: "
        f"{ui.pct(uplift)}."
    )


def customers() -> None:
    snap = ui.snapshot()
    st.title("Customers")
    coverage = views.loyalty_coverage(snap.table("store_week"))
    st.info(
        f"Customer analysis covers loyalty-card baskets only: {ui.pct(coverage['basket_share'])} "
        f"of baskets and {ui.pct(coverage['revenue_share'])} of revenue. The rest are anonymous."
    )
    segments = views.segment_table(snap.table("mart_rfm_segment_summary"))
    left, right = st.columns(2)
    with left:
        st.subheader("RFM segments")
        st.plotly_chart(charts.segment_chart(segments), width="stretch")
    with right:
        st.subheader("Segment detail")
        st.dataframe(
            segments,
            hide_index=True,
            width="stretch",
            column_config={
                "rfm_segment": "Segment",
                "customers": st.column_config.NumberColumn("Customers", format="localized"),
                "customer_share": st.column_config.NumberColumn("Customer share", format="percent"),
                "revenue": st.column_config.NumberColumn("Revenue", format="compact"),
                "revenue_share": st.column_config.NumberColumn("Revenue share", format="percent"),
                "avg_baskets": st.column_config.NumberColumn("Avg baskets", format="%.1f"),
                "avg_revenue": st.column_config.NumberColumn("Avg revenue", format="compact"),
                "avg_recency_weeks": st.column_config.NumberColumn("Recency (wk)", format="%.1f"),
            },
        )
    st.subheader("Cohort retention")
    st.plotly_chart(
        charts.heatmap(
            views.retention_matrix(snap.table("mart_customer_cohort")),
            "Periods (4 weeks) since first purchase",
            "Cohort period",
            percent=True,
        ),
        width="stretch",
    )
    st.caption(
        "Share of each cohort's customers still buying. Cohorts can only be tracked from the "
        "first week of data, so the earliest cohort mixes new and existing customers."
    )
    ui.footer()
