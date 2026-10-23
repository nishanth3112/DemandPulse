import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import demandpulse.app as app_package
from demandpulse.app.snapshot import APP_DATA_ENV

APP_DIR = Path(app_package.__file__).parent
TIMEOUT = 90


@pytest.fixture(autouse=True)
def snapshot_env(monkeypatch, app_snapshot_dir):
    monkeypatch.setenv(APP_DATA_ENV, str(app_snapshot_dir))


def page(name: str, **state) -> AppTest:
    at = AppTest.from_string(
        f"from demandpulse.app import pages\npages.{name}()", default_timeout=TIMEOUT
    )
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def widget(elements, label: str):
    matches = [e for e in elements if e.label == label]
    assert matches, f"no widget labelled {label!r}; have {[e.label for e in elements]}"
    return matches[0]


def values(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_overview_shows_kpis_charts_and_the_disclaimer():
    at = page("overview")
    assert not at.exception
    assert at.title[0].value == "Executive overview"
    assert {"Revenue", "Like-for-like growth", "Baskets", "Average basket value"} <= set(values(at))
    assert any("no currency" in c.value for c in at.caption)


def test_overview_follows_the_selected_comparison_window():
    year_ago = values(page("overview", comparison="window_vs_year_ago"))
    prior = values(page("overview", comparison="window_vs_prior"))
    assert year_ago["Like-for-like growth"] != prior["Like-for-like growth"]


def test_overview_filters_by_region_and_says_when_nothing_matches():
    at = page("overview")
    widget(at.multiselect, "Regions").set_value(["E01"])
    widget(at.multiselect, "Store formats").set_value(["LS"]).run()
    assert not at.exception
    assert not at.info


def test_stores_page_lists_the_scorecard_and_a_drilldown():
    at = page("stores")
    assert not at.exception
    assert any("45 stores" in c.value for c in at.caption)
    assert {"Revenue", "Baskets", "Average basket"} <= set(values(at))


def test_stores_filters_narrow_the_scorecard():
    at = page("stores")
    widget(at.checkbox, "Like-for-like only").check().run()
    assert any("44 stores" in c.value for c in at.caption)
    widget(at.text_input, "Store code contains").set_value("s07").run()
    assert any("1 stores" in c.value for c in at.caption)
    assert widget(at.selectbox, "Store").value == "S07"


def test_stores_with_no_match_have_no_drilldown_and_no_error():
    at = page("stores")
    widget(at.text_input, "Store code contains").set_value("zzz").run()
    assert not at.exception
    assert any("0 stores" in c.value for c in at.caption)
    assert not [s for s in at.subheader if s.value == "Store drilldown"]


def test_forecasting_page_compares_models_and_switches_runs():
    at = page("forecasting")
    assert not at.exception
    assert at.title[0].value == "Forecast accuracy"
    widget(at.radio, "Forecast run").set_value("robustness").run()
    assert not at.exception
    widget(at.multiselect, "Models").set_value([]).run()
    assert any("at least one model" in i.value for i in at.info)


def test_risk_page_warns_that_targets_are_hypothetical():
    at = page("risk")
    assert not at.exception
    assert any("hypothetical" in w.value for w in at.warning)
    assert {"Stores", "Forecast below target", "Actually missed (hindsight)"} <= set(values(at))


def test_risk_page_changes_with_target_and_origin():
    at = page("risk")
    flat = values(at)["Target growth assumption"]
    widget(at.selectbox, "Target").set_value("trailing_growth").run()
    assert values(at)["Target growth assumption"] != flat
    widget(at.selectbox, "Forecast origin (week)").set_value(78).run()
    assert not at.exception
    assert values(at)["Expected revenue below target"] == "n/a"


def test_risk_filters_can_empty_the_table_without_breaking_the_page():
    at = page("risk")
    widget(at.multiselect, "Regions").set_value(["E01"]).run()
    assert not at.exception
    assert int(values(at)["Stores"]) < 45


def test_scenario_planner_closes_more_of_the_gap_with_bigger_changes():
    at = page("risk")
    widget(at.slider, "Traffic (baskets per week), %").set_value(0).run()
    none = values(at)["Gap closed"]
    widget(at.slider, "Traffic (baskets per week), %").set_value(10).run()
    more = values(at)["Gap closed"]
    assert float(none.rstrip("%")) == 0.0
    assert float(more.rstrip("%")) > 0.0


def test_ranking_evidence_switches_level_group_size_and_metric():
    at = page("risk")
    widget(at.radio, "Level").set_value("Store x category").run()
    widget(at.radio, "Group size").set_value("Top 5%").run()
    widget(at.radio, "Metric").set_value("lift").run()
    assert not at.exception


def test_classifier_tab_states_whether_it_beat_the_forecast_probability():
    at = page("risk")
    assert any("Calibrated classifier Brier" in i.value for i in at.info)


def test_customers_page_says_it_covers_loyalty_baskets_only():
    at = page("customers")
    assert not at.exception
    assert any("loyalty-card baskets only" in i.value for i in at.info)


def test_every_page_stops_with_the_export_command_when_there_is_no_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv(APP_DATA_ENV, str(tmp_path / "missing"))
    for name in ("overview", "stores", "forecasting", "risk", "customers"):
        at = page(name)
        assert any("app export" in e.value for e in at.error), name


def test_the_full_app_opens_on_the_overview_with_a_sidebar_window_picker():
    at = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.title[0].value == "Executive overview"
    assert widget(at.sidebar.selectbox, "Comparison window").value == "window_vs_year_ago"
    widget(at.sidebar.selectbox, "Comparison window").set_value("window_vs_prior").run()
    assert not at.exception


BANNED = {"boto3", "botocore", "duckdb", "fsspec"}
BANNED_PREFIXES = (
    "demandpulse.storage",
    "demandpulse.ingest",
    "demandpulse.warehouse",
    "demandpulse.config",
)


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_the_app_cannot_reach_aws_duckdb_or_the_basket_fact():
    runtime = [p for p in APP_DIR.glob("*.py") if p.name != "export.py"]
    assert {p.name for p in runtime} >= {"main.py", "pages.py", "snapshot.py", "ui.py"}
    for path in runtime:
        modules = imports_of(path)
        assert "demandpulse.app.export" not in modules, path.name
        assert not {m.split(".")[0] for m in modules} & BANNED, path.name
        assert not [m for m in modules if m.startswith(BANNED_PREFIXES)], path.name
        assert "fct_basket" not in path.read_text(), path.name
        assert "AWS_" not in path.read_text(), path.name
