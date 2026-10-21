"""Opt-in tests against the real dunnhumby CDN. Run with: uv run pytest -m integration"""

import pytest

from demandpulse.ingest.bronze import ingest_source
from demandpulse.ingest.silver import build_silver
from demandpulse.ingest.sources import get_source
from demandpulse.quality import run_quality_checks
from demandpulse.storage import LocalStore

pytestmark = pytest.mark.integration


def test_5k_customer_sample_flows_through_every_layer_with_clean_reconciliation(tmp_path):
    store = LocalStore(tmp_path / "lake")
    source = get_source("sample-5k-customers")

    weeks = {200607, 200609}
    ingested = ingest_source(source, store, weeks=weeks)
    built = build_silver(source.name, store, weeks=weeks)
    report = run_quality_checks(source.name, store)

    assert ingested.ok
    assert built.ok
    assert ingested.processed == [200607, 200609]
    failing = [c for c in report.checks if c.status == "fail"]
    assert failing == []
    assert report.rows > 0
