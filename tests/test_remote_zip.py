import zipfile

import pytest

from demandpulse.ingest.remote_zip import list_members, open_archive, read_member
from demandpulse.ingest.sources import Source, get_source, load_sources

PREFIX = "dunnhumby_Let's-Get-Sort-of-Real-(Sample-5K-customers)"


@pytest.fixture
def archive_path(tmp_path):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{PREFIX}/", "")
        zf.writestr(f"{PREFIX}/time.csv", "shop_week,date_from,date_to\n")
        zf.writestr(f"{PREFIX}/transactions_200610.csv", "b\n")
        zf.writestr(f"{PREFIX}/transactions_200609.csv", "a\n")
        zf.writestr("__MACOSX/._junk", "x")
        zf.writestr(f"{PREFIX}/readme.txt", "ignored")
    return path


def test_lists_only_ingestable_members_in_week_order(archive_path):
    with open_archive(archive_path.as_uri()) as zf:
        members = list_members(zf)
    assert [(m.kind, m.shop_week) for m in members] == [
        ("calendar", None),
        ("transactions", 200609),
        ("transactions", 200610),
    ]


def test_reads_member_bytes_and_checks_size(archive_path):
    with open_archive(archive_path.as_uri()) as zf:
        by_week = {m.shop_week: m for m in list_members(zf)}
        assert read_member(zf, by_week[200609]) == b"a\n"
        assert by_week[200610].size == 2


def test_source_registry_loads_all_published_sources():
    sources = load_sources()
    assert len(sources) == 13
    assert {f"full-part-{n}" for n in range(1, 10)} <= set(sources)
    assert isinstance(sources["sample-5k-customers"], Source)
    assert all(s.url.startswith("https://") for s in sources.values())


def test_unknown_source_lists_alternatives():
    with pytest.raises(KeyError, match="sample-5k-customers"):
        get_source("nope")
