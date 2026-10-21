import polars as pl
import pytest

from demandpulse import cli
from demandpulse.ingest.bronze import bronze_key
from demandpulse.ingest.silver import silver_key


@pytest.fixture
def sources_file(make_zip, tmp_path):
    source = make_zip()
    path = tmp_path / "sources.yaml"
    path.write_text(f"sources:\n  - name: {source.name}\n    kind: sample\n    url: {source.url}\n")
    return path, source


def run(args, sources_path, lake):
    return cli.main(["--sources-file", str(sources_path), "--local-root", str(lake), *args])


def test_run_builds_every_layer_and_reports_quality(sources_file, tmp_path, capsys):
    path, source = sources_file
    lake = tmp_path / "lake"

    assert run(["run", source.name], path, lake) == 0

    assert (lake / bronze_key(source.name, 200607)).exists()
    assert pl.read_parquet(lake / silver_key(source.name, 200609)).height == 3
    assert f"# Data quality report: {source.name}" in capsys.readouterr().out


def test_stages_can_run_separately_and_resume(sources_file, tmp_path):
    path, source = sources_file
    lake = tmp_path / "lake"

    assert run(["ingest", source.name, "--max-weeks", "1"], path, lake) == 0
    assert run(["ingest", source.name], path, lake) == 0
    assert run(["silver", source.name, "--weeks", "200607"], path, lake) == 0
    assert (lake / silver_key(source.name, 200607)).exists()
    assert not (lake / silver_key(source.name, 200609)).exists()


def test_failed_quality_gate_returns_nonzero(sources_file, tmp_path):
    path, source = sources_file
    # Quality on an empty lake has no data, which must fail rather than pass vacuously.
    assert run(["quality", source.name], path, tmp_path / "empty-lake") == 1


def test_sources_command_lists_registry(sources_file, tmp_path, capsys):
    path, source = sources_file
    assert run(["sources"], path, tmp_path / "x") == 0
    assert source.name in capsys.readouterr().out


def test_unknown_source_is_a_clear_error(sources_file, tmp_path):
    path, _ = sources_file
    with pytest.raises(KeyError, match="Unknown source"):
        run(["ingest", "nope"], path, tmp_path / "lake")
