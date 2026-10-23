import argparse

import pytest

from demandpulse import cli
from demandpulse.app.snapshot import Snapshot
from demandpulse.demo import DemoResult, render_benchmark
from demandpulse.synthetic import SyntheticSummary


def test_demo_builds_the_lake_prints_timings_and_the_dashboard_command(tmp_path, capsys):
    out = tmp_path / "demo"
    assert cli.main(["demo", "--out", str(out), "--stores", "34"]) == 0
    text = capsys.readouterr().out
    assert "| total |" in text
    assert f"DEMANDPULSE_APP_DATA={out / 'app'} uv run streamlit run" in text
    assert Snapshot(out / "app").has("risk_windows_store")


def test_demo_refuses_a_directory_that_already_has_files(tmp_path, capsys):
    out = tmp_path / "demo"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    assert cli.main(["demo", "--out", str(out)]) == 2
    assert "already has files" in capsys.readouterr().err
    assert (out / "keep.txt").read_text() == "mine"


def test_benchmark_prints_one_column_per_size(capsys):
    assert cli.main(["benchmark", "--stores", "34,36"]) == 0
    text = capsys.readouterr().out
    assert "| stage | 34 stores | 36 stores |" in text
    assert "invented data" in text


@pytest.mark.parametrize("bad", ["", "0", "-5", "a"])
def test_bad_store_counts_are_rejected(bad):
    with pytest.raises((argparse.ArgumentTypeError, ValueError, SystemExit)):
        cli.build_parser().parse_args(["benchmark", "--stores", bad])


def test_benchmark_table_has_a_row_per_stage_and_a_total(tmp_path):
    def result(stores: int, rows: int) -> DemoResult:
        data = SyntheticSummary(rows=rows, baskets=1, weeks=117, stores=stores)
        return DemoResult(
            root=tmp_path, app_dir=tmp_path, data=data, seconds={"gold": 1.3, "risk": 0.5}
        )

    table = render_benchmark([result(10, 1000), result(20, 2000)])
    assert "| basket lines | 1,000 | 2,000 |" in table
    assert "| gold | 1.3s | 1.3s |" in table
    assert "| risk | 0.5s | 0.5s |" in table
    assert "| total | 1.8s | 1.8s |" in table
