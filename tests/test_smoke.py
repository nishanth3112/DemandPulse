import duckdb
import polars as pl
import pytest

from demandpulse import cli


def test_cli_reports_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("demandpulse ")


def test_parquet_round_trip_through_polars_and_duckdb(tmp_path):
    path = tmp_path / "sales.parquet"
    pl.DataFrame({"store_code": ["S1", "S1", "S2"], "spend": [1.5, 2.5, 4.0]}).write_parquet(path)

    total = duckdb.sql(
        f"SELECT store_code, SUM(spend) AS spend FROM '{path}' GROUP BY 1 ORDER BY 1"
    )
    assert total.fetchall() == [("S1", 4.0), ("S2", 4.0)]


@pytest.mark.parametrize("module", ["lightgbm", "pyarrow", "plotly", "statsmodels", "streamlit"])
def test_core_dependencies_import(module):
    __import__(module)
