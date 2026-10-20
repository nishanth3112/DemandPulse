import duckdb
import polars as pl
import pytest

import demandpulse


def test_package_exposes_cli_entrypoint(capsys):
    demandpulse.main()
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
