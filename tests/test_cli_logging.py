import json

import pytest

from demandpulse import cli


def test_cli_logs_command_duration_as_json_lines_on_stderr(tmp_path, capsys):
    cli.main(["--local-root", str(tmp_path), "--log-format", "json", "sources"])
    lines = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    messages = [x["message"] for x in lines]
    assert "command sources started" in messages
    assert any(m.startswith("command sources finished in") for m in messages)


def test_a_failing_command_is_logged_and_the_error_propagates(tmp_path, capsys):
    with pytest.raises(KeyError, match="Unknown source"):
        cli.main(["--local-root", str(tmp_path), "ingest", "nope"])
    assert "command ingest failed after" in capsys.readouterr().err
