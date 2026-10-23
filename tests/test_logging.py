import json
import logging

import pytest

from demandpulse.logging_setup import FORMAT_ENV, LEVEL_ENV, JsonFormatter, configure_logging, timed


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:], root.level = handlers, level


def record(message="hello %s", args=("world",), exc_info=None) -> logging.LogRecord:
    return logging.LogRecord("demandpulse.x", logging.WARNING, __file__, 1, message, args, exc_info)


def test_json_formatter_writes_one_valid_object_per_record():
    line = JsonFormatter().format(record())
    payload = json.loads(line)
    assert "\n" not in line
    assert payload["message"] == "hello world"
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "demandpulse.x"
    assert "time" in payload


def test_json_formatter_includes_the_traceback_for_errors():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        payload = json.loads(JsonFormatter().format(record(exc_info=sys.exc_info())))
    assert "ValueError: boom" in payload["exception"]


def test_json_format_applies_to_every_handler():
    configure_logging("INFO", "json")
    assert all(isinstance(h.formatter, JsonFormatter) for h in logging.getLogger().handlers)


def test_text_is_the_default_format():
    configure_logging("INFO")
    assert not any(isinstance(h.formatter, JsonFormatter) for h in logging.getLogger().handlers)


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="Unknown log format"):
        configure_logging("INFO", "xml")


def test_level_and_format_come_from_the_environment(monkeypatch):
    monkeypatch.setenv(LEVEL_ENV, "DEBUG")
    monkeypatch.setenv(FORMAT_ENV, "json")
    configure_logging()
    assert logging.getLogger().level == logging.DEBUG
    assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)


def test_an_explicit_level_beats_the_environment(monkeypatch):
    monkeypatch.setenv(LEVEL_ENV, "DEBUG")
    configure_logging("ERROR")
    assert logging.getLogger().level == logging.ERROR


def test_chatty_libraries_are_held_at_warning():
    configure_logging("DEBUG")
    assert logging.getLogger("botocore").level == logging.WARNING


def test_timed_logs_start_and_duration(caplog):
    logger = logging.getLogger("demandpulse.test")
    with caplog.at_level(logging.INFO, logger="demandpulse.test"), timed(logger, "stage one"):
        pass
    assert [r.getMessage().split(" in ")[0] for r in caplog.records][0] == "stage one started"
    assert "stage one finished in" in caplog.records[1].getMessage()


def test_timed_logs_the_failure_and_still_raises(caplog):
    logger = logging.getLogger("demandpulse.test")
    with (
        caplog.at_level(logging.INFO, logger="demandpulse.test"),
        pytest.raises(RuntimeError, match="nope"),
        timed(logger, "stage two"),
    ):
        raise RuntimeError("nope")
    assert caplog.records[-1].levelno == logging.ERROR
    assert "stage two failed after" in caplog.records[-1].getMessage()
