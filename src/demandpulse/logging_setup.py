"""Logging for the CLI and pipelines: plain text or one JSON object per line, plus run timings."""

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LEVEL_ENV = "DEMANDPULSE_LOG_LEVEL"
FORMAT_ENV = "DEMANDPULSE_LOG_FORMAT"
LOG_FORMATS = ("text", "json")
# These log every credential lookup and HTTP connection at INFO, which buries pipeline messages.
NOISY_LOGGERS = ("botocore", "boto3", "urllib3", "s3transfer", "fsspec", "asyncio")


class JsonFormatter(logging.Formatter):
    """One JSON object per record, so run logs can be filtered with jq or shipped as is."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: int | str | None = None, fmt: str | None = None) -> None:
    """Set up the root logger; `level` and `fmt` fall back to the environment, then INFO / text."""
    level = level if level is not None else os.environ.get(LEVEL_ENV, "INFO")
    fmt = fmt if fmt is not None else os.environ.get(FORMAT_ENV, "text")
    if fmt not in LOG_FORMATS:
        raise ValueError(f"Unknown log format {fmt!r}; choose from {LOG_FORMATS}")
    # force=True so repeated calls (tests, notebooks, Streamlit reruns) don't stack handlers.
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    if fmt == "json":
        for handler in logging.getLogger().handlers:
            handler.setFormatter(JsonFormatter())
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


@contextmanager
def timed(logger: logging.Logger, name: str) -> Iterator[None]:
    """Log how long a stage took, including when it fails (the error still propagates)."""
    started = time.monotonic()
    logger.info("%s started", name)
    try:
        yield
    except BaseException:
        logger.error("%s failed after %.1fs", name, time.monotonic() - started)
        raise
    logger.info("%s finished in %.1fs", name, time.monotonic() - started)
