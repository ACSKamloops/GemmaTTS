"""Structured JSON logging configuration for GemmaTTS.

Provides request-correlated JSON logs to stdout with configurable level
via the LOG_LEVEL environment variable (default: INFO).
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

# Context variable for per-request correlation
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

SERVICE_NAME = "gemma-tts"


def get_request_id() -> str:
    """Return the current request_id or generate a new one."""
    rid = request_id_ctx.get()
    if rid is None:
        rid = uuid.uuid4().hex
        request_id_ctx.set(rid)
    return rid


def set_request_id(rid: str) -> None:
    """Set the request_id for the current context."""
    request_id_ctx.set(rid)


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "request_id": request_id_ctx.get(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields attached via `logging.info("msg", extra={...})`
        for key in ("extra",):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry, default=str)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the root logger with JSON output to stdout.

    Parameters
    ----------
    log_level:
        Standard Python log-level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    # Remove any existing handlers to avoid duplicate output
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
