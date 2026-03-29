"""
LucilleLLM - Structured Logging

JSON-formatted logging for Cloud Run compatibility.
Automatically activated when LOG_FORMAT=json (production).
Falls back to standard formatting for local development.
"""

import json
import logging
import os
import sys


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for Cloud Run structured logging.
    Maps Python log levels to Google Cloud severity levels.
    """

    _SEVERITY_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "severity": self._SEVERITY_MAP.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging():
    """
    Configure logging based on environment.

    - LOG_FORMAT=json  -> JSON structured logging (production/Cloud Run)
    - LOG_FORMAT=text  -> Human-readable logging (local development)
    """
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s [%(module)s:%(lineno)d] %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
