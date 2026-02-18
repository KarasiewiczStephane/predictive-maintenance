"""Structured logging utilities with JSON formatting and performance tracking.

Provides a JSON log formatter, configurable logging setup, and a
timing decorator for measuring function execution duration.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable


class JSONFormatter(logging.Formatter):
    """Log formatter that outputs structured JSON log lines.

    Produces JSON objects with timestamp, level, logger name, message,
    and optional request_id and exception info.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-encoded string representation of the log record.
        """
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if record.exc_info and record.exc_info[1]:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(log_level: str = "INFO", json_format: bool = False) -> None:
    """Configure application-wide logging.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, use JSON formatting; otherwise use plain text.
    """
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(log_level)


def timing_decorator(func: Callable) -> Callable:
    """Decorator that logs the execution time of a function.

    Args:
        func: The function to wrap with timing.

    Returns:
        Wrapped function that logs duration after completion.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logging.getLogger(func.__module__).info("%s completed in %.4fs", func.__name__, duration)
        return result

    return wrapper
