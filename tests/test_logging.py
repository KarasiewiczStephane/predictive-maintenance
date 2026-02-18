"""Tests for the logging and monitoring utilities."""

import json
import logging
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware import RequestLoggingMiddleware
from src.utils.logging import JSONFormatter, setup_logging, timing_decorator


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_produces_valid_json(self) -> None:
        """Output is valid JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "test message"
        assert parsed["logger"] == "test"
        assert "timestamp" in parsed

    def test_includes_request_id(self) -> None:
        """Request ID included when present on record."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        record.request_id = "abc123"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "abc123"

    def test_includes_exception(self) -> None:
        """Exception info included when present."""
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="error occurred",
                args=None,
                exc_info=True,
            )
            # Need to capture exc_info
            import sys

            record.exc_info = sys.exc_info()

        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_plain_format(self) -> None:
        """Plain text format configures correctly."""
        setup_logging(log_level="DEBUG", json_format=False)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1

    def test_json_format(self) -> None:
        """JSON format configures JSONFormatter."""
        setup_logging(log_level="INFO", json_format=True)
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_log_level_filtering(self) -> None:
        """Log level filters lower-priority messages."""
        setup_logging(log_level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING


class TestTimingDecorator:
    """Tests for timing_decorator."""

    def test_logs_duration(self, caplog: pytest.LogCaptureFixture) -> None:
        """Decorated function logs execution duration."""

        @timing_decorator
        def slow_function() -> str:
            time.sleep(0.01)
            return "done"

        with caplog.at_level(logging.INFO):
            result = slow_function()

        assert result == "done"
        assert "slow_function" in caplog.text
        assert "completed in" in caplog.text

    def test_preserves_return_value(self) -> None:
        """Decorated function returns original value."""

        @timing_decorator
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_name(self) -> None:
        """Decorated function preserves original __name__."""

        @timing_decorator
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"


class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware."""

    def test_adds_request_id_header(self) -> None:
        """Middleware adds X-Request-ID to response headers."""
        test_app = FastAPI()
        test_app.add_middleware(RequestLoggingMiddleware)

        @test_app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(test_app)
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8

    def test_logs_request_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Middleware logs method, path, status, and duration."""
        test_app = FastAPI()
        test_app.add_middleware(RequestLoggingMiddleware)

        @test_app.get("/test-log")
        async def test_log_endpoint():
            return {"status": "ok"}

        client = TestClient(test_app)
        with caplog.at_level(logging.INFO):
            client.get("/test-log")

        assert "method=GET" in caplog.text
        assert "path=/test-log" in caplog.text
        assert "status=200" in caplog.text
        assert "duration=" in caplog.text
