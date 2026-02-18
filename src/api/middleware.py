"""FastAPI middleware for request logging and tracking.

Adds request ID generation, timing, and structured request/response
logging for all API calls.
"""

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs request details and adds a tracking request ID.

    Logs method, path, status code, and duration for every request.
    Adds X-Request-ID header to all responses.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process a request with logging and request ID tracking.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            Response with X-Request-ID header added.
        """
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start
        logger.info(
            "request_id=%s method=%s path=%s status=%d duration=%.4fs",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        response.headers["X-Request-ID"] = request_id
        return response
