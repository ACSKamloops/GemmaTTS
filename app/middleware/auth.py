"""Authentication middleware for GemmaTTS.

Supports three modes configured via AUTH_MODE:
  - 'none'  : No authentication (default, for local development).
  - 'token' : Bearer token in the Authorization header, matched against API_TOKEN.
  - 'hmac'  : HMAC-SHA256 signature of the request body in X-Signature header,
               using the existing SECRET_KEY from settings.

Auth is enforced only on mutation methods (POST, PUT, DELETE, PATCH).
GET requests and any request to a ``/health`` path are always allowed through.
"""

import hashlib
import hmac
import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings
from app.logging_config import get_request_id, set_request_id

logger = logging.getLogger(__name__)

# Methods that require authentication
_MUTATION_METHODS = {"POST", "PUT", "DELETE", "PATCH"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Starlette/FastAPI middleware that gates mutation endpoints behind auth."""

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Assign or propagate a request-id for correlation
        rid = request.headers.get("x-request-id") or get_request_id()
        set_request_id(rid)

        # Always allow non-mutation methods
        if request.method not in _MUTATION_METHODS:
            return await call_next(request)

        # Always allow health endpoints regardless of method
        if self._is_health_endpoint(request.url.path):
            return await call_next(request)

        auth_mode = settings.auth_mode.lower()

        if auth_mode == "none":
            return await call_next(request)

        if auth_mode == "token":
            return await self._check_token(request, call_next)

        if auth_mode == "hmac":
            return await self._check_hmac(request, call_next)

        # Unknown mode — reject safely
        logger.error("Unknown AUTH_MODE '%s'; rejecting request", auth_mode)
        return self._unauthorized("Server authentication misconfigured")

    # ------------------------------------------------------------------
    # Token auth
    # ------------------------------------------------------------------
    async def _check_token(
        self, request: Request, call_next: Callable
    ) -> Response:
        token = settings.api_token
        if not token:
            logger.error("AUTH_MODE=token but API_TOKEN is not set")
            return self._unauthorized("Server authentication misconfigured")

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "Missing or malformed Authorization header from %s",
                request.client.host if request.client else "unknown",
            )
            return self._unauthorized(
                "Missing or malformed Authorization header. Expected: Bearer <token>"
            )

        provided = auth_header[len("Bearer ") :]
        if not hmac.compare_digest(provided, token):
            logger.warning(
                "Invalid bearer token from %s",
                request.client.host if request.client else "unknown",
            )
            return self._unauthorized("Invalid authentication token")

        return await call_next(request)

    # ------------------------------------------------------------------
    # HMAC auth
    # ------------------------------------------------------------------
    async def _check_hmac(
        self, request: Request, call_next: Callable
    ) -> Response:
        secret = settings.secret_key
        if not secret:
            logger.error("AUTH_MODE=hmac but SECRET_KEY is not set")
            return self._unauthorized("Server authentication misconfigured")

        signature = request.headers.get("x-signature")
        if not signature:
            logger.warning(
                "Missing X-Signature header from %s",
                request.client.host if request.client else "unknown",
            )
            return self._unauthorized(
                "Missing X-Signature header for HMAC authentication"
            )

        # Read body (will be cached by Starlette for downstream handlers)
        body = await request.body()
        expected = hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            logger.warning(
                "HMAC signature mismatch from %s",
                request.client.host if request.client else "unknown",
            )
            return self._unauthorized("Invalid HMAC signature")

        return await call_next(request)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_health_endpoint(path: str) -> bool:
        """Return True for paths that end with ``/health``."""
        normalised = path.rstrip("/")
        return normalised == "/health" or normalised.endswith("/health")

    @staticmethod
    def _unauthorized(detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"detail": detail},
        )
