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


import time
import threading

class NonceStore:
    def __init__(self, expiry_seconds: int = 300):
        self.nonces = {}
        self.expiry_seconds = expiry_seconds
        self.lock = threading.Lock()

    def is_valid_and_add(self, nonce: str, timestamp: int) -> bool:
        now = int(time.time())
        # Check window
        if abs(now - timestamp) > self.expiry_seconds:
            return False
            
        with self.lock:
            # Clean expired nonces
            expired = [n for n, t in self.nonces.items() if now - t > self.expiry_seconds]
            for n in expired:
                del self.nonces[n]
                
            if nonce in self.nonces:
                return False
                
            self.nonces[nonce] = timestamp
            return True

nonce_store = NonceStore()

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

        # Require a strong SECRET_KEY outside test mode
        if settings.mode == "real" and (not secret or len(secret) < 32):
            logger.error("Encountered insecure/empty SECRET_KEY in real mode")
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

        timestamp_str = request.headers.get("x-timestamp")
        nonce = request.headers.get("x-nonce")

        # In real mode, enforce replay mitigation headers
        if settings.mode == "real" and (not timestamp_str or not nonce):
            logger.warning("Missing X-Timestamp or X-Nonce header in real mode")
            return self._unauthorized("Missing X-Timestamp or X-Nonce header for replay mitigation")

        # Read body (will be cached by Starlette for downstream handlers)
        body = await request.body()

        if timestamp_str and nonce:
            try:
                timestamp = int(timestamp_str)
            except ValueError:
                return self._unauthorized("Invalid X-Timestamp header")

            # Verify signature with full replay context
            body_hash = hashlib.sha256(body).hexdigest()
            payload = f"{request.method}:{request.url.path}:{timestamp_str}:{nonce}:{body_hash}"
            expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

            if not hmac.compare_digest(signature, expected):
                logger.warning("HMAC replay signature mismatch")
                return self._unauthorized("Invalid HMAC signature")

            # Validate timestamp window and unique nonce
            if not nonce_store.is_valid_and_add(nonce, timestamp):
                logger.warning("HMAC request replay detected or expired timestamp")
                return self._unauthorized("Request expired or nonce already used")
        else:
            # Fallback simple body-only signature for backward compatibility in dev/test
            expected = hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected):
                logger.warning(
                    "HMAC simple signature mismatch from %s",
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
