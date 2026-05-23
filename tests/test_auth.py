"""Tests for the AuthMiddleware.

Covers the four required scenarios:
  1. auth_mode='none'  → all requests pass
  2. auth_mode='token' → missing/bad token is 401, valid token passes
  3. auth_mode='token' → GET /health always bypasses auth
  4. auth_mode='hmac'  → valid/invalid HMAC signatures
"""

import hashlib
import hmac as hmac_mod

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings
from app.middleware.auth import AuthMiddleware


# ---------------------------------------------------------------------------
# Helpers: tiny FastAPI app used only by these tests
# ---------------------------------------------------------------------------

def _make_app() -> FastAPI:
    """Build a minimal FastAPI app with the AuthMiddleware attached."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/health")
    def health_post():
        return {"status": "ok"}

    @app.post("/v1/dialogue")
    async def dialogue(request: Request):
        body = await request.json()
        return {"echo": body}

    @app.get("/v1/info")
    def info():
        return {"info": "public"}

    @app.delete("/v1/item")
    def delete_item():
        return {"deleted": True}

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _save_settings():
    """Save and restore settings fields touched by these tests."""
    original_mode = settings.auth_mode
    original_token = settings.api_token
    original_secret = settings.secret_key
    yield
    settings.auth_mode = original_mode
    settings.api_token = original_token
    settings.secret_key = original_secret


@pytest.fixture()
def client(_save_settings) -> TestClient:
    return TestClient(_make_app())


# ---------------------------------------------------------------------------
# 1. auth_mode = 'none' — everything is allowed
# ---------------------------------------------------------------------------

class TestAuthNone:
    def test_post_allowed(self, client: TestClient):
        settings.auth_mode = "none"
        resp = client.post("/v1/dialogue", json={"msg": "hello"})
        assert resp.status_code == 200
        assert resp.json()["echo"]["msg"] == "hello"

    def test_delete_allowed(self, client: TestClient):
        settings.auth_mode = "none"
        resp = client.delete("/v1/item")
        assert resp.status_code == 200

    def test_get_allowed(self, client: TestClient):
        settings.auth_mode = "none"
        resp = client.get("/v1/info")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. auth_mode = 'token'
# ---------------------------------------------------------------------------

class TestAuthToken:
    TOKEN = "super-secret-test-token-12345"

    def test_missing_header_returns_401(self, client: TestClient):
        settings.auth_mode = "token"
        settings.api_token = self.TOKEN
        resp = client.post("/v1/dialogue", json={"msg": "hello"})
        assert resp.status_code == 401
        assert "Authorization" in resp.json()["detail"]

    def test_wrong_token_returns_401(self, client: TestClient):
        settings.auth_mode = "token"
        settings.api_token = self.TOKEN
        resp = client.post(
            "/v1/dialogue",
            json={"msg": "hello"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    def test_valid_token_passes(self, client: TestClient):
        settings.auth_mode = "token"
        settings.api_token = self.TOKEN
        resp = client.post(
            "/v1/dialogue",
            json={"msg": "hello"},
            headers={"Authorization": f"Bearer {self.TOKEN}"},
        )
        assert resp.status_code == 200
        assert resp.json()["echo"]["msg"] == "hello"

    def test_get_request_skips_auth(self, client: TestClient):
        """GET requests are never gated, even in token mode."""
        settings.auth_mode = "token"
        settings.api_token = self.TOKEN
        resp = client.get("/v1/info")
        assert resp.status_code == 200

    def test_malformed_bearer_prefix(self, client: TestClient):
        settings.auth_mode = "token"
        settings.api_token = self.TOKEN
        resp = client.post(
            "/v1/dialogue",
            json={"msg": "hello"},
            headers={"Authorization": "Token some-val"},
        )
        assert resp.status_code == 401

    def test_api_token_not_set_returns_401(self, client: TestClient):
        """If the server has token mode but no API_TOKEN, reject safely."""
        settings.auth_mode = "token"
        settings.api_token = None
        resp = client.post("/v1/dialogue", json={"msg": "hello"})
        assert resp.status_code == 401
        assert "misconfigured" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 3. Health endpoint bypass
# ---------------------------------------------------------------------------

class TestHealthBypass:
    TOKEN = "test-token-health"

    def test_get_health_bypasses_token_auth(self, client: TestClient):
        settings.auth_mode = "token"
        settings.api_token = self.TOKEN
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_post_health_bypasses_token_auth(self, client: TestClient):
        """Even POST to /health should be exempt."""
        settings.auth_mode = "token"
        settings.api_token = self.TOKEN
        resp = client.post("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 4. auth_mode = 'hmac'
# ---------------------------------------------------------------------------

class TestAuthHmac:
    SECRET = "hmac-test-secret-key-must-be-long-enough"

    def _sign(self, body: bytes) -> str:
        return hmac_mod.new(self.SECRET.encode(), body, hashlib.sha256).hexdigest()

    def test_missing_signature_returns_401(self, client: TestClient):
        settings.auth_mode = "hmac"
        settings.secret_key = self.SECRET
        resp = client.post("/v1/dialogue", json={"msg": "hello"})
        assert resp.status_code == 401
        assert "X-Signature" in resp.json()["detail"]

    def test_wrong_signature_returns_401(self, client: TestClient):
        settings.auth_mode = "hmac"
        settings.secret_key = self.SECRET
        resp = client.post(
            "/v1/dialogue",
            json={"msg": "hello"},
            headers={"X-Signature": "deadbeef"},
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    def test_valid_signature_passes(self, client: TestClient):
        import json as _json

        settings.auth_mode = "hmac"
        settings.secret_key = self.SECRET
        body = _json.dumps({"msg": "hello"}).encode()
        sig = self._sign(body)
        resp = client.post(
            "/v1/dialogue",
            content=body,
            headers={
                "X-Signature": sig,
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 200

    def test_get_skips_hmac(self, client: TestClient):
        settings.auth_mode = "hmac"
        settings.secret_key = self.SECRET
        resp = client.get("/v1/info")
        assert resp.status_code == 200
