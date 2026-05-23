import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import time
import os

from app.services.gemma_service import app, rate_limiter, model_manager

@pytest.fixture
def client():
    # Clear rate limiter timestamps before each test
    rate_limiter.timestamps.clear()
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "gemma-service"}

def test_generate_test_mode(client):
    response = client.post("/generate", json={
        "prompt": "Hello world",
        "test_mode": True
    })
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "generation_time_ms" in data
    assert data["text"] == "MOCK_RESPONSE: Hello world"

def test_generate_thinking_enabled_and_disabled(client):
    # thinking enabled
    response = client.post("/generate", json={
        "prompt": "Hello",
        "enable_thinking": True,
        "test_mode": True
    })
    assert response.status_code == 200
    data = response.json()
    assert "<think>Thinking...</think>" in data["text"]

    # thinking disabled
    response = client.post("/generate", json={
        "prompt": "Hello",
        "enable_thinking": False,
        "test_mode": True
    })
    assert response.status_code == 200
    data = response.json()
    assert "<think>" not in data["text"]
    assert data["text"] == "MOCK_RESPONSE: Hello"

def test_generate_empty_prompt(client):
    response = client.post("/generate", json={
        "prompt": "",
        "test_mode": True
    })
    assert response.status_code == 422

    # Whitespace only prompt
    response = client.post("/generate", json={
        "prompt": "   ",
        "test_mode": True
    })
    assert response.status_code == 422

def test_generate_negative_max_words(client):
    response = client.post("/generate", json={
        "prompt": "Hello",
        "max_words": -5,
        "test_mode": True
    })
    assert response.status_code == 422

def test_generate_non_integer_max_words(client):
    # float max_words
    response = client.post("/generate", json={
        "prompt": "Hello",
        "max_words": 150.5,
        "test_mode": True
    })
    assert response.status_code == 422

    # string non-integer max_words
    response = client.post("/generate", json={
        "prompt": "Hello",
        "max_words": "five",
        "test_mode": True
    })
    assert response.status_code == 422

    # boolean max_words
    response = client.post("/generate", json={
        "prompt": "Hello",
        "max_words": True,
        "test_mode": True
    })
    assert response.status_code == 422

def test_generate_rate_limiting(client):
    # Send 40 requests, all should pass
    for i in range(40):
        response = client.post("/generate", json={
            "prompt": "Hello",
            "test_mode": True
        })
        assert response.status_code == 200
    
    # 41st request should be rate limited
    response = client.post("/generate", json={
        "prompt": "Hello",
        "test_mode": True
    })
    assert response.status_code == 429
    assert response.json()["detail"] == "Too Many Requests"

def test_generate_real_model(client):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    gemma_test_path = PROJECT_ROOT / "models" / "gemma_test"
    
    if gemma_test_path.exists() and (gemma_test_path / "config.json").exists():
        response = client.post("/generate", json={
            "prompt": "Hello",
            "test_mode": False,
            "max_words": 10
        })
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "generation_time_ms" in data
        assert isinstance(data["text"], str)
        assert len(data["text"]) > 0
    else:
        pytest.skip("gemma_test model not found, skipping real model generation test")
