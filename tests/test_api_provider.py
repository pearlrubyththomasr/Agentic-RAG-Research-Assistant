import os
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from agent.llm_client import LLMClient


def test_llm_client_default_provider_is_mock():
    client = LLMClient(provider=None)
    assert client.provider == "mock"
    assert client.generate("prompt") == "Mock answer based on retrieved context."


def test_llm_client_openai_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(provider=None)
    assert client.provider == "openai"
    try:
        client.generate("prompt")
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected runtime error for missing OPENAI_API_KEY")


def test_api_respects_llm_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    client = TestClient(app)
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Mock answer based on retrieved context."
