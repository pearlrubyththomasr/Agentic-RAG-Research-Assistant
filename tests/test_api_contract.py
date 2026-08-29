import os

from fastapi.testclient import TestClient

from api.main import app


def test_query_endpoint_returns_expected_schema(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    client = TestClient(app)

    response = client.post("/query", json={"query": "What is RAG?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mock answer based on retrieved context."
    assert isinstance(data["citations"], list)
    assert isinstance(data["confidence"], str)
    assert isinstance(data["retries"], int)
    assert isinstance(data["latency_ms"], int)
    assert data["latency_ms"] >= 0
