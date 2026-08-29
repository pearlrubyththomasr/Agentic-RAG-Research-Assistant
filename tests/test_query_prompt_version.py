from fastapi.testclient import TestClient
import os

from agent.llm_client import LLMClient

import api.main as main


def test_query_records_prompt_version(monkeypatch):
    # Ensure we use mock provider so no external LLM required
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    client = TestClient(main.app)
    resp = client.post("/query", json={"query": "test query for prompt version"})
    assert resp.status_code == 200
    data = resp.json()
    # mock provider uses inline prompts so prompt_version may be present or None
    assert "request_id" in data
    assert "answer" in data
