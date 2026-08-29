import sys
import types

from agent.llm_client import LLMClient


def test_ollama_generate_and_critique_with_mocked_requests(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama2")

    def mocked_post(url, json, timeout):
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                message = json["messages"][0]["content"]
                if "Rate the answer" in message:
                    return {"choices": [{"message": {"content": "0.78"}}]}
                return {"choices": [{"message": {"content": "Generated answer from Ollama"}}]}

        return Response()

    requests_module = types.SimpleNamespace(post=mocked_post)
    monkeypatch.setitem(sys.modules, "requests", requests_module)

    client = LLMClient(provider="ollama")
    generated = client.generate("Test prompt")
    assert generated == "Generated answer from Ollama"

    score = client.critique("Test answer")
    assert abs(score - 0.78) < 1e-6
