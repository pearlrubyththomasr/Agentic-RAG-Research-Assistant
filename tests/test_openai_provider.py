import sys
import types

from agent.llm_client import LLMClient


def test_openai_generate_and_critique_with_mocked_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def create_generate(**kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Generated answer from OpenAI"))]
        )

    def create_critique(**kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="0.8"))]
        )

    openai_module = types.SimpleNamespace()
    openai_module.ChatCompletion = types.SimpleNamespace(create=create_generate)
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    client = LLMClient(provider="openai")
    generated = client.generate("Test prompt")
    assert generated == "Generated answer from OpenAI"

    openai_module.ChatCompletion = types.SimpleNamespace(create=create_critique)
    score = client.critique("Test answer")
    assert score == 0.8
