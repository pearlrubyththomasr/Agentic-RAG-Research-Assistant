"""Provider-agnostic LLM client. Uses provider implementations in the `llm` package."""

from __future__ import annotations

import os
from typing import Any

from llm.base import LLMProvider
from llm.ollama import OllamaProvider
from prompts.loader import load_prompt


class LLMClient:
    """Thin wrapper to select an LLM provider based on environment configuration.

    Supported providers: 'mock', 'ollama', 'openai' (optional — requires openai package).
    Default is 'mock' to keep local/free operation by default.
    """

    def __init__(self, provider: str | None = None) -> None:
        self.provider_name = provider or os.getenv("LLM_PROVIDER", "mock")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.provider: LLMProvider | None = None
        if self.provider_name == "ollama":
            self.provider = OllamaProvider()

    def generate(self, prompt: str) -> str:
        if self.provider_name == "mock":
            return self._mock_generate(prompt)
        if self.provider_name == "ollama":
            assert self.provider is not None
            return self.provider.generate(prompt)
        if self.provider_name == "openai":
            return self._openai_generate(prompt)
        raise NotImplementedError(f"LLM provider '{self.provider_name}' is not implemented")

    def critique(self, answer: str) -> float:
        if self.provider_name == "mock":
            return self._mock_critique(answer)
        if self.provider_name == "ollama":
            assert self.provider is not None
            return self.provider.critique(answer)
        if self.provider_name == "openai":
            return self._openai_critique(answer)
        raise NotImplementedError(f"LLM provider '{self.provider_name}' is not implemented")

    def format_prompt(self, question: str, context: str) -> str:
        # Use versioned prompt templates when available
        try:
            prompt_meta = load_prompt("answer", "v1")
            template = prompt_meta["text"]
            # replace placeholders
            prompt = template.replace("{context}", context).replace("{question}", question)
            # record last used prompt version for observability
            self.last_prompt_version = f"answer:{prompt_meta.get('version','v1')}"
            return prompt
        except Exception:
            # fallback to simple inline prompt
            self.last_prompt_version = "answer:inline"
            return f"Context:\n{context}\nQuestion:\n{question}"

    def _mock_generate(self, prompt: str) -> str:
        return "Mock answer based on retrieved context."

    def _mock_critique(self, answer: str) -> float:
        return 0.82

    def _load_openai(self) -> Any:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("openai package is required for the openai provider") from exc
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for openai provider")
        openai.api_key = self.api_key
        return openai

    def _openai_generate(self, prompt: str) -> str:
        openai = self._load_openai()
        response = openai.ChatCompletion.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def _openai_critique(self, answer: str) -> float:
        openai = self._load_openai()
        prompt = (
            "You are evaluating the faithfulness of an answer given the retrieved context. "
            "Rate the answer from 0.0 to 1.0, where 1.0 means the answer is fully supported by the context. "
            "Answer only with a single decimal number.\n\n"
            f"Answer:\n{answer}\n"
        )
        response = openai.ChatCompletion.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5

