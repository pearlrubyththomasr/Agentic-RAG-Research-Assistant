from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError()

    @abstractmethod
    def critique(self, answer: str) -> float:
        raise NotImplementedError()

    def format_prompt(self, question: str, context: str) -> str:
        return f"Context:\n{context}\nQuestion:\n{question}"
