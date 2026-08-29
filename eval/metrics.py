"""Simple evaluation metrics for RAG-style retrieval experiments."""

from __future__ import annotations

from typing import Iterable, List, Sequence


def precision(relevant: Sequence[int], retrieved: Sequence[int]) -> float:
    if not retrieved:
        return 0.0
    return sum(1 for item in retrieved if item in relevant) / len(retrieved)


def recall(relevant: Sequence[int], retrieved: Sequence[int]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in retrieved if item in relevant) / len(relevant)


def ragas_relevance_score(answer: str, reference: str) -> float:
    answer_tokens = set(answer.lower().split())
    reference_tokens = set(reference.lower().split())
    if not reference_tokens:
        return 0.0
    return len(answer_tokens & reference_tokens) / len(reference_tokens)


def ragas_faithfulness_score(answer: str, retrieved: Sequence[str]) -> float:
    answer_tokens = set(answer.lower().split())
    retrieved_tokens = set(" ".join(retrieved).lower().split())
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & retrieved_tokens) / len(answer_tokens)
