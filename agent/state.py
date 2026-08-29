"""Typed state for the LangGraph agent."""

from __future__ import annotations

from typing import List, TypedDict, Optional


class AgentState(TypedDict, total=False):
    query: str
    retrieved_chunks: List[dict]
    draft_answer: str
    critique_score: Optional[float]
    retry_count: int
    final_answer: str
    confidence: str
    citations: List[dict]
