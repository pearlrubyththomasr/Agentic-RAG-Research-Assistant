"""Agent node implementations for the research assistant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agent.state import AgentState
from retriever.vector_store import Chunk, FaissVectorStore, InMemoryVectorStore


DEFAULT_FAISS_INDEX = Path("data/index/faiss.index")
DEFAULT_FAISS_METADATA = Path("data/index/metadata.json")


def load_default_vector_store() -> FaissVectorStore | InMemoryVectorStore:
    if DEFAULT_FAISS_INDEX.exists() and DEFAULT_FAISS_METADATA.exists():
        try:
            return FaissVectorStore.from_files(DEFAULT_FAISS_INDEX, DEFAULT_FAISS_METADATA)
        except Exception:
            fallback_store = InMemoryVectorStore()
            try:
                metadata = json.loads(DEFAULT_FAISS_METADATA.read_text(encoding="utf-8"))
                texts = [item.get("text", "") for item in metadata]
                fallback_store.add_texts(texts, metadata)
            except Exception:
                pass
            return fallback_store
    return InMemoryVectorStore()


from agent.llm_client import LLMClient
from observability.observability import record_llm_metric


class MockLLM:
    """Simple deterministic LLM used for tests and offline smoke runs."""

    def generate(self, prompt: str) -> str:
        return "Mock answer based on retrieved context."

    def critique(self, answer: str) -> float:
        return 0.82

    def format_prompt(self, question: str, context: str) -> str:
        return f"Context:\n{context}\nQuestion:\n{question}"


def query_node(state: AgentState, llm: Any | None = None) -> AgentState:
    """Refine the user query before retrieval."""
    state["query"] = state.get("query", "").strip()
    state["retry_count"] = state.get("retry_count", 0)
    return state


def retrieve_node(state: AgentState, retriever: FaissVectorStore | InMemoryVectorStore | None = None) -> AgentState:
    """Retrieve top-k chunk matches using the vector store."""
    if retriever is None:
        retriever = load_default_vector_store()
    chunks = retriever.retrieve(state.get("query", ""), k=3)
    state["retrieved_chunks"] = [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks]
    return state


def generate_node(state: AgentState, llm: Any | None = None) -> AgentState:
    """Generate a draft answer from retrieved chunks."""
    llm_client = llm or LLMClient()
    chunks = state.get("retrieved_chunks", [])
    context = "\n".join(chunk["text"] for chunk in chunks)
    prompt = (
        llm_client.format_prompt(state.get("query", ""), context)
        if hasattr(llm_client, "format_prompt")
        else f"Context:\n{context}\nQuestion:\n{state.get('query')}"
    )
    start = __import__("time").perf_counter()
    state["draft_answer"] = llm_client.generate(prompt)
    llm_latency_ms = int((__import__("time").perf_counter() - start) * 1000)
    state["llm_latency_ms"] = llm_latency_ms
    # write a lightweight metric for monitoring
    try:
        record_llm_metric({
            "ts": int(__import__("time").time()),
            "node": "generate",
            "latency_ms": llm_latency_ms,
            "prompt_version": getattr(llm_client, "last_prompt_version", None),
        })
    except Exception:
        pass
    # record prompt version used for this invocation when available
    if hasattr(llm_client, "last_prompt_version"):
        state["prompt_version"] = getattr(llm_client, "last_prompt_version")
    return state


def critique_node(state: AgentState, llm: Any | None = None) -> AgentState:
    """Score the draft answer for grounding and relevance."""
    llm_client = llm or LLMClient()
    start = __import__("time").perf_counter()
    state["critique_score"] = llm_client.critique(state.get("draft_answer", ""))
    critique_latency_ms = int((__import__("time").perf_counter() - start) * 1000)
    state["critique_latency_ms"] = critique_latency_ms
    try:
        record_llm_metric({
            "ts": int(__import__("time").time()),
            "node": "critique",
            "latency_ms": critique_latency_ms,
            "prompt_version": getattr(llm_client, "last_prompt_version", None),
        })
    except Exception:
        pass
    return state


def reformulate_node(state: AgentState) -> AgentState:
    """Rewrite the query when the first retrieval pass looks weak."""
    current_query = state.get("query", "")
    state["query"] = f"{current_query} with more context"
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


def final_answer_node(state: AgentState) -> AgentState:
    """Write the final answer and confidence label."""
    score = state.get("critique_score", 0.0)
    state["final_answer"] = state.get("draft_answer", "")
    state["confidence"] = "high" if score >= 0.75 else "low"
    state["citations"] = state.get("retrieved_chunks", [])
    return state
