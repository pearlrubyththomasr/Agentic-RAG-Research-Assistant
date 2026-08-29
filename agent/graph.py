"""Build and compile the LangGraph-based agent workflow."""

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from agent.nodes import (
    critique_node,
    final_answer_node,
    generate_node,
    query_node,
    reformulate_node,
    retrieve_node,
    MockLLM,
)
from agent.state import AgentState


def build_agent_graph(use_mock_llm: bool = True, enable_reformulation: bool = True) -> Any:
    """Compile a simple graph with a bounded self-correction loop."""
    workflow = StateGraph(AgentState)
    llm_client = MockLLM() if use_mock_llm else None

    workflow.add_node("query_node", lambda state: query_node(state))
    workflow.add_node("retrieve_node", lambda state: retrieve_node(state))
    workflow.add_node("generate_node", lambda state: generate_node(state, llm=llm_client))
    workflow.add_node("critique_node", lambda state: critique_node(state, llm=llm_client))
    # Add reformulation node only when the reformulation feature is enabled
    workflow.add_node("final_node", lambda state: final_answer_node(state))

    workflow.set_entry_point("query_node")
    workflow.add_edge("query_node", "retrieve_node")
    workflow.add_edge("retrieve_node", "generate_node")
    workflow.add_edge("generate_node", "critique_node")

    def should_retry(state: AgentState) -> str:
        score = state.get("critique_score", 0.0) or 0.0
        retries = state.get("retry_count", 0)
        if score >= 0.75 or retries >= 2:
            return "final"
        return "reformulate"

    if enable_reformulation:
        # add reformulation node when enabled
        workflow.add_node("reformulate_node", lambda state: reformulate_node(state))
        workflow.add_conditional_edges(
            "critique_node",
            should_retry,
            {"final": "final_node", "reformulate": "reformulate_node"},
        )
        workflow.add_edge("reformulate_node", "retrieve_node")
    else:
        workflow.add_edge("critique_node", "final_node")
    workflow.add_edge("final_node", END)

    return workflow.compile()
