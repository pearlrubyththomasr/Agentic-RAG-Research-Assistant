from agent.graph import build_agent_graph


def test_graph_runs_with_mock_llm():
    graph = build_agent_graph(use_mock_llm=True, enable_reformulation=False)
    state = {"query": "Summarize claim-centered systems", "retry_count": 0}
    result = graph.invoke(state)
    assert "final_answer" in result
    assert "confidence" in result
