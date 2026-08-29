from agent.graph import build_agent_graph


def test_graph_builds_and_runs():
    graph = build_agent_graph(use_mock_llm=True)
    result = graph.invoke({"query": "What is the paper about?"})
    assert "final_answer" in result
    assert result["retry_count"] <= 2
