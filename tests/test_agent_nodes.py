from agent.nodes import MockLLM, generate_node, critique_node


def test_generate_and_critique_with_mockllm():
    state = {
        "query": "What is claim-centered search?",
        "retrieved_chunks": [{"text": "Paper about claim-centered search.", "metadata": {"paper_id": "2607.28618v1"}}],
    }
    mock = MockLLM()
    state = generate_node(state, llm=mock)
    assert "draft_answer" in state
    state = critique_node(state, llm=mock)
    assert "critique_score" in state
    assert isinstance(state["critique_score"], float)
