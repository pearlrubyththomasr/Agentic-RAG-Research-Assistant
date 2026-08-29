from prompts.loader import load_prompt


def test_load_prompt_answer_v1():
    meta = load_prompt("answer", "v1")
    assert isinstance(meta, dict)
    assert "text" in meta and "version" in meta
    assert "Answer:" in meta["text"]
