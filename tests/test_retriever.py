from retriever.vector_store import InMemoryVectorStore


def test_in_memory_retriever_returns_matches():
    store = InMemoryVectorStore()
    store.add_texts(["alpha beta", "gamma delta"], [{"id": 1}, {"id": 2}])
    results = store.retrieve("alpha", k=1)
    assert len(results) == 1
    assert results[0].metadata["id"] == 1
