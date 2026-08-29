from pathlib import Path

from retriever.vector_store import FaissVectorStore


def test_faiss_retriever_loads_index():
    index_path = Path("data/index/faiss.index")
    metadata_path = Path("data/index/metadata.json")
    store = FaissVectorStore.from_files(index_path, metadata_path)
    results = store.retrieve("artificial intelligence", k=2)
    assert isinstance(results, list)
    assert all(hasattr(chunk, "text") for chunk in results)
