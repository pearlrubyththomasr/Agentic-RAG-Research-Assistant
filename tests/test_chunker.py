from ingestion.chunker import chunk_text, chunk_text_recursive


def test_chunk_text_returns_multiple_chunks():
    text = "word " * 40
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)


def test_recursive_chunker_splits_sentences():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_text_recursive(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 1
    assert any("Second" in chunk for chunk in chunks)
