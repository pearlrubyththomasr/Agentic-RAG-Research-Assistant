"""Vector retrieval wrappers for local FAISS and in-memory backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import faiss
import numpy as np


def _load_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence_transformers is required for FAISS retrieval. "
            "Install it in the active environment or use the mock/in-memory retriever."
        ) from exc
    return SentenceTransformer(model_name)


@dataclass
class Chunk:
    text: str
    metadata: dict


class InMemoryVectorStore:
    """A minimal vector store for local development and tests."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []

    def add_texts(self, texts: List[str], metadata: Optional[List[dict]] = None) -> None:
        metadata = metadata or [{} for _ in texts]
        for text, meta in zip(texts, metadata):
            self._chunks.append(Chunk(text=text, metadata=meta))

    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        query_terms = set(query.lower().split())
        scored = []
        for chunk in self._chunks:
            text_terms = set(chunk.text.lower().split())
            overlap = len(query_terms & text_terms)
            if overlap > 0:
                scored.append((overlap, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]


class FaissVectorStore:
    """FAISS-backed vector store with persisted index and metadata."""

    def __init__(self, index: faiss.IndexFlatIP, metadata: List[dict], embedder: SentenceTransformer) -> None:
        self.index = index
        self.metadata = metadata
        self.embedder = embedder

    @classmethod
    def from_files(cls, index_path: Path, metadata_path: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> "FaissVectorStore":
        index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        embedder = _load_sentence_transformer(model_name)
        return cls(index=index, metadata=metadata, embedder=embedder)

    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        if not query.strip():
            return []
        query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(query_embedding, k)
        results: List[Chunk] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            results.append(Chunk(text=item.get("text", ""), metadata=item))
        return results
