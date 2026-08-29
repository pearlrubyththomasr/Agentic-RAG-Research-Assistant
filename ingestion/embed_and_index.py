"""Embed chunk text and build a lightweight FAISS index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.chunker import chunk_text


@dataclass
class IndexedChunk:
    text: str
    metadata: Dict[str, Any]
    embedding: List[float]


class SimpleEmbedder:
    """Small wrapper around sentence-transformers for the prototype."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()


def build_documents_from_raw(raw_documents: List[Dict[str, Any]], chunk_size: int = 400, overlap: int = 80) -> List[Dict[str, Any]]:
    """Create chunks from raw document dictionaries."""
    chunks: List[Dict[str, Any]] = []
    for doc in raw_documents:
        # Prefer full-text when available, otherwise fall back to title+abstract
        full = doc.get("fulltext") or ""
        text = full.strip() if full.strip() else f"{doc.get('title', '')}\n{doc.get('abstract', '')}"
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append({
                "text": chunk,
                "paper_id": doc.get("id"),
                "title": doc.get("title"),
                "source": doc.get("link"),
            })
    return chunks


def save_metadata(path: Path, metadata: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_faiss_index(index: faiss.IndexFlatIP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path) -> faiss.IndexFlatIP:
    return faiss.read_index(str(path))


def build_and_persist_index(
    raw_documents: List[Dict[str, Any]],
    index_path: Path,
    metadata_path: Path,
    chunk_size: int = 400,
    overlap: int = 80,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    documents = build_documents_from_raw(raw_documents, chunk_size=chunk_size, overlap=overlap)
    embedder = SimpleEmbedder(model_name=model_name)
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed(texts)

    dimension = len(embeddings[0]) if embeddings else 0
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    save_faiss_index(index, index_path)
    save_metadata(metadata_path, documents)
