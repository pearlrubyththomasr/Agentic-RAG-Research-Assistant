"""Chunking utilities for arXiv paper text."""

from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split text into overlapping fixed-size chunks."""
    if not text.strip():
        return []
    words = text.split()
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def chunk_text_recursive(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split a string into sentence-aware chunks with simple overlap."""
    if not text.strip():
        return []
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    if overlap > 0 and len(chunks) > 1:
        return [chunk[:chunk_size] for chunk in chunks]
    return chunks
