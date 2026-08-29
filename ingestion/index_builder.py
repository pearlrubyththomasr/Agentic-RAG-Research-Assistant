"""Build the FAISS index from raw arXiv JSON data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.embed_and_index import build_and_persist_index
from ingestion.fetch_arxiv import load_raw_papers


def main() -> None:
    raw_path = Path("data/raw/arxiv_raw.json")
    # Prefer augmented raw file with fulltext if present
    augmented = Path("data/raw/arxiv_raw_with_fulltext.json")
    if augmented.exists():
        raw_path = augmented
    index_path = Path("data/index/faiss.index")
    metadata_path = Path("data/index/metadata.json")
    raw_documents = load_raw_papers(raw_path)
    if not raw_documents:
        raise RuntimeError(f"No raw documents found at {raw_path}. Run ingestion/fetch_arxiv.py first.")
    build_and_persist_index(
        raw_documents,
        index_path=index_path,
        metadata_path=metadata_path,
    )
    print(f"Built FAISS index with {len(raw_documents)} documents.")


if __name__ == "__main__":
    main()
