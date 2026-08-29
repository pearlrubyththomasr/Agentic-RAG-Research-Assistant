"""Download arXiv PDFs for raw documents and extract text using PyMuPDF (pymupdf).

Produces an augmented JSON file at `data/raw/arxiv_raw_with_fulltext.json` where each
entry has a `fulltext` key containing extracted text.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import fitz  # pymupdf


PDF_DIR = Path("data/raw/pdfs")


def pdf_url_for_id(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def download_pdf(arxiv_id: str, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = pdf_url_for_id(arxiv_id)
    out_path = out_dir / f"{arxiv_id}.pdf"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return out_path
    except Exception:
        return None


def extract_text_from_pdf(path: Path) -> str:
    text_parts: List[str] = []
    try:
        doc = fitz.open(path)
        for page in doc:
            text_parts.append(page.get_text())
        return "\n".join(text_parts)
    except Exception:
        return ""


def augment_with_fulltext(raw_path: Path, out_path: Path, sleep: float = 1.0) -> None:
    if not raw_path.exists():
        raise RuntimeError(f"Raw file not found: {raw_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    papers = raw
    augmented: List[Dict[str, Any]] = []
    for p in papers:
        arxiv_id = p.get("id")
        if not arxiv_id:
            augmented.append(p)
            continue
        pdf_path = download_pdf(arxiv_id, PDF_DIR)
        fulltext = ""
        if pdf_path:
            fulltext = extract_text_from_pdf(pdf_path)
        p["fulltext"] = fulltext
        augmented.append(p)
        time.sleep(sleep)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(augmented, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PDFs and extract full text for raw arXiv JSON.")
    parser.add_argument("--input", default=Path("data/raw/arxiv_raw.json"))
    parser.add_argument("--output", default=Path("data/raw/arxiv_raw_with_fulltext.json"))
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    augment_with_fulltext(Path(args.input), Path(args.output), sleep=args.sleep)
    print(f"Wrote augmented file to {args.output}")


if __name__ == "__main__":
    main()
