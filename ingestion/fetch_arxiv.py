"""Fetch arXiv metadata and abstracts with respectful throttling."""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def parse_arxiv_entry(entry: ET.Element) -> Dict[str, Any]:
    def text(tag: str) -> str:
        element = entry.find(f"atom:{tag}", ATOM_NS)
        return element.text.strip() if element is not None and element.text else ""

    authors = [
        a.find("atom:name", ATOM_NS).text.strip()
        for a in entry.findall("atom:author", ATOM_NS)
        if a.find("atom:name", ATOM_NS) is not None and a.find("atom:name", ATOM_NS).text
    ]
    categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ATOM_NS)]
    link = ""
    for link_element in entry.findall("atom:link", ATOM_NS):
        if link_element.attrib.get("rel") == "alternate":
            link = link_element.attrib.get("href", "")
            break
    return {
        "id": text("id").split("/")[-1],
        "title": text("title"),
        "abstract": text("summary"),
        "authors": authors,
        "published": text("published"),
        "categories": categories,
        "link": link,
    }


def fetch_arxiv_papers(query: str, max_papers: int = 20, sleep_seconds: float = 3.0) -> List[Dict[str, Any]]:
    """Fetch a batch of arXiv papers by query string."""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_papers,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    papers: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        papers.append(parse_arxiv_entry(entry))
        time.sleep(sleep_seconds)
    return papers


def save_raw_papers(papers: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(papers, indent=2), encoding="utf-8")


def load_raw_papers(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv papers and save raw JSON.")
    parser.add_argument("--query", default="cat:cs.AI OR cat:cs.LG", help="arXiv search query")
    parser.add_argument("--max-papers", type=int, default=20, help="Maximum number of papers to fetch")
    parser.add_argument("--output", type=Path, default=Path("data/raw/arxiv_raw.json"), help="Output path for raw paper JSON")
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds to wait between arXiv requests")
    args = parser.parse_args()

    papers = fetch_arxiv_papers(args.query, max_papers=args.max_papers, sleep_seconds=args.sleep)
    save_raw_papers(papers, args.output)
    print(f"Saved {len(papers)} papers to {args.output}")


if __name__ == "__main__":
    main()
