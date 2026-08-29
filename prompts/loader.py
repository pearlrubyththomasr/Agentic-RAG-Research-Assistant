from __future__ import annotations

from pathlib import Path
from typing import Dict


PROMPTS_ROOT = Path(__file__).resolve().parent


def load_prompt(category: str, version: str = "v1") -> Dict[str, str]:
    """Load a prompt file and return its text and version.

    Returns {'text': str, 'version': 'v1'}
    """
    path = PROMPTS_ROOT / category / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    text = path.read_text(encoding="utf-8")
    # simple parse for header version if present
    first_line = text.splitlines()[0] if text.splitlines() else ""
    ver = version
    if first_line.startswith("## version:"):
        ver = first_line.split(":", 1)[1].strip()
    return {"text": text, "version": ver}
