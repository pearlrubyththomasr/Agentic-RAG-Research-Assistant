"""Quick connectivity check for an Ollama server.

Usage:
    python scripts/check_ollama.py

It reads `OLLAMA_HOST` or `OLLAMA_BASE_URL` and `OLLAMA_MODEL` from the environment and
sends a short prompt to verify the server responds as expected.
"""
from __future__ import annotations

import os
import sys
import requests

BASE = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

URL = f"{BASE.rstrip('/')}/v1/chat/completions"

try:
    resp = requests.post(URL, json={"model": MODEL, "messages": [{"role": "user", "content": "Say hello"}]}, timeout=10)
except Exception as exc:
    print(f"Failed to contact Ollama at {BASE}: {exc}")
    sys.exit(2)

if not resp.ok:
    print(f"Ollama responded with {resp.status_code}: {resp.text[:500]}")
    sys.exit(3)

try:
    data = resp.json()
    content = data.get("choices", [])[0].get("message", {}).get("content", "")
    print("Ollama responded:")
    print(content)
    sys.exit(0)
except Exception as exc:
    print("Failed to parse Ollama response:", exc)
    print(resp.text[:1000])
    sys.exit(4)
