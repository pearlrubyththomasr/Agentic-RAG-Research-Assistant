from __future__ import annotations

import os
from typing import Any

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: int | None = None) -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3:latest")
        self.timeout = int(timeout or os.getenv("OLLAMA_TIMEOUT", "300"))

    def _complete(self, prompt: str) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests package is required for the ollama provider") from exc

        endpoints = [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/generate",
        ]

        payloads = [
            {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
            {"model": self.model, "prompt": prompt, "temperature": 0.0},
            {"model": self.model, "input": prompt, "temperature": 0.0},
        ]

        last_exc = None
        for endpoint, payload in zip(endpoints, payloads):
            url = f"{self.base_url.rstrip('/')}{endpoint}"
            for attempt in range(1, 4):
                try:
                    resp = requests.post(url, json=payload, timeout=self.timeout)
                except Exception as exc:
                    last_exc = exc
                    wait = 0.5 * (2 ** (attempt - 1))
                    try:
                        __import__("time").sleep(wait)
                    except Exception:
                        pass
                    continue

                if not resp.ok:
                    last_exc = RuntimeError(f"Ollama request failed ({resp.status_code}): {resp.text[:500]}")
                    # try again with backoff
                    wait = 0.5 * (2 ** (attempt - 1))
                    try:
                        __import__("time").sleep(wait)
                    except Exception:
                        pass
                    continue

                try:
                    data = resp.json()
                except Exception as exc:
                    last_exc = exc
                    break

                # Attempt several plausible response shapes
                text = None
                try:
                    text = data["choices"][0]["message"]["content"].strip()
                except Exception:
                    pass
                if not text:
                    try:
                        text = data["choices"][0]["text"].strip()
                    except Exception:
                        pass
                if not text:
                    try:
                        # some Ollama builds return 'output': [{'content': '...'}]
                        text = data.get("output", [])[0].get("content", "").strip()
                    except Exception:
                        pass

                if text:
                    return text

        # If we reach here, nothing worked
        raise RuntimeError(f"Ollama request failed; last error: {last_exc}")

    def generate(self, prompt: str) -> str:
        return self._complete(prompt)

    def critique(self, answer: str) -> float:
        prompt = (
            "You are evaluating the faithfulness of an answer given the retrieved context. "
            "Rate the answer from 0.0 to 1.0, where 1.0 means the answer is fully supported by the context. "
            "Answer only with a single decimal number.\n\n"
            f"Answer:\n{answer}\n"
        )
        try:
            return float(self._complete(prompt))
        except ValueError:
            return 0.5
