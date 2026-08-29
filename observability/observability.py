from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import requests

OBS_ENABLED = os.getenv("OBSERVABILITY_ENABLED", "false").lower() in ("1", "true", "yes")
LANGFUSE_URL = os.getenv("LANGFUSE_URL")
LANGFUSE_API_KEY = os.getenv("LANGFUSE_API_KEY")


def _write_local(request_id: str, payload: Dict[str, Any]) -> None:
    out_dir = Path("logs/observability")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{request_id}.json"
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _post_langfuse(payload: Dict[str, Any]) -> None:
    if not LANGFUSE_URL:
        return
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if LANGFUSE_API_KEY:
        headers["Authorization"] = f"Bearer {LANGFUSE_API_KEY}"
    try:
        requests.post(LANGFUSE_URL, json=payload, headers=headers, timeout=5)
    except Exception:
        pass


def record_request(request_id: str, payload: Dict[str, Any], forward: Optional[bool] = True) -> None:
    """Record an observability payload locally and optionally forward it to a configured Langfuse URL.

    The record is only written when OBS_ENABLED is true. Forwarding is optional and will not raise.
    """
    if not OBS_ENABLED:
        return

    payload.setdefault("_observed_at", int(time.time()))
    _write_local(request_id, payload)
    if forward:
        _post_langfuse(payload)


def record_llm_metric(metric: Dict[str, Any]) -> None:
    """Append a metric line to logs/llm_metrics.jsonl for local monitoring.

    The metric should be a JSON-serializable dict. This function always writes
    locally (independent of OBS_ENABLED) so the monitoring UI can show recent
    LLM latencies.
    """
    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "llm_metrics.jsonl"
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metric) + "\n")
    except Exception:
        pass

