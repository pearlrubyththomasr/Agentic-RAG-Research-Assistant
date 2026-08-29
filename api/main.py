import logging
import os
import time
import uuid
import json
from pathlib import Path
from typing import Any, Dict

import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from agent.graph import build_agent_graph
from monitoring.monitor import get_system_metrics
from observability.observability import record_request, OBS_ENABLED
from fastapi import BackgroundTasks

import eval.run_eval as eval_runner


class EvaluateRequest(BaseModel):
    provider: str = "mock"
    mlflow: bool = False
    output_dir: str = "eval/results"
    background: bool = False


def _run_eval_and_save(provider: str, mlflow_flag: bool, out_dir: Path) -> None:
    use_mock = provider == "mock"
    baseline = eval_runner.run_eval("baseline", use_agentic=False, use_mock_llm=use_mock)
    agentic = eval_runner.run_eval("agentic", use_agentic=True, use_mock_llm=use_mock)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    Path(out_dir / "agentic.json").write_text(json.dumps(agentic, indent=2), encoding="utf-8")
    Path(out_dir / "latest.json").write_text(json.dumps({"baseline": baseline, "agentic": agentic}, indent=2), encoding="utf-8")
    baseline_summary = eval_runner.summarize(baseline)
    agentic_summary = eval_runner.summarize(agentic)
    Path(out_dir / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2), encoding="utf-8")
    Path(out_dir / "agentic_summary.json").write_text(json.dumps(agentic_summary, indent=2), encoding="utf-8")

    if mlflow_flag:
        try:
            if eval_runner.mlflow is None:
                logger.warning("MLflow requested but mlflow package is not installed; skipping MLflow logging")
            else:
                # reuse eval_runner's mlflow logic by setting args and calling main is intrusive; instead replicate minimal logging here
                mlflow = eval_runner.mlflow
                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
                run_name = f"agentic_rag_eval_{provider}_{int(time.time())}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("provider", provider)
                    mlflow.log_param("num_questions", len(baseline))
                    mlflow.log_param("use_mock", str(use_mock))
                    for k, v in baseline_summary.items():
                        try:
                            mlflow.log_metric(f"baseline/{k}", float(v))
                        except Exception:
                            pass
                    for k, v in agentic_summary.items():
                        try:
                            mlflow.log_metric(f"agentic/{k}", float(v))
                        except Exception:
                            pass
                    mlflow.log_artifact(str(out_dir / "latest.json"), artifact_path="eval")
        except Exception:
            logger.exception("Failed to log to MLflow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic RAG Research Assistant")

provider = os.getenv("LLM_PROVIDER", "mock")
openai_key_present = bool(os.getenv("OPENAI_API_KEY"))
use_mock_llm = provider == "mock"
agent = build_agent_graph(use_mock_llm=use_mock_llm)

MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))

logger.info("Starting API with LLM_PROVIDER=%s, OPENAI_API_KEY=%s", provider, openai_key_present)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list
    confidence: str
    retries: int
    latency_ms: int
    prompt_version: str | None = None
    llm_latency_ms: int | None = None
    critique_latency_ms: int | None = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _check_ollama() -> tuple[bool, str]:
    """Return (ready, message). Gracefully handle missing service or network errors."""
    try:
        url = f"{OLLAMA_BASE.rstrip('/')}/v1/models"
        resp = requests.get(url, timeout=2)
        if resp.ok:
            return True, "ollama available"
        return False, f"ollama returned {resp.status_code}"
    except Exception as exc:  # network error or requests not installed
        return False, f"ollama check failed: {exc}"


@app.get("/ready")
def ready() -> Dict[str, Any]:
    """Readiness probe. Ensures LLM backend is reachable when required."""
    if provider == "ollama":
        ok, msg = _check_ollama()
        status = "ready" if ok else "not ready"
        return {"status": status, "detail": msg}
    # if using mock/openai without key, we consider app ready (openai requires env key to be used)
    return {"status": "ready", "detail": "provider ok"}


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Return lightweight system metrics. Does not require MLflow or Ollama."""
    try:
        return get_system_metrics()
    except Exception as exc:
        logger.exception("Failed to collect system metrics: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to collect system metrics")


@app.get("/mlflow")
def mlflow_info() -> Dict[str, Any]:
    """Return MLflow tracking URI and recent experiments if mlflow package is installed."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        return {"available": False, "reason": "MLFLOW_TRACKING_URI not set"}
    try:
        if eval_runner.mlflow is None:
            return {"available": False, "reason": "mlflow package not installed"}
        mlflow = eval_runner.mlflow
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        exps = client.list_experiments()[:10]
        return {"available": True, "tracking_uri": tracking_uri, "experiments": [{"name": e.name, "id": e.experiment_id} for e in exps]}
    except Exception as exc:
        logger.exception("Failed to query mlflow: %s", exc)
        return {"available": False, "reason": str(exc)}


@app.get("/metrics/history")
def metrics_history(limit: int = 100) -> Dict[str, Any]:
    """Return recent LLM latency metrics collected in logs/llm_metrics.jsonl."""
    path = Path("logs/llm_metrics.jsonl")
    if not path.exists():
        return {"metrics": []}
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.read().strip().splitlines()
        if limit and len(lines) > limit:
            lines = lines[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return {"metrics": out}
    except Exception:
        logger.exception("Failed to read metrics history")
        raise HTTPException(status_code=500, detail="Failed to read metrics history")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, fastapi_request: Request | None = None) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    logger.info("[%s] /query received", request_id)

    if not request.query or len(request.query) > MAX_QUERY_LENGTH:
        logger.warning("[%s] query too long or empty", request_id)
        raise HTTPException(status_code=413, detail="Query is empty or exceeds maximum allowed length")

    if provider == "ollama":
        ok, msg = _check_ollama()
        if not ok:
            logger.error("[%s] Ollama not available: %s", request_id, msg)
            raise HTTPException(status_code=503, detail=("Ollama is not available: " + msg + ". "
                                                         "Start Ollama and set OLLAMA_BASE_URL accordingly."))

    start = time.perf_counter()
    try:
        result = agent.invoke({"query": request.query, "retry_count": 0})
    except Exception as exc:
        logger.exception("[%s] Agent invocation failed: %s", request_id, exc)
        raise HTTPException(status_code=500, detail="Internal error while processing the request")
    latency_ms = int((time.perf_counter() - start) * 1000)

    response = {
        "request_id": request_id,
        "answer": result.get("final_answer", ""),
        "citations": result.get("citations", []),
        "confidence": result.get("confidence", "low"),
        "retries": result.get("retry_count", 0),
        "latency_ms": latency_ms,
    }
    # include optional observability fields when available
    response["prompt_version"] = result.get("prompt_version")
    response["llm_latency_ms"] = result.get("llm_latency_ms")
    response["critique_latency_ms"] = result.get("critique_latency_ms")
    logger.info("[%s] /query completed in %dms, confidence=%s", request_id, latency_ms, response["confidence"])
    # Record observability payload when enabled
    if OBS_ENABLED:
        # Build a richer observability payload including retrieved document ids and titles
        retrieved = []
        for c in result.get("retrieved_chunks", []):
            meta = c.get("metadata", {}) if isinstance(c, dict) else {}
            retrieved.append({
                "paper_id": meta.get("paper_id") or meta.get("id") or meta.get("paperId"),
                "title": meta.get("title"),
                "source": meta.get("source") or meta.get("link"),
            })

        obs_payload = {
            "request_id": request_id,
            "query": request.query,
            "latency_ms": latency_ms,
            "llm_latency_ms": response.get("llm_latency_ms"),
            "critique_latency_ms": response.get("critique_latency_ms"),
            "prompt_version": response.get("prompt_version"),
            "retrieved_count": len(retrieved),
            "retrieved": retrieved,
            "retry_count": result.get("retry_count", 0),
            "confidence": result.get("confidence"),
            "provider": provider,
        }
        try:
            record_request(request_id, obs_payload)
        except Exception:
            logger.exception("Failed to record observability for %s", request_id)
    return response


@app.post("/evaluate")
def evaluate(request: EvaluateRequest, background_tasks: BackgroundTasks | None = None) -> Dict[str, Any]:
    """Run the evaluation harness synchronously. This can be long-running.

    For interactive use, call with provider='mock' to avoid requiring a live LLM.
    """
    out_dir = Path(request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if request.background and background_tasks is not None:
        # schedule background run
        background_tasks.add_task(_run_eval_and_save, request.provider, request.mlflow, out_dir)
        return {"status": "scheduled", "output_dir": str(out_dir)}

    # run synchronously
    _run_eval_and_save(request.provider, request.mlflow, out_dir)
    baseline_summary = json.loads(Path(out_dir / "baseline_summary.json").read_text(encoding="utf-8"))
    agentic_summary = json.loads(Path(out_dir / "agentic_summary.json").read_text(encoding="utf-8"))
    return {
        "status": "completed",
        "output_dir": str(out_dir),
        "baseline_summary": baseline_summary,
        "agentic_summary": agentic_summary,
    }
