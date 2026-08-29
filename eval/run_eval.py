"""Run the evaluation harness for baseline and self-correcting agent variants."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from agent.graph import build_agent_graph
from eval.metrics import (
    precision,
    recall,
    ragas_faithfulness_score,
    ragas_relevance_score,
)

try:
    import mlflow
except Exception:  # pragma: no cover - optional
    mlflow = None


def load_test_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def make_metrics(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    retrieved_texts = [chunk["text"] for chunk in result.get("retrieved_chunks", [])]
    gold_ids = item.get("gold_chunks", [])
    retrieved_ids = [chunk["metadata"].get("paper_id") for chunk in result.get("retrieved_chunks", []) if chunk.get("metadata")]
    return {
        "question": item["question"],
        "answer": result.get("final_answer", ""),
        "faithfulness": ragas_faithfulness_score(result.get("final_answer", ""), retrieved_texts),
        "answer_relevance": ragas_relevance_score(result.get("final_answer", ""), item.get("reference", "")),
        "context_precision": precision(gold_ids, retrieved_ids),
        "context_recall": recall(gold_ids, retrieved_ids),
        "retrieval_precision_at_5": precision(gold_ids, retrieved_ids[:5]),
        "retrieval_precision_at_10": precision(gold_ids, retrieved_ids[:10]),
        "latency_ms": result.get("latency_ms", 0),
        "retry_count": result.get("retry_count", 0),
        "mode": item.get("mode", ""),
    }


def run_eval(mode: str = "agentic", use_agentic: bool = True, use_mock_llm: bool = True) -> list[dict]:
    graph = build_agent_graph(use_mock_llm=use_mock_llm, enable_reformulation=use_agentic)
    questions = load_test_questions(Path("eval/test_questions.json"))
    results: list[dict[str, Any]] = []
    for item in questions:
        state = {"query": item["question"], "retry_count": 0}
        start = time.time()
        result = graph.invoke(state)
        latency_ms = int((time.time() - start) * 1000)
        result["latency_ms"] = latency_ms
        # include prompt version if available
        item_prompt_version = result.get("prompt_version")
        if item_prompt_version:
            result["prompt_version"] = item_prompt_version
        result_metrics = make_metrics({**item, "mode": mode}, result)
        results.append(result_metrics)
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    summary = {
        "faithfulness": sum(r["faithfulness"] for r in results) / len(results),
        "answer_relevance": sum(r["answer_relevance"] for r in results) / len(results),
        "context_precision": sum(r["context_precision"] for r in results) / len(results),
        "context_recall": sum(r["context_recall"] for r in results) / len(results),
        "retrieval_precision_at_5": sum(r["retrieval_precision_at_5"] for r in results) / len(results),
        "retrieval_precision_at_10": sum(r["retrieval_precision_at_10"] for r in results) / len(results),
        "latency_ms": sum(r["latency_ms"] for r in results) / len(results),
        "retry_rate": sum(1 for r in results if r["retry_count"] > 0) / len(results),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline and agentic evaluation.")
    parser.add_argument("--provider", choices=["mock", "openai", "ollama"], default=os.getenv("LLM_PROVIDER", "mock"))
    parser.add_argument("--output-dir", default="eval/results")
    parser.add_argument("--mlflow", action="store_true", help="Log experiment to MLflow if available")
    args = parser.parse_args()

    # Ensure the selected provider is visible to `LLMClient` and other modules
    os.environ["LLM_PROVIDER"] = args.provider

    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set to use the openai provider.")
    # Accept either OLLAMA_HOST or OLLAMA_BASE_URL for compatibility
    if args.provider == "ollama" and not (os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL")):
        raise RuntimeError("OLLAMA_HOST or OLLAMA_BASE_URL must be set to use the ollama provider.")

    use_mock_llm = args.provider == "mock"
    baseline = run_eval("baseline", use_agentic=False, use_mock_llm=use_mock_llm)
    agentic = run_eval("agentic", use_agentic=True, use_mock_llm=use_mock_llm)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    Path(output_dir / "agentic.json").write_text(json.dumps(agentic, indent=2), encoding="utf-8")
    Path(output_dir / "baseline_summary.json").write_text(json.dumps(summarize(baseline), indent=2), encoding="utf-8")
    Path(output_dir / "agentic_summary.json").write_text(json.dumps(summarize(agentic), indent=2), encoding="utf-8")
    # Write a combined latest file for quick access
    Path(output_dir / "latest.json").write_text(json.dumps({"baseline": baseline, "agentic": agentic}, indent=2), encoding="utf-8")

    # Optionally log a detailed run to MLflow
    if args.mlflow:
        if mlflow is None:
            raise RuntimeError("mlflow is not installed but --mlflow was requested. Install mlflow or omit --mlflow.")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        baseline_summary = summarize(baseline)
        agentic_summary = summarize(agentic)
        run_name = f"agentic_rag_eval_{args.provider}_{int(time.time())}"
        with mlflow.start_run(run_name=run_name):
            # Params
            mlflow.log_param("provider", args.provider)
            mlflow.log_param("num_questions", len(baseline))
            mlflow.log_param("use_mock", str(use_mock_llm))
            if args.provider == "ollama":
                mlflow.log_param("ollama_model", os.getenv("OLLAMA_MODEL", "unknown"))

            # Metrics: log summarized metrics for baseline and agentic
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

            # Artifacts: save JSON outputs
            mlflow.log_artifact(str(output_dir / "baseline.json"), artifact_path="eval")
            mlflow.log_artifact(str(output_dir / "agentic.json"), artifact_path="eval")
            mlflow.log_artifact(str(output_dir / "baseline_summary.json"), artifact_path="eval")
            mlflow.log_artifact(str(output_dir / "agentic_summary.json"), artifact_path="eval")
            mlflow.log_artifact(str(output_dir / "latest.json"), artifact_path="eval")
    print(f"Wrote eval results to {output_dir}")


if __name__ == "__main__":
    main()
