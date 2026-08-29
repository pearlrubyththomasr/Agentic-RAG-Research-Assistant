from pathlib import Path

from eval.run_eval import load_test_questions, make_metrics, run_eval, summarize


def test_load_test_questions_reads_json():
    questions = load_test_questions(Path("eval/test_questions.json"))
    assert isinstance(questions, list)
    assert questions[0]["question"].startswith("What is the main contribution")


def test_make_metrics_with_empty_result():
    item = {
        "question": "Test question",
        "reference": "A reference sentence.",
        "gold_chunks": ["2607.28618v1"],
        "mode": "test",
    }
    result = {"retrieved_chunks": [], "final_answer": "", "retry_count": 0}
    metrics = make_metrics(item, result)
    assert metrics["question"] == item["question"]
    assert metrics["faithfulness"] == 0.0
    assert metrics["answer_relevance"] == 0.0
    assert metrics["context_precision"] == 0.0
    assert metrics["context_recall"] == 0.0
    assert metrics["retrieval_precision_at_5"] == 0.0
    assert metrics["retrieval_precision_at_10"] == 0.0


def test_summarize_aggregates_metrics():
    results = [
        {"faithfulness": 0.5, "answer_relevance": 0.5, "context_precision": 1.0, "context_recall": 0.5,
         "retrieval_precision_at_5": 1.0, "retrieval_precision_at_10": 0.5, "latency_ms": 100, "retry_count": 0},
        {"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 0.0, "context_recall": 0.5,
         "retrieval_precision_at_5": 0.5, "retrieval_precision_at_10": 1.0, "latency_ms": 200, "retry_count": 1},
    ]
    summary = summarize(results)
    assert summary["faithfulness"] == 0.75
    assert summary["answer_relevance"] == 0.75
    assert summary["context_precision"] == 0.5
    assert summary["context_recall"] == 0.5
    assert summary["retrieval_precision_at_5"] == 0.75
    assert summary["retrieval_precision_at_10"] == 0.75
    assert summary["latency_ms"] == 150
    assert summary["retry_rate"] == 0.5


def test_run_eval_returns_results():
    results = run_eval("baseline", use_agentic=False)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all("question" in item for item in results)
    assert all("mode" in item for item in results)
