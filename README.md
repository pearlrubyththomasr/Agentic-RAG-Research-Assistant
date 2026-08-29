# Agentic RAG Research Assistant

This project builds a production-style research assistant that answers technical questions from a corpus of arXiv papers using an explicit LangGraph agent loop rather than a single prompt call. The defining feature is the self-critique and re-retrieval cycle: when the first retrieval pass looks weak, the system reformulates the query and tries again before returning a final answer.

## Architecture

```text
User Query
  -> Query Node
  -> Retrieve Node
  -> Generate Node
  -> Critique Node
      -> Final Answer (if score above threshold)
      -> Reformulate -> Retrieve (if below threshold, retry capped at 2)
```

## Results

| Metric | Baseline (mock) | Self-correcting (mock) |
| --- | ---: | ---: |
| Faithfulness | 0.1667 | 0.1667 |
| Answer relevance | 0.0104 | 0.0104 |
| Context precision | 0.2222 | 0.2222 |
| Context recall | 0.4167 | 0.4167 |
| Retrieval precision@5 | 0.2222 | 0.2222 |
| Retrieval precision@10 | 0.2222 | 0.2222 |
| Latency (ms) | 12147.17 | 5565.83 |
| Retry rate | 0% | 0% |

> These numbers were produced by the evaluation harness in `eval/run_eval.py` using the repository's mock LLM provider. The numeric summaries are taken from `eval/results/baseline_summary.json` and `eval/results/agentic_summary.json`.

## Design decisions

- Chunking strategy: fixed-size chunking is used first for simplicity and predictable retrieval; recursive chunking is available as a second option for better coherence.
- Retry cap: the agent is capped at two reformulation attempts to control cost and prevent endless loops.
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 is the default because it is local, free, and works well for a prototype.
- LangGraph: explicit graph orchestration makes the self-correction loop observable, testable, and easier to evolve than a single monolithic prompt.

## Running locally

```bash
docker compose up
```

### Evaluation

Run the evaluation harness locally with the mock LLM:

```bash
python -m eval.run_eval --provider mock
```

To use a real OpenAI provider, set `OPENAI_API_KEY` and run:

```bash
OPENAI_API_KEY=your_key python -m eval.run_eval --provider openai
```

To use Ollama instead, set `LLM_PROVIDER=ollama` and optionally configure `OLLAMA_HOST` and `OLLAMA_MODEL`:

```bash
OLLAMA_HOST=http://127.0.0.1:11434 OLLAMA_MODEL=llama2 LLM_PROVIDER=ollama python -m eval.run_eval --provider ollama
```

#### Quick Ollama connectivity check

Before running a full evaluation, verify Ollama is reachable:

```bash
OLLAMA_HOST=http://127.0.0.1:11434 OLLAMA_MODEL=llama2 python scripts/check_ollama.py
```

If the script prints a reply from the model, proceed to run the evaluation.

### Background evaluation via API

You can run the evaluation from the frontend or call the API directly. The API supports background scheduling to avoid blocking the request.

POST `/evaluate` JSON body example:

```json
{
  "provider": "mock",
  "mlflow": false,
  "output_dir": "eval/results",
  "background": true
}
```

If `background` is `true`, the API schedules the evaluation and returns immediately with `status: scheduled`. Results are written to `eval/results` when complete.

### MLflow

If you want MLflow logging:

1. Install MLflow: `pip install mlflow`
2. Start MLflow server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

3. Set `MLFLOW_TRACKING_URI` environment variable to `http://127.0.0.1:5000` and enable `mlflow` in the evaluate API or Streamlit UI.

The API exposes `/mlflow` to show the configured tracking URI and recent experiments.

### Backend provider mode

The API backend also uses `LLM_PROVIDER` to choose the LLM mode:

```bash
LLM_PROVIDER=mock uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```bash
OLLAMA_HOST=http://127.0.0.1:11434 OLLAMA_MODEL=llama2 LLM_PROVIDER=ollama uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Live demo: TBD

## Ingest full-text PDFs (optional)

To build a richer index from full paper text instead of abstracts, run the PDF fetch and extraction step. This will download PDFs under `data/raw/pdfs` and produce an augmented JSON at `data/raw/arxiv_raw_with_fulltext.json` that `ingestion/index_builder.py` will prefer when building the FAISS index.

```bash
python ingestion/fetch_fulltext.py --input data/raw/arxiv_raw.json --output data/raw/arxiv_raw_with_fulltext.json
python ingestion/index_builder.py
```

Note: PDF download and extraction can be slow and may fail for some arXiv IDs; the script is tolerant and will continue. Ensure `pymupdf` (a.k.a. `pymupdf`/`fitz`) is installed (`pip install pymupdf`).

## Limitations

- Full-text PDF ingestion is not yet implemented; the current version works from arXiv metadata and abstracts.
- Semantic chunking and reranking are planned next.
- A managed vector database and stronger LLM providers can be swapped in later.
