import streamlit as st
import requests

st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("Agentic RAG Research Assistant")
query = st.text_input("Ask a research question")
show_monitoring = st.sidebar.checkbox("Show monitoring", value=False)

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("Submit") and query:
        try:
            response = requests.post("http://localhost:8000/query", json={"query": query}, timeout=30)
        except requests.exceptions.RequestException as exc:
            st.error(f"Failed to contact backend: {exc}")
        else:
            if response.ok:
                payload = response.json()
                st.write(payload["answer"])
                st.caption(f"Confidence: {payload['confidence']} | Retries: {payload['retries']} | Latency: {payload['latency_ms']} ms")
                st.subheader("Citations")
                for citation in payload.get("citations", []):
                    metadata = citation.get("metadata", {})
                    title = metadata.get("title") or metadata.get("paper_id") or "Unknown source"
                    source = metadata.get("source") or metadata.get("link")
                    st.markdown(f"- **{title}** {'[' + source + ']' + '(' + source + ')' if source else ''}")
                    snippet = citation.get("text", "")
                    if snippet:
                        st.caption(snippet[:240] + ("..." if len(snippet) > 240 else ""))
            else:
                st.error(f"API returned {response.status_code}: {response.text}")

with col2:
    if show_monitoring:
        st.subheader("System Metrics")
        try:
            resp = requests.get("http://localhost:8000/metrics", timeout=3)
        except requests.exceptions.RequestException as exc:
            st.error(f"Failed to fetch metrics: {exc}")
        else:
            if resp.ok:
                metrics = resp.json()
                st.write(metrics)
                # fetch recent LLM latency history and plot
                try:
                    hist = requests.get("http://localhost:8000/metrics/history?limit=200", timeout=3)
                    if hist.ok:
                        data = hist.json().get("metrics", [])
                        if data:
                            import pandas as pd

                            df = pd.DataFrame(data)
                            if "latency_ms" in df.columns:
                                df_ts = df.set_index(pd.to_datetime(df["ts"], unit="s"))
                                st.line_chart(df_ts["latency_ms"])
                except Exception:
                    pass
            else:
                st.error(f"Metrics endpoint returned {resp.status_code}")

st.sidebar.markdown("---")
st.sidebar.subheader("Evaluation")
eval_provider = st.sidebar.selectbox("Provider", ["mock", "ollama"], index=0)
run_eval = st.sidebar.button("Run evaluation")
run_in_background = st.sidebar.checkbox("Run in background", value=True)
use_mlflow = st.sidebar.checkbox("Log to MLflow if available", value=False)
if run_eval:
    payload = {"provider": eval_provider, "mlflow": use_mlflow, "background": run_in_background}
    try:
        resp = requests.post("http://localhost:8000/evaluate", json=payload, timeout=600)
    except requests.exceptions.RequestException as exc:
        st.sidebar.error(f"Evaluation failed: {exc}")
    else:
        if resp.ok:
            data = resp.json()
            if data.get("status") == "scheduled":
                st.sidebar.info(f"Evaluation scheduled. Results will be written to {data.get('output_dir')}")
            else:
                st.sidebar.success("Evaluation completed")
                st.sidebar.write(data.get("baseline_summary"))
                st.sidebar.write(data.get("agentic_summary"))
        else:
            st.sidebar.error(f"Evaluate API returned {resp.status_code}: {resp.text}")

st.sidebar.markdown("---")
st.sidebar.subheader("MLflow")
try:
    mlflow_info = requests.get("http://localhost:8000/mlflow", timeout=3).json()
    if mlflow_info.get("available"):
        st.sidebar.markdown(f"MLflow tracking: {mlflow_info.get('tracking_uri')}")
        for e in mlflow_info.get("experiments", []):
            st.sidebar.write(f"- {e['name']} (id={e['id']})")
    else:
        st.sidebar.write(mlflow_info.get("reason"))
except Exception:
    st.sidebar.write("MLflow info unavailable")
