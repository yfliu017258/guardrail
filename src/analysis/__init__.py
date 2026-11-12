"""Read-only analysis of evaluation results (``results/<run>/``): loader + interactive Streamlit app.

    streamlit run src/analysis/app.py -- --results results/

Depends only on the results contract (run.json, predictions.jsonl, metrics.json) — never on the
evaluation runtime or the guardrail server.
"""
