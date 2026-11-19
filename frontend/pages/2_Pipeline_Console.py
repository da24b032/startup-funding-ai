"""Pipeline console: the ML pipeline end to end, in one screen.

Shows the DVC pipeline graph, data-engineering stage timings and throughput,
model comparison, and a console of successful and failed pipeline runs.
Everything comes from the backend's ``/pipeline/*`` endpoints.
"""

from __future__ import annotations

import client
import pandas as pd
import plotly.express as px
import streamlit as st
from client import BackendError
from ui import page_setup, require_backend, show_error

page_setup("Pipeline console", "🛠️")

if not require_backend():
    st.stop()

try:
    summary = client.pipeline_summary()
    info = client.model_info()
except BackendError as exc:
    show_error(exc)
    st.stop()

if st.button("🔄 Refresh"):
    st.rerun()

# ------------------------------------------------------------------ run health
counts = summary["run_counts"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Successful runs", counts["success"])
c2.metric("Failed runs", counts["failed"], delta=None if not counts["failed"] else "needs attention",
          delta_color="inverse")
report = summary.get("data_report") or {}
c3.metric("Data throughput", f"{report.get('pipeline_throughput_rows_per_second', 0):,.0f} rows/s")
c4.metric("Served model", info["model_name"])

tab_flow, tab_data, tab_models, tab_runs, tab_trace = st.tabs(
    ["Pipeline flow", "Data engineering", "Model training", "Run console", "Traceability"]
)

# ------------------------------------------------------------------- flow (DAG)
with tab_flow:
    stages = summary.get("dvc_stages", [])
    st.write("Each box is a pipeline stage; arrows show which stage produces files another stage needs. "
             "Only stages whose inputs changed are re-run.")
    if stages:
        produced = {out: s["name"] for s in stages for out in s["outs"]}
        lines = ["digraph G { rankdir=LR; node [shape=box, style=\"rounded,filled\", "
                 "fillcolor=\"#dbe9f6\", fontname=Helvetica];",
                 '"raw data" [fillcolor="#e8f5e9"]; "served API" [fillcolor="#fff3e0"];']
        for s in stages:
            for dep in s["deps"]:
                src = produced.get(dep) or ("raw data" if dep.startswith("data/raw") else None)
                if src and src != s["name"]:
                    lines.append(f'"{src}" -> "{s["name"]}";')
        lines.append('"train" -> "served API";')
        lines.append("}")
        st.graphviz_chart("\n".join(lines))
        st.dataframe(pd.DataFrame([{"stage": s["name"], "command": s["cmd"],
                                    "produces": ", ".join(s["outs"])} for s in stages]),
                     hide_index=True, width="stretch")
    else:
        st.info("No pipeline definition found.")

# ----------------------------------------------------------- data engineering
with tab_data:
    if report:
        a, b, c, d = st.columns(4)
        a.metric("Rows ingested", report["rows_ingested"])
        b.metric("After cleaning", report["rows_after_cleaning"], delta=f"-{report['duplicates_removed']} duplicate")
        c.metric("Training rows", report["train_rows"])
        d.metric("Test rows", report["test_rows"])
        stage_df = pd.DataFrame(
            [{"stage": k, "seconds": v["seconds"], "rows_per_second": v["rows_per_second"]}
             for k, v in report["stages"].items()]
        )
        st.plotly_chart(px.bar(stage_df, x="stage", y="seconds", title="Time spent per data stage (seconds)"),
                        width="stretch")
        st.dataframe(stage_df, hide_index=True, width="stretch")
        st.caption(f"Whole data pipeline: {report['pipeline_seconds']} s "
                   f"({report['pipeline_throughput_rows_per_second']:,.0f} rows/s). "
                   f"Class balance — acquired share: train {report['train_acquired_rate']:.1%}, "
                   f"test {report['test_acquired_rate']:.1%}.")
        with st.expander("Data quality issues found (and how they are handled)"):
            issues = report["tolerated_issues"]
            st.write(f"- Funding dated before founding: **{issues['negative_funding_ages']}** rows "
                     "(kept, flagged as a feature)")
            st.write(f"- Missing milestone dates: **{issues['missing_milestone_age']}** rows "
                     "(treated as 'no milestone')")
            st.write(f"- Duplicate startup ids: **{issues['duplicate_ids']}** (removed)")
    else:
        st.info("No data report yet. Run the data stage first.")

# ------------------------------------------------------------ model comparison
with tab_models:
    comparison = pd.DataFrame(summary.get("model_comparison", []))
    if comparison.empty:
        st.info("No training results yet.")
    else:
        st.write(f"The **{info['model_name']}** model was chosen because it had the best average "
                 "cross-validated ROC-AUC on the training data. The test set was never used to choose.")
        long = comparison.melt(id_vars="model", value_vars=["cv_roc_auc_mean", "test_roc_auc", "test_f1",
                                                           "test_accuracy"], var_name="metric")
        st.plotly_chart(px.bar(long, x="model", y="value", color="metric", barmode="group",
                               title="Model comparison"), width="stretch")
        st.dataframe(comparison, hide_index=True, width="stretch")
        imp = summary.get("feature_importance") or {}
        if imp:
            top = pd.DataFrame([{"input": k, "importance": v["mean"]} for k, v in imp.items()]).head(10)
            st.plotly_chart(px.bar(top.iloc[::-1], x="importance", y="input", orientation="h",
                                   title="Most influential inputs (permutation importance)"),
                            width="stretch")

# ------------------------------------------------------------------ run console
with tab_runs:
    try:
        runs = client.pipeline_runs(100)
    except BackendError as exc:
        show_error(exc)
        runs = []
    if not runs:
        st.info("No pipeline runs recorded yet.")
    else:
        df = pd.DataFrame(runs)
        df["status"] = df["status"].map({"success": "✅ success", "failed": "❌ failed"}).fillna(df["status"])
        cols = [c for c in ["timestamp", "stage", "status", "duration_seconds",
                            "throughput_rows_per_second", "selected_model", "error"] if c in df]
        only_failed = st.toggle("Show only failures")
        shown = df[df["status"].str.contains("failed")] if only_failed else df
        st.dataframe(shown[cols], hide_index=True, width="stretch")
        failed = df[df["status"].str.contains("failed")]
        if not failed.empty:
            st.error(f"{len(failed)} failed run(s). Latest: {failed.iloc[0].get('error')}")

# ----------------------------------------------------------------- traceability
with tab_trace:
    st.write("Every served model can be traced to the exact code version, data version and experiment run.")
    st.json({k: info.get(k) for k in ["mlflow_run_id", "git_commit", "train_data_sha256",
                                      "trained_at", "tracking_backend"]})
    st.markdown(f"Open the tools: [MLflow experiments]({client.MLFLOW_URL}) · "
                f"[Grafana dashboards]({client.GRAFANA_URL}) · [Prometheus]({client.PROMETHEUS_URL})")
