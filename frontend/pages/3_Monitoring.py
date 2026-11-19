"""Live service monitoring and data-drift view."""

from __future__ import annotations

import client
import pandas as pd
import plotly.express as px
import streamlit as st
from client import BackendError
from ui import page_setup, require_backend, show_error

page_setup("Monitoring", "📈")
st.write("Checks whether the startups people are entering still look like the ones the model learned from.")

if not require_backend():
    st.stop()

try:
    result = client.drift()
except BackendError as exc:
    show_error(exc)
    st.stop()

col_a, col_b = st.columns([3, 1])
col_a.info(result["message"])
if col_b.button("Reset window"):
    client.reset_drift()
    st.rerun()

c1, c2 = st.columns(2)
c1.metric("Predictions in window", result["window_size"])
c2.metric("Inputs showing drift", len(result["drifted_features"]))

if result["ready"]:
    df = pd.DataFrame([{"input": k, "drift score": v["statistic"], "drifted": v["drifted"],
                        "test": "KS test" if v["type"] == "numeric" else "category shift"}
                       for k, v in result["features"].items()]).sort_values("drift score", ascending=False)
    st.plotly_chart(px.bar(df, x="drift score", y="input", orientation="h", color="drifted",
                           color_discrete_map={True: "#c62828", False: "#1f4e79"},
                           title="Drift score per input"), width="stretch")
    if result["drifted_features"]:
        st.warning("Drift means incoming data looks different from training data. "
                   "It is a signal to review the model, not proof that predictions are wrong.")
else:
    st.caption("Make a few predictions (the Predict or Batch pages) and come back.")

st.markdown(f"For live charts of request rates, latency and predictions see "
            f"[Grafana]({client.GRAFANA_URL}).")
