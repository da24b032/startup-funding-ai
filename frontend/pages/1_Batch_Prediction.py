"""Predict many startups from a CSV file."""

from __future__ import annotations

import io

import client
import pandas as pd
import streamlit as st
from client import BackendError
from ui import page_setup, require_backend, show_error

page_setup("Batch prediction", "📄")
st.write("Upload a spreadsheet (CSV) with one startup per row to score many startups at once.")

if not require_backend():
    st.stop()

try:
    fields = client.get_schema()
except BackendError as exc:
    show_error(exc)
    st.stop()

columns = [f["name"] for f in fields]
required = [f["name"] for f in fields if f["required"]]

template = pd.DataFrame([{c: "" for c in columns}])
st.download_button("⬇️ Download a blank CSV template", template.to_csv(index=False),
                   file_name="startup_template.csv", mime="text/csv")
with st.expander("What columns are needed?"):
    st.dataframe(pd.DataFrame([{"column": f["name"], "meaning": f["label"], "required": f["required"]}
                               for f in fields]), hide_index=True, width="stretch")

upload = st.file_uploader("Choose a CSV file", type="csv")
if upload is None:
    st.stop()

try:
    frame = pd.read_csv(upload)
except Exception:  # noqa: BLE001 - any parse problem is a user-facing "bad file"
    st.error("That file could not be read as a CSV. Please upload a plain .csv file.")
    st.stop()

missing = [c for c in required if c not in frame.columns]
if missing:
    st.error(f"The file is missing these columns: {', '.join(missing)}")
    st.stop()
if len(frame) > 500:
    st.error("Please upload at most 500 startups at a time.")
    st.stop()

st.write(f"Found **{len(frame)}** startups.")
if st.button("Run predictions", type="primary"):
    records = [
        {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
         for k, v in row.items() if k in columns}
        for row in frame.to_dict("records")
    ]
    try:
        with st.spinner("Scoring…"):
            out = client.predict_batch(records)
    except BackendError as exc:
        show_error(exc)
        st.stop()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", out["total"])
    c2.metric("Predicted acquired", out["predicted_acquired"])
    c3.metric("Predicted closed", out["predicted_closed"])
    result = frame.copy()
    result["outcome"] = [p["outcome"] for p in out["predictions"]]
    result["acquired_probability"] = [p["acquired_probability"] for p in out["predictions"]]
    st.dataframe(result, hide_index=True, width="stretch")
    buf = io.StringIO()
    result.to_csv(buf, index=False)
    st.download_button("⬇️ Download results", buf.getvalue(), file_name="predictions.csv", mime="text/csv")
