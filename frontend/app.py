"""Home page: predict the outcome for one startup.

Run:  streamlit run frontend/app.py
"""

from __future__ import annotations

import client
import streamlit as st
from client import BackendError
from ui import page_setup, require_backend, show_error

page_setup("Predict a startup's outcome", "🚀")
st.write(
    "Describe a startup below. The tool estimates how likely it is to be "
    "**acquired** rather than **closed**, and shows which details pushed the estimate up or down."
)

if not require_backend():
    st.stop()

try:
    fields = client.get_schema()
except BackendError as exc:
    show_error(exc)
    st.stop()

EXAMPLES = {
    "Well-funded software startup": dict(
        state_code="CA", category_code="software", age_first_funding_year=1.5,
        age_last_funding_year=3.5, funding_rounds=3, funding_total_usd=12_000_000,
        avg_participants=3.0, has_VC=1, has_angel=0, has_roundA=1, has_roundB=1,
        has_roundC=0, has_roundD=0, is_top500=1, relationships=8, milestones=2,
        age_first_milestone_year=2.0, age_last_milestone_year=4.0),
    "Early, lightly funded startup": dict(
        state_code="TX", category_code="web", age_first_funding_year=0.5,
        age_last_funding_year=0.5, funding_rounds=1, funding_total_usd=250_000,
        avg_participants=1.0, has_VC=0, has_angel=1, has_roundA=0, has_roundB=0,
        has_roundC=0, has_roundD=0, is_top500=0, relationships=2, milestones=0,
        age_first_milestone_year=None, age_last_milestone_year=None),
}
choice = st.selectbox("Start from an example (optional)", ["— blank form —", *EXAMPLES])
preset = EXAMPLES.get(choice, {})

groups: dict[str, list[dict]] = {}
for spec in fields:
    groups.setdefault(spec["group"], []).append(spec)


def widget(spec: dict, key_prefix: str):
    """Render one input for a field spec coming from the API."""
    name, label, help_ = spec["name"], spec["label"], spec["help"]
    default = preset.get(name)
    key = f"{key_prefix}_{name}_{choice}"
    if spec["type"] == "category":
        options = spec["options"]
        index = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=index, help=help_, key=key)
    if spec["type"] == "flag":
        return int(st.toggle(label, value=bool(default), help=help_, key=key))
    optional = not spec["required"]
    if optional:
        use = st.checkbox(f"{label} — provide", value=default is not None, key=key + "_on", help=help_)
        if not use:
            return None
    start = float(default if default is not None else spec["min"] if spec["min"] > 0 else 0)
    if spec["type"] == "integer":
        return int(st.number_input(label, int(spec["min"]), int(spec["max"]), int(start), step=1,
                                   help=help_ if not optional else None, key=key + "_v"))
    return float(st.number_input(label, float(spec["min"]), float(spec["max"]), start, step=0.5,
                                 help=help_ if not optional else None, key=key + "_v"))


with st.form("profile_form"):
    values: dict = {}
    tabs = st.tabs(list(groups))
    for tab, (_group, specs) in zip(tabs, groups.items(), strict=False):
        with tab:
            cols = st.columns(2)
            for i, spec in enumerate(specs):
                with cols[i % 2]:
                    values[spec["name"]] = widget(spec, "single")
    submitted = st.form_submit_button("Predict outcome", type="primary", width="stretch")

if submitted:
    if values.get("milestones", 0) == 0:  # keep the form foolproof: no milestone ages without milestones
        values["age_first_milestone_year"] = values["age_last_milestone_year"] = None
    with st.spinner("Analysing…"):
        try:
            result = client.predict(values, explain=True)
        except BackendError as exc:
            show_error(exc)
            st.stop()

    acquired = result["prediction"] == 1
    prob = result["acquired_probability"]
    st.markdown(
        f'<div class="result-card {"result-acquired" if acquired else "result-closed"}">'
        f'<h2>{"✅ Likely to be acquired" if acquired else "⚠️ Likely to close"}</h2>'
        f'<div>Estimated chance of acquisition: <b>{prob:.0%}</b> · chance of closing: '
        f'<b>{result["closed_probability"]:.0%}</b></div></div>',
        unsafe_allow_html=True,
    )
    st.progress(prob, text=f"Acquisition likelihood {prob:.0%}")

    factors = result.get("top_factors", [])
    if factors:
        st.subheader("What influenced this estimate")
        for f in factors:
            arrow = "🟢 raises" if f["direction"] == "raises" else "🔴 lowers"
            st.write(f"- **{f['label']}** (you entered `{f['value']}`) {arrow} the acquisition chance "
                     f"by about {abs(f['impact']) * 100:.1f} percentage points")
        st.caption("These are model-based comparisons against a typical startup in the training data. "
                   "They describe the model, not guaranteed causes.")
    st.caption(f"Model: {result['model_name']} · response time {result['latency_ms']:.0f} ms · "
               "This is a statistical estimate, not financial advice.")
