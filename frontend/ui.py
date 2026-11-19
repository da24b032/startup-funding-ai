"""Shared look-and-feel helpers for all pages."""

from __future__ import annotations

import streamlit as st
from client import BackendError, health

ACCENT = "#1f4e79"
GOOD = "#2e7d32"
BAD = "#c62828"

_CSS = f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1100px; }}
.result-card {{ border-radius: 14px; padding: 1.2rem 1.5rem; margin: .5rem 0 1rem 0;
               border: 1px solid rgba(128,128,128,.25); }}
.result-acquired {{ border-left: 8px solid {GOOD}; }}
.result-closed {{ border-left: 8px solid {BAD}; }}
.result-card h2 {{ margin: 0 0 .2rem 0; }}
.small-note {{ font-size: .85rem; opacity: .75; }}
@media (max-width: 700px) {{ .block-container {{ padding: 1rem .8rem; }} }}
</style>
"""


def page_setup(title: str, icon: str) -> None:
    """Common page config + styling, and a friendly banner if the backend is down."""
    st.set_page_config(page_title=f"{title} · Startup Outcome Predictor", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title(f"{icon} {title}")


def require_backend() -> bool:
    """Show a clear message and return False when the backend cannot serve requests."""
    try:
        status = health()
    except BackendError as exc:
        st.error(str(exc))
        return False
    if status.get("status") != "ok":
        st.warning("The service is running but no model is loaded. Run the training step first.")
        return False
    return True


def show_error(exc: BackendError) -> None:
    st.error(str(exc))
