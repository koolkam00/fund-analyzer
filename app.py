from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="PE Fund Analyzer", layout="wide")

st.title("PE Fund Analyzer")
st.caption("Redirecting to Track Record…")

try:
    # Prefer switching by page title to avoid brittle file-path issues after renames
    st.switch_page("Track Record")
except Exception:
    try:
        st.switch_page("pages/3_Track Record.py")
    except Exception:
        st.page_link("pages/3_Track Record.py", label="Go to Track Record")
        st.stop()

