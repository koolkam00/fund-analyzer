from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="PE Fund Analyzer", layout="wide")

st.title("PE Fund Analyzer")
st.caption("Redirecting to Track Record…")

try:
    st.switch_page("pages/1_Track Record.py")
except Exception:
    st.page_link("pages/1_Track Record.py", label="Go to Track Record")
    st.stop()

