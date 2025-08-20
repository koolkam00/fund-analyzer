from __future__ import annotations

import pandas as pd
import streamlit as st

from data_loader import ensure_workbook_loaded


st.set_page_config(page_title="Fund Manager Profile", layout="wide")

st.title("Fund Manager Profile")
st.caption("Displays high-level information about the GP/manager from the uploaded workbook.")

sheets, _, _, _ = ensure_workbook_loaded()
if not sheets:
    st.info("Upload the master workbook to view the manager profile.")
    st.stop()

# Identify the profile sheet by common names or position
sheet_names = list(sheets.keys())
mgr_sheet_name = None
for nm in sheet_names:
    low = str(nm).strip().lower()
    if low in {"fund manager profile", "manager profile", "fund manager"}:
        mgr_sheet_name = nm
        break
if mgr_sheet_name is None and len(sheet_names) > 4:
    mgr_sheet_name = sheet_names[4]

df_mgr = sheets.get(mgr_sheet_name, pd.DataFrame())
if df_mgr.empty:
    st.info("Manager profile sheet not found or is empty.")
    st.stop()

def _norm(s: str) -> str:
    return str(s).strip()

df_show = df_mgr.copy()
df_show.columns = [_norm(c) for c in df_show.columns]

# Display the first row as primary profile; show entire table below
primary = df_show.iloc[0:1]
st.subheader("Profile Summary")
st.dataframe(primary, use_container_width=True)

st.subheader("Full Profile Table")
st.dataframe(df_show, use_container_width=True)


