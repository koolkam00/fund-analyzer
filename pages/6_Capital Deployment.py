from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr
from filters import render_and_filter


st.set_page_config(page_title="Capital Deployment & Realizations", layout="wide")


@st.cache_data(show_spinner=False)
def _read_excel_or_csv(upload, header_row_index: int) -> Dict[str, pd.DataFrame]:
    if upload is None:
        return {}
    header_zero = max(0, int(header_row_index) - 1)
    name = upload.name.lower()
    if name.endswith(".csv"):
        content = upload.getvalue()
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = content.decode("latin-1", errors="ignore")
        df = pd.read_csv(io.StringIO(text), header=header_zero)
        return {"Sheet1": df}
    else:
        content = upload.getvalue()
        bio = io.BytesIO(content)
        xls = pd.ExcelFile(bio, engine="openpyxl")
        return {sheet: xls.parse(sheet, header=header_zero) for sheet in xls.sheet_names}


st.title("Capital Deployment & Realizations")
st.caption("Analyze invested, realized (proceeds), and unrealized (current value) capital across time and by segments.")

with st.sidebar:
    upload = st.file_uploader("Upload Portfolio Metrics file (.xlsx or .csv)", type=["xlsx", "csv"])  # type: ignore
    header_row_index = st.number_input("Header row (1-based)", min_value=1, max_value=100, value=int(st.session_state.get("header_row_index", 2)), step=1)

sheets = _read_excel_or_csv(upload, header_row_index)
if not sheets:
    sheets = st.session_state.get("sheets", {})
if not sheets:
    st.info("Upload a file to begin.")
    st.stop()

sheet_name = st.selectbox(
    "Select sheet",
    list(sheets.keys()),
    index=max(0, list(sheets.keys()).index(st.session_state.get("selected_sheet", list(sheets.keys())[0])) if sheets else 0),
)
df = sheets[sheet_name]
st.caption(f"Loaded sheet '{sheet_name}' with rows: {len(df):,}")
st.session_state["sheets"] = sheets
st.session_state["selected_sheet"] = sheet_name
st.session_state["header_row_index"] = header_row_index

ops_df_raw, _ = extract_operational_by_template_order(df, list(df.columns))
if ops_df_raw.empty:
    st.error("No metrics detected from the uploaded sheet. Confirm the header row and column order.")
    st.stop()

ops_df = add_growth_and_cagr(ops_df_raw)

ops_df["invest_date"] = pd.to_datetime(ops_df.get("invest_date"), errors="coerce")
ops_df["exit_date"] = pd.to_datetime(ops_df.get("exit_date"), errors="coerce")
ops_df["year_invest"] = ops_df["invest_date"].dt.year
ops_df["year_exit"] = ops_df["exit_date"].dt.year
# New: determine realized year only when truly realized AND the uploaded Exit Date exists
ops_df["proceeds_num"] = pd.to_numeric(ops_df.get("proceeds"), errors="coerce")
status_l = ops_df.get("status").astype(str).str.lower() if "status" in ops_df.columns else pd.Series("", index=ops_df.index)
has_realization = (ops_df["proceeds_num"] > 0) | status_l.isin(["realized", "fully realized", "partially realized"])  # type: ignore
# Use raw uploaded exit_date (pre add_growth fill) to avoid counting current month placeholders
exit_date_raw = pd.to_datetime(ops_df_raw.get("exit_date"), errors="coerce")
ops_df["year_exit_real"] = np.where(has_realization & exit_date_raw.notna(), exit_date_raw.dt.year, np.nan)

st.subheader("Filters")
f = render_and_filter(ops_df, key_prefix="cap")

st.subheader("Deployment by Investment Year (stacked by Sector) & WA Entry TEV/EBITDA")
if {"invested", "sector", "year_invest"}.issubset(f.columns):
    dep_stack = f.groupby(["year_invest", "sector"])['invested'].sum(min_count=1).reset_index().dropna(subset=["year_invest"]) 
else:
    dep_stack = pd.DataFrame(columns=["year_invest", "sector", "invested"]) 

wa_mult = pd.DataFrame()
if {"entry_tev_ebitda", "invested", "year_invest"}.issubset(f.columns):
    tmp = f[["year_invest", "entry_tev_ebitda", "invested"]].dropna(subset=["year_invest"]) 
    def _wa(g: pd.DataFrame) -> float:
        vals = pd.to_numeric(g["entry_tev_ebitda"], errors="coerce")
        w = pd.to_numeric(g["invested"], errors="coerce").clip(lower=0)
        if w.notna().sum() and float(w.sum()) > 0:
            return float(np.average(vals.fillna(0), weights=w.fillna(0)))
        return np.nan
    wa_mult = tmp.groupby("year_invest").apply(_wa).reset_index(name="wa_entry_tev_ebitda")

fig_dep = go.Figure()
if not dep_stack.empty:
    for sec in dep_stack['sector'].dropna().unique().tolist():
        seg = dep_stack[dep_stack['sector'] == sec]
        fig_dep.add_bar(name=str(sec), x=seg['year_invest'], y=seg['invested'])
fig_dep.update_layout(barmode='stack', title='Invested Capital by Year (Stacked by Sector)', xaxis_title='Investment Year', yaxis_title='Invested')
if not wa_mult.empty:
    fig_dep.add_scatter(x=wa_mult['year_invest'], y=wa_mult['wa_entry_tev_ebitda'], name='WA Entry TEV/EBITDA', mode='lines+markers', yaxis='y2')
    fig_dep.update_layout(
        yaxis2=dict(title='WA Entry TEV/EBITDA', overlaying='y', side='right', showgrid=False)
    )
st.plotly_chart(fig_dep, use_container_width=True)

st.subheader("Deployment by Investment Year (stacked by Sector) & WA Entry TEV/Revenue")
# Weighted average Entry TEV/Revenue by invest year
wa_rev = pd.DataFrame()
if {"entry_tev_revenue", "invested", "year_invest"}.issubset(f.columns):
    tmp_rev = f[["year_invest", "entry_tev_revenue", "invested"]].dropna(subset=["year_invest"]) 
    def _wa_rev(g: pd.DataFrame) -> float:
        vals = pd.to_numeric(g["entry_tev_revenue"], errors="coerce")
        w = pd.to_numeric(g["invested"], errors="coerce").clip(lower=0)
        if w.notna().sum() and float(w.sum()) > 0:
            return float(np.average(vals.fillna(0), weights=w.fillna(0)))
        return np.nan
    wa_rev = tmp_rev.groupby("year_invest").apply(_wa_rev).reset_index(name="wa_entry_tev_revenue")

fig_dep_rev = go.Figure()
if not dep_stack.empty:
    for sec in dep_stack['sector'].dropna().unique().tolist():
        seg = dep_stack[dep_stack['sector'] == sec]
        fig_dep_rev.add_bar(name=str(sec), x=seg['year_invest'], y=seg['invested'])
fig_dep_rev.update_layout(barmode='stack', title='Invested Capital by Year (Stacked by Sector)', xaxis_title='Investment Year', yaxis_title='Invested')
if not wa_rev.empty:
    fig_dep_rev.add_scatter(x=wa_rev['year_invest'], y=wa_rev['wa_entry_tev_revenue'], name='WA Entry TEV/Revenue', mode='lines+markers', yaxis='y2')
    fig_dep_rev.update_layout(
        yaxis2=dict(title='WA Entry TEV/Revenue', overlaying='y', side='right', showgrid=False)
    )
st.plotly_chart(fig_dep_rev, use_container_width=True)

st.subheader("Realizations by Exit Year (stacked by Sector)")
if {"proceeds", "sector", "year_exit_real"}.issubset(f.columns):
    real_stack = (
        f.dropna(subset=["year_exit_real"])  # only when realized
         .groupby(["year_exit_real", "sector"])['proceeds']
         .sum(min_count=1)
         .reset_index()
         .rename(columns={"year_exit_real": "year_exit"})
    )
else:
    real_stack = pd.DataFrame(columns=["year_exit", "sector", "proceeds"]) 
fig_real = go.Figure()
if not real_stack.empty:
    for sec in real_stack['sector'].dropna().unique().tolist():
        seg = real_stack[real_stack['sector'] == sec]
        fig_real.add_bar(name=str(sec), x=seg['year_exit'], y=seg['proceeds'])
fig_real.update_layout(barmode='stack', title='Realized Proceeds by Year (Stacked by Sector)', xaxis_title='Exit Year', yaxis_title='Proceeds')
st.plotly_chart(fig_real, use_container_width=True)

st.subheader("Cumulative Deployment vs Realizations")
by_invest_year = (
    f.groupby("year_invest")["invested"].sum(min_count=1).reset_index().dropna(subset=["year_invest"]) if "invested" in f.columns else pd.DataFrame(columns=["year_invest", "invested"]) 
)
by_exit_year = (
    f.dropna(subset=["year_exit_real"]).groupby("year_exit_real")["proceeds"].sum(min_count=1).reset_index().rename(columns={"year_exit_real": "year_exit"}) if ("proceeds" in f.columns and "year_exit_real" in f.columns) else pd.DataFrame(columns=["year_exit", "proceeds"]) 
)
years = sorted(set(by_invest_year.get("year_invest", pd.Series(dtype=int))).union(set(by_exit_year.get("year_exit", pd.Series(dtype=int)))))
cum_df = pd.DataFrame({"year": years})
cum_df = cum_df.sort_values("year")
inv_map = dict(zip(by_invest_year.get("year_invest", []), by_invest_year.get("invested", [])))
real_map = dict(zip(by_exit_year.get("year_exit", []), by_exit_year.get("proceeds", [])))
cum_df["invested"] = cum_df["year"].map(inv_map).fillna(0)
cum_df["proceeds"] = cum_df["year"].map(real_map).fillna(0)
cum_df["cum_invested"] = cum_df["invested"].cumsum()
cum_df["cum_proceeds"] = cum_df["proceeds"].cumsum()
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(x=cum_df["year"], y=cum_df["cum_invested"], name="Cumulative Invested", mode="lines+markers"))
fig_cum.add_trace(go.Scatter(x=cum_df["year"], y=cum_df["cum_proceeds"], name="Cumulative Proceeds", mode="lines+markers"))
fig_cum.update_layout(xaxis_title="Year", yaxis_title="Amount", title="Cumulative Invested vs Proceeds")
st.plotly_chart(fig_cum, use_container_width=True)

st.subheader("Deployment by Sector")
if "sector" in f.columns and "invested" in f.columns:
    by_sector = f.groupby("sector")["invested"].sum(min_count=1).sort_values(ascending=False).reset_index()
    fig_sec = px.bar(by_sector, x="sector", y="invested", title="Invested by Sector", labels={"invested": "Invested", "sector": "Sector"})
    fig_sec.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_sec, use_container_width=True)


