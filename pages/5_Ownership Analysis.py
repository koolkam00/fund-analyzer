from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr


st.set_page_config(page_title="Ownership % Analysis", layout="wide")


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


st.title("Ownership % Analysis")
st.caption("Analyze how ownership varies with MOIC and IRR.")

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

with np.errstate(divide="ignore", invalid="ignore"):
    if {"kam_equity_entry", "equity_entry_total"}.issubset(ops_df.columns):
        ops_df["kam_ownership_entry_pct"] = pd.to_numeric(ops_df["kam_equity_entry"], errors="coerce") / pd.to_numeric(ops_df["equity_entry_total"], errors="coerce")

st.subheader("Filters")
c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)
c7, c8, _ = st.columns(3)

def _vals(col: str) -> List[str]:
    return sorted(ops_df[col].dropna().unique().tolist()) if col in ops_df.columns else []

sectors = _vals("sector")
geos = _vals("geography")
statuses = _vals("status")
funds = _vals("fund_name")
strategies = _vals("investment_strategy")
instruments = _vals("instrument_type")
purchases = _vals("purchase_process")
exit_types = _vals("exit_type")

sel_sector = c1.multiselect("Sector", sectors, default=sectors)
sel_geo = c2.multiselect("Geography", geos, default=geos)
sel_status = c3.multiselect("Status", statuses, default=statuses)
sel_fund = c4.multiselect("Fund Name (GP)", funds, default=funds)
sel_strategy = c5.multiselect("Investment strategy", strategies, default=strategies)
sel_instrument = c6.multiselect("Instrument type", instruments, default=instruments)
sel_purchase = c7.multiselect("Purchase Process", purchases, default=purchases)
sel_exit_type = c8.multiselect("Exit type", exit_types, default=exit_types)

f = ops_df.copy()
if sel_sector and "sector" in f.columns:
    f = f[f["sector"].isin(sel_sector)]
if sel_geo and "geography" in f.columns:
    f = f[f["geography"].isin(sel_geo)]
if sel_status and "status" in f.columns:
    f = f[f["status"].isin(sel_status)]
if sel_fund and "fund_name" in f.columns:
    f = f[f["fund_name"].isin(sel_fund)]
if sel_strategy and "investment_strategy" in f.columns:
    f = f[f["investment_strategy"].isin(sel_strategy)]
if sel_instrument and "instrument_type" in f.columns:
    f = f[f["instrument_type"].isin(sel_instrument)]
if sel_purchase and "purchase_process" in f.columns:
    f = f[f["purchase_process"].isin(sel_purchase)]
if sel_exit_type and "exit_type" in f.columns:
    f = f[f["exit_type"].isin(sel_exit_type)]

opt1, opt2, opt3 = st.columns(3)
exclude_outliers = opt1.checkbox("Exclude outliers (percentile)", value=True)
low_pct = opt2.number_input("Lower pct", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
high_pct = opt3.number_input("Upper pct", min_value=50.0, max_value=100.0, value=99.0, step=0.5)

st.subheader("Ownership vs MOIC")
if "kam_ownership_entry_pct" in f.columns and "gross_moic" in f.columns:
    plot1 = f.copy()
    plot1["y_val"] = pd.to_numeric(plot1["gross_moic"], errors="coerce")
    if exclude_outliers and plot1["y_val"].notna().sum() > 5:
        ql = plot1["y_val"].quantile(low_pct / 100.0)
        qh = plot1["y_val"].quantile(high_pct / 100.0)
        plot1 = plot1[(plot1["y_val"] >= ql) & (plot1["y_val"] <= qh)]
    fig1 = px.scatter(
        plot1,
        x="kam_ownership_entry_pct",
        y="y_val",
        size="invested" if "invested" in plot1.columns else None,
        color="fund_name" if "fund_name" in plot1.columns else None,
        hover_name="portfolio_company" if "portfolio_company" in plot1.columns else None,
        labels={"kam_ownership_entry_pct": "Entry Ownership %", "y_val": "MOIC"},
        title="Entry Ownership % vs MOIC",
    )
    fig1.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig1, use_container_width=True)

if "kam_ownership_exit_pct" in f.columns and "gross_moic" in f.columns:
    plot2 = f.copy()
    plot2["y_val"] = pd.to_numeric(plot2["gross_moic"], errors="coerce")
    if exclude_outliers and plot2["y_val"].notna().sum() > 5:
        ql = plot2["y_val"].quantile(low_pct / 100.0)
        qh = plot2["y_val"].quantile(high_pct / 100.0)
        plot2 = plot2[(plot2["y_val"] >= ql) & (plot2["y_val"] <= qh)]
    fig2 = px.scatter(
        plot2,
        x="kam_ownership_exit_pct",
        y="y_val",
        size="invested" if "invested" in plot2.columns else None,
        color="fund_name" if "fund_name" in plot2.columns else None,
        hover_name="portfolio_company" if "portfolio_company" in plot2.columns else None,
        labels={"kam_ownership_exit_pct": "Exit Ownership %", "y_val": "MOIC"},
        title="Exit Ownership % vs MOIC",
    )
    fig2.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Ownership vs IRR (Gross)")
if "kam_ownership_entry_pct" in f.columns and "gross_irr" in f.columns:
    plot3 = f.copy()
    plot3["y_val"] = pd.to_numeric(plot3["gross_irr"], errors="coerce")
    if exclude_outliers and plot3["y_val"].notna().sum() > 5:
        ql = plot3["y_val"].quantile(low_pct / 100.0)
        qh = plot3["y_val"].quantile(high_pct / 100.0)
        plot3 = plot3[(plot3["y_val"] >= ql) & (plot3["y_val"] <= qh)]
    fig3 = px.scatter(
        plot3,
        x="kam_ownership_entry_pct",
        y="y_val",
        size="invested" if "invested" in plot3.columns else None,
        color="fund_name" if "fund_name" in plot3.columns else None,
        hover_name="portfolio_company" if "portfolio_company" in plot3.columns else None,
        labels={"kam_ownership_entry_pct": "Entry Ownership %", "y_val": "IRR (Gross)"},
        title="Entry Ownership % vs IRR (Gross)",
    )
    fig3.update_xaxes(tickformat=".1%")
    fig3.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig3, use_container_width=True)

if "kam_ownership_exit_pct" in f.columns and "gross_irr" in f.columns:
    plot4 = f.copy()
    plot4["y_val"] = pd.to_numeric(plot4["gross_irr"], errors="coerce")
    if exclude_outliers and plot4["y_val"].notna().sum() > 5:
        ql = plot4["y_val"].quantile(low_pct / 100.0)
        qh = plot4["y_val"].quantile(high_pct / 100.0)
        plot4 = plot4[(plot4["y_val"] >= ql) & (plot4["y_val"] <= qh)]
    fig4 = px.scatter(
        plot4,
        x="kam_ownership_exit_pct",
        y="y_val",
        size="invested" if "invested" in plot4.columns else None,
        color="fund_name" if "fund_name" in plot4.columns else None,
        hover_name="portfolio_company" if "portfolio_company" in plot4.columns else None,
        labels={"kam_ownership_exit_pct": "Exit Ownership %", "y_val": "IRR (Gross)"},
        title="Exit Ownership % vs IRR (Gross)",
    )
    fig4.update_xaxes(tickformat=".1%")
    fig4.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig4, use_container_width=True)


