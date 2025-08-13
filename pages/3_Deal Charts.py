from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr


st.set_page_config(page_title="Deal Charts: Entry vs Exit", layout="wide")


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


st.title("Deal Charts: Entry vs Exit")
st.caption("Grouped bar charts by Portfolio Company for Revenue, EBITDA, and EBITDA Margin. Filters match other pages.")

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

portfolio_header = df.columns[0] if len(df.columns) > 0 else "Portfolio Company"
if portfolio_header not in f.columns and "portfolio_company" in f.columns:
    f.insert(0, portfolio_header, f["portfolio_company"])  # ensure display col exists

# Build unique x-axis label combining company and fund to separate cross-fund investments
x_label_col = f"{portfolio_header} (Fund)"
if portfolio_header in f.columns and "fund_name" in f.columns:
    f[x_label_col] = f[portfolio_header].astype(str) + " — " + f["fund_name"].astype(str)
else:
    f[x_label_col] = f.get(portfolio_header, pd.Series(dtype=str)).astype(str)

with np.errstate(divide="ignore", invalid="ignore"):
    if {"entry_ebitda", "entry_revenue"}.issubset(f.columns):
        f["entry_margin_pct"] = pd.to_numeric(f["entry_ebitda"], errors="coerce") / pd.to_numeric(f["entry_revenue"], errors="coerce")
    if {"exit_ebitda", "exit_revenue"}.issubset(f.columns):
        f["exit_margin_pct"] = pd.to_numeric(f["exit_ebitda"], errors="coerce") / pd.to_numeric(f["exit_revenue"], errors="coerce")

f["sort_change_ebitda"] = pd.to_numeric(f.get("exit_ebitda"), errors="coerce") - pd.to_numeric(f.get("entry_ebitda"), errors="coerce")
f = f.sort_values("sort_change_ebitda", ascending=False, na_position="last")
x_order = f[x_label_col].astype(str).tolist() if x_label_col in f.columns else []

def grouped_bar(df_in: pd.DataFrame, y_entry: str, y_exit: str, y_format: str, title: str, x_order: List[str]):
    x_vals = df_in[x_label_col].astype(str).tolist() if x_label_col in df_in.columns else list(range(len(df_in)))
    y1 = pd.to_numeric(df_in.get(y_entry), errors="coerce")
    y2 = pd.to_numeric(df_in.get(y_exit), errors="coerce")
    df_plot = df_in.copy()
    df_plot["y1"] = y1
    df_plot["y2"] = y2
    if exclude_outliers:
        both = pd.concat([y1, y2]).dropna()
        if not both.empty and both.size > 5:
            q_low = both.quantile(low_pct / 100.0)
            q_high = both.quantile(high_pct / 100.0)
            mask = (df_plot["y1"].between(q_low, q_high)) & (df_plot["y2"].between(q_low, q_high))
            df_plot = df_plot[mask]
    x_vals = df_plot[x_label_col].astype(str).tolist() if x_label_col in df_plot.columns else list(range(len(df_plot)))
    y1 = df_plot["y1"]
    y2 = df_plot["y2"]
    fig = go.Figure()
    fund = df_plot.get("fund_name")
    moic = pd.to_numeric(df_plot.get("gross_moic"), errors="coerce")
    irr = pd.to_numeric(df_plot.get("gross_irr"), errors="coerce")
    status = df_plot.get("status")
    sector = df_plot.get("sector")
    custom = np.column_stack([
        fund.fillna("") if fund is not None else np.repeat("", len(df_plot)),
        moic.fillna(np.nan) if moic is not None else np.repeat(np.nan, len(df_plot)),
        irr.fillna(np.nan) if irr is not None else np.repeat(np.nan, len(df_plot)),
        status.fillna("") if status is not None else np.repeat("", len(df_plot)),
        sector.fillna("") if sector is not None else np.repeat("", len(df_plot)),
    ]) if len(df_plot) > 0 else np.empty((0,5))
    fig.add_bar(name="Entry", x=x_vals, y=y1, marker_color="#1f77b4")
    fig.add_bar(name="Exit", x=x_vals, y=y2, marker_color="#ff7f0e")
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title=x_label_col,
        yaxis_title=y_entry.replace("entry_", "").replace("_", " ").title(),
        xaxis_tickangle=-45,
        legend_title_text="Point in Time",
        margin=dict(t=60, r=20, b=80, l=60),
        height=500,
    )
    if x_order:
        fig.update_xaxes(categoryorder="array", categoryarray=x_order)
    fig.update_yaxes(tickformat=y_format)
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Fund: %{customdata[0]}<br>MOIC: %{customdata[1]:.1f}<br>IRR: %{customdata[2]:.1%}<br>Status: %{customdata[3]}<br>Sector: %{customdata[4]}<extra></extra>",
        customdata=custom,
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Entry vs Exit Revenue")
grouped_bar(f, "entry_revenue", "exit_revenue", ",.1f", "Revenue: Entry vs Exit", x_order)

st.subheader("Entry vs Exit EBITDA")
grouped_bar(f, "entry_ebitda", "exit_ebitda", ",.1f", "EBITDA: Entry vs Exit", x_order)

st.subheader("Entry vs Exit EBITDA Margin")
grouped_bar(f, "entry_margin_pct", "exit_margin_pct", ".1%", "EBITDA Margin: Entry vs Exit", x_order)

st.subheader("Entry vs Exit TEV/EBITDA")
grouped_bar(f, "entry_tev_ebitda", "exit_tev_ebitda", ",.1f", "TEV/EBITDA: Entry vs Exit", x_order)

st.subheader("Entry vs Exit TEV/Revenue")
grouped_bar(f, "entry_tev_revenue", "exit_tev_revenue", ",.1f", "TEV/Revenue: Entry vs Exit", x_order)

st.subheader("Entry vs Exit Leverage (Net Debt / EBITDA)")
grouped_bar(f, "entry_leverage", "exit_leverage", ",.1f", "Leverage: Entry vs Exit", x_order)


