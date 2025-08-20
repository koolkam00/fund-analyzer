from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr
from data_loader import ensure_workbook_loaded
from filters import render_and_filter


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
# Show firm on top if available
firm = st.session_state.get("firm_name")
if firm:
    st.markdown(f"**Firm:** {firm}")
st.caption("Grouped bar charts by Portfolio Company for Revenue, EBITDA, EBITDA Margin, TEV/EBITDA, TEV/Revenue, Leverage, Net Debt, and TEV. Filters match other pages.")

sheets, ops_sheet_name, _, _ = ensure_workbook_loaded()
if not sheets:
    st.info("Upload a workbook to begin.")
    st.stop()
sheet_name = ops_sheet_name or list(sheets.keys())[0]
df = sheets[sheet_name]
st.caption(f"Loaded workbook — operational sheet '{sheet_name}' with rows: {len(df):,}")

ops_df_raw, _ = extract_operational_by_template_order(df, list(df.columns))
if ops_df_raw.empty:
    st.error("No metrics detected from the uploaded sheet. Confirm the header row and column order.")
    st.stop()

ops_df = add_growth_and_cagr(ops_df_raw)

st.subheader("Filters")
f = render_and_filter(ops_df, key_prefix="dc")

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

st.subheader("Entry vs Exit Net Debt")
grouped_bar(f, "entry_net_debt", "exit_net_debt", ",.1f", "Net Debt: Entry vs Exit", x_order)

st.subheader("Entry vs Exit TEV")
grouped_bar(f, "entry_tev", "exit_tev", ",.1f", "TEV: Entry vs Exit", x_order)


