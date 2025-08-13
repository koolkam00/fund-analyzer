from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr


st.set_page_config(page_title="Scatter Plots", layout="wide")


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


st.title("Scatter Plots by Investment Date")
st.caption("Upload the Portfolio Metrics file and visualize metrics vs Investment Date. Color = Fund Name (GP), Size = Invested.")

with st.sidebar:
    upload = st.file_uploader("Upload Portfolio Metrics file (.xlsx or .csv)", type=["xlsx", "csv"])  # type: ignore
    header_row_index = st.number_input("Header row (1-based)", min_value=1, max_value=100, value=int(st.session_state.get("header_row_index", 2)), step=1)

sheets = _read_excel_or_csv(upload, header_row_index)
if not sheets:
    sheets = st.session_state.get("sheets", {})
if not sheets:
    st.info("Upload a file to begin.")
    st.stop()

sheet_name = st.selectbox("Select sheet", list(sheets.keys()), index=max(0, list(sheets.keys()).index(st.session_state.get("selected_sheet", list(sheets.keys())[0])) if sheets else 0))
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

flt_col1, flt_col2, flt_col3 = st.columns(3)
sectors = sorted(ops_df["sector"].dropna().unique().tolist()) if "sector" in ops_df.columns else []
funds = sorted(ops_df["fund_name"].dropna().unique().tolist()) if "fund_name" in ops_df.columns else []
geos = sorted(ops_df["geography"].dropna().unique().tolist()) if "geography" in ops_df.columns else []
sel_sector = flt_col1.multiselect("Sector", sectors, default=sectors)
sel_fund = flt_col2.multiselect("Fund Name (GP)", funds, default=funds)
sel_geo = flt_col3.multiselect("Geography", geos, default=geos)

filtered = ops_df.copy()
if sel_sector and "sector" in filtered.columns:
    filtered = filtered[filtered["sector"].isin(sel_sector)]
if sel_fund and "fund_name" in filtered.columns:
    filtered = filtered[filtered["fund_name"].isin(sel_fund)]
if sel_geo and "geography" in filtered.columns:
    filtered = filtered[filtered["geography"].isin(sel_geo)]

metric_choices: List[str] = [
    "entry_revenue",
    "exit_revenue",
    "entry_ebitda",
    "exit_ebitda",
    "entry_tev",
    "exit_tev",
    "entry_net_debt",
    "exit_net_debt",
    "entry_tev_ebitda",
    "exit_tev_ebitda",
    "entry_leverage",
    "exit_leverage",
    "entry_tev_revenue",
    "exit_tev_revenue",
    "gross_moic",
    "gross_irr",
]
available_metrics = [m for m in metric_choices if m in filtered.columns]
if {"entry_ebitda", "entry_revenue"}.issubset(filtered.columns):
    available_metrics.append("entry_margin_pct")
if {"exit_ebitda", "exit_revenue"}.issubset(filtered.columns):
    available_metrics.append("exit_margin_pct")
selected_metrics = st.multiselect("Select metrics (Y-axis)", available_metrics, default=available_metrics[:4])

if not selected_metrics:
    st.info("Select at least one metric to plot.")
    st.stop()

filtered["invest_date"] = pd.to_datetime(filtered.get("invest_date"), errors="coerce")
# Ensure numeric types for plotting
if "invested" in filtered.columns:
    filtered["invested"] = pd.to_numeric(filtered["invested"], errors="coerce").clip(lower=0)
for col in ["gross_moic", "gross_irr"]:
    if col in filtered.columns:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

opt_col1, opt_col2, opt_col3 = st.columns(3)
exclude_outliers = opt_col1.checkbox("Exclude outliers (percentile)", value=True)
low_pct = opt_col2.number_input("Lower pct", min_value=0.0, max_value=50.0, value=1.0, step=0.5, help="Keeps data above this percentile")
high_pct = opt_col3.number_input("Upper pct", min_value=50.0, max_value=100.0, value=99.0, step=0.5, help="Keeps data below this percentile")

x_range = None
y_range = None

plots_per_row = 2
rows = (len(selected_metrics) + plots_per_row - 1) // plots_per_row
for r in range(rows):
    cols = st.columns(plots_per_row)
    for c_idx in range(plots_per_row):
        i = r * plots_per_row + c_idx
        if i >= len(selected_metrics):
            break
        metric = selected_metrics[i]
        with cols[c_idx]:
            plot_df = filtered.copy()
            if metric == "entry_margin_pct" and {"entry_ebitda", "entry_revenue"}.issubset(plot_df.columns):
                with np.errstate(divide="ignore", invalid="ignore"):
                    plot_df["y_val"] = pd.to_numeric(plot_df["entry_ebitda"], errors="coerce") / pd.to_numeric(plot_df["entry_revenue"], errors="coerce")
                is_percent = True
            elif metric == "exit_margin_pct" and {"exit_ebitda", "exit_revenue"}.issubset(plot_df.columns):
                with np.errstate(divide="ignore", invalid="ignore"):
                    plot_df["y_val"] = pd.to_numeric(plot_df["exit_ebitda"], errors="coerce") / pd.to_numeric(plot_df["exit_revenue"], errors="coerce")
                is_percent = True
            else:
                plot_df["y_val"] = pd.to_numeric(plot_df.get(metric), errors="coerce")
                is_percent = False
            if exclude_outliers and plot_df["y_val"].notna().sum() > 5:
                q_low = plot_df["y_val"].quantile(low_pct / 100.0)
                q_high = plot_df["y_val"].quantile(high_pct / 100.0)
                plot_df = plot_df[(plot_df["y_val"] >= q_low) & (plot_df["y_val"] <= q_high)]
            # Drop invalid points
            plot_df = plot_df.dropna(subset=["invest_date", "y_val"]) if not plot_df.empty else plot_df
            size_col = "invested" if ("invested" in plot_df.columns and plot_df["invested"].notna().any()) else None
            fig = px.scatter(
                plot_df,
                x="invest_date",
                y="y_val",
                color="fund_name" if "fund_name" in plot_df.columns else None,
                size=size_col,
                title=f"{metric} vs Investment Date",
            )
            fig.update_layout(legend_title_text="Fund")
            fig.update_yaxes(title=metric, tickformat=(".1%" if is_percent else None))
            fig.update_xaxes(title="Investment Date")
            comp = plot_df.get("portfolio_company")
            if comp is None and len(df.columns) > 0:
                comp = plot_df.iloc[:, 0].astype(str)
            fund = plot_df.get("fund_name")
            moic = pd.to_numeric(plot_df.get("gross_moic"), errors="coerce")
            irr = pd.to_numeric(plot_df.get("gross_irr"), errors="coerce")
            status = plot_df.get("status")
            sector = plot_df.get("sector")
            custom = np.column_stack([
                fund.fillna("") if fund is not None else np.repeat("", len(plot_df)),
                moic.fillna(np.nan) if moic is not None else np.repeat(np.nan, len(plot_df)),
                irr.fillna(np.nan) if irr is not None else np.repeat(np.nan, len(plot_df)),
                status.fillna("") if status is not None else np.repeat("", len(plot_df)),
                sector.fillna("") if sector is not None else np.repeat("", len(plot_df)),
            ]) if len(plot_df) > 0 else np.empty((0,5))
            fig.update_traces(
                hovertext=comp.astype(str) if comp is not None else None,
                customdata=custom,
                hovertemplate="<b>%{hovertext}</b><br>Fund: %{customdata[0]}<br>MOIC: %{customdata[1]:.1f}<br>IRR: %{customdata[2]:.1%}<br>Status: %{customdata[3]}<br>Sector: %{customdata[4]}<extra></extra>",
            )
            st.plotly_chart(fig, use_container_width=True)


