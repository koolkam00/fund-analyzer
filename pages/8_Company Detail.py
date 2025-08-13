from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr


st.set_page_config(page_title="Company Detail", layout="wide")


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


st.title("Company Detail")
st.caption("Single-deal view for a selected Portfolio Company and Fund.")

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
# Recompute value creation components to ensure waterfall inputs exist
def _compute_value_creation_local(df: pd.DataFrame) -> pd.DataFrame:
    e0 = pd.to_numeric(df.get("entry_ebitda"), errors="coerce")
    e1 = pd.to_numeric(df.get("exit_ebitda"), errors="coerce")
    r0 = pd.to_numeric(df.get("entry_revenue"), errors="coerce")
    r1 = pd.to_numeric(df.get("exit_revenue"), errors="coerce")
    tev0 = pd.to_numeric(df.get("entry_tev"), errors="coerce")
    tev1 = pd.to_numeric(df.get("exit_tev"), errors="coerce")
    nd0 = pd.to_numeric(df.get("entry_net_debt"), errors="coerce")
    nd1 = pd.to_numeric(df.get("exit_net_debt"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        mult0 = np.where(e0 > 0, tev0 / e0, np.nan)
        mult1 = np.where(e1 > 0, tev1 / e1, np.nan)
        marg0 = np.where(r0 > 0, e0 / r0, np.nan)
        marg1 = np.where(r1 > 0, e1 / r1, np.nan)
        mult0 = pd.Series(mult0).replace([np.inf, -np.inf], np.nan)
        mult1 = pd.Series(mult1).replace([np.inf, -np.inf], np.nan)
        marg0 = pd.Series(marg0).replace([np.inf, -np.inf], np.nan)
        marg1 = pd.Series(marg1).replace([np.inf, -np.inf], np.nan)
        rev_growth = (r1 - r0) * marg0 * mult0
        margin_exp = r1 * (marg1 - marg0) * mult0
        e1_safe = e1.where(e1 > 0)
        multiple_change = (mult1 - mult0) * e1_safe
        deleveraging = -(nd1 - nd0)
        eq0 = tev0 - nd0
        eq1 = tev1 - nd1
        bridge_sum = rev_growth + margin_exp + multiple_change + deleveraging
    out = df.copy()
    out["equity_entry"] = eq0
    out["equity_exit"] = eq1
    out["vc_rev_growth"] = rev_growth
    out["vc_margin_expansion"] = margin_exp
    out["vc_multiple_change"] = multiple_change
    out["vc_deleveraging"] = deleveraging
    out["vc_bridge_sum"] = bridge_sum
    return out

ops_df = _compute_value_creation_local(ops_df)

portfolio_header = df.columns[0] if len(df.columns) > 0 else "Portfolio Company"
if portfolio_header not in ops_df.columns and "portfolio_company" in ops_df.columns:
    ops_df.insert(0, portfolio_header, ops_df["portfolio_company"])  # ensure display col exists

# Determine target selection from session or query params
qp = st.query_params
sel_company = st.session_state.get("detail_company") or qp.get("company")
sel_fund = st.session_state.get("detail_fund") or qp.get("fund")

# Fallback selector if not provided
if not sel_company or not sel_fund:
    st.info("Choose a company and fund to view details.")
    companies = sorted(ops_df[portfolio_header].dropna().astype(str).unique().tolist()) if portfolio_header in ops_df.columns else []
    funds = sorted(ops_df["fund_name"].dropna().astype(str).unique().tolist()) if "fund_name" in ops_df.columns else []
    c1, c2 = st.columns(2)
    sel_company = c1.selectbox("Portfolio Company", companies)
    sel_fund = c2.selectbox("Fund Name (GP)", funds)

# Filter to the specific deal (company + fund)
g = ops_df.copy()
if portfolio_header in g.columns:
    g = g[g[portfolio_header].astype(str) == str(sel_company)]
if "fund_name" in g.columns:
    g = g[g["fund_name"].astype(str) == str(sel_fund)]

if g.empty:
    st.warning("No matching deal found for the selected Company and Fund.")
    st.stop()

row = g.iloc[0]

# Summary KPIs
st.subheader(f"{sel_company} — {sel_fund}")
k1, k2, k3, k4, k5 = st.columns(5)
inv = float(pd.to_numeric(row.get("invested"), errors="coerce")) if pd.notna(row.get("invested")) else np.nan
proc = float(pd.to_numeric(row.get("proceeds"), errors="coerce")) if pd.notna(row.get("proceeds")) else np.nan
nav = float(pd.to_numeric(row.get("current_value"), errors="coerce")) if pd.notna(row.get("current_value")) else np.nan
moic = float(pd.to_numeric(row.get("gross_moic"), errors="coerce")) if pd.notna(row.get("gross_moic")) else np.nan
irr = float(pd.to_numeric(row.get("gross_irr"), errors="coerce")) if pd.notna(row.get("gross_irr")) else np.nan
k1.metric("Invested", f"${inv:,.1f}" if pd.notna(inv) else "—")
k2.metric("Proceeds", f"${proc:,.1f}" if pd.notna(proc) else "—")
k3.metric("NAV", f"${nav:,.1f}" if pd.notna(nav) else "—")
k4.metric("Total MOIC", f"{moic:.1f}x" if pd.notna(moic) else "—")
k5.metric("Gross IRR", f"{irr:.1%}" if pd.notna(irr) else "—")

# Entry vs Exit table
st.subheader("Entry vs Exit Metrics")
cols = [
    "entry_revenue", "exit_revenue",
    "entry_ebitda", "exit_ebitda",
    "entry_tev", "exit_tev",
    "entry_net_debt", "exit_net_debt",
    "entry_tev_ebitda", "exit_tev_ebitda",
    "entry_tev_revenue", "exit_tev_revenue",
    "entry_leverage", "exit_leverage",
    "revenue_growth_pct", "revenue_cagr",
    "ebitda_growth_pct", "ebitda_cagr",
    "tev_growth_pct", "tev_cagr",
]
cols = [c for c in cols if c in g.columns]
tbl = g[cols].copy()
fmt: Dict[str, object] = {}
for c in tbl.columns:
    if c.endswith("_growth_pct") or c.endswith("_cagr") or c.endswith("_pct"):
        fmt[c] = "{:.1%}"
for c in ["entry_tev_ebitda", "exit_tev_ebitda", "entry_tev_revenue", "exit_tev_revenue", "entry_leverage", "exit_leverage", "entry_revenue", "exit_revenue", "entry_ebitda", "exit_ebitda", "entry_tev", "exit_tev", "entry_net_debt", "exit_net_debt"]:
    if c in tbl.columns and c not in fmt:
        fmt[c] = "{:.1f}"
st.dataframe(tbl.style.format(fmt), use_container_width=True)

# Value creation waterfall for the deal, if available
st.subheader("Value Creation Waterfall")
start = float(row.get("equity_entry", np.nan))
rev = float(row.get("vc_rev_growth", 0.0))
marg = float(row.get("vc_margin_expansion", 0.0))
mult = float(row.get("vc_multiple_change", 0.0))
debt = float(row.get("vc_deleveraging", 0.0))
end = float(row.get("equity_exit", np.nan))
if pd.notna(start) and pd.notna(end):
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Equity at Entry", "Revenue Growth", "Margin Expansion", "Multiple Change", "Deleveraging", "Equity at Exit"],
            textposition="outside",
            text=[f"{v:,.1f}" for v in [start, rev, marg, mult, debt, end]],
            y=[start, rev, marg, mult, debt, end],
        )
    )
    fig.update_layout(showlegend=False, waterfallgap=0.3)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Value creation inputs not available for this deal.")


