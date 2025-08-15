from __future__ import annotations

import io
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr
from filters import render_and_filter


st.set_page_config(page_title="Compare Views", layout="wide")


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


st.title("Compare Views")
st.caption("Side-by-side analysis with independent filters on each side.")

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


def _track_record_table(frame: pd.DataFrame, portfolio_header: str) -> pd.io.formats.style.Styler:
    cols = [
        portfolio_header,
        "sector",
        "status",
        "invest_date",
        "exit_date",
        "holding_years",
        "ownership_pct",
        "invested",
        "proceeds",
        "current_value",
        "gross_moic",
        "gross_irr",
        "fund_name",
    ]
    out = frame.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if "ownership_pct" not in out.columns:
            if "kam_ownership_exit_pct" in out.columns:
                out["ownership_pct"] = pd.to_numeric(out["kam_ownership_exit_pct"], errors="coerce")
            elif {"kam_equity_entry", "equity_entry_total"}.issubset(out.columns):
                out["ownership_pct"] = pd.to_numeric(out["kam_equity_entry"], errors="coerce") / pd.to_numeric(out["equity_entry_total"], errors="coerce")
            else:
                out["ownership_pct"] = np.nan
    view = out[[c for c in cols if c in out.columns or c == portfolio_header]].copy()
    for dc in ["invest_date", "exit_date"]:
        if dc in view.columns:
            view[dc] = pd.to_datetime(view[dc], errors="coerce").dt.strftime("%b %Y")
    fmt = {}
    if "ownership_pct" in view.columns:
        fmt["ownership_pct"] = "{:.1%}"
    if "gross_irr" in view.columns:
        fmt["gross_irr"] = "{:.1%}"
    for c in view.select_dtypes(include=["number"]).columns:
        if c not in fmt:
            fmt[c] = "{:.1f}"
    return view.style.format(fmt, na_rep="—")


def _portfolio_vc_waterfall(frame: pd.DataFrame) -> go.Figure | None:
    def _sum(series: pd.Series) -> float:
        return float(pd.to_numeric(series, errors="coerce").sum(skipna=True))
    eq0 = _sum(frame.get("equity_entry", pd.Series(dtype=float)))
    rev = _sum(frame.get("vc_rev_growth", pd.Series(dtype=float)))
    marg = _sum(frame.get("vc_margin_expansion", pd.Series(dtype=float)))
    mult = _sum(frame.get("vc_multiple_change", pd.Series(dtype=float)))
    debt = _sum(frame.get("vc_deleveraging", pd.Series(dtype=float)))
    eq1 = _sum(frame.get("equity_exit", pd.Series(dtype=float)))
    if not (np.isfinite(eq0) and np.isfinite(eq1)):
        return None
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=[
                "Equity at Entry",
                "Revenue Growth",
                "Margin Expansion",
                "Multiple Change",
                "Deleveraging",
                "Equity at Exit",
            ],
            textposition="outside",
            text=[f"{v:,.1f}" for v in [eq0, rev, marg, mult, debt, eq1]],
            y=[eq0, rev, marg, mult, debt, eq1],
        )
    )
    fig.update_layout(showlegend=False, waterfallgap=0.3)
    return fig


portfolio_header = df.columns[0] if len(df.columns) > 0 else "Portfolio Company"
if portfolio_header not in ops_df.columns and "portfolio_company" in ops_df.columns:
    ops_df.insert(0, portfolio_header, ops_df["portfolio_company"])  # ensure display col exists

left, right = st.columns(2)

with left:
    st.subheader("Left View")
    f_left = render_and_filter(ops_df, key_prefix="cmp_left")
    st.markdown("**Track Record**")
    st.dataframe(_track_record_table(f_left, portfolio_header), use_container_width=True)
    st.markdown("**Value Creation (Portfolio)**")
    fig_l = _portfolio_vc_waterfall(f_left)
    if fig_l is not None:
        st.plotly_chart(fig_l, use_container_width=True)

with right:
    st.subheader("Right View")
    f_right = render_and_filter(ops_df, key_prefix="cmp_right")
    st.markdown("**Track Record**")
    st.dataframe(_track_record_table(f_right, portfolio_header), use_container_width=True)
    st.markdown("**Value Creation (Portfolio)**")
    fig_r = _portfolio_vc_waterfall(f_right)
    if fig_r is not None:
        st.plotly_chart(fig_r, use_container_width=True)


