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


# Ensure value creation components exist
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


def _status_bucket_row(row: pd.Series) -> str:
    invested = float(pd.to_numeric(row.get("invested"), errors="coerce") or 0.0)
    proceeds = float(pd.to_numeric(row.get("proceeds"), errors="coerce") or 0.0)
    nav = float(pd.to_numeric(row.get("current_value"), errors="coerce") or 0.0)
    if invested <= 0:
        return "Unrealized"
    if nav <= 0 and proceeds > 0:
        return "Fully Realized"
    if nav > 0 and proceeds > 0:
        return "Partially Realized"
    if nav > 0 and proceeds <= 0:
        return "Unrealized"
    return str(row.get("status", "")).strip() or "Unrealized"


def _subtotals_table(frame: pd.DataFrame, portfolio_header: str) -> pd.io.formats.style.Styler:
    dfp = frame.copy()
    # Ensure numeric
    for c in ["invested", "proceeds", "current_value", "gross_moic", "gross_irr", "holding_years"]:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    # Bucket
    dfp["_bucket"] = dfp.apply(_status_bucket_row, axis=1)
    # Compute totals per bucket
    rows = []
    def _row_for(label: str, sub: pd.DataFrame) -> dict:
        inv = float(sub.get("invested", pd.Series(dtype=float)).sum(skipna=True))
        real = float(sub.get("proceeds", pd.Series(dtype=float)).sum(skipna=True))
        nav = float(sub.get("current_value", pd.Series(dtype=float)).sum(skipna=True))
        total_val = real + nav
        moic = (total_val / inv) if inv > 0 else np.nan
        # Weighted avg IRR by invested
        w = pd.to_numeric(sub.get("invested", pd.Series(dtype=float)), errors="coerce").clip(lower=0).fillna(0)
        irr_vals = pd.to_numeric(sub.get("gross_irr", pd.Series(dtype=float)), errors="coerce").fillna(0)
        irr_wa = float(np.average(irr_vals, weights=w)) if float(w.sum()) > 0 else np.nan
        # Weighted avg holding years by invested
        yrs = pd.to_numeric(sub.get("holding_years", pd.Series(dtype=float)), errors="coerce").fillna(0)
        yrs_wa = float(np.average(yrs, weights=w)) if float(w.sum()) > 0 else np.nan
        return {
            portfolio_header: label,
            "fund_name": "—",
            "status": "—",
            "holding_years": yrs_wa,
            "invested": inv,
            "proceeds": real,
            "current_value": nav,
            "total_value": total_val,
            "gross_moic": moic,
            "gross_irr": irr_wa,
        }
    # Buckets and total
    buckets = ["Fully Realized", "Partially Realized", "Unrealized"]
    for b in buckets:
        rows.append(_row_for(b, dfp[dfp["_bucket"] == b]))
    rows.append(_row_for("Total", dfp))
    out = pd.DataFrame(rows)
    # Format
    fmt = {
        "holding_years": "{:.1f}",
        "invested": "{:,.1f}",
        "proceeds": "{:,.1f}",
        "current_value": "{:,.1f}",
        "total_value": "{:,.1f}",
        "gross_moic": "{:.1f}",
        "gross_irr": "{:.1%}",
    }
    # Select display columns per request
    cols = [
        portfolio_header,
        "fund_name",
        "status",
        "holding_years",
        "invested",
        "proceeds",
        "current_value",
        "total_value",
        "gross_moic",
        "gross_irr",
    ]
    out = out[[c for c in cols if c in out.columns]]
    return out.style.format(fmt, na_rep="—")

with left:
    st.subheader("Left View")
    f_left = render_and_filter(ops_df, key_prefix="cmp_left")
    st.markdown("**Track Record**")
    st.dataframe(_track_record_table(f_left, portfolio_header), use_container_width=True, key="cmp_tr_left")
    st.markdown("**Subtotals**")
    st.dataframe(_subtotals_table(f_left, portfolio_header), use_container_width=True, key="cmp_sub_left")
    st.markdown("**Value Creation (Portfolio)**")
    fig_l = _portfolio_vc_waterfall(f_left)
    if fig_l is not None:
        st.plotly_chart(fig_l, use_container_width=True, key="cmp_vc_left")

with right:
    st.subheader("Right View")
    f_right = render_and_filter(ops_df, key_prefix="cmp_right")
    st.markdown("**Track Record**")
    st.dataframe(_track_record_table(f_right, portfolio_header), use_container_width=True, key="cmp_tr_right")
    st.markdown("**Subtotals**")
    st.dataframe(_subtotals_table(f_right, portfolio_header), use_container_width=True, key="cmp_sub_right")
    st.markdown("**Value Creation (Portfolio)**")
    fig_r = _portfolio_vc_waterfall(f_right)
    if fig_r is not None:
        st.plotly_chart(fig_r, use_container_width=True, key="cmp_vc_right")


