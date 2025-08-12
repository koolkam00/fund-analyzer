from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr


st.set_page_config(page_title="Value Creation Analysis", layout="wide")


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


def compute_value_creation(df: pd.DataFrame) -> pd.DataFrame:
    e0 = pd.to_numeric(df.get("entry_ebitda"), errors="coerce")
    e1 = pd.to_numeric(df.get("exit_ebitda"), errors="coerce")
    r0 = pd.to_numeric(df.get("entry_revenue"), errors="coerce")
    r1 = pd.to_numeric(df.get("exit_revenue"), errors="coerce")
    tev0 = pd.to_numeric(df.get("entry_tev"), errors="coerce")
    tev1 = pd.to_numeric(df.get("exit_tev"), errors="coerce")
    nd0 = pd.to_numeric(df.get("entry_net_debt"), errors="coerce")
    nd1 = pd.to_numeric(df.get("exit_net_debt"), errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        mult0 = tev0 / e0
        mult1 = tev1 / e1
        marg0 = e0 / r0
        marg1 = e1 / r1

        rev_growth = (r1 - r0) * marg0 * mult0
        margin_exp = r1 * (marg1 - marg0) * mult0
        multiple_change = (mult1 - mult0) * e1
        deleveraging = -(nd1 - nd0)

        eq0 = tev0 - nd0
        eq1 = tev1 - nd1
        eq_change = eq1 - eq0
        bridge_sum = rev_growth + margin_exp + multiple_change + deleveraging

    out = df.copy()
    out["equity_entry"] = eq0
    out["equity_exit"] = eq1
    out["equity_change"] = eq_change
    out["vc_rev_growth"] = rev_growth
    out["vc_margin_expansion"] = margin_exp
    out["vc_multiple_change"] = multiple_change
    out["vc_deleveraging"] = deleveraging
    out["vc_bridge_sum"] = bridge_sum
    return out


st.title("Value Creation Analysis")
st.caption("Decompose change in equity value into Revenue Growth, Margin Expansion, Multiple Change, and Deleveraging.")

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
ops_df = compute_value_creation(ops_df)

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

st.subheader("Deal value creation table")
portfolio_header = df.columns[0] if len(df.columns) > 0 else "Portfolio Company"
display_cols = [
    portfolio_header,
    "fund_name",
    "sector",
    "geography",
    "invest_date",
    "exit_date",
    "equity_entry",
    "equity_exit",
    "equity_change",
    "vc_rev_growth",
    "vc_margin_expansion",
    "vc_multiple_change",
    "vc_deleveraging",
    "vc_bridge_sum",
]
view = f.copy()
if portfolio_header not in view.columns and "portfolio_company" in view.columns:
    view.insert(0, portfolio_header, view["portfolio_company"])  # ensure first col
view = view[[c for c in display_cols if c in view.columns]]
fmt = {col: "{:.1f}" for col in view.select_dtypes(include=["number"]).columns}
st.dataframe(view.style.format(fmt), use_container_width=True)

# Aggregated waterfall across all filtered deals
st.subheader("Portfolio Waterfall (filtered)")
def _sum_num(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").sum(skipna=True))

eq0_total = _sum_num(f.get("equity_entry", pd.Series(dtype=float)))
rev_total = _sum_num(f.get("vc_rev_growth", pd.Series(dtype=float)))
marg_total = _sum_num(f.get("vc_margin_expansion", pd.Series(dtype=float)))
mult_total = _sum_num(f.get("vc_multiple_change", pd.Series(dtype=float)))
debt_total = _sum_num(f.get("vc_deleveraging", pd.Series(dtype=float)))
eq1_total = _sum_num(f.get("equity_exit", pd.Series(dtype=float)))

fig_port = go.Figure(
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
        text=[f"{v:,.1f}" for v in [eq0_total, rev_total, marg_total, mult_total, debt_total, eq1_total]],
        y=[eq0_total, rev_total, marg_total, mult_total, debt_total, eq1_total],
    )
)
fig_port.update_layout(showlegend=False, waterfallgap=0.3)

fund = f.get("fund_name")
moic = pd.to_numeric(f.get("gross_moic"), errors="coerce")
irr = pd.to_numeric(f.get("gross_irr"), errors="coerce")
status = f.get("status")
sector = f.get("sector")
custom_port = np.column_stack([
    fund.fillna("") if fund is not None else np.repeat("", len(f)),
    moic.fillna(np.nan) if moic is not None else np.repeat(np.nan, len(f)),
    irr.fillna(np.nan) if irr is not None else np.repeat(np.nan, len(f)),
    status.fillna("") if status is not None else np.repeat("", len(f)),
    sector.fillna("") if sector is not None else np.repeat("", len(f)),
]) if len(f) > 0 else np.empty((0,5))
fig_port.update_traces(
    hovertemplate="<b>%{x}</b><br>Fund: %{customdata[0]}<br>MOIC: %{customdata[1]:.1f}<br>IRR: %{customdata[2]:.1%}<br>Status: %{customdata[3]}<br>Sector: %{customdata[4]}<extra></extra>",
    customdata=custom_port,
)
st.plotly_chart(fig_port, use_container_width=True)

labels = ["Revenue Growth", "Margin Expansion", "Multiple Change", "Deleveraging"]
port_values_abs = [abs(rev_total), abs(marg_total), abs(mult_total), abs(debt_total)]
if sum(v for v in port_values_abs if pd.notna(v)) > 0:
    pie_port = px.pie(
        names=labels,
        values=port_values_abs,
        title="Portfolio Attribution Mix (%)",
        hole=0.4,
    )
    pie_port.update_traces(textposition="inside", texttemplate="%{percent:.1%}")
    st.plotly_chart(pie_port, use_container_width=True)

st.subheader("Waterfall (per deal)")
deal_names = view[portfolio_header].dropna().astype(str).unique().tolist() if portfolio_header in view.columns else []
sel_deal = st.selectbox("Select deal", deal_names)
if sel_deal:
    row = view.loc[view[portfolio_header] == sel_deal].iloc[0]
    start = float(row.get("equity_entry", np.nan))
    rev = float(row.get("vc_rev_growth", 0.0))
    marg = float(row.get("vc_margin_expansion", 0.0))
    mult = float(row.get("vc_multiple_change", 0.0))
    debt = float(row.get("vc_deleveraging", 0.0))
    end = float(row.get("equity_exit", np.nan))

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
    custom_deal = np.array([[row.get("fund_name", ""), row.get("gross_moic", np.nan), row.get("gross_irr", np.nan), row.get("status", ""), row.get("sector", "")]])
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Fund: %{customdata[0]}<br>MOIC: %{customdata[1]:.1f}<br>IRR: %{customdata[2]:.1%}<br>Status: %{customdata[3]}<br>Sector: %{customdata[4]}<extra></extra>",
        customdata=custom_deal,
    )
    st.plotly_chart(fig, use_container_width=True)

    deal_values_abs = [abs(rev), abs(marg), abs(mult), abs(debt)]
    if sum(v for v in deal_values_abs if pd.notna(v)) > 0:
        pie_deal = px.pie(
            names=labels,
            values=deal_values_abs,
            title=f"Attribution Mix (%) – {sel_deal}",
            hole=0.4,
        )
        pie_deal.update_traces(textposition="inside", texttemplate="%{percent:.1%}")
        st.plotly_chart(pie_deal, use_container_width=True)


# Fund drop-down tables: company, cumulative growths, CAGRs, change in EBITDA margin
st.subheader("Fund tables")

# Ensure portfolio company display column exists
if portfolio_header not in f.columns and "portfolio_company" in f.columns:
    f.insert(0, portfolio_header, f["portfolio_company"])  # ensure first col

# Compute margin columns and change
with np.errstate(divide="ignore", invalid="ignore"):
    if {"entry_ebitda", "entry_revenue"}.issubset(f.columns):
        f["entry_margin_pct"] = pd.to_numeric(f["entry_ebitda"], errors="coerce") / pd.to_numeric(f["entry_revenue"], errors="coerce")
    if {"exit_ebitda", "exit_revenue"}.issubset(f.columns):
        f["exit_margin_pct"] = pd.to_numeric(f["exit_ebitda"], errors="coerce") / pd.to_numeric(f["exit_revenue"], errors="coerce")
    if {"entry_margin_pct", "exit_margin_pct"}.issubset(f.columns):
        f["ebitda_margin_change_pct"] = pd.to_numeric(f["exit_margin_pct"], errors="coerce") - pd.to_numeric(f["entry_margin_pct"], errors="coerce")

def _neg_red(v):
    try:
        v_float = float(v)
        if pd.notna(v_float) and v_float < 0:
            return "color: red"
    except Exception:
        return ""
    return ""

if "fund_name" in f.columns:
    for fund, g in f.groupby("fund_name"):
        with st.expander(str(fund)):
            cols = [
                portfolio_header,
                "ebitda_growth_pct",
                "ebitda_cagr",
                "revenue_growth_pct",
                "revenue_cagr",
                "ebitda_margin_change_pct",
            ]
            cols = [c for c in cols if c in g.columns or c == portfolio_header]
            tbl = g[cols].copy()
            # Format as %
            fmt = {}
            for c in ["ebitda_growth_pct", "ebitda_cagr", "revenue_growth_pct", "revenue_cagr", "ebitda_margin_change_pct"]:
                if c in tbl.columns:
                    fmt[c] = "{:.1%}"
            # Build subtotal rows: Weighted Avg (by Invested), Average, Median, Max, Min
            agg_rows = []
            label_col = portfolio_header
            numeric_cols = [c for c in tbl.columns if c != portfolio_header and pd.api.types.is_numeric_dtype(tbl[c])]
            # Weighted Average by Invested
            w = pd.to_numeric(g.get("invested"), errors="coerce").clip(lower=0)
            if len(numeric_cols) and w.notna().sum() and float(w.fillna(0).sum()) > 0:
                wa_row = {label_col: "Weighted Avg (by Invested)"}
                for c in numeric_cols:
                    vals = pd.to_numeric(g[c], errors="coerce") if c in g.columns else pd.Series(dtype=float)
                    mask = (~vals.isna()) & (~w.isna()) & (w > 0)
                    wa_row[c] = float(np.average(vals[mask].fillna(0), weights=w[mask].fillna(0))) if mask.any() and float(w[mask].sum()) > 0 else np.nan
                agg_rows.append(wa_row)
            # Average
            avg_row = {label_col: "Average"}
            for c in numeric_cols:
                avg_row[c] = pd.to_numeric(g.get(c), errors="coerce").mean(skipna=True)
            agg_rows.append(avg_row)
            # Median
            med_row = {label_col: "Median"}
            for c in numeric_cols:
                med_row[c] = pd.to_numeric(g.get(c), errors="coerce").median(skipna=True)
            agg_rows.append(med_row)
            # Max
            max_row = {label_col: "Max"}
            for c in numeric_cols:
                max_row[c] = pd.to_numeric(g.get(c), errors="coerce").max(skipna=True)
            agg_rows.append(max_row)
            # Min
            min_row = {label_col: "Min"}
            for c in numeric_cols:
                min_row[c] = pd.to_numeric(g.get(c), errors="coerce").min(skipna=True)
            agg_rows.append(min_row)
            # Render deals table and separate subtotals table
            num_cols = [c for c in tbl.columns if c != portfolio_header and pd.api.types.is_numeric_dtype(tbl[c])]
            st.caption("Deals")
            st.dataframe(tbl.style.format(fmt).applymap(_neg_red, subset=num_cols), use_container_width=True)

            if agg_rows:
                sub_df = pd.DataFrame(agg_rows)
                # Ensure same column order as deals table where present
                sub_df = sub_df[[c for c in tbl.columns if c in sub_df.columns]]
                sub_num_cols = [c for c in sub_df.columns if c != portfolio_header and pd.api.types.is_numeric_dtype(sub_df[c])]
                st.caption("Subtotals")
                st.dataframe(sub_df.style.format(fmt).applymap(_neg_red, subset=sub_num_cols), use_container_width=True)

