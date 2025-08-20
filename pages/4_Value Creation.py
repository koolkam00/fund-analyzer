from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr, compute_value_creation
from data_loader import ensure_workbook_loaded
from filters import render_and_filter


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


st.title("Value Creation Analysis")
# Show firm on top if available
firm = st.session_state.get("firm_name")
if firm:
    st.markdown(f"**Firm:** {firm}")
st.caption("Decompose change in equity value into Revenue Growth, Margin Expansion, Multiple Change, and Deleveraging.")

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
ops_df = compute_value_creation(ops_df)

st.subheader("Filters")
f = render_and_filter(ops_df, key_prefix="vc")
# Only include deals with complete inputs for value creation
f = f[f.get("vc_valid", True).astype(bool)]

with st.expander("Deal Value Creation Table", expanded=False):
    # Compute additional change columns for TEV and leverage
    with np.errstate(all='ignore'):
        f["tev_change"] = pd.to_numeric(f.get("exit_tev"), errors="coerce") - pd.to_numeric(f.get("entry_tev"), errors="coerce")
        f["leverage_change"] = pd.to_numeric(f.get("exit_leverage"), errors="coerce") - pd.to_numeric(f.get("entry_leverage"), errors="coerce")
        # TEV/EBITDA multiple change
        f["tev_ebitda_change"] = pd.to_numeric(f.get("exit_tev_ebitda"), errors="coerce") - pd.to_numeric(f.get("entry_tev_ebitda"), errors="coerce")
    portfolio_header = df.columns[0] if len(df.columns) > 0 else "Portfolio Company"
    display_cols = [
        portfolio_header,
        "status",
        "holding_years",
        "fund_name",
        "sector",
        "geography",
        "invest_date",
        "exit_date",
        "entry_tev",
        "exit_tev",
        "entry_tev_ebitda",
        "exit_tev_ebitda",
        "entry_leverage",
        "exit_leverage",
        "tev_change",
        "tev_ebitda_change",
        "leverage_change",
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
    # Build a unique label (Company — Fund) to distinguish cross-fund investments for selections
    label_col_unique = f"{portfolio_header} (Fund)"
    if portfolio_header in view.columns and "fund_name" in view.columns:
        view[label_col_unique] = view[portfolio_header].astype(str) + " — " + view["fund_name"].astype(str)
    else:
        view[label_col_unique] = view.get(portfolio_header, pd.Series(dtype=str)).astype(str)
    fmt = {col: "{:.1f}" for col in view.select_dtypes(include=["number"]).columns}
    # Percent formatting for leverage change is in turns of x, keep as numeric; dates below formatted elsewhere
    st.dataframe(view.style.format(fmt), use_container_width=True)

# Quick navigation to Company Detail
if {portfolio_header, "fund_name"}.issubset(view.columns):
    nav_opts = view[[portfolio_header, "fund_name"]].dropna().astype(str).drop_duplicates()
    nav_opts[label_col_unique] = nav_opts[portfolio_header] + " — " + nav_opts["fund_name"]
    c_nav1, c_nav2 = st.columns([3,1])
    sel = c_nav1.selectbox("Open in Company Detail", nav_opts[label_col_unique].tolist())
    if c_nav2.button("Open", use_container_width=True):
        parts = sel.split(" — ", 1)
        st.session_state["detail_company"] = parts[0]
        st.session_state["detail_fund"] = parts[1] if len(parts) > 1 else ""
        try:
            st.switch_page("pages/5_Company Detail.py")
        except Exception:
            pass

# Aggregated waterfall across all filtered deals
st.subheader("Portfolio Waterfall (filtered)")

def _sum_num(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").sum(skipna=True))

# Exclude invalid rows for portfolio-level viz
f_vc = f[f.get("vc_valid", True).astype(bool)]

eq0_total = _sum_num(f_vc.get("equity_entry", pd.Series(dtype=float)))
rev_total = _sum_num(f_vc.get("vc_rev_growth", pd.Series(dtype=float)))
marg_total = _sum_num(f_vc.get("vc_margin_expansion", pd.Series(dtype=float)))
mult_total = _sum_num(f_vc.get("vc_multiple_change", pd.Series(dtype=float)))
debt_total = _sum_num(f_vc.get("vc_deleveraging", pd.Series(dtype=float)))
eq1_total = _sum_num(f_vc.get("equity_exit", pd.Series(dtype=float)))

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

fund = f_vc.get("fund_name")
moic = pd.to_numeric(f_vc.get("gross_moic"), errors="coerce")
irr = pd.to_numeric(f_vc.get("gross_irr"), errors="coerce")
status = f_vc.get("status")
sector = f_vc.get("sector")
custom_port = np.column_stack([
    fund.fillna("") if fund is not None else np.repeat("", len(f_vc)),
    moic.fillna(np.nan) if moic is not None else np.repeat(np.nan, len(f_vc)),
    irr.fillna(np.nan) if irr is not None else np.repeat(np.nan, len(f_vc)),
    status.fillna("") if status is not None else np.repeat("", len(f_vc)),
    sector.fillna("") if sector is not None else np.repeat("", len(f_vc)),
]) if len(f_vc) > 0 else np.empty((0,5))
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
deal_names = view[label_col_unique].dropna().astype(str).unique().tolist() if label_col_unique in view.columns else []
sel_deal = st.selectbox("Select deal", deal_names)
if sel_deal:
    row = view.loc[view[label_col_unique] == sel_deal].iloc[0]
    # Skip invalid rows
    if bool(row.get("vc_valid", True)):
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
            # Column order per request
            cols = [
                portfolio_header,
                "status",
                "holding_years",
                "revenue_growth_pct",
                "revenue_cagr",
                "ebitda_growth_pct",
                "ebitda_cagr",
                "ebitda_margin_change_pct",
                "entry_tev",
                "exit_tev",
                "entry_tev_ebitda",
                "exit_tev_ebitda",
                "entry_leverage",
                "exit_leverage",
                "tev_ebitda_change",
            ]
            cols = [c for c in cols if c in g.columns or c == portfolio_header]
            tbl = g[cols].copy()
            # Server-side sorting to ensure numeric-correct order
            sort_left, sort_right = st.columns([3,1])
            sort_options = [c for c in tbl.columns]
            sort_col = sort_left.selectbox("Sort by", sort_options, index=0, key=f"vc_sort_col_{fund}")
            sort_asc = sort_right.toggle("Asc", value=False, key=f"vc_sort_dir_{fund}")
            # Convert selected column to numeric when possible for correct ordering
            if sort_col != portfolio_header:
                tbl = tbl.sort_values(by=sort_col, key=lambda s: pd.to_numeric(s, errors="coerce"), ascending=sort_asc, na_position="last")
            else:
                tbl = tbl.sort_values(by=sort_col, ascending=sort_asc, na_position="last")
            # Format: percentages with one decimal; numeric values one decimal
            fmt = {}
            for c in ["ebitda_growth_pct", "ebitda_cagr", "revenue_growth_pct", "revenue_cagr", "ebitda_margin_change_pct"]:
                if c in tbl.columns:
                    fmt[c] = "{:.1%}"
            for c in tbl.select_dtypes(include=["number"]).columns:
                if c not in fmt:
                    fmt[c] = "{:.1f}"
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
            sty = tbl.style.format(fmt, na_rep="—")
            if num_cols:
                sty = sty.applymap(_neg_red, subset=num_cols)
            st.dataframe(sty, use_container_width=True)

            if agg_rows:
                sub_df = pd.DataFrame(agg_rows)
                # Ensure same column order as deals table where present
                sub_df = sub_df[[c for c in tbl.columns if c in sub_df.columns]]
                sub_num_cols = [c for c in sub_df.columns if c != portfolio_header and pd.api.types.is_numeric_dtype(sub_df[c])]
                st.caption("Subtotals")
                sty2 = sub_df.style.format(fmt, na_rep="—")
                if sub_num_cols:
                    sty2 = sty2.applymap(_neg_red, subset=sub_num_cols)
                st.dataframe(sty2, use_container_width=True)

