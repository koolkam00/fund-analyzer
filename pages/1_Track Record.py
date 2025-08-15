from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr
from filters import render_and_filter


st.set_page_config(page_title="Track Record", layout="wide")


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


st.title("Track Record")
st.caption("Grouped by fund. Expand a fund to view per-deal details; collapse to view totals.")

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

# Build operating dataframe using fixed order mapping
ops_df_raw, _ = extract_operational_by_template_order(df, list(df.columns))
if ops_df_raw.empty:
    st.error("No metrics detected from the uploaded sheet. Confirm the header row and column order.")
    st.stop()

ops_df = add_growth_and_cagr(ops_df_raw)

# Ensure identity/display columns
portfolio_header = df.columns[0] if len(df.columns) > 0 else "Portfolio Company"
if portfolio_header not in ops_df.columns and "portfolio_company" in ops_df.columns:
    ops_df.insert(0, portfolio_header, ops_df["portfolio_company"])  # ensure display col exists

# Compute ownership column (prefer exit if available)
with np.errstate(divide="ignore", invalid="ignore"):
    if "kam_ownership_exit_pct" in ops_df.columns:
        ops_df["ownership_pct"] = pd.to_numeric(ops_df["kam_ownership_exit_pct"], errors="coerce")
    elif {"kam_equity_entry", "equity_entry_total"}.issubset(ops_df.columns):
        ops_df["ownership_pct"] = pd.to_numeric(ops_df["kam_equity_entry"], errors="coerce") / pd.to_numeric(ops_df["equity_entry_total"], errors="coerce")
    else:
        ops_df["ownership_pct"] = np.nan

# Core numeric fields
ops_df["invested"] = pd.to_numeric(ops_df.get("invested"), errors="coerce")
ops_df["proceeds"] = pd.to_numeric(ops_df.get("proceeds"), errors="coerce")
ops_df["current_value"] = pd.to_numeric(ops_df.get("current_value"), errors="coerce")
ops_df["gross_moic"] = pd.to_numeric(ops_df.get("gross_moic"), errors="coerce")
ops_df["gross_irr"] = pd.to_numeric(ops_df.get("gross_irr"), errors="coerce")

# Realized/Unrealized MOIC per deal
with np.errstate(divide="ignore", invalid="ignore"):
    ops_df["realized_moic"] = ops_df["proceeds"] / ops_df["invested"]
    ops_df["unrealized_moic"] = ops_df["current_value"] / ops_df["invested"]

st.subheader("Filters")
f = render_and_filter(ops_df, key_prefix="tr")

st.subheader("Track Record by Fund")
if "fund_name" not in f.columns:
    st.info("Fund column not found. Ensure 'Fund Name (GP)' exists in the sheet.")
    st.stop()

def _fmt(df_in: pd.DataFrame) -> pd.io.formats.style.Styler:
    # Build column-specific formatters
    formatters: Dict[str, object] = {}
    # Percent columns by suffix
    for col in df_in.columns:
        if col.endswith("pct"):
            formatters[col] = "{:.1%}"
    # Explicit percent columns
    if "gross_irr" in df_in.columns:
        formatters["gross_irr"] = "{:.1%}"
    if "pct_of_fund_invested" in df_in.columns:
        formatters["pct_of_fund_invested"] = "{:.1%}"
    if "ownership_pct" in df_in.columns:
        formatters["ownership_pct"] = "{:.1%}"

    # MOIC columns as 0.0x
    def moic_fmt(v):
        try:
            return (f"{float(v):.1f}x") if pd.notna(v) else ""
        except Exception:
            return ""
    for moic_col in ["realized_moic", "unrealized_moic", "gross_moic"]:
        if moic_col in df_in.columns:
            formatters[moic_col] = moic_fmt

    # Default numeric formatting for remaining numeric columns
    for col in df_in.select_dtypes(include=["number"]).columns:
        if col not in formatters:
            formatters[col] = "{:,.1f}"

    return df_in.style.format(formatters)

# Overall (all funds) expanders
overall_invest = float(f.get("invested", pd.Series(dtype=float)).sum(skipna=True)) if "invested" in f.columns else 0.0
overall_proc = float(f.get("proceeds", pd.Series(dtype=float)).sum(skipna=True)) if "proceeds" in f.columns else 0.0
overall_nav = float(f.get("current_value", pd.Series(dtype=float)).sum(skipna=True)) if "current_value" in f.columns else 0.0
overall_moic = (overall_proc + overall_nav) / overall_invest if overall_invest else np.nan
wa_irr_overall = np.nan
if {"gross_irr", "invested"}.issubset(f.columns):
    irr_vals = pd.to_numeric(f["gross_irr"], errors="coerce")
    w = pd.to_numeric(f["invested"], errors="coerce").clip(lower=0)
    if w.notna().sum() and float(w.sum()) > 0:
        wa_irr_overall = float(np.average(irr_vals.fillna(0), weights=w.fillna(0)))

overall_moic_str = f"{overall_moic:.1f}x" if pd.notna(overall_moic) else "—"
wa_irr_overall_str = f"{wa_irr_overall:.1%}" if pd.notna(wa_irr_overall) else "—"
overall_dpi = (overall_proc / overall_invest) if overall_invest else np.nan
overall_rvpi = (overall_nav / overall_invest) if overall_invest else np.nan
overall_dpi_str = f"{overall_dpi:.1f}x" if pd.notna(overall_dpi) else "—"
overall_rvpi_str = f"{overall_rvpi:.1f}x" if pd.notna(overall_rvpi) else "—"

summary_cols = [
    portfolio_header,
    "sector",
    "status",
    "ownership_pct",
    "pct_of_fund_invested",
    "invested",
    "proceeds",
    "current_value",
    "realized_moic",
    "unrealized_moic",
    "gross_moic",
    "gross_irr",
]
summary_cols = [c for c in summary_cols if c in f.columns or c in [portfolio_header]]

overall_header = (
    f"All Funds - TVPI: {overall_moic_str} | DPI: {overall_dpi_str} | RVPI: {overall_rvpi_str} | "
    f"Invested: ${overall_invest:,.1f} | Realized: ${overall_proc:,.1f} | NAV: ${overall_nav:,.1f} | WA IRR: {wa_irr_overall_str}"
)
with st.expander(overall_header):
    overall_row = pd.DataFrame([
        {
            portfolio_header: "Total",
            "sector": "—",
            "status": "—",
            "ownership_pct": np.nan,
            "pct_of_fund_invested": 1.0 if overall_invest > 0 else np.nan,
            "invested": overall_invest,
            "proceeds": overall_proc,
            "current_value": overall_nav,
            "realized_moic": (overall_proc / overall_invest) if overall_invest else np.nan,
            "unrealized_moic": (overall_nav / overall_invest) if overall_invest else np.nan,
            "gross_moic": overall_moic,
            "gross_irr": wa_irr_overall,
        }
    ])
    st.dataframe(_fmt(overall_row[summary_cols]), use_container_width=True)
    # Quick nav to Company Detail from All Deals list
    if {portfolio_header, "fund_name"}.issubset(f.columns):
        c_nav1, c_nav2 = st.columns([3,1])
        opts_df = f[[portfolio_header, "fund_name"]].dropna().astype(str).drop_duplicates()
        opts_df["label"] = opts_df[portfolio_header] + " — " + opts_df["fund_name"]
        sel = c_nav1.selectbox("Open a deal in Company Detail", opts_df["label"].tolist())
        if c_nav2.button("Open", use_container_width=True):
            parts = sel.split(" — ", 1)
            st.session_state["detail_company"] = parts[0]
            st.session_state["detail_fund"] = parts[1] if len(parts) > 1 else ""
            try:
                st.switch_page("pages/8_Company Detail.py")
            except Exception:
                pass

    # Subtotals by realization status (Fully Realized, Partially Realized, Unrealized)
    proceeds_num = pd.to_numeric(f.get("proceeds"), errors="coerce").fillna(0)
    nav_num = pd.to_numeric(f.get("current_value"), errors="coerce").fillna(0)
    invested_num = pd.to_numeric(f.get("invested"), errors="coerce").fillna(0)

    is_fully_realized = (nav_num <= 0) & (proceeds_num > 0)
    is_partially_realized = (proceeds_num > 0) & (nav_num > 0)
    is_unrealized = (proceeds_num <= 0) & (nav_num > 0)

    def _subtotal(label: str, mask: pd.Series) -> dict:
        sub = f[mask].copy()
        inv = float(pd.to_numeric(sub.get("invested"), errors="coerce").sum(skipna=True)) if not sub.empty else 0.0
        proc = float(pd.to_numeric(sub.get("proceeds"), errors="coerce").sum(skipna=True)) if not sub.empty else 0.0
        navv = float(pd.to_numeric(sub.get("current_value"), errors="coerce").sum(skipna=True)) if not sub.empty else 0.0
        invested_safe = inv if inv != 0 else np.nan
        total_moic_sub = (proc + navv) / invested_safe if invested_safe else np.nan
        realized_moic_sub = proc / invested_safe if invested_safe else np.nan
        unrealized_moic_sub = navv / invested_safe if invested_safe else np.nan
        wa_irr_sub = np.nan
        if {"gross_irr", "invested"}.issubset(sub.columns) and inv > 0:
            irr_vals = pd.to_numeric(sub["gross_irr"], errors="coerce").fillna(0)
            weights = pd.to_numeric(sub["invested"], errors="coerce").clip(lower=0).fillna(0)
            if float(weights.sum()) > 0:
                wa_irr_sub = float(np.average(irr_vals, weights=weights))
        return {
            portfolio_header: label,
            "sector": "—",
            "status": "—",
            "ownership_pct": np.nan,
            "pct_of_fund_invested": (inv / overall_invest) if overall_invest > 0 else np.nan,
            "invested": inv,
            "proceeds": proc,
            "current_value": navv,
            "realized_moic": realized_moic_sub,
            "unrealized_moic": unrealized_moic_sub,
            "gross_moic": total_moic_sub,
            "gross_irr": wa_irr_sub,
        }

    subtotals_rows = [
        _subtotal("Fully Realized", is_fully_realized),
        _subtotal("Partially Realized", is_partially_realized),
        _subtotal("Unrealized", is_unrealized),
    ]
    subtotals_df = pd.DataFrame(subtotals_rows)
    st.caption("Subtotals by realization status")
    st.dataframe(_fmt(subtotals_df[summary_cols]), use_container_width=True)

with st.expander("All Funds — All Deals"):
    # reuse per-deal display columns used in fund sections
    all_deals_cols = [
        portfolio_header,
        "sector",
        "status",
        "invest_date",
        "exit_date",
        "holding_years",
        "ownership_pct",
        "pct_of_fund_invested",
        "invested",
        "proceeds",
        "current_value",
        "realized_moic",
        "unrealized_moic",
        "gross_moic",
        "gross_irr",
        "fund_name",
    ]
    all_deals_cols = [c for c in all_deals_cols if c in f.columns or c == portfolio_header]
    df_all = f[all_deals_cols].copy()
    for dc in ["invest_date", "exit_date"]:
        if dc in df_all.columns:
            df_all[dc] = pd.to_datetime(df_all[dc], errors="coerce").dt.strftime("%b %Y")
    st.dataframe(_fmt(df_all), use_container_width=True)

for fund, g in f.groupby("fund_name"):
    # Fund totals
    fund_invest = float(g["invested"].sum(skipna=True)) if "invested" in g.columns else 0.0
    fund_proceeds = float(g["proceeds"].sum(skipna=True)) if "proceeds" in g.columns else 0.0
    fund_nav = float(g["current_value"].sum(skipna=True)) if "current_value" in g.columns else 0.0
    invested_safe = fund_invest if fund_invest != 0 else np.nan
    total_moic = (fund_proceeds + fund_nav) / invested_safe if invested_safe else np.nan
    realized_moic = fund_proceeds / invested_safe if invested_safe else np.nan
    unrealized_moic = fund_nav / invested_safe if invested_safe else np.nan
    # Weighted average Gross IRR by invested
    wa_irr = np.nan
    if {"gross_irr", "invested"}.issubset(g.columns):
        irr_vals = pd.to_numeric(g["gross_irr"], errors="coerce")
        w = pd.to_numeric(g["invested"], errors="coerce").clip(lower=0)
        if w.notna().sum() and float(w.sum()) > 0:
            wa_irr = float(np.average(irr_vals.fillna(0), weights=w.fillna(0)))

    # Build per-deal rows with required columns and computed % of total invested
    rows = g.copy()
    if fund_invest and fund_invest > 0:
        rows["pct_of_fund_invested"] = rows["invested"] / fund_invest
    else:
        rows["pct_of_fund_invested"] = np.nan

    display_cols = [
        portfolio_header,          # 1
        "sector",                 # 2
        "status",                 # 3
        "invest_date",            # 4
        "exit_date",              # 5
        "holding_years",          # 6
        "ownership_pct",
        "pct_of_fund_invested",
        "invested",
        "proceeds",
        "current_value",
        "realized_moic",
        "unrealized_moic",
        "gross_moic",
        "gross_irr",
    ]
    display_cols = [c for c in display_cols if c in rows.columns]

    total_moic_str = f"{total_moic:.1f}" if pd.notna(total_moic) else "—"
    wa_irr_str = f"{wa_irr:.1%}" if pd.notna(wa_irr) else "—"
    fund_dpi = (fund_proceeds / fund_invest) if fund_invest else np.nan
    fund_rvpi = (fund_nav / fund_invest) if fund_invest else np.nan
    fund_dpi_str = f"{fund_dpi:.1f}x" if pd.notna(fund_dpi) else "—"
    fund_rvpi_str = f"{fund_rvpi:.1f}x" if pd.notna(fund_rvpi) else "—"
    fund_header = (
        f"{fund} - TVPI: {total_moic_str} | DPI: {fund_dpi_str} | RVPI: {fund_rvpi_str} | "
        f"Invested: ${fund_invest:,.1f} | Realized: ${fund_proceeds:,.1f} | NAV: ${fund_nav:,.1f} | WA IRR: {wa_irr_str}"
    )
    with st.expander(fund_header):
        # Top-level summary row
        summary = pd.DataFrame([
            {
                portfolio_header: "Total",
                "sector": "—",
                "status": "—",
                "invest_date": pd.NaT,
                "exit_date": pd.NaT,
                "holding_years": np.nan,
                "ownership_pct": np.nan,
                "pct_of_fund_invested": 1.0 if fund_invest and fund_invest > 0 else np.nan,
                "invested": fund_invest,
                "proceeds": fund_proceeds,
                "current_value": fund_nav,
                "realized_moic": realized_moic,
                "unrealized_moic": unrealized_moic,
                "gross_moic": total_moic,
                "gross_irr": np.nan,
            }
        ])
        # Server-side sorting selector for per-fund table (sort only deal rows, keep summary on top)
        sleft, sright = st.columns([3,1])
        sortable_cols = [c for c in display_cols if c in rows.columns]
        sort_col = sleft.selectbox("Sort by", sortable_cols, index=0, key=f"tr_sort_col_{fund}")
        sort_asc = sright.toggle("Asc", value=False, key=f"tr_sort_dir_{fund}")
        rows_sorted = rows[display_cols].copy()
        if sort_col in ["invest_date", "exit_date"] and sort_col in rows_sorted.columns:
            rows_sorted[sort_col] = pd.to_datetime(rows_sorted[sort_col], errors="coerce")
            rows_sorted = rows_sorted.sort_values(by=sort_col, ascending=sort_asc, na_position="last")
            rows_sorted[sort_col] = rows_sorted[sort_col].dt.strftime("%b %Y")
        elif sort_col not in [portfolio_header, "sector", "status"] and sort_col in rows_sorted.columns:
            rows_sorted = rows_sorted.sort_values(by=sort_col, key=lambda s: pd.to_numeric(s, errors="coerce"), ascending=sort_asc, na_position="last")
        else:
            if sort_col in rows_sorted.columns:
                rows_sorted = rows_sorted.sort_values(by=sort_col, ascending=sort_asc, na_position="last")
        table = pd.concat([summary[display_cols], rows_sorted], ignore_index=True)
        # Format date columns as Mon YYYY
        for dc in ["invest_date", "exit_date"]:
            if dc in table.columns:
                table[dc] = pd.to_datetime(table[dc], errors="coerce").dt.strftime("%b %Y")
        st.dataframe(_fmt(table), use_container_width=True)
        # Per-fund quick nav
        if {portfolio_header, "fund_name"}.issubset(rows.columns):
            c_navf1, c_navf2 = st.columns([3,1])
            o2 = rows[[portfolio_header, "fund_name"]].dropna().astype(str).drop_duplicates()
            o2["label"] = o2[portfolio_header] + " — " + o2["fund_name"]
            sel2 = c_navf1.selectbox("Open a deal in Company Detail", o2["label"].tolist(), key=f"nav_{fund}")
            if c_navf2.button("Open", use_container_width=True, key=f"nav_btn_{fund}"):
                parts2 = sel2.split(" — ", 1)
                st.session_state["detail_company"] = parts2[0]
                st.session_state["detail_fund"] = parts2[1] if len(parts2) > 1 else ""
                try:
                    st.switch_page("pages/8_Company Detail.py")
                except Exception:
                    pass


