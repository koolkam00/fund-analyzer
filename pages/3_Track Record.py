from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr
from data_loader import ensure_workbook_loaded
from filters import render_and_filter


st.set_page_config(page_title="Track Record", layout="wide")

# Ensure consistent header font styling for all expanders on this page
st.markdown(
    """
    <style>
    /* Keep expander header text on one line and consistent styling */
    div[data-testid="stExpander"] button,
    div[data-testid="stExpander"] button p,
    div[data-testid="stExpander"] button span {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        font-family: inherit !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        word-break: normal !important;
        overflow-wrap: normal !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
# Show firm on top if available
firm = st.session_state.get("firm_name")
if firm:
    st.markdown(f"**Firm:** {firm}")
st.caption("Grouped by fund. Expand a fund to view per-deal details; collapse to view totals.")

sheets, ops_sheet_name, funds_sheet_name, _ = ensure_workbook_loaded()
if not sheets:
    st.info("Upload a workbook to begin.")
    st.stop()
sheet_name = ops_sheet_name or list(sheets.keys())[0]
df = sheets[sheet_name]
st.caption(f"Loaded workbook — operational sheet '{sheet_name}' with rows: {len(df):,}")

# Build operating dataframe using fixed order mapping
ops_df_raw, _ = extract_operational_by_template_order(df, list(df.columns))
if ops_df_raw.empty:
    st.error("No metrics detected from the uploaded sheet. Confirm the header row and column order.")
    st.stop()

ops_df = add_growth_and_cagr(ops_df_raw)

# Map Fund-level data (Net Returns and Fund Size) from Funds sheet (sheet 2)
fund_net_map: Dict[str, Dict[str, float]] = {}
fund_size_map: Dict[str, float] = {}
try:
    if funds_sheet_name and funds_sheet_name in sheets:
        df_funds = sheets[funds_sheet_name].copy()
        # Normalize columns
        def _norm(s: str) -> str:
            return str(s).strip().lower().replace(" ", "_")
        df_funds.columns = [_norm(c) for c in df_funds.columns]
        # Expected columns: fund, fund_size, vintage_year, net_irr, net_tvpi, net_dpi
        # Clean fund size (in $MM)
        if "fund_size" in df_funds.columns:
            df_funds["fund_size"] = pd.to_numeric(
                df_funds["fund_size"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                errors="coerce",
            )
        if "fund" in df_funds.columns:
            # Clean numeric fields
            if "net_irr" in df_funds.columns:
                irr_str = (
                    df_funds["net_irr"].astype(str)
                    .str.replace(r"[^0-9.\-%]", "", regex=True)
                    .str.strip()
                )
                has_pct = irr_str.str.contains("%", regex=False)
                irr_num = pd.to_numeric(irr_str.str.replace("%", ""), errors="coerce")
                df_funds["net_irr"] = irr_num.where(~(has_pct | (irr_num > 1.0)), irr_num / 100.0)
            for c in ["net_tvpi", "net_dpi"]:
                if c in df_funds.columns:
                    df_funds[c] = pd.to_numeric(
                        df_funds[c].astype(str).str.replace(r"[^0-9.\-\.]", "", regex=True),
                        errors="coerce",
                    )
            for _, r in df_funds.iterrows():
                key = str(r.get("fund", "")).strip().lower()
                if not key:
                    continue
                fund_net_map[key] = {
                    "net_irr": float(r.get("net_irr")) if pd.notna(r.get("net_irr")) else float("nan"),
                    "net_tvpi": float(r.get("net_tvpi")) if pd.notna(r.get("net_tvpi")) else float("nan"),
                    "net_dpi": float(r.get("net_dpi")) if pd.notna(r.get("net_dpi")) else float("nan"),
                }
                if pd.notna(r.get("fund_size")):
                    fund_size_map[key] = float(r.get("fund_size"))
except Exception:
    fund_net_map = {}

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

## Removed top-level All Funds expander; will render All Funds subtotals at bottom

## Removed All Funds — All Deals table per request

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

    # Resolve fund size (in $MM) if available
    fund_key = str(fund).strip().lower()
    fund_size_mm = fund_size_map.get(fund_key, np.nan)

    # Build per-deal rows with required columns and computed % of fund size invested
    rows = g.copy()
    if pd.notna(fund_size_mm) and fund_size_mm > 0:
        rows["pct_of_fund_invested"] = pd.to_numeric(rows["invested"], errors="coerce") / float(fund_size_mm)
    else:
        # Fallback to share of this fund's invested total
        if fund_invest and fund_invest > 0:
            rows["pct_of_fund_invested"] = pd.to_numeric(rows["invested"], errors="coerce") / fund_invest
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

    total_moic_str = f"{total_moic:.1f}x" if pd.notna(total_moic) else "—"
    wa_irr_str = f"{wa_irr:.1%}" if pd.notna(wa_irr) else "—"
    fund_dpi = (fund_proceeds / fund_invest) if fund_invest else np.nan
    fund_dpi_str = f"{fund_dpi:.1f}x" if pd.notna(fund_dpi) else "—"
    # Append Net returns if available from Funds sheet
    net_irr = net_tvpi = net_dpi = None
    if fund_net_map:
        rec = fund_net_map.get(str(fund).strip().lower())
        if rec:
            net_irr = rec.get("net_irr")
            net_tvpi = rec.get("net_tvpi")
            net_dpi = rec.get("net_dpi")
    net_irr_str = f"{net_irr:.1%}" if isinstance(net_irr, (int, float)) and pd.notna(net_irr) else "—"
    net_tvpi_str = f"{net_tvpi:.1f}x" if isinstance(net_tvpi, (int, float)) and pd.notna(net_tvpi) else "—"
    net_dpi_str = f"{net_dpi:.1f}x" if isinstance(net_dpi, (int, float)) and pd.notna(net_dpi) else "—"
    deployed_pct = (fund_invest / fund_size_mm) if (pd.notna(fund_size_mm) and fund_size_mm > 0) else np.nan
    deployed_str = f"{deployed_pct:.1%}" if pd.notna(deployed_pct) else "—"

    def _fmt_size_mm(mm: float) -> str:
        try:
            val = float(mm)
        except Exception:
            return "—"
        if not np.isfinite(val):
            return "—"
        # Use B for billions to reduce width
        if val >= 1000.0:
            return f"${val/1000.0:,.1f}B"
        return f"${val:,.1f}MM"

    fund_size_str = _fmt_size_mm(fund_size_mm) if pd.notna(fund_size_mm) else "—"
    invested_str = _fmt_size_mm(fund_invest) if pd.notna(fund_invest) else "—"
    realized_str = _fmt_size_mm(fund_proceeds) if pd.notna(fund_proceeds) else "—"
    nav_str = _fmt_size_mm(fund_nav) if pd.notna(fund_nav) else "—"

    fund_header = (
        f"{fund} - Gross MOIC: {total_moic_str} | Gross IRR: {wa_irr_str} | Gross DPI: {fund_dpi_str} | "
        f"Net TVPI: {net_tvpi_str} | Net IRR: {net_irr_str} | Net DPI: {net_dpi_str} | "
        f"FS: {fund_size_str} | Dep: {deployed_str} | "
        f"Inv: {invested_str} | Real: {realized_str} | NAV: {nav_str}"
    )
    with st.expander(fund_header):
        # Server-side sorting selector for per-fund table (deal rows only)
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
        table = rows_sorted
        # Format date columns as Mon YYYY for deal rows
        for dc in ["invest_date", "exit_date"]:
            if dc in table.columns:
                table[dc] = pd.to_datetime(table[dc], errors="coerce").dt.strftime("%b %Y")
        st.dataframe(_fmt(table), use_container_width=True)
        # Subtotals by realization status with WA holding period by invested
        proceeds_num_f = pd.to_numeric(g.get("proceeds"), errors="coerce").fillna(0)
        nav_num_f = pd.to_numeric(g.get("current_value"), errors="coerce").fillna(0)
        invested_num_f = pd.to_numeric(g.get("invested"), errors="coerce").fillna(0)
        is_fully_realized_f = (nav_num_f <= 0) & (proceeds_num_f > 0)
        is_partially_realized_f = (proceeds_num_f > 0) & (nav_num_f > 0)
        is_unrealized_f = (proceeds_num_f <= 0) & (nav_num_f > 0)
        def _subtotal_f(label: str, mask: pd.Series) -> dict:
            sub = g[mask].copy()
            inv = float(pd.to_numeric(sub.get("invested"), errors="coerce").sum(skipna=True)) if not sub.empty else 0.0
            proc = float(pd.to_numeric(sub.get("proceeds"), errors="coerce").sum(skipna=True)) if not sub.empty else 0.0
            navv = float(pd.to_numeric(sub.get("current_value"), errors="coerce").sum(skipna=True)) if not sub.empty else 0.0
            invested_safe = inv if inv != 0 else np.nan
            total_val = proc + navv
            total_moic_sub = (total_val / invested_safe) if invested_safe else np.nan
            realized_moic_sub = (proc / invested_safe) if invested_safe else np.nan
            unrealized_moic_sub = (navv / invested_safe) if invested_safe else np.nan
            # Weighted average IRR and holding years by invested
            wa_irr_sub = np.nan
            wa_hold_sub = np.nan
            if not sub.empty and inv > 0:
                w = pd.to_numeric(sub.get("invested"), errors="coerce").clip(lower=0).fillna(0)
                if float(w.sum()) > 0:
                    irr_vals = pd.to_numeric(sub.get("gross_irr"), errors="coerce").fillna(0)
                    wa_irr_sub = float(np.average(irr_vals, weights=w))
                    yrs = pd.to_numeric(sub.get("holding_years"), errors="coerce").fillna(0)
                    wa_hold_sub = float(np.average(yrs, weights=w))
            return {
                portfolio_header: label,
                "holding_years": wa_hold_sub,
                "invested": inv,
                "proceeds": proc,
                "current_value": navv,
                "total_value": total_val,
                "realized_moic": realized_moic_sub,
                "unrealized_moic": unrealized_moic_sub,
                "gross_moic": total_moic_sub,
                "gross_irr": wa_irr_sub,
            }
        sub_rows_f = [
            _subtotal_f("Fully Realized", is_fully_realized_f),
            _subtotal_f("Partially Realized", is_partially_realized_f),
            _subtotal_f("Unrealized", is_unrealized_f),
            _subtotal_f("Total", pd.Series(True, index=g.index)),
        ]
        sub_df_f = pd.DataFrame(sub_rows_f)
        sub_cols = [
            portfolio_header,
            "holding_years",
            "invested",
            "proceeds",
            "current_value",
            "total_value",
            "realized_moic",
            "unrealized_moic",
            "gross_moic",
            "gross_irr",
        ]
        sub_df_f = sub_df_f[[c for c in sub_cols if c in sub_df_f.columns]]
        st.caption("Subtotals by realization status")
        st.dataframe(_fmt(sub_df_f), use_container_width=True)
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
                st.switch_page("pages/5_Company Detail.py")


# All Funds subtotals at bottom (not a dropdown)
st.subheader("All Funds — Subtotals by realization status")
proceeds_num = pd.to_numeric(f.get("proceeds"), errors="coerce").fillna(0)
nav_num = pd.to_numeric(f.get("current_value"), errors="coerce").fillna(0)
invested_num = pd.to_numeric(f.get("invested"), errors="coerce").fillna(0)

is_fully_realized = (nav_num <= 0) & (proceeds_num > 0)
is_partially_realized = (proceeds_num > 0) & (nav_num > 0)
is_unrealized = (proceeds_num <= 0) & (nav_num > 0)

def _subtotal_all(label: str, mask: pd.Series) -> dict:
    sub = f[mask].copy()
    inv = float(pd.to_numeric(sub.get("invested"), errors="coerce").sum(skipna=True)) if not sub.empty else 0.0
    proc = float(pd.to_numeric(sub.get("proceeds"), errors="coerce").sum(skipna=True)) if not sub.empty else 0.0
    navv = float(pd.to_numeric(sub.get("current_value"), errors="coerce").sum(skipna=True)) if not sub.empty else 0.0
    invested_safe = inv if inv != 0 else np.nan
    total_moic_sub = (proc + navv) / invested_safe if invested_safe else np.nan
    realized_moic_sub = proc / invested_safe if invested_safe else np.nan
    unrealized_moic_sub = navv / invested_safe if invested_safe else np.nan
    total_val = proc + navv
    wa_irr_sub = np.nan
    if {"gross_irr", "invested"}.issubset(sub.columns) and inv > 0:
        irr_vals = pd.to_numeric(sub["gross_irr"], errors="coerce").fillna(0)
        weights = pd.to_numeric(sub["invested"], errors="coerce").clip(lower=0).fillna(0)
        if float(weights.sum()) > 0:
            wa_irr_sub = float(np.average(irr_vals, weights=weights))
    # Weighted average holding years by invested
    wa_hold_sub = np.nan
    if {"holding_years", "invested"}.issubset(sub.columns) and inv > 0:
        yrs_vals = pd.to_numeric(sub["holding_years"], errors="coerce").fillna(0)
        weights = pd.to_numeric(sub["invested"], errors="coerce").clip(lower=0).fillna(0)
        if float(weights.sum()) > 0:
            wa_hold_sub = float(np.average(yrs_vals, weights=weights))
    return {
        portfolio_header: label,
        "holding_years": wa_hold_sub,
        "invested": inv,
        "proceeds": proc,
        "current_value": navv,
        "total_value": total_val,
        "realized_moic": realized_moic_sub,
        "unrealized_moic": unrealized_moic_sub,
        "gross_moic": total_moic_sub,
        "gross_irr": wa_irr_sub,
    }

subtotals_rows_all = [
    _subtotal_all("Fully Realized", is_fully_realized),
    _subtotal_all("Partially Realized", is_partially_realized),
    _subtotal_all("Unrealized", is_unrealized),
    _subtotal_all("Total", pd.Series(True, index=f.index)),
]
subtotals_df_all = pd.DataFrame(subtotals_rows_all)
# Display specific columns (remove sector/status), include WA holding years and total value
all_cols = [
    portfolio_header,
    "holding_years",
    "invested",
    "proceeds",
    "current_value",
    "total_value",
    "realized_moic",
    "unrealized_moic",
    "gross_moic",
    "gross_irr",
]
subtotals_df_all = subtotals_df_all[[c for c in all_cols if c in subtotals_df_all.columns]]
st.dataframe(_fmt(subtotals_df_all), use_container_width=True)
