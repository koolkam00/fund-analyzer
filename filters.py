from __future__ import annotations

from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st


FILTER_FIELDS: List[Tuple[str, str]] = [
    ("fund_name", "Fund Name (GP)"),
    # ("fund_currency", "Fund Currency"),  # removed per request
    ("cross_fund_investment", "Cross-Fund Investment"),
    ("geography", "Country (HQ)"),
    ("region", "Region of majority operations"),
    ("sector", "Kam Vertical"),
    ("vertical_description", "Vertical Description"),
    # ("company_currency", "Company Currency"),  # removed per request
    ("investment_strategy", "Investment strategy"),
    ("instrument_type", "Instrument type"),
    ("public_private", "Public/ Private"),
    ("purchase_process", "Purchase Process"),
    ("purchase_type", "Purchase Type"),
    ("deal_role", "Deal Role"),
    ("seller_type", "Seller Type"),
    ("exit_type", "Exit type"),
    ("status", "Fund Position Status"),
]


def _inject_compact_css() -> None:
    st.markdown(
        """
        <style>
        /* Compact multi-selects and labels */
        .stMultiSelect > div[data-baseweb="select"] div { min-height: 28px; }
        .stSelectbox > div[data-baseweb="select"] div { min-height: 28px; }
        .stMultiSelect label, .stSelectbox label { font-size: 0.85rem; margin-bottom: 0.1rem; }
        .stMultiSelect, .stSelectbox { margin-bottom: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_filters(df: pd.DataFrame, key_prefix: str = "flt") -> Dict[str, Any]:
    """Render standardized, compact filters with empty defaults. Return selections dict."""
    _inject_compact_css()

    # Build options per field
    selections: Dict[str, Any] = {}
    cols = st.columns(3)
    for idx, (field, label) in enumerate(FILTER_FIELDS):
        if field not in df.columns:
            continue
        options = sorted(pd.Series(df[field]).dropna().astype(str).unique().tolist())
        col = cols[idx % 3]
        sel = col.multiselect(label, options, default=[], key=f"{key_prefix}_{field}")
        selections[field] = sel

    # Year sliders for Investment and Exit years (start as no-op by default: full range)
    # Parse years from date columns if present
    c1, c2 = st.columns(2)
    if "invest_date" in df.columns:
        inv_years = pd.to_datetime(df["invest_date"], errors="coerce").dt.year.dropna().astype(int)
        if not inv_years.empty:
            inv_min, inv_max = int(inv_years.min()), int(inv_years.max())
            inv_range = c1.slider(
                "Investment Year",
                min_value=inv_min,
                max_value=inv_max,
                value=(inv_min, inv_max),
                step=1,
                key=f"{key_prefix}_invest_year_range",
            )
            selections["invest_year_range"] = inv_range
    if "exit_date" in df.columns:
        exit_years = pd.to_datetime(df["exit_date"], errors="coerce").dt.year.dropna().astype(int)
        if not exit_years.empty:
            ex_min, ex_max = int(exit_years.min()), int(exit_years.max())
            ex_range = c2.slider(
                "Exit Year",
                min_value=ex_min,
                max_value=ex_max,
                value=(ex_min, ex_max),
                step=1,
                key=f"{key_prefix}_exit_year_range",
            )
            selections["exit_year_range"] = ex_range
    return selections


def apply_filters(df: pd.DataFrame, selections: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for field, sel in selections.items():
        if field in out.columns and isinstance(sel, list) and sel:
            out = out[out[field].astype(str).isin([str(v) for v in sel])]
    # Apply year range filters only when narrowed (not the full range)
    inv_range = selections.get("invest_year_range")
    if inv_range and "invest_date" in out.columns:
        years = pd.to_datetime(out["invest_date"], errors="coerce").dt.year
        full_years = years.dropna().astype(int)
        if not full_years.empty:
            lo, hi = int(inv_range[0]), int(inv_range[1])
            if lo > full_years.min() or hi < full_years.max():
                out = out[(years >= lo) & (years <= hi)]
    ex_range = selections.get("exit_year_range")
    if ex_range and "exit_date" in out.columns:
        years = pd.to_datetime(out["exit_date"], errors="coerce").dt.year
        full_years = years.dropna().astype(int)
        if not full_years.empty:
            lo, hi = int(ex_range[0]), int(ex_range[1])
            if lo > full_years.min() or hi < full_years.max():
                out = out[(years >= lo) & (years <= hi)]
    return out


def render_and_filter(df: pd.DataFrame, key_prefix: str = "flt") -> pd.DataFrame:
    sels = render_filters(df, key_prefix=key_prefix)
    return apply_filters(df, sels)


