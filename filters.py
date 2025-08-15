from __future__ import annotations

from typing import Dict, List, Tuple

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


def render_filters(df: pd.DataFrame, key_prefix: str = "flt") -> Dict[str, List[str]]:
    """Render standardized, compact filters with empty defaults. Return selections dict."""
    _inject_compact_css()

    # Build options per field
    selections: Dict[str, List[str]] = {}
    cols = st.columns(3)
    for idx, (field, label) in enumerate(FILTER_FIELDS):
        if field not in df.columns:
            continue
        options = sorted(pd.Series(df[field]).dropna().astype(str).unique().tolist())
        col = cols[idx % 3]
        sel = col.multiselect(label, options, default=[], key=f"{key_prefix}_{field}")
        selections[field] = sel
    return selections


def apply_filters(df: pd.DataFrame, selections: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    for field, sel in selections.items():
        if field in out.columns and sel:
            out = out[out[field].astype(str).isin([str(v) for v in sel])]
    return out


def render_and_filter(df: pd.DataFrame, key_prefix: str = "flt") -> pd.DataFrame:
    sels = render_filters(df, key_prefix=key_prefix)
    return apply_filters(df, sels)


