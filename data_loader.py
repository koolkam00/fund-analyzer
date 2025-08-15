from __future__ import annotations

import io
from typing import Dict, Tuple, Optional

import pandas as pd
import streamlit as st


def ensure_workbook_loaded() -> Tuple[Dict[str, pd.DataFrame], Optional[str], Optional[str], Optional[str]]:
    """Ensure a single Excel workbook is loaded into session state for all non-PME pages.

    Returns (sheets, ops_sheet_name, funds_sheet_name, bench_sheet_name)
    """
    with st.sidebar:
        st.markdown("**Workbook**")
        header_row_index = int(st.session_state.get("header_row_index", 2))
        header_row_index = st.number_input(
            "Header row (1-based) for uploaded sheets",
            min_value=1,
            max_value=100,
            value=header_row_index,
            step=1,
            key="wb_header_row_index",
        )
        upload = st.file_uploader("Upload Excel workbook (.xlsx)", type=["xlsx"], key="wb_file_uploader")  # type: ignore

    # If already loaded and no new upload, reuse existing
    if "workbook" in st.session_state and upload is None:
        wb = st.session_state["workbook"]
        return wb["sheets"], wb.get("ops_sheet_name"), wb.get("funds_sheet_name"), wb.get("bench_sheet_name")

    if upload is None:
        return {}, None, None, None

    # Read all sheets with selected header row
    header_zero = max(0, int(header_row_index) - 1)
    content = upload.getvalue()
    bio = io.BytesIO(content)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    sheets: Dict[str, pd.DataFrame] = {sheet: xls.parse(sheet, header=header_zero) for sheet in xls.sheet_names}

    sheet_names = list(sheets.keys())
    ops_sheet_name = sheet_names[0] if sheet_names else None
    funds_sheet_name = sheet_names[1] if len(sheet_names) > 1 else None
    bench_sheet_name = sheet_names[2] if len(sheet_names) > 2 else None

    st.session_state["workbook"] = {
        "sheets": sheets,
        "ops_sheet_name": ops_sheet_name,
        "funds_sheet_name": funds_sheet_name,
        "bench_sheet_name": bench_sheet_name,
        "header_row_index": header_row_index,
    }
    # Back-compat for existing pages
    st.session_state["sheets"] = sheets
    st.session_state["selected_sheet"] = ops_sheet_name
    st.session_state["header_row_index"] = header_row_index

    return sheets, ops_sheet_name, funds_sheet_name, bench_sheet_name


