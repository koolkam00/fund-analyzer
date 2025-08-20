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


# ===== Master workbook template (Portfolio Metrics, Funds Net Returns, Benchmarks) =====
@st.cache_data(show_spinner=False)
def build_master_workbook_template() -> bytes:
    """Create an Excel workbook with:
    - Sheet "Portfolio Metrics": instructions in row 1, headers in row 2 (55 columns)
    - Sheet "Funds Net Returns": instructions in row 1, headers in row 2
    - Sheet "Benchmarks": instructions in row 1, headers in row 2

    The template aligns with the expected fixed-order columns referenced across the app.
    """
    import io
    from openpyxl.utils import get_column_letter
    
    # Portfolio Metrics headers (ordered 1..55)
    portfolio_headers = [
        "Portfolio Company",
        "Acquisition Financial Date",
        "Fund Name (GP)",
        "Fund Currency",
        "Cross-Fund Investment",
        "Country (HQ)",
        "Region of majority operations",
        "Kam Vertical",
        "Vertical Description",
        "Company Currency",
        "Investment strategy",
        "Instrument type",
        "Public/ Private",
        "Purchase Process",
        "Purchase Type",
        "Deal Role",
        "Seller Type",
        "Date Final Exit",
        "Exit type",
        "Fund Position Status",
        "Date Financials",
        "Date Investment",
        "Date IPO",
        "Total Investment Cost ($MM)",
        "Gross Proceeds Distributed to Fund",
        "Current Fair Value ($MM)",
        "Total Value ($MM)",
        "MOIC (Gross)",
        "IRR (Gross)",
        "Equity (Entry) ($MM)",
        "Kam Equity (Entry) ($MM)",
        "Net Debt (Entry) ($MM)",
        "Enterprise Value (Entry) ($MM)",
        "Revenue (Entry) ($MM)",
        "EBITDA (Entry) ($MM)",
        "Multiple (Entry)",
        "M&A Number of Transactions (Net)",
        "Most Recently Available Financial Statements",
        "Equity (Exit/Current) ($MM)",
        "Net Debt (Exit/Current) ($MM)",
        "Enterprise Value (Exit/Current) ($MM)",
        "Revenue (Exit/Current) ($MM)",
        "EBITDA (Exit/Current) ($MM)",
        "Multiple (Exit/Current) ($MM)",
        "Multiple Methodology (Entry)",
        "Multiple Methodology (Exit/Current) ($MM)",
        "Kam Equity (Entry)",
        "Kam Equity Ownership Fund (Exit/Current)",
        "Co-Invest Equity Ownership (Entry)",
        "Co-Invest Equity Ownership (Exit/Current)",
        "Lender Equity Ownership (Entry)",
        "Lender Equity Ownership (Exit/Current)",
        "Mgmt./Seller Equity Ownership (Entry)",
        "Mgmt./Seller Equity Ownership (Exit/Current)",
        "Post Rollover New Majority Owner(s) Ownership (Current)",
    ]

    portfolio_instructions = [
        "Text: Company name",
        "Date: YYYY-MM-DD (or Excel date)",
        "Text: Fund name",
        "Text: Fund currency code (e.g., USD, EUR)",
        "Text: Yes/No or blank",
        "Text: Country (HQ)",
        "Text: Region name",
        "Text: Sector/Vertical",
        "Text: Vertical description",
        "Text: Company currency code",
        "Text: Strategy name",
        "Text: Instrument type",
        "Text: Public or Private",
        "Text: Purchase process",
        "Text: Purchase type",
        "Text: Deal role",
        "Text: Seller type",
        "Date: Final exit date if realized, else blank",
        "Text: Exit type (e.g., Sale, IPO) or blank",
        "Text: Fund position status (e.g., Realized/Unrealized/Partial)",
        "Date: Most recent financials date",
        "Date: Investment date",
        "Date: IPO date if applicable",
        "Number: $MM invested (total investment cost)",
        "Number: $MM gross proceeds distributed",
        "Number: $MM current fair value (NAV)",
        "Number: $MM total value (Proceeds + NAV)",
        "Number: Gross MOIC (e.g., 1.8)",
        "Percent: Gross IRR (e.g., 0.18 or 18%)",
        "Number: $MM equity at entry (optional)",
        "Number: $MM Kam equity at entry (optional)",
        "Number: $MM net debt at entry",
        "Number: $MM enterprise value at entry",
        "Number: $MM revenue at entry",
        "Number: $MM EBITDA at entry",
        "Number: Entry TEV/EBITDA multiple (e.g., 9.5)",
        "Number: Net M&A transactions (optional)",
        "Date: Most recently available financial statements",
        "Number: $MM equity at exit/current (optional)",
        "Number: $MM net debt at exit/current",
        "Number: $MM enterprise value at exit/current",
        "Number: $MM revenue at exit/current",
        "Number: $MM EBITDA at exit/current",
        "Number: Exit/Current TEV/EBITDA multiple",
        "Text: Multiple methodology at entry",
        "Text: Multiple methodology at exit/current",
        "Number: $MM Kam equity (entry) (optional)",
        "Percent: Kam equity ownership fund (exit/current)",
        "Percent: Co-invest equity ownership (entry)",
        "Percent: Co-invest equity ownership (exit/current)",
        "Percent: Lender equity ownership (entry)",
        "Percent: Lender equity ownership (exit/current)",
        "Percent: Mgmt./Seller equity ownership (entry)",
        "Percent: Mgmt./Seller equity ownership (exit/current)",
        "Percent: Post-rollover new majority owner(s) ownership (current)",
    ]

    # Funds Net Returns headers
    funds_headers = ["Fund", "Fund Size", "Vintage Year", "Net IRR", "Net TVPI", "Net DPI"]
    funds_instructions = [
        "Text: Fund name",
        "Number: Fund size ($MM)",
        "Integer: Vintage year (e.g., 2019)",
        "Percent: Net IRR (0.18 or 18%)",
        "Multiple: Net TVPI (e.g., 1.8)",
        "Multiple: Net DPI (e.g., 0.9)",
    ]

    # Benchmarks headers
    bench_headers = ["Vintage Year", "Metric", "Top5%", "Upper Quartile", "Median", "Lower Quartile"]
    bench_instructions = [
        "Integer: Vintage year (e.g., 2019)",
        "Text: One of 'Net IRR', 'Net TVPI', 'Net DPI'",
        "Threshold: Top 5% value (percent or multiple)",
        "Threshold: Upper quartile (percent or multiple)",
        "Threshold: Median (percent or multiple)",
        "Threshold: Lower quartile (percent or multiple)",
    ]

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        # Portfolio Metrics sheet
        pd.DataFrame(columns=portfolio_headers).to_excel(
            writer, sheet_name="Portfolio Metrics", index=False, startrow=1
        )
        ws = writer.sheets["Portfolio Metrics"]
        for idx, instr in enumerate(portfolio_instructions, start=1):
            ws.cell(row=1, column=idx, value=instr)
        # Autosize a bit
        for i, hdr in enumerate(portfolio_headers, start=1):
            col_letter = get_column_letter(i)
            ws.column_dimensions[col_letter].width = max(14, min(50, len(str(hdr)) + 4))

        # Funds Net Returns sheet
        pd.DataFrame(columns=funds_headers).to_excel(
            writer, sheet_name="Funds Net Returns", index=False, startrow=1
        )
        ws2 = writer.sheets["Funds Net Returns"]
        for idx, instr in enumerate(funds_instructions, start=1):
            ws2.cell(row=1, column=idx, value=instr)
        for i, hdr in enumerate(funds_headers, start=1):
            col_letter = get_column_letter(i)
            ws2.column_dimensions[col_letter].width = max(12, min(30, len(str(hdr)) + 4))

        # Benchmarks sheet
        pd.DataFrame(columns=bench_headers).to_excel(
            writer, sheet_name="Benchmarks", index=False, startrow=1
        )
        ws3 = writer.sheets["Benchmarks"]
        for idx, instr in enumerate(bench_instructions, start=1):
            ws3.cell(row=1, column=idx, value=instr)
        for i, hdr in enumerate(bench_headers, start=1):
            col_letter = get_column_letter(i)
            ws3.column_dimensions[col_letter].width = max(14, min(32, len(str(hdr)) + 4))

    return bio.getvalue()

