from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from analysis import add_growth_and_cagr, extract_operational_by_template_order


st.set_page_config(page_title="PE Fund Analyzer", layout="wide")


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


def _download_df_button(label: str, df: pd.DataFrame, filename: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label=label, data=buf.getvalue(), file_name=filename, mime="text/csv")


def sidebar_inputs() -> Tuple[str, Dict[str, str], Dict[str, object], Optional[pd.DataFrame]]:
    st.sidebar.header("1) Data upload")
    upload = st.sidebar.file_uploader("Upload Portfolio Metrics file (.xlsx or .csv)", type=["xlsx", "csv"])  # type: ignore
    header_row_index = st.sidebar.number_input("Header row (1-based)", min_value=1, max_value=100, value=int(st.session_state.get("header_row_index", 2)), step=1)
    sheets = _read_excel_or_csv(upload, header_row_index)
    if not sheets and "sheets" in st.session_state:
        sheets = st.session_state.get("sheets", {})
    df = None
    if sheets:
        sheet_names = list(sheets.keys())
        selected_sheet = st.sidebar.selectbox("Select sheet", sheet_names, index=max(0, sheet_names.index(st.session_state.get("selected_sheet", sheet_names[0]))) if sheet_names else 0)
        df = sheets.get(selected_sheet)
        if df is not None:
            st.sidebar.caption(f"Loaded sheet '{selected_sheet}' with rows: {len(df):,}")
            with st.expander("Preview uploaded sheet (first 10 rows)", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
        st.session_state["sheets"] = sheets
        st.session_state["selected_sheet"] = selected_sheet
        st.session_state["header_row_index"] = header_row_index

    st.sidebar.header("2) File format")
    st.sidebar.caption("The uploaded file must match the Portfolio Metric Sheet format. First row = column names.")
    data_shape = "Deal summary"

    options: Dict[str, object] = {}
    mapping: Dict[str, str] = {}

    st.sidebar.header("3) Settings")
    options["include_nav_in_irr"] = False

    return data_shape, mapping, options, df


def main():
    st.title("Private Equity Fund Analyzer")
    st.caption("Upload your Portfolio Metrics file and explore deal-level and portfolio summary tables.")

    data_shape, mapping, options, df = sidebar_inputs()

    if df is None:
        st.info("Upload a Portfolio Metrics file to begin. First row must be column names.")
        st.stop()

    st.subheader("Operating & Valuation Metrics (Entry vs Exit)")
    use_template_mode = True
    ops_df_raw = pd.DataFrame()
    ops_mapping: Dict[str, str] = {}
    if use_template_mode:
        template_headers = list(df.columns)
        ops_df_raw, ops_mapping = extract_operational_by_template_order(df, template_headers)
    if not ops_df_raw.empty:
        ops_df = add_growth_and_cagr(ops_df_raw)

        ops_filtered = ops_df.copy()

        st.subheader("Filters")
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        sectors = sorted([s for s in ops_df["sector"].dropna().unique().tolist()]) if "sector" in ops_df.columns else []
        geos = sorted([g for g in ops_df["geography"].dropna().unique().tolist()]) if "geography" in ops_df.columns else []
        statuses = sorted(ops_df["status"].dropna().unique().tolist()) if "status" in ops_df.columns else []
        fund_names = sorted(ops_df["fund_name"].dropna().unique().tolist()) if "fund_name" in ops_df.columns else []
        strategies = sorted(ops_df["investment_strategy"].dropna().unique().tolist()) if "investment_strategy" in ops_df.columns else []
        instruments = sorted(ops_df["instrument_type"].dropna().unique().tolist()) if "instrument_type" in ops_df.columns else []
        purchases = sorted(ops_df["purchase_process"].dropna().unique().tolist()) if "purchase_process" in ops_df.columns else []
        exit_types = sorted(ops_df["exit_type"].dropna().unique().tolist()) if "exit_type" in ops_df.columns else []

        sel_sector = c1.multiselect("Sector", sectors, default=sectors)
        sel_geo = c2.multiselect("Geography", geos, default=geos)
        sel_status = c3.multiselect("Status", statuses, default=statuses)
        sel_fund = c4.multiselect("Fund Name (GP)", fund_names, default=fund_names)
        sel_strategy = c5.multiselect("Investment strategy", strategies, default=strategies)
        sel_instrument = c6.multiselect("Instrument type", instruments, default=instruments)
        c7, c8, _ = st.columns(3)
        sel_purchase = c7.multiselect("Purchase Process", purchases, default=purchases)
        sel_exit_type = c8.multiselect("Exit type", exit_types, default=exit_types)

        if sel_sector and "sector" in ops_filtered.columns:
            ops_filtered = ops_filtered[ops_filtered["sector"].isin(sel_sector)]
        if sel_geo and "geography" in ops_filtered.columns:
            ops_filtered = ops_filtered[ops_filtered["geography"].isin(sel_geo)]
        if sel_status and "status" in ops_filtered.columns:
            ops_filtered = ops_filtered[ops_filtered["status"].isin(sel_status)]
        if sel_fund and "fund_name" in ops_filtered.columns:
            ops_filtered = ops_filtered[ops_filtered["fund_name"].isin(sel_fund)]
        if sel_strategy and "investment_strategy" in ops_filtered.columns:
            ops_filtered = ops_filtered[ops_filtered["investment_strategy"].isin(sel_strategy)]
        if sel_instrument and "instrument_type" in ops_filtered.columns:
            ops_filtered = ops_filtered[ops_filtered["instrument_type"].isin(sel_instrument)]
        if sel_purchase and "purchase_process" in ops_filtered.columns:
            ops_filtered = ops_filtered[ops_filtered["purchase_process"].isin(sel_purchase)]
        if sel_exit_type and "exit_type" in ops_filtered.columns:
            ops_filtered = ops_filtered[ops_filtered["exit_type"].isin(sel_exit_type)]

        def _sum_pair(df_in: pd.DataFrame, entry_col: str, exit_col: str) -> Tuple[float, float]:
            return float(pd.to_numeric(df_in[entry_col], errors="coerce").sum(skipna=True)), float(pd.to_numeric(df_in[exit_col], errors="coerce").sum(skipna=True))

        rev_entry_sum, rev_exit_sum = _sum_pair(ops_filtered, "entry_revenue", "exit_revenue")
        ebitda_entry_sum, ebitda_exit_sum = _sum_pair(ops_filtered, "entry_ebitda", "exit_ebitda")
        tev_entry_sum, tev_exit_sum = _sum_pair(ops_filtered, "entry_tev", "exit_tev")
        nd_entry_sum, nd_exit_sum = _sum_pair(ops_filtered, "entry_net_debt", "exit_net_debt")

        def _weighted_cagr(entry_sum: float, exit_sum: float, years_series: pd.Series) -> Optional[float]:
            if not np.isfinite(entry_sum) or not np.isfinite(exit_sum) or entry_sum <= 0 or exit_sum <= 0:
                return None
            yrs = pd.to_numeric(years_series, errors="coerce")
            if yrs.notna().any():
                entry_vals = pd.to_numeric(ops_filtered["entry_revenue"], errors="coerce")
                weights = entry_vals.where(entry_vals > 0).fillna(0)
                if weights.sum() > 0 and len(weights) == len(yrs):
                    avg_years = float((weights * yrs.fillna(0)).sum() / weights.sum())
                else:
                    avg_years = float(yrs.mean(skipna=True))
            else:
                avg_years = np.nan
            if not np.isfinite(avg_years) or avg_years <= 0:
                return None
            return float((exit_sum / entry_sum) ** (1 / avg_years) - 1)

        portfolio_ops = {
            "Revenue Entry": rev_entry_sum,
            "Revenue Exit": rev_exit_sum,
            "Revenue Growth %": (rev_exit_sum - rev_entry_sum) / rev_entry_sum if rev_entry_sum else np.nan,
            "Revenue CAGR": _weighted_cagr(rev_entry_sum, rev_exit_sum, ops_filtered.get("holding_years", pd.Series(dtype=float))),
            "EBITDA Entry": ebitda_entry_sum,
            "EBITDA Exit": ebitda_exit_sum,
            "EBITDA Growth %": (ebitda_exit_sum - ebitda_entry_sum) / ebitda_entry_sum if ebitda_entry_sum else np.nan,
            "EBITDA CAGR": _weighted_cagr(ebitda_entry_sum, ebitda_exit_sum, ops_filtered.get("holding_years", pd.Series(dtype=float))),
            "TEV Entry": tev_entry_sum,
            "TEV Exit": tev_exit_sum,
            "TEV Growth %": (tev_exit_sum - tev_entry_sum) / tev_entry_sum if tev_entry_sum else np.nan,
            "TEV CAGR": _weighted_cagr(tev_entry_sum, tev_exit_sum, ops_filtered.get("holding_years", pd.Series(dtype=float))),
            "Net Debt Entry": nd_entry_sum,
            "Net Debt Exit": nd_exit_sum,
            "Net Debt Change %": (nd_exit_sum - nd_entry_sum) / nd_entry_sum if nd_entry_sum else np.nan,
        }

        show_cols = [
            "sector",
            "geography",
            "status",
            "invest_date",
            "exit_date",
            "holding_years",
            "entry_revenue",
            "exit_revenue",
            "revenue_growth_pct",
            "revenue_cagr",
            "entry_ebitda",
            "exit_ebitda",
            "ebitda_growth_pct",
            "ebitda_cagr",
            "entry_tev",
            "exit_tev",
            "entry_tev_ebitda",
            "exit_tev_ebitda",
            "entry_leverage",
            "exit_leverage",
            "entry_tev_revenue",
            "exit_tev_revenue",
            "tev_growth_pct",
            "tev_cagr",
            "entry_net_debt",
            "exit_net_debt",
            "net_debt_change_pct",
            "gross_moic",
            "gross_irr",
            "invested",
            "proceeds",
            "current_value",
        ]
        show_cols = [c for c in show_cols if c in ops_filtered.columns]
        view_df = ops_filtered.copy()
        for dc in ["invest_date", "exit_date", "valuation_date"]:
            if dc in view_df.columns:
                view_df[dc] = pd.to_datetime(view_df[dc], errors="coerce").dt.strftime("%b %Y")
        percent_cols = [
            "revenue_growth_pct",
            "revenue_cagr",
            "ebitda_growth_pct",
            "ebitda_cagr",
            "tev_growth_pct",
            "tev_cagr",
            "net_debt_change_pct",
        ]
        fmt: Dict[str, str] = {}
        for pc in percent_cols:
            if pc in view_df.columns:
                fmt[pc] = "{:.1%}"
        for nc in view_df.select_dtypes(include=["number"]).columns:
            if nc not in fmt:
                fmt[nc] = "{:.1f}"
        portfolio_header = df.columns[0] if len(df.columns) > 0 else "Portfolio Company"
        uploaded_names = df.iloc[:, 0].astype(str)
        if portfolio_header not in view_df.columns:
            view_df.insert(0, portfolio_header, uploaded_names)
        else:
            if view_df[portfolio_header].isna().all() or (view_df[portfolio_header].astype(str).str.strip() == "").all():
                view_df[portfolio_header] = uploaded_names
        left_cols = [portfolio_header] if portfolio_header in view_df.columns else []
        table_df = view_df[left_cols + [c for c in show_cols if c in view_df.columns and c != "deal"]]
        st.dataframe(table_df.style.format(fmt), use_container_width=True)
        # Quick nav to Company Detail
        portfolio_header = df.columns[0] if len(df.columns) > 0 else "Portfolio Company"
        if {portfolio_header, "fund_name"}.issubset(view_df.columns):
            cnav1, cnav2 = st.columns([3,1])
            nav_df = view_df[[portfolio_header, "fund_name"]].dropna().astype(str).drop_duplicates()
            nav_df["label"] = nav_df[portfolio_header] + " — " + nav_df["fund_name"]
            choice = cnav1.selectbox("Open a deal in Company Detail", nav_df["label"].tolist())
            if cnav2.button("Open", use_container_width=True):
                parts = choice.split(" — ", 1)
                st.session_state["detail_company"] = parts[0]
                st.session_state["detail_fund"] = parts[1] if len(parts) > 1 else ""
                try:
                    st.switch_page("pages/8_Company Detail.py")
                except Exception:
                    pass
        download_df = table_df.copy()
        _download_df_button("Download operating metrics (CSV)", download_df, "operating_metrics.csv")

        st.subheader("Deal-level aggregations")
        numeric_cols = table_df.select_dtypes(include=["number"]).columns.tolist()
        label_col = "Metric"
        def _row(label: str) -> Dict[str, object]:
            return {label_col: label, **{col: np.nan for col in numeric_cols}}

        agg_rows: List[Dict[str, object]] = []
        total = _row("Total")
        for col in numeric_cols:
            total[col] = pd.to_numeric(table_df[col], errors="coerce").sum(skipna=True)
        agg_rows.append(total)

        if "invested" in table_df.columns:
            w = pd.to_numeric(table_df["invested"], errors="coerce")
            wa = _row("Weighted Avg (by Invested)")
            for col in numeric_cols:
                vals = pd.to_numeric(table_df[col], errors="coerce")
                mask = (~vals.isna()) & (~w.isna()) & (w > 0)
                if mask.any() and float(w[mask].sum()) > 0:
                    wa[col] = float(np.average(vals[mask], weights=w[mask]))
            agg_rows.append(wa)

        avg = _row("Average")
        for col in numeric_cols:
            avg[col] = pd.to_numeric(table_df[col], errors="coerce").mean(skipna=True)
        agg_rows.append(avg)

        med = _row("Median")
        for col in numeric_cols:
            med[col] = pd.to_numeric(table_df[col], errors="coerce").median(skipna=True)
        agg_rows.append(med)

        mx = _row("Max")
        for col in numeric_cols:
            mx[col] = pd.to_numeric(table_df[col], errors="coerce").max(skipna=True)
        agg_rows.append(mx)

        mn = _row("Min")
        for col in numeric_cols:
            mn[col] = pd.to_numeric(table_df[col], errors="coerce").min(skipna=True)
        agg_rows.append(mn)

        agg_df = pd.DataFrame(agg_rows)
        agg_fmt: Dict[str, str] = {}
        for col in numeric_cols:
            if col in percent_cols:
                agg_fmt[col] = "{:.1%}"
            else:
                agg_fmt[col] = "{:.1f}"
        st.dataframe(agg_df.style.format(agg_fmt), use_container_width=True)
        _download_df_button("Download aggregations (CSV)", agg_df, "deal_aggregations.csv")
    else:
        st.info("No operating metrics detected. Ensure headers are on the selected header row and match the specified order.")


if __name__ == "__main__":
    main()


