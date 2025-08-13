from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="PME Benchmarking (KS-PME)", layout="wide")


@st.cache_data(show_spinner=False)
def _read_excel_or_csv(upload, header_row_index: int) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    header_zero = max(0, int(header_row_index) - 1)
    name = upload.name.lower()
    if name.endswith(".csv"):
        content = upload.getvalue()
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = content.decode("latin-1", errors="ignore")
        return pd.read_csv(io.StringIO(text), header=header_zero)
    else:
        content = upload.getvalue()
        bio = io.BytesIO(content)
        return pd.read_excel(bio, header=header_zero, engine="openpyxl")


@st.cache_data(show_spinner=False)
def _fetch_index_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data = yf.download(ticker, start=start.date(), end=end.date(), progress=False, auto_adjust=True)
    if data is None or data.empty:
        return pd.DataFrame(columns=["Date", "Close"])  # empty
    out = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _normalize_cf(df: pd.DataFrame) -> pd.DataFrame:
    # Expect: Date (A), Type (B), Value (C), Portfolio Company (D)
    cols = list(df.columns)
    out = pd.DataFrame()
    def _parse_date_col(s: pd.Series) -> pd.Series:
        # Native parse first
        d = pd.to_datetime(s, errors="coerce")
        numeric = pd.to_numeric(s, errors="coerce")
        is_num = numeric.notna().mean() > 0.5
        if is_num:
            # If native parse produced many 1969-1971 dates (epoch ns mis-parse), switch to Excel serial logic
            years = d.dt.year.where(d.notna())
            many_epoch = (d.notna().mean() > 0.5) and (years.between(1969, 1971).mean() > 0.5)
            looks_excel_range = numeric.between(20000, 60000).mean() > 0.5
            if many_epoch or looks_excel_range:
                origin = pd.Timestamp("1899-12-30")
                return origin + pd.to_timedelta(numeric, unit="D")
        # If many NaT and mostly digits, also try Excel serial
        if (d.isna().mean() > 0.5) and (s.astype(str).str.fullmatch(r"\d+").mean() > 0.5):
            origin = pd.Timestamp("1899-12-30")
            return origin + pd.to_timedelta(numeric, unit="D")
        return d
    if len(cols) >= 1:
        out["date"] = _parse_date_col(df.iloc[:, 0])
    if len(cols) >= 2:
        out["type"] = df.iloc[:, 1].astype(str).str.strip().str.lower()
    if len(cols) >= 3:
        out["amount"] = pd.to_numeric(
            df.iloc[:, 2].astype(str).str.replace(r"[,$%]", "", regex=True).str.replace(r"^\((.*)\)$", r"-\\1", regex=True),
            errors="coerce",
        )
    if len(cols) >= 4:
        out["portfolio_company"] = df.iloc[:, 3].astype(str).str.strip()
    else:
        out["portfolio_company"] = "(Unknown)"
    # Map type to standard categories
    out["cat"] = np.select(
        [out["type"].str.contains("capital call|contribution|call", case=False, na=False),
         out["type"].str.contains("distribution|proceed|dist", case=False, na=False),
         out["type"].str.contains("nav|valuation|fair value|current", case=False, na=False)],
        ["call", "dist", "nav"],
        default="other",
    )
    out = out.dropna(subset=["date", "amount"])  # keep necessary
    return out


def _ks_pme_index_multiple(cf: pd.DataFrame, index_df: pd.DataFrame) -> float | None:
    # Kaplan–Schoar PME = (PV of distributions and NAV discounted by index) / (PV of contributions discounted by index)
    if cf.empty or index_df.empty:
        return None
    idx_series = index_df.set_index("date")["close"].sort_index()
    # align index level for each cash flow date (forward/backfill to nearest available trading day)
    cf = cf.copy()
    cf["date"] = pd.to_datetime(cf["date"], errors="coerce")
    cf = cf.dropna(subset=["date"]).sort_values("date")
    cf["index_level"] = idx_series.reindex(cf["date"]).ffill().bfill().values
    if cf["index_level"].isna().all():
        return None
    # For KS-PME, scale cash flows by index level at the date relative to the last date level
    last_level = float(idx_series.iloc[-1]) if len(idx_series) else np.nan
    if not np.isfinite(last_level) or last_level <= 0:
        return None
    cf["scale"] = last_level / cf["index_level"].replace(0, np.nan)
    # Separate contributions (calls, negative outflow for investor) and distributions/NAV (positive inflow)
    calls = cf[cf["cat"] == "call"]["amount"].fillna(0) * cf[cf["cat"] == "call"]["scale"].fillna(0)
    dists = cf[cf["cat"] == "dist"]["amount"].fillna(0) * cf[cf["cat"] == "dist"]["scale"].fillna(0)
    navs = cf[cf["cat"] == "nav"]["amount"].fillna(0) * cf[cf["cat"] == "nav"]["scale"].fillna(0)
    denom = float(calls.sum())
    numer = float(dists.sum() + navs.sum())
    if denom == 0:
        return None
    return numer / denom


st.title("Benchmarking: KS-PME vs Public Indices")
st.caption("Upload gross cash flows by deal to compute KS-PME against S&P 500 / Nasdaq / Dow Jones.")

def _download_template_button():
    example = pd.DataFrame(
        [
            {"Date": pd.Timestamp("2020-01-15"), "Type": "Capital Call", "Value": 25.0, "Portfolio Company": "Alpha"},
            {"Date": pd.Timestamp("2022-07-10"), "Type": "Distribution", "Value": 10.0, "Portfolio Company": "Alpha"},
            {"Date": pd.Timestamp("2024-12-31"), "Type": "NAV", "Value": 20.0, "Portfolio Company": "Alpha"},
        ]
    )
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        example.to_excel(writer, index=False, sheet_name="CashFlows")
    bio.seek(0)
    st.download_button(
        label="Download cash flow template (Excel)",
        data=bio.getvalue(),
        file_name="cash_flows_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

_download_template_button()

with st.sidebar:
    upload = st.file_uploader("Upload cash flows (.xlsx or .csv) — columns: Date | Type | Value | Portfolio Company", type=["xlsx", "csv"])  # type: ignore
    header_row_index = st.number_input("Header row (1-based)", min_value=1, max_value=100, value=1, step=1)
    index_choice = st.selectbox("Benchmark index", ["S&P 500 (\u005EGSPC)", "Nasdaq (\u005EIXIC)", "Dow Jones (\u005EDJI)"])

raw_df = _read_excel_or_csv(upload, header_row_index)
if raw_df.empty:
    st.info("Upload a cash flow file to begin.")
    st.stop()

cf = _normalize_cf(raw_df)
if cf.empty:
    st.error("No valid rows found. Ensure Date, Type, Value, and Portfolio Company are provided.")
    st.stop()

min_dt = pd.to_datetime(cf["date"], errors="coerce").min()
max_dt = pd.to_datetime(cf["date"], errors="coerce").max()
if pd.isna(min_dt) or pd.isna(max_dt):
    st.error("Could not determine date range from the cash flows.")
    st.stop()

ticker_map = {
    "S&P 500 (\u005EGSPC)": "^GSPC",
    "Nasdaq (\u005EIXIC)": "^IXIC",
    "Dow Jones (\u005EDJI)": "^DJI",
}
ticker = ticker_map.get(index_choice, "^GSPC")
index_df = _fetch_index_history(ticker, min_dt - pd.Timedelta(days=7), max_dt + pd.Timedelta(days=7))
if index_df.empty:
    st.error("Failed to load index history. Try another index or check the date range.")
    st.stop()

st.subheader("Per-Deal KS-PME")
rows: List[Dict[str, object]] = []
for deal, g in cf.groupby("portfolio_company"):
    kspme = _ks_pme_index_multiple(g, index_df)
    rows.append({
        "Portfolio Company": deal,
        "KS-PME": kspme,
        "First CF": pd.to_datetime(g["date"]).min(),
        "Last CF": pd.to_datetime(g["date"]).max(),
        "Total Calls": float(g.loc[g["cat"] == "call", "amount"].sum()),
        "Total Dists": float(g.loc[g["cat"] == "dist", "amount"].sum()),
        "Last NAV": float(g.loc[g["cat"] == "nav", "amount"].tail(1).sum()),
    })
out = pd.DataFrame(rows)
if not out.empty:
    # Ensure numeric for formatter and handle None
    if "KS-PME" in out.columns:
        out["KS-PME"] = pd.to_numeric(out["KS-PME"], errors="coerce")
    out = out.sort_values("KS-PME", ascending=False, na_position="last")
    # Format
    fmt = {
        "KS-PME": "{:.2f}",
        "Total Calls": "{:,.1f}",
        "Total Dists": "{:,.1f}",
        "Last NAV": "{:,.1f}",
    }
    for dc in ["First CF", "Last CF"]:
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce").dt.strftime("%b %Y")
    st.dataframe(out.style.format(fmt, na_rep="—"), use_container_width=True)

    # Portfolio summary
    st.subheader("Portfolio Summary")
    tot_calls = float(cf.loc[cf["cat"] == "call", "amount"].sum())
    tot_dists = float(cf.loc[cf["cat"] == "dist", "amount"].sum())
    last_nav = float(cf.loc[cf["cat"] == "nav", "amount"].tail(1).sum())
    st.write(f"Total Calls: ${tot_calls:,.1f} | Total Distributions: ${tot_dists:,.1f} | Last NAV: ${last_nav:,.1f}")
else:
    st.info("No per-deal results to show.")


