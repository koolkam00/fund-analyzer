from __future__ import annotations

import io
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Net Returns Benchmarking", layout="wide")


@st.cache_data(show_spinner=False)
def _template_excel() -> bytes:
    # Build a two-sheet template: Funds and Benchmarks
    funds = pd.DataFrame(
        {
            "Fund": ["Fund A", "Fund B"],
            "Fund Size": [500.0, 800.0],
            "Vintage Year": [2017, 2019],
            "Net IRR": [0.18, 0.12],
            "Net TVPI": [1.9, 1.5],
            "Net DPI": [1.1, 0.6],
        }
    )
    # Build per-row thresholds to avoid unequal length errors
    bm_rows = [
        {"Vintage Year": 2017, "Metric": "Net IRR", "Top5%": 0.30, "Upper Quartile": 0.20, "Median": 0.15, "Lower Quartile": 0.10},
        {"Vintage Year": 2017, "Metric": "Net TVPI", "Top5%": 2.50, "Upper Quartile": 2.00, "Median": 1.70, "Lower Quartile": 1.40},
        {"Vintage Year": 2017, "Metric": "Net DPI", "Top5%": 1.50, "Upper Quartile": 1.00, "Median": 0.70, "Lower Quartile": 0.50},
        {"Vintage Year": 2019, "Metric": "Net IRR", "Top5%": 0.28, "Upper Quartile": 0.18, "Median": 0.13, "Lower Quartile": 0.09},
        {"Vintage Year": 2019, "Metric": "Net TVPI", "Top5%": 2.30, "Upper Quartile": 1.90, "Median": 1.60, "Lower Quartile": 1.30},
        {"Vintage Year": 2019, "Metric": "Net DPI", "Top5%": 1.40, "Upper Quartile": 0.95, "Median": 0.65, "Lower Quartile": 0.45},
    ]
    benchmarks = pd.DataFrame(bm_rows)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        funds.to_excel(writer, index=False, sheet_name="Funds")
        benchmarks.to_excel(writer, index=False, sheet_name="Benchmarks")
    return bio.getvalue()


@st.cache_data(show_spinner=False)
def _read_excel(upload) -> Dict[str, pd.DataFrame]:
    if upload is None:
        return {}
    content = upload.getvalue()
    bio = io.BytesIO(content)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    return {sheet: xls.parse(sheet) for sheet in xls.sheet_names}


st.title("Net Returns Benchmarking")
st.caption("Upload net returns and vintage benchmarks, then classify funds into Top 5% and quartiles.")

# Template download
templ = _template_excel()
st.download_button(
    label="Download Excel Template",
    data=templ,
    file_name="net_returns_benchmarking_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

with st.sidebar:
    upload = st.file_uploader("Upload Net Returns & Benchmarks (.xlsx)", type=["xlsx"])  # type: ignore

sheets = _read_excel(upload)
if not sheets:
    st.info("Upload the template with your data to begin.")
    st.stop()

# Expect two sheets: Funds and Benchmarks
funds_sheet_name = "Funds" if "Funds" in sheets else list(sheets.keys())[0]
bench_sheet_name = "Benchmarks" if "Benchmarks" in sheets else (list(sheets.keys())[1] if len(sheets) > 1 else None)
if bench_sheet_name is None:
    st.error("Could not find a 'Benchmarks' sheet. Include a second sheet with benchmark thresholds.")
    st.stop()

df_funds = sheets[funds_sheet_name].copy()
df_bm = sheets[bench_sheet_name].copy()

# Normalize columns
def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

df_funds.columns = [_norm(c) for c in df_funds.columns]
df_bm.columns = [_norm(c) for c in df_bm.columns]

required_funds = {"fund", "fund_size", "vintage_year", "net_irr", "net_tvpi", "net_dpi"}
required_bm = {"vintage_year", "metric", "top5%", "upper_quartile", "median", "lower_quartile"}
if not required_funds.issubset(df_funds.columns):
    st.error(f"Funds sheet missing columns: {sorted(required_funds - set(df_funds.columns))}")
    st.stop()
if not required_bm.issubset(df_bm.columns):
    st.error(f"Benchmarks sheet missing columns: {sorted(required_bm - set(df_bm.columns))}")
    st.stop()

# Coerce types
for c in ["fund_size", "net_tvpi", "net_dpi"]:
    df_funds[c] = pd.to_numeric(df_funds[c], errors="coerce")
df_funds["net_irr"] = pd.to_numeric(df_funds["net_irr"], errors="coerce")
df_funds["vintage_year"] = pd.to_numeric(df_funds["vintage_year"], errors="coerce").astype("Int64")

df_bm["vintage_year"] = pd.to_numeric(df_bm["vintage_year"], errors="coerce").astype("Int64")
df_bm["metric"] = df_bm["metric"].astype(str).str.strip().str.lower()

# Map metric names to tokens used in funds
metric_map = {
    "net irr": "net_irr",
    "net_tvpi": "net_tvpi",
    "net tvpi": "net_tvpi",
    "net_dpi": "net_dpi",
    "net dpi": "net_dpi",
}
df_bm["metric"] = df_bm["metric"].map(lambda x: metric_map.get(x, x))

# Build thresholds per (vintage_year, metric)
key_cols = ["vintage_year", "metric"]
thresh_cols = ["top5%", "upper_quartile", "median", "lower_quartile"]
bm_valid = df_bm[key_cols + thresh_cols].dropna(subset=["vintage_year", "metric"])

def classify(value: float, row: pd.Series) -> str:
    if pd.isna(value):
        return "—"
    top5 = row.get("top5%")
    uq = row.get("upper_quartile")
    med = row.get("median")
    lq = row.get("lower_quartile")
    # Higher is better
    try:
        if pd.notna(top5) and value >= float(top5):
            return "Top 5%"
        if pd.notna(uq) and value >= float(uq):
            return "1st Quartile"
        if pd.notna(med) and value >= float(med):
            return "2nd Quartile"
        if pd.notna(lq) and value >= float(lq):
            return "3rd Quartile"
        return "4th Quartile"
    except Exception:
        return "—"

# Merge thresholds for each metric by vintage
def add_bucket(funds: pd.DataFrame, metric: str, label: str) -> pd.Series:
    left = funds[["vintage_year", metric]].copy()
    left = left.rename(columns={metric: "value"})
    right = bm_valid[bm_valid["metric"] == metric]
    merged = pd.merge(left, right, on="vintage_year", how="left")
    return merged.apply(lambda r: classify(r["value"], r), axis=1).rename(label)

df_out = df_funds.copy()
df_out["irr_bucket"] = add_bucket(df_out, "net_irr", "IRR Bucket")
df_out["tvpi_bucket"] = add_bucket(df_out, "net_tvpi", "TVPI Bucket")
df_out["dpi_bucket"] = add_bucket(df_out, "net_dpi", "DPI Bucket")

# Display table
show_cols = [
    "fund",
    "fund_size",
    "vintage_year",
    "net_irr",
    "IRR Bucket",
    "net_tvpi",
    "TVPI Bucket",
    "net_dpi",
    "DPI Bucket",
]
show_cols = [c for c in show_cols if c in df_out.columns]
tbl = df_out[show_cols].copy()

# Formatting
def _moic_fmt(v):
    try:
        return f"{float(v):.1f}x" if pd.notna(v) else "—"
    except Exception:
        return "—"
fmt = {"fund_size": "{:,.1f}", "net_dpi": "{:.1f}", "net_irr": "{:.1%}"}
sty = tbl.style.format(fmt, na_rep="—")
if "net_tvpi" in tbl.columns:
    sty = sty.format({"net_tvpi": _moic_fmt})
st.dataframe(sty, use_container_width=True)

# Summary counts by bucket
st.subheader("Summary Counts by Bucket")
def _bucket_counts(col: str) -> pd.DataFrame:
    if col not in df_out.columns:
        return pd.DataFrame()
    return (
        df_out[col]
        .value_counts(dropna=False)
        .rename_axis("Bucket")
        .reset_index(name="Count")
        .sort_values(by="Bucket")
    )
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**IRR**")
    st.dataframe(_bucket_counts("IRR Bucket"), use_container_width=True)
with c2:
    st.markdown("**TVPI**")
    st.dataframe(_bucket_counts("TVPI Bucket"), use_container_width=True)
with c3:
    st.markdown("**DPI**")
    st.dataframe(_bucket_counts("DPI Bucket"), use_container_width=True)


