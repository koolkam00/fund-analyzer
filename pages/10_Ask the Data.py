from __future__ import annotations

import io
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import os
import requests
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr


st.set_page_config(page_title="Ask Your Data (LLM)", layout="wide")


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


st.title("Ask Your Data (LLM)")
st.caption("Query your uploaded Portfolio Metrics using a language model. The model sees a compact summary + sample rows.")

with st.sidebar:
    st.markdown("**Data source**")
    use_session = st.toggle("Use existing uploaded Portfolio Metrics from other pages", value=True)
    upload = None
    header_row_index = int(st.session_state.get("header_row_index", 2))
    if not use_session:
        upload = st.file_uploader("Upload Portfolio Metrics (.xlsx or .csv)", type=["xlsx", "csv"])  # type: ignore
        header_row_index = st.number_input("Header row (1-based)", min_value=1, max_value=100, value=header_row_index, step=1)

    st.markdown("**LLM provider (OpenRouter)**")
    # Pull default key from secrets or environment without hardcoding
    default_key = ""
    try:
        if hasattr(st, "secrets") and isinstance(st.secrets, dict):
            if "OPENROUTER_API_KEY" in st.secrets:
                default_key = st.secrets.get("OPENROUTER_API_KEY", "")
            elif "openrouter" in st.secrets and isinstance(st.secrets["openrouter"], dict):
                default_key = st.secrets["openrouter"].get("api_key", "")
    except Exception:
        default_key = ""
    if not default_key:
        default_key = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_key = st.text_input("OpenRouter API Key", value=default_key, type="password")
    # Normalize pasted key (avoid leading/trailing spaces/newlines)
    openrouter_key = (openrouter_key or "").strip()
    # Fallback to default (hardcoded/secrets/env) if input is empty
    openrouter_key = openrouter_key or default_key
    model = st.selectbox(
        "Model",
        [
            "google/gemini-2.5-flash",
            "mistralai/mixtral-8x7b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemma-2-9b-it",
        ],
        index=0,
    )


def _load_ops_df() -> pd.DataFrame:
    # Load from session or newly uploaded
    sheets: Dict[str, pd.DataFrame] = {}
    if use_session:
        sheets = st.session_state.get("sheets", {})
    else:
        sheets = _read_excel_or_csv(upload, header_row_index)
        if sheets:
            st.session_state["sheets"] = sheets
            st.session_state["selected_sheet"] = list(sheets.keys())[0]
            st.session_state["header_row_index"] = header_row_index
    if not sheets:
        return pd.DataFrame()
    sheet_name = st.session_state.get("selected_sheet", list(sheets.keys())[0])
    df = sheets.get(sheet_name, pd.DataFrame())
    ops_df_raw, _ = extract_operational_by_template_order(df, list(df.columns))
    if ops_df_raw.empty:
        return pd.DataFrame()
    return add_growth_and_cagr(ops_df_raw)


def _build_context(df: pd.DataFrame, max_rows: int = 200, pme_raw: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    # Keep a curated subset of columns that are most relevant
    preferred_cols = [
        "portfolio_company",
        "fund_name",
        "sector",
        "geography",
        "status",
        "invest_date",
        "exit_date",
        "invested",
        "proceeds",
        "current_value",
        "gross_moic",
        "gross_irr",
        "entry_revenue",
        "exit_revenue",
        "entry_ebitda",
        "exit_ebitda",
        "entry_net_debt",
        "exit_net_debt",
        "entry_tev",
        "exit_tev",
        "holding_years",
        "revenue_cagr",
        "ebitda_cagr",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    view = df[cols].copy()
    # Coerce datetimes to string for serialization
    for dc in ["invest_date", "exit_date"]:
        if dc in view.columns:
            view[dc] = pd.to_datetime(view[dc], errors="coerce").dt.strftime("%b %Y")
    # Basic summary for numeric columns
    num_cols = view.select_dtypes(include=["number"]).columns.tolist()
    summary: Dict[str, Dict[str, float]] = {}
    for c in num_cols:
        s = pd.to_numeric(view[c], errors="coerce")
        if s.notna().any():
            summary[c] = {
                "count": float(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)) if s.count() > 1 else np.nan,
                "min": float(s.min()),
                "max": float(s.max()),
            }
    sample = view.head(max_rows).fillna("")
    ctx: Dict[str, object] = {
        "ops_columns": cols,
        "ops_summary": summary,
        "ops_rows": json.loads(sample.to_json(orient="records")),
        "ops_row_count_total": int(len(view)),
    }
    # Include PME cash flow raw data if available
    if pme_raw is not None and not pme_raw.empty:
        # Coerce dates to string for serialization to avoid NaT issues
        pme_ser = pme_raw.copy()
        for c in pme_ser.columns:
            if "date" in str(c).lower():
                pme_ser[c] = pd.to_datetime(pme_ser[c], errors="coerce").dt.strftime("%Y-%m-%d")
        ctx["pme_columns"] = [str(c) for c in pme_ser.columns]
        ctx["pme_rows"] = json.loads(pme_ser.fillna("").to_json(orient="records"))
        ctx["pme_row_count_total"] = int(len(pme_ser))
    return ctx


def _openrouter_chat(api_key: str, model: str, system_prompt: str, user_prompt: str) -> Optional[str]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Recommended by OpenRouter for routing/limits; safe defaults for local/Cloud use
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "http://localhost"),
        "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "PE Fund Analyzer"),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"Request failed: {e}"


ops_df = _load_ops_df()
if ops_df.empty:
    st.info("Upload data on the main app page or here, then return to this page.")
    st.stop()

st.success(f"Data loaded. Rows: {len(ops_df):,}")

all_rows = len(ops_df)
ctx = _build_context(ops_df, max_rows=all_rows, pme_raw=None)
with st.expander("Preview: sample rows used for the model context"):
    preview = {
        "ops_row_count_total": ctx.get("ops_row_count_total"),
        "ops_columns": ctx.get("ops_columns"),
        "ops_sample_rows": (ctx.get("ops_rows") or [])[:10],
    }
    st.json(preview)

st.subheader("Ask a question about your data")
question = st.text_area("Question", placeholder="Examples: Which fund has the highest TVPI? What is the median EBITDA growth by sector?", height=100)
btn = st.button("Generate answer", type="primary")

if btn:
    if not openrouter_key:
        st.error("Provide an OpenRouter API key in the sidebar or st.secrets.")
        st.stop()

    system_prompt = (
        "You are a private equity data analyst. You are given a compact summary and a sample of a portfolio metrics table. "
        "Answer the user's question using ONLY this data. If the answer is ambiguous, explain clearly what is missing. "
        "Prefer numerical summaries (counts, sums, averages, medians) and include units (x for MOIC, % for IRR)."
    )
    user_prompt = (
        "Operational metrics summary (JSON):\n"
        + json.dumps({"columns": ctx.get("ops_columns"), "summary": ctx.get("ops_summary")}, ensure_ascii=False)
        + "\n\nOperational metrics rows (JSON records):\n"
        + json.dumps(ctx.get("ops_rows"), ensure_ascii=False)
    )
    user_prompt += f"\n\nQuestion: {question}"

    with st.spinner("Asking the model..."):
        answer = _openrouter_chat(openrouter_key, model, system_prompt, user_prompt)
    st.markdown("**Answer**")
    st.write(answer if answer else "(No response)")


