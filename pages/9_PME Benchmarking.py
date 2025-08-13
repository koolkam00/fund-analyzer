from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
try:
    import numpy_financial as npf
except Exception:  # pragma: no cover
    npf = None
try:
    from pyxirr import xirr as px_xirr
except Exception:  # pragma: no cover
    px_xirr = None


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
        out["portfolio_company"] = (
            df.iloc[:, 3]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
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
    # Normalize signs: calls negative, dists/nav positive
    out.loc[out["cat"] == "call", "amount"] = -out.loc[out["cat"] == "call", "amount"].abs()
    out.loc[out["cat"].isin(["dist", "nav"]), "amount"] = out.loc[out["cat"].isin(["dist", "nav"]), "amount"].abs()
    # Keep only recognized categories
    out = out[out["cat"].isin(["call", "dist", "nav"])].copy()
    # Canonicalize portfolio company for stable grouping
    out["portfolio_company"] = out["portfolio_company"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
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
    # Separate contributions (calls negative) and distributions; include ONLY last NAV
    calls_scaled = cf.loc[cf["cat"] == "call", ["amount", "scale"]]
    calls_scaled = (calls_scaled["amount"].fillna(0) * calls_scaled["scale"].fillna(0))
    dists_scaled = cf.loc[cf["cat"] == "dist", ["amount", "scale"]]
    dists_scaled = (dists_scaled["amount"].fillna(0) * dists_scaled["scale"].fillna(0))
    nav_scaled = 0.0
    nav_rows = cf.loc[cf["cat"] == "nav"].sort_values("date")
    if not nav_rows.empty:
        last_nav_row = nav_rows.iloc[-1]
        nav_amt = float(pd.to_numeric(last_nav_row.get("amount"), errors="coerce") or 0.0)
        nav_scl = float(pd.to_numeric(last_nav_row.get("scale"), errors="coerce") or 0.0)
        nav_scaled = nav_amt * nav_scl
    # Denominator: positive magnitude of scaled calls
    denom = float(-calls_scaled.sum())
    numer = float(dists_scaled.sum() + nav_scaled)
    if denom == 0:
        return None
    return numer / denom


def _xnpv(rate: float, dates: list[pd.Timestamp], amounts: list[float]) -> float:
    if rate <= -0.999999:
        return np.inf
    t0 = min(dates)
    total = 0.0
    for d, a in zip(dates, amounts):
        days = (d - t0).days
        total += a / ((1 + rate) ** (days / 365.2425))
    return total


def _xirr_newton(dates: list[pd.Timestamp], amounts: list[float]) -> float | None:
    if not dates or not amounts or len(dates) != len(amounts):
        return None
    if all(a == 0 for a in amounts):
        return None
    # Require at least one negative and one positive cash flow
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        return None
    # Newton-Raphson
    rate = 0.10
    for _ in range(100):
        f = _xnpv(rate, dates, amounts)
        h = 1e-6
        f1 = _xnpv(rate + h, dates, amounts)
        d = (f1 - f) / h
        if not np.isfinite(d) or abs(d) < 1e-12:
            break
        step = f / d
        # Dampen step to improve stability
        new_rate = rate - max(min(step, 1.0), -1.0)
        if not np.isfinite(new_rate) or new_rate <= -0.999999:
            new_rate = (rate + max(-0.9, min(1.0, new_rate))) / 2
        if abs(new_rate - rate) < 1e-9:
            rate = new_rate
            break
        rate = new_rate
    return float(rate) if np.isfinite(rate) and rate > -0.999999 else None


def _xirr_bracket(dates: list[pd.Timestamp], amounts: list[float]) -> float | None:
    # Simple bracketing + bisection
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        return None
    low, high = -0.9, 5.0
    f_low = _xnpv(low, dates, amounts)
    f_high = _xnpv(high, dates, amounts)
    # Expand high if same sign
    iters = 0
    while f_low * f_high > 0 and high < 100 and iters < 20:
        high *= 2
        f_high = _xnpv(high, dates, amounts)
        iters += 1
    if f_low * f_high > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2
        f_mid = _xnpv(mid, dates, amounts)
        if abs(f_mid) < 1e-9:
            return float(mid)
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return float((low + high) / 2)


def _xirr(dates: list[pd.Timestamp], amounts: list[float]) -> float | None:
    # Try pyxirr first (robust)
    try:
        if px_xirr is not None:
            val = px_xirr([pd.to_datetime(d).date() for d in dates], amounts)
            if np.isfinite(val):
                return float(val)
    except Exception:
        pass
    # Try numpy_financial if available
    try:
        if npf is not None:
            val = npf.xirr(amounts, [pd.to_datetime(d).date() for d in dates])
            if np.isfinite(val):
                return float(val)
    except Exception:
        pass
    # Fallbacks
    val = _xirr_newton(dates, amounts)
    if val is not None:
        return val
    return _xirr_bracket(dates, amounts)


st.title("Benchmarking: KS-PME vs Public Indices")
st.caption("Upload gross cash flows by deal to compute KS-PME against S&P 500 / Nasdaq / Dow Jones.")

st.markdown(
    """
    KS-PME compares a deal's performance to a public index by scaling all cash flows by index levels over time.
    - KS-PME > 1.0: outperformed the selected index
    - KS-PME < 1.0: underperformed the selected index
    """
)

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
for deal, g in cf.groupby("portfolio_company", sort=False):
    kspme = _ks_pme_index_multiple(g, index_df)
    # Compute gross deal IRR from aggregated dated flows.
    g_sorted = g.sort_values("date").copy()
    # Aggregate per date: contributions negative, distributions positive; add NAV only on last date
    last_dt = g_sorted["date"].max()
    agg = (
        g_sorted.assign(
            flow=lambda d: np.where(d["cat"].eq("nav"), 0.0, d["amount"].astype(float))
        )
        .groupby("date", as_index=False)["flow"].sum()
    )
    nav_last = float(g_sorted.loc[(g_sorted["cat"] == "nav") & (g_sorted["date"] == last_dt), "amount"].sum()) if pd.notna(last_dt) else 0.0
    if pd.notna(last_dt) and nav_last != 0:
        # add NAV on last date
        idx_last = agg.index[agg["date"] == last_dt]
        if len(idx_last):
            agg.loc[idx_last, "flow"] = agg.loc[idx_last, "flow"].astype(float) + nav_last
        else:
            agg = pd.concat([agg, pd.DataFrame({"date": [last_dt], "flow": [nav_last]})], ignore_index=True)
    agg = agg.sort_values("date")
    irr_dates = [pd.to_datetime(d).date() for d in agg["date"].tolist()]
    irr_amounts = [float(x) for x in agg["flow"].tolist()]
    # Require at least one negative call and at least one positive (dist or last NAV) for XIRR
    # If no distributions but there is a NAV, include only the last NAV
    has_neg = any(a < 0 for a in irr_amounts)
    has_pos = any(a > 0 for a in irr_amounts)
    if not has_pos:
        # try to use only last NAV as positive terminal
        nav_series = g_sorted.loc[g_sorted["cat"] == "nav", ["date", "amount"]].sort_values("date")
        if not nav_series.empty:
            last_date = nav_series.iloc[-1]["date"]
            # replace any earlier NAV to 0 and keep only last NAV on its date
            agg["flow"] = np.where(agg["date"] == last_date, agg["flow"], np.where(agg["flow"] > 0, 0.0, agg["flow"]))
            irr_amounts = [float(x) for x in agg["flow"].tolist()]
            has_pos = any(a > 0 for a in irr_amounts)
    deal_irr = _xirr(irr_dates, irr_amounts) if (has_neg and has_pos) else None

    # Index-equivalent IRR using scaled flows to last index level (exclude NAV except last date)
    idx_series = index_df.set_index("date")["close"].sort_index()
    last_level = float(idx_series.iloc[-1]) if len(idx_series) else np.nan
    if np.isfinite(last_level):
        # Build scaled aggregated flows aligned to agg dates
        scaled = []
        for d, amt in zip(agg["date"], agg["flow"]):
            if len(idx_series):
                upto = idx_series.loc[:d]
                idx_val = float(upto.iloc[-1]) if not upto.empty else np.nan
            else:
                idx_val = np.nan
            scale = last_level / idx_val if np.isfinite(idx_val) and (idx_val > 0) else np.nan
            scaled.append(float(amt) * (scale if np.isfinite(scale) else 0.0))
        # Index IRR requires both negative and positive
        if any(a < 0 for a in scaled) and any(a > 0 for a in scaled):
            idx_irr = _xirr(irr_dates, scaled)
        else:
            idx_irr = None
    else:
        idx_irr = None

    # MOIC and index-equivalent MOIC
    # Positive magnitude for total calls
    calls_sum = float(-g.loc[g["cat"] == "call", "amount"].sum())
    dists_sum = float(g.loc[g["cat"] == "dist", "amount"].sum())
    nav_last = float(g.loc[g["cat"] == "nav", "amount"].tail(1).sum())
    moic = (dists_sum + nav_last) / calls_sum if calls_sum > 0 else np.nan
    moic_pme = float(kspme) if kspme is not None else np.nan
    rows.append({
        "Portfolio Company": deal,
        "KS-PME": kspme,
        "Deal IRR": deal_irr,
        "Index IRR": idx_irr,
        "IRR Alpha": (deal_irr - idx_irr) if (deal_irr is not None and idx_irr is not None) else np.nan,
        "MOIC": moic,
        "PME Multiple": moic_pme,
        "MOIC Alpha": (moic - moic_pme) if (pd.notna(moic) and pd.notna(moic_pme)) else np.nan,
        "First CF": pd.to_datetime(g["date"]).min(),
        "Last CF": pd.to_datetime(g["date"]).max(),
        "Total Calls": calls_sum,
        "Total Dists": dists_sum,
        "Last NAV": nav_last,
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
        "PME Multiple": "{:.2f}",
        "Deal IRR": "{:.1%}",
        "Index IRR": "{:.1%}",
        "IRR Alpha": "{:.1%}",
        "MOIC": "{:.2f}",
        "MOIC Alpha": "{:.2f}",
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

    # Visualization: KS-PME by company
    import plotly.express as px
    import plotly.graph_objects as go
    st.subheader("KS-PME by Portfolio Company")
    viz = out.copy()
    viz["Perf vs Index"] = np.where(pd.to_numeric(viz["KS-PME"], errors="coerce") >= 1.0, "Outperform", "Underperform")
    fig_bar = px.bar(
        viz.dropna(subset=["KS-PME"]),
        x="Portfolio Company",
        y="KS-PME",
        color="Perf vs Index",
        color_discrete_map={"Outperform": "#2ca02c", "Underperform": "#d62728"},
        category_orders={"Portfolio Company": viz["Portfolio Company"].tolist()},
    )
    fig_bar.add_hline(y=1.0, line_dash="dash", line_color="#7f7f7f")
    fig_bar.update_layout(xaxis_tickangle=-45, yaxis_title="KS-PME", legend_title_text="Performance")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Distribution of KS-PME
    st.caption("Distribution of KS-PME")
    fig_hist = px.histogram(viz.dropna(subset=["KS-PME"]), x="KS-PME", nbins=25)
    fig_hist.add_vline(x=1.0, line_dash="dash", line_color="#7f7f7f")
    fig_hist.update_layout(yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

    # IRR Alpha vs PME Multiple scatter
    st.subheader("IRR Alpha vs PME Multiple")
    scatter_df = viz.dropna(subset=["IRR Alpha", "PME Multiple"]).copy()
    if not scatter_df.empty:
        fig_sc = px.scatter(
            scatter_df,
            x="PME Multiple",
            y="IRR Alpha",
            hover_name="Portfolio Company",
        )
        fig_sc.add_vline(x=1.0, line_dash="dash", line_color="#7f7f7f")
        fig_sc.add_hline(y=0.0, line_dash="dash", line_color="#7f7f7f")
        fig_sc.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_sc, use_container_width=True)
else:
    st.info("No per-deal results to show.")


