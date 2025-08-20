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
def _fetch_index_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data = yf.download(ticker, start=start.date(), end=end.date(), progress=False, auto_adjust=True)
    if data is None or data.empty:
        return pd.DataFrame(columns=["Date", "Close"])  # empty
    out = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _normalize_cf(df: pd.DataFrame) -> pd.DataFrame:
    # Expect: Date (A), Type (B), Value (C), Portfolio Company (D), Fund (E)
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
    if len(cols) >= 5:
        out["fund_name"] = (
            df.iloc[:, 4]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    else:
        out["fund_name"] = "(Unknown Fund)"
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
    out["fund_name"] = out["fund_name"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    return out


def _ks_pme_index_multiple(cf: pd.DataFrame, index_df: pd.DataFrame) -> float | None:
    # Kaplan–Schoar PME = (PV of distributions and NAV discounted by index) / (PV of contributions discounted by index)
    if cf.empty or index_df.empty:
        return None
    # Align index level at-or-before each CF date (same logic as diagnostics)
    cf = cf.copy()
    cf["date"] = pd.to_datetime(cf["date"], errors="coerce")
    cf = cf.dropna(subset=["date"]).sort_values("date")
    idx_series = index_df.set_index("date")["close"].sort_index()
    idx_last = float(idx_series.iloc[-1]) if len(idx_series) else np.nan
    if not np.isfinite(idx_last) or idx_last <= 0:
        return None
    idx_levels = []
    for d in cf["date"].tolist():
        upto = idx_series.loc[:d]
        lvl = float(upto.iloc[-1]) if not upto.empty else (float(idx_series.iloc[0]) if len(idx_series) else np.nan)
        idx_levels.append(lvl)
    cf["index_level"] = idx_levels
    if cf["index_level"].isna().all():
        return None
    cf["scale"] = idx_last / cf["index_level"].replace(0, np.nan)
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
st.caption("Uses the 'PME Cash Flows' sheet from the uploaded master workbook to compute KS-PME.")

from data_loader import ensure_workbook_loaded

with st.sidebar:
    sheets, _, _, _ = ensure_workbook_loaded()
    index_options = [
        "S&P 500 (\u005EGSPC)",
        "Nasdaq Composite (\u005EIXIC)",
        "Dow Jones (\u005EDJI)",
        "Russell 2000 (\u005ERUT)",
        "Russell 3000 (\u005ERUA)",
        "S&P Comm Services (XLC)",
        "S&P Consumer Discretionary (XLY)",
        "S&P Consumer Staples (XLP)",
        "S&P Energy (XLE)",
        "S&P Financials (XLF)",
        "S&P Health Care (XLV)",
        "S&P Industrials (XLI)",
        "S&P Materials (XLB)",
        "S&P Real Estate (XLRE)",
        "S&P Technology (XLK)",
        "S&P Utilities (XLU)",
    ]
    prev_idx = st.session_state.get("pme_index_choice", index_options[0])
    index_choice = st.selectbox(
        "Benchmark index",
        index_options,
        index=(index_options.index(prev_idx) if prev_idx in index_options else 0),
        key="pme_index_choice",
    )

if not sheets:
    st.info("Upload the master workbook (with 'PME Cash Flows' sheet) using the sidebar.")
    st.stop()

# Locate PME Cash Flows sheet
pme_sheet_name = None
for name in sheets.keys():
    if str(name).strip().lower() in {"pme cash flows", "pme_cash_flows", "pme", "cash flows", "cashflows"}:
        pme_sheet_name = name
        break
if pme_sheet_name is None:
    # Fallback to last sheet (template uses 'PME Cash Flows' as 4th)
    pme_sheet_name = list(sheets.keys())[-1]
raw_df = sheets.get(pme_sheet_name, pd.DataFrame())

cf = _normalize_cf(raw_df)
if cf.empty:
    st.error("No valid rows found in 'PME Cash Flows'. Ensure Date, Type, Value, Portfolio Company, and Fund are provided.")
    st.stop()

min_dt = pd.to_datetime(cf["date"], errors="coerce").min()
max_dt = pd.to_datetime(cf["date"], errors="coerce").max()
if pd.isna(min_dt) or pd.isna(max_dt):
    st.error("Could not determine date range from the cash flows.")
    st.stop()

ticker_map = {
    "S&P 500 (\u005EGSPC)": "^GSPC",
    "Nasdaq Composite (\u005EIXIC)": "^IXIC",
    "Dow Jones (\u005EDJI)": "^DJI",
    "Russell 2000 (\u005ERUT)": "^RUT",
    "Russell 3000 (\u005ERUA)": "^RUA",
    "S&P Comm Services (XLC)": "XLC",
    "S&P Consumer Discretionary (XLY)": "XLY",
    "S&P Consumer Staples (XLP)": "XLP",
    "S&P Energy (XLE)": "XLE",
    "S&P Financials (XLF)": "XLF",
    "S&P Health Care (XLV)": "XLV",
    "S&P Industrials (XLI)": "XLI",
    "S&P Materials (XLB)": "XLB",
    "S&P Real Estate (XLRE)": "XLRE",
    "S&P Technology (XLK)": "XLK",
    "S&P Utilities (XLU)": "XLU",
}
ticker = ticker_map.get(index_choice, "^GSPC")
index_df = _fetch_index_history(ticker, min_dt - pd.Timedelta(days=7), max_dt + pd.Timedelta(days=7))
if index_df.empty:
    st.error("Failed to load index history. Try another index or check the date range.")
    st.stop()

st.subheader("Per-Deal KS-PME")
rows: List[Dict[str, object]] = []
for (deal, fund), g in cf.groupby(["portfolio_company", "fund_name"], sort=False):
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

    # Index-equivalent IRR removed per request
    idx_irr = None

    # MOIC and index-equivalent MOIC
    # Positive magnitude for total calls
    calls_sum = float(-g.loc[g["cat"] == "call", "amount"].sum())
    dists_sum = float(g.loc[g["cat"] == "dist", "amount"].sum())
    nav_last = float(g.loc[g["cat"] == "nav", "amount"].tail(1).sum())
    moic = (dists_sum + nav_last) / calls_sum if calls_sum > 0 else np.nan
    moic_pme = float(kspme) if kspme is not None else np.nan
    # Index MOIC: invest $1 at first CF date, hold to last CF date (at-or-before levels)
    idx_series = index_df.set_index("date")["close"].sort_index()
    first_dt = pd.to_datetime(g["date"]).min()
    last_dt = pd.to_datetime(g["date"]).max()
    if pd.notna(first_dt) and pd.notna(last_dt) and len(idx_series):
        upto_first = idx_series.loc[:first_dt]
        upto_last = idx_series.loc[:last_dt]
        idx_first = float(upto_first.iloc[-1]) if not upto_first.empty else np.nan
        idx_last = float(upto_last.iloc[-1]) if not upto_last.empty else np.nan
        index_moic = (idx_last / idx_first) if (np.isfinite(idx_first) and np.isfinite(idx_last) and idx_first > 0) else np.nan
    else:
        index_moic = np.nan
    first_cf = pd.to_datetime(g["date"]).min()
    last_cf = pd.to_datetime(g["date"]).max()
    hold_years = float(((last_cf - first_cf).days) / 365.2425) if pd.notna(first_cf) and pd.notna(last_cf) else np.nan
    realized_moic = (dists_sum / calls_sum) if calls_sum > 0 else np.nan
    unrealized_moic = (nav_last / calls_sum) if calls_sum > 0 else np.nan
    # Deal status from realized/unrealized composition
    status_str = "—"
    if calls_sum > 0:
        if nav_last <= 0 and dists_sum > 0:
            status_str = "Fully Realized"
        elif nav_last > 0 and dists_sum > 0:
            status_str = "Partially Realized"
        elif nav_last > 0 and (dists_sum <= 0 or pd.isna(dists_sum)):
            status_str = "Unrealized"
        else:
            status_str = "Unrealized"
    rows.append({
        "Portfolio Company": deal,
        "Fund": fund,
        "Status": status_str,
        "First Cash Flow": first_cf,
        "Last Cash Flow": last_cf,
        "Hold Period (yrs)": hold_years,
        "KS-PME": kspme,
        "Realized MOIC": realized_moic,
        "Unrealized MOIC": unrealized_moic,
        "Total MOIC": moic,
        "Index MOIC": index_moic,
        "IRR": deal_irr,
        "Invested Capital": calls_sum,
        "Realized Capital": dists_sum,
        "Unrealized Capital": nav_last,
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
        "Realized MOIC": "{:.2f}",
        "Unrealized MOIC": "{:.2f}",
        "Total MOIC": "{:.2f}",
        "Index MOIC": "{:.2f}",
        "IRR": "{:.1%}",
        "Hold Period (yrs)": "{:.1f}",
        "Invested Capital": "{:,.1f}",
        "Realized Capital": "{:,.1f}",
        "Unrealized Capital": "{:,.1f}",
    }
    for dc in ["First Cash Flow", "Last Cash Flow"]:
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce").dt.strftime("%b %Y")
    # Reorder columns as requested
    desired_order = [
        "Portfolio Company",
        "Fund",
        "Status",
        "First Cash Flow",
        "Last Cash Flow",
        "Hold Period (yrs)",
        "KS-PME",
        "Realized MOIC",
        "Unrealized MOIC",
        "Total MOIC",
        "Index MOIC",
        "IRR",
        "Invested Capital",
        "Realized Capital",
        "Unrealized Capital",
    ]
    show_cols = [c for c in desired_order if c in out.columns]
    # Per-deal table removed per request

    # By-fund tables and summaries
    st.subheader("By Fund")
    if "Fund" in out.columns and "fund_name" in cf.columns:
        for fund in sorted(out["Fund"].dropna().unique().tolist()):
            st.markdown(f"**{fund}**")
            of = out[out["Fund"] == fund]
            st.dataframe(of[show_cols].style.format(fmt, na_rep="—"), use_container_width=True)
            # Fund summary
            cf_f = cf[cf["fund_name"] == fund]
            # Segmented summaries by status
            def _seg_summary(df_deals: pd.DataFrame, cf_scope: pd.DataFrame) -> tuple[str, str, str]:
                if df_deals.empty:
                    return "—", "—", "—"
                deals_set = set(df_deals["Portfolio Company"].astype(str))
                cf_seg = cf_scope[cf_scope["portfolio_company"].isin(deals_set)]
                # KS-PME on flows subset
                ksp = _ks_pme_index_multiple(cf_seg, index_df)
                # MOIC
                calls = float(-cf_seg.loc[cf_seg["cat"] == "call", "amount"].sum())
                dists = float(cf_seg.loc[cf_seg["cat"] == "dist", "amount"].sum())
                navv = float(
                    cf_seg[cf_seg["cat"] == "nav"]
                    .sort_values(["portfolio_company", "date"]) 
                    .groupby("portfolio_company")["amount"].tail(1)
                    .sum()
                )
                moic_val = (dists + navv) / calls if calls > 0 else np.nan
                # IRR from aggregated flows
                flows = []
                for _, gfund in cf_seg.groupby(["portfolio_company"], sort=False):
                    g_sorted = gfund.sort_values("date")
                    if (g_sorted["cat"] == "nav").any():
                        nav_date = g_sorted.loc[g_sorted["cat"] == "nav", "date"].max()
                        g_use = g_sorted[(g_sorted["cat"] != "nav") | (g_sorted["date"] == nav_date)].copy()
                    else:
                        g_use = g_sorted.copy()
                    flows.append(g_use[["date", "amount"]])
                if flows:
                    fl_df = pd.concat(flows, ignore_index=True).groupby("date", as_index=False)["amount"].sum().sort_values("date")
                    f_dates = [pd.to_datetime(d).date() for d in fl_df["date"].tolist()]
                    f_amts = [float(x) for x in fl_df["amount"].tolist()]
                    irr_val = _xirr(f_dates, f_amts) if (any(a < 0 for a in f_amts) and any(a > 0 for a in f_amts)) else None
                else:
                    irr_val = None
                return (
                    f"{ksp:.2f}" if pd.notna(ksp) else "—",
                    f"{moic_val:.2f}" if pd.notna(moic_val) else "—",
                    f"{irr_val:.1%}" if irr_val is not None else "—",
                )
            # Build segments
            segs = {
                "Fully Realized": of[of["Status"] == "Fully Realized"],
                "Partially Realized": of[of["Status"] == "Partially Realized"],
                "Unrealized": of[of["Status"] == "Unrealized"],
                "Total": of,
            }
            parts = []
            for name, df_deals in segs.items():
                ksp_s, moic_s, irr_s = _seg_summary(df_deals, cf_f)
                parts.append(f"{name}: KS-PME {ksp_s} | MOIC {moic_s} | IRR {irr_s}")
            st.caption(" | ".join(parts))

    # Charts: KS-PME by Portfolio Company and distribution
    import plotly.express as px
    import plotly.graph_objects as go
    st.subheader("KS-PME by Portfolio Company")
    viz = out.copy()
    if "KS-PME" in viz.columns and "Portfolio Company" in viz.columns:
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
        # Add MOIC and IRR to hover
        fig_bar.update_traces(
            customdata=np.column_stack([
                pd.to_numeric(viz.get("Total MOIC"), errors="coerce"),
                pd.to_numeric(viz.get("IRR"), errors="coerce"),
            ]),
            hovertemplate="<b>%{x}</b><br>KS-PME: %{y:.2f}<br>Total MOIC: %{customdata[0]:.2f}<br>IRR: %{customdata[1]:.1%}<extra></extra>",
        )
        fig_bar.update_layout(xaxis_tickangle=-45, yaxis_title="KS-PME", legend_title_text="Performance")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.caption("Distribution of KS-PME")
        viz_nonan = viz.dropna(subset=["KS-PME"]).copy()
        fig_hist = px.histogram(viz_nonan, x="KS-PME", nbins=25)
        fig_hist.add_vline(x=1.0, line_dash="dash", line_color="#7f7f7f")
        fig_hist.update_layout(yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Portfolio summary: KS-PME, MOIC, IRR
    st.subheader("Portfolio Summary")
    # Portfolio KS-PME over all deals
    port_kspme = _ks_pme_index_multiple(cf, index_df)
    # Portfolio MOIC: sum dists + sum of last NAV per deal, over abs sum calls
    calls_port = float(-cf.loc[cf["cat"] == "call", "amount"].sum())
    dists_port = float(cf.loc[cf["cat"] == "dist", "amount"].sum())
    nav_port = float(cf[cf["cat"] == "nav"].sort_values(["portfolio_company", "date"]).groupby("portfolio_company")["amount"].tail(1).sum())
    port_moic = (dists_port + nav_port) / calls_port if calls_port > 0 else np.nan
    # Portfolio IRR: aggregate flows by date across deals, using only last NAV per deal
    flows = []
    for deal, g in cf.groupby("portfolio_company", sort=False):
        g_sorted = g.sort_values("date")
        # include only last NAV
        if (g_sorted["cat"] == "nav").any():
            nav_date = g_sorted.loc[g_sorted["cat"] == "nav", "date"].max()
            g_use = g_sorted[(g_sorted["cat"] != "nav") | (g_sorted["date"] == nav_date)].copy()
        else:
            g_use = g_sorted.copy()
        flows.append(g_use[["date", "amount"]])
    if flows:
        port_flows = pd.concat(flows, ignore_index=True).groupby("date", as_index=False)["amount"].sum().sort_values("date")
        port_dates = [pd.to_datetime(d).date() for d in port_flows["date"].tolist()]
        port_amts = [float(x) for x in port_flows["amount"].tolist()]
        port_irr = _xirr(port_dates, port_amts) if (any(a < 0 for a in port_amts) and any(a > 0 for a in port_amts)) else None
    else:
        port_irr = None
    ks_str = f"{port_kspme:.2f}" if pd.notna(port_kspme) else "—"
    moic_str = f"{port_moic:.2f}" if pd.notna(port_moic) else "—"
    irr_str = f"{port_irr:.1%}" if port_irr is not None else "—"
    st.write(f"Portfolio KS-PME: {ks_str} | Portfolio MOIC: {moic_str} | Portfolio IRR: {irr_str}")
    # KS-PME >= 1 vs < 1 breakout (counts and capital)
    out_nonan = out.dropna(subset=["KS-PME"]).copy()
    if not out_nonan.empty:
        over = out_nonan[pd.to_numeric(out_nonan["KS-PME"], errors="coerce") >= 1.0]
        under = out_nonan[pd.to_numeric(out_nonan["KS-PME"], errors="coerce") < 1.0]
        def _caps(df_part: pd.DataFrame) -> tuple[float, float, float, int]:
            inv = float(pd.to_numeric(df_part.get("Invested Capital"), errors="coerce").sum()) if not df_part.empty else 0.0
            real = float(pd.to_numeric(df_part.get("Realized Capital"), errors="coerce").sum()) if not df_part.empty else 0.0
            navv = float(pd.to_numeric(df_part.get("Unrealized Capital"), errors="coerce").sum()) if not df_part.empty else 0.0
            cnt = int(df_part.shape[0])
            return inv, real, navv, cnt
        inv_o, real_o, nav_o, cnt_o = _caps(over)
        inv_u, real_u, nav_u, cnt_u = _caps(under)
        st.caption(
            f"KS-PME ≥ 1.0 — Deals: {cnt_o} | Invested: ${inv_o:,.1f} | Realized: ${real_o:,.1f} | NAV: ${nav_o:,.1f}"
        )
        st.caption(
            f"KS-PME < 1.0 — Deals: {cnt_u} | Invested: ${inv_u:,.1f} | Realized: ${real_u:,.1f} | NAV: ${nav_u:,.1f}"
        )
    # Segmented portfolio summary across statuses using 'out'
    if not out.empty and "Status" in out.columns:
        def _seg_port(name: str, df_deals: pd.DataFrame) -> str:
            deals_set = set(df_deals["Portfolio Company"].astype(str))
            cf_seg = cf[cf["portfolio_company"].isin(deals_set)]
            ksp = _ks_pme_index_multiple(cf_seg, index_df)
            calls = float(-cf_seg.loc[cf_seg["cat"] == "call", "amount"].sum())
            dists = float(cf_seg.loc[cf_seg["cat"] == "dist", "amount"].sum())
            navv = float(cf_seg[cf_seg["cat"] == "nav"].sort_values(["portfolio_company", "date"]).groupby("portfolio_company")["amount"].tail(1).sum())
            moic_val = (dists + navv) / calls if calls > 0 else np.nan
            # IRR
            flows = []
            for _, gseg in cf_seg.groupby(["portfolio_company"], sort=False):
                g_sorted = gseg.sort_values("date")
                if (g_sorted["cat"] == "nav").any():
                    nav_date = g_sorted.loc[g_sorted["cat"] == "nav", "date"].max()
                    g_use = g_sorted[(g_sorted["cat"] != "nav") | (g_sorted["date"] == nav_date)].copy()
                else:
                    g_use = g_sorted.copy()
                flows.append(g_use[["date", "amount"]])
            if flows:
                fl_df = pd.concat(flows, ignore_index=True).groupby("date", as_index=False)["amount"].sum().sort_values("date")
                f_dates = [pd.to_datetime(d).date() for d in fl_df["date"].tolist()]
                f_amts = [float(x) for x in fl_df["amount"].tolist()]
                irr_val = _xirr(f_dates, f_amts) if (any(a < 0 for a in f_amts) and any(a > 0 for a in f_amts)) else None
            else:
                irr_val = None
            return f"{name}: KS-PME {ksp:.2f} | MOIC {moic_val:.2f} | IRR {irr_val:.1%}" if pd.notna(ksp) else f"{name}: KS-PME — | MOIC {moic_val:.2f if pd.notna(moic_val) else float('nan')} | IRR {'—' if irr_val is None else f'{irr_val:.1%}'}"
        segs_port = {
            "Fully Realized": out[out["Status"] == "Fully Realized"],
            "Partially Realized": out[out["Status"] == "Partially Realized"],
            "Unrealized": out[out["Status"] == "Unrealized"],
            "Total": out,
        }
        lines = []
        for name, df_deals in segs_port.items():
            lines.append(_seg_port(name, df_deals))
        st.caption(" | ".join(lines))

    # Per-deal diagnostics and charts removed per request
else:
    st.info("No per-deal results to show.")


