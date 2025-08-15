from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from analysis import extract_operational_by_template_order, add_growth_and_cagr, compute_value_creation
from filters import render_and_filter


st.set_page_config(page_title="Compare Views", layout="wide")


# Consistent color mapping for value creation pie charts across both panes
VC_PIE_LABELS = ["Revenue Growth", "Margin Expansion", "Multiple Change", "Deleveraging"]
VC_PIE_COLOR_MAP = {
    "Revenue Growth": "#1f77b4",   # blue
    "Margin Expansion": "#2ca02c", # green
    "Multiple Change": "#ff7f0e",  # orange
    "Deleveraging": "#d62728",     # red
}


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


st.title("Compare Views")
st.caption("Side-by-side analysis with independent filters on each side.")

with st.sidebar:
    upload = st.file_uploader("Upload Portfolio Metrics file (.xlsx or .csv)", type=["xlsx", "csv"])  # type: ignore
    header_row_index = st.number_input("Header row (1-based)", min_value=1, max_value=100, value=int(st.session_state.get("header_row_index", 2)), step=1)

sheets = _read_excel_or_csv(upload, header_row_index)
if not sheets:
    sheets = st.session_state.get("sheets", {})
if not sheets:
    st.info("Upload a file to begin.")
    st.stop()

sheet_name = st.selectbox(
    "Select sheet",
    list(sheets.keys()),
    index=max(0, list(sheets.keys()).index(st.session_state.get("selected_sheet", list(sheets.keys())[0])) if sheets else 0),
)
df = sheets[sheet_name]
st.caption(f"Loaded sheet '{sheet_name}' with rows: {len(df):,}")
st.session_state["sheets"] = sheets
st.session_state["selected_sheet"] = sheet_name
st.session_state["header_row_index"] = header_row_index

ops_df_raw, _ = extract_operational_by_template_order(df, list(df.columns))
if ops_df_raw.empty:
    st.error("No metrics detected from the uploaded sheet. Confirm the header row and column order.")
    st.stop()

ops_df = add_growth_and_cagr(ops_df_raw)


# Ensure value creation components exist
def _compute_value_creation_local(df: pd.DataFrame) -> pd.DataFrame:
    e0 = pd.to_numeric(df.get("entry_ebitda"), errors="coerce")
    e1 = pd.to_numeric(df.get("exit_ebitda"), errors="coerce")
    r0 = pd.to_numeric(df.get("entry_revenue"), errors="coerce")
    r1 = pd.to_numeric(df.get("exit_revenue"), errors="coerce")
    tev0 = pd.to_numeric(df.get("entry_tev"), errors="coerce")
    tev1 = pd.to_numeric(df.get("exit_tev"), errors="coerce")
    nd0 = pd.to_numeric(df.get("entry_net_debt"), errors="coerce")
    nd1 = pd.to_numeric(df.get("exit_net_debt"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        mult0 = np.where(e0 > 0, tev0 / e0, np.nan)
        mult1 = np.where(e1 > 0, tev1 / e1, np.nan)
        marg0 = np.where(r0 > 0, e0 / r0, np.nan)
        marg1 = np.where(r1 > 0, e1 / r1, np.nan)
        mult0 = pd.Series(mult0).replace([np.inf, -np.inf], np.nan)
        mult1 = pd.Series(mult1).replace([np.inf, -np.inf], np.nan)
        marg0 = pd.Series(marg0).replace([np.inf, -np.inf], np.nan)
        marg1 = pd.Series(marg1).replace([np.inf, -np.inf], np.nan)
        rev_growth = (r1 - r0) * marg0 * mult0
        margin_exp = r1 * (marg1 - marg0) * mult0
        e1_safe = e1.where(e1 > 0)
        multiple_change = (mult1 - mult0) * e1_safe
        deleveraging = -(nd1 - nd0)
        eq0 = tev0 - nd0
        eq1 = tev1 - nd1
        bridge_sum = rev_growth + margin_exp + multiple_change + deleveraging
    out = df.copy()
    out["equity_entry"] = eq0
    out["equity_exit"] = eq1
    out["vc_rev_growth"] = rev_growth
    out["vc_margin_expansion"] = margin_exp
    out["vc_multiple_change"] = multiple_change
    out["vc_deleveraging"] = deleveraging
    out["vc_bridge_sum"] = bridge_sum
    return out

ops_df = compute_value_creation(ops_df)


def _track_record_table(frame: pd.DataFrame, portfolio_header: str) -> pd.io.formats.style.Styler:
    cols = [
        portfolio_header,
        "sector",
        "status",
        "invest_date",
        "exit_date",
        "holding_years",
        "ownership_pct",
        "invested",
        "proceeds",
        "current_value",
        "gross_moic",
        "gross_irr",
        "fund_name",
    ]
    out = frame.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if "ownership_pct" not in out.columns:
            if "kam_ownership_exit_pct" in out.columns:
                out["ownership_pct"] = pd.to_numeric(out["kam_ownership_exit_pct"], errors="coerce")
            elif {"kam_equity_entry", "equity_entry_total"}.issubset(out.columns):
                out["ownership_pct"] = pd.to_numeric(out["kam_equity_entry"], errors="coerce") / pd.to_numeric(out["equity_entry_total"], errors="coerce")
            else:
                out["ownership_pct"] = np.nan
    view = out[[c for c in cols if c in out.columns or c == portfolio_header]].copy()
    for dc in ["invest_date", "exit_date"]:
        if dc in view.columns:
            view[dc] = pd.to_datetime(view[dc], errors="coerce").dt.strftime("%b %Y")
    fmt = {}
    if "ownership_pct" in view.columns:
        fmt["ownership_pct"] = "{:.1%}"
    if "gross_irr" in view.columns:
        fmt["gross_irr"] = "{:.1%}"
    for c in view.select_dtypes(include=["number"]).columns:
        if c not in fmt:
            fmt[c] = "{:.1f}"
    return view.style.format(fmt, na_rep="—")


def _portfolio_vc_waterfall(frame: pd.DataFrame) -> go.Figure | None:
    def _sum(series: pd.Series) -> float:
        return float(pd.to_numeric(series, errors="coerce").sum(skipna=True))
    eq0 = _sum(frame.get("equity_entry", pd.Series(dtype=float)))
    rev = _sum(frame.get("vc_rev_growth", pd.Series(dtype=float)))
    marg = _sum(frame.get("vc_margin_expansion", pd.Series(dtype=float)))
    mult = _sum(frame.get("vc_multiple_change", pd.Series(dtype=float)))
    debt = _sum(frame.get("vc_deleveraging", pd.Series(dtype=float)))
    eq1 = _sum(frame.get("equity_exit", pd.Series(dtype=float)))
    if not (np.isfinite(eq0) and np.isfinite(eq1)):
        return None
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=[
                "Equity at Entry",
                "Revenue Growth",
                "Margin Expansion",
                "Multiple Change",
                "Deleveraging",
                "Equity at Exit",
            ],
            textposition="outside",
            text=[f"{v:,.1f}" for v in [eq0, rev, marg, mult, debt, eq1]],
            y=[eq0, rev, marg, mult, debt, eq1],
        )
    )
    fig.update_layout(showlegend=False, waterfallgap=0.3)
    return fig


def _portfolio_vc_pie(frame: pd.DataFrame):
    def _sum(series: pd.Series) -> float:
        return float(pd.to_numeric(series, errors="coerce").sum(skipna=True))
    rev = _sum(frame.get("vc_rev_growth", pd.Series(dtype=float)))
    marg = _sum(frame.get("vc_margin_expansion", pd.Series(dtype=float)))
    mult = _sum(frame.get("vc_multiple_change", pd.Series(dtype=float)))
    debt = _sum(frame.get("vc_deleveraging", pd.Series(dtype=float)))
    vals_abs = [abs(rev), abs(marg), abs(mult), abs(debt)]
    if sum(v for v in vals_abs if np.isfinite(v)) <= 0:
        return None
    labels = VC_PIE_LABELS
    # Use shared color mapping so left/right pies have identical colors
    fig = px.pie(
        names=labels,
        values=vals_abs,
        color=labels,
        color_discrete_map=VC_PIE_COLOR_MAP,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", texttemplate="%{percent:.1%}")
    fig.update_layout(showlegend=True)
    return fig


portfolio_header = df.columns[0] if len(df.columns) > 0 else "Portfolio Company"
if portfolio_header not in ops_df.columns and "portfolio_company" in ops_df.columns:
    ops_df.insert(0, portfolio_header, ops_df["portfolio_company"])  # ensure display col exists

left, right = st.columns(2)


def _status_bucket_row(row: pd.Series) -> str:
    invested = float(pd.to_numeric(row.get("invested"), errors="coerce") or 0.0)
    proceeds = float(pd.to_numeric(row.get("proceeds"), errors="coerce") or 0.0)
    nav = float(pd.to_numeric(row.get("current_value"), errors="coerce") or 0.0)
    if invested <= 0:
        return "Unrealized"
    if nav <= 0 and proceeds > 0:
        return "Fully Realized"
    if nav > 0 and proceeds > 0:
        return "Partially Realized"
    if nav > 0 and proceeds <= 0:
        return "Unrealized"
    return str(row.get("status", "")).strip() or "Unrealized"


def _subtotals_table(frame: pd.DataFrame, portfolio_header: str) -> pd.io.formats.style.Styler:
    dfp = frame.copy()
    # Ensure numeric
    for c in ["invested", "proceeds", "current_value", "gross_moic", "gross_irr", "holding_years"]:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
    # Bucket
    dfp["_bucket"] = dfp.apply(_status_bucket_row, axis=1)
    # Compute totals per bucket
    rows = []
    def _row_for(label: str, sub: pd.DataFrame) -> dict:
        inv = float(sub.get("invested", pd.Series(dtype=float)).sum(skipna=True))
        real = float(sub.get("proceeds", pd.Series(dtype=float)).sum(skipna=True))
        nav = float(sub.get("current_value", pd.Series(dtype=float)).sum(skipna=True))
        total_val = real + nav
        moic = (total_val / inv) if inv > 0 else np.nan
        # Weighted avg IRR by invested
        w = pd.to_numeric(sub.get("invested", pd.Series(dtype=float)), errors="coerce").clip(lower=0).fillna(0)
        irr_vals = pd.to_numeric(sub.get("gross_irr", pd.Series(dtype=float)), errors="coerce").fillna(0)
        irr_wa = float(np.average(irr_vals, weights=w)) if float(w.sum()) > 0 else np.nan
        # Weighted avg holding years by invested
        yrs = pd.to_numeric(sub.get("holding_years", pd.Series(dtype=float)), errors="coerce").fillna(0)
        yrs_wa = float(np.average(yrs, weights=w)) if float(w.sum()) > 0 else np.nan
        return {
            portfolio_header: label,
            "holding_years": yrs_wa,
            "invested": inv,
            "proceeds": real,
            "current_value": nav,
            "total_value": total_val,
            "gross_moic": moic,
            "gross_irr": irr_wa,
        }
    # Buckets and total
    buckets = ["Fully Realized", "Partially Realized", "Unrealized"]
    for b in buckets:
        rows.append(_row_for(b, dfp[dfp["_bucket"] == b]))
    rows.append(_row_for("Total", dfp))
    out = pd.DataFrame(rows)
    # Format
    fmt = {
        "holding_years": "{:.1f}",
        "invested": "{:,.1f}",
        "proceeds": "{:,.1f}",
        "current_value": "{:,.1f}",
        "total_value": "{:,.1f}",
        "gross_moic": "{:.1f}",
        "gross_irr": "{:.1%}",
    }
    # Select display columns per request
    cols = [
        portfolio_header,
        "holding_years",
        "invested",
        "proceeds",
        "current_value",
        "total_value",
        "gross_moic",
        "gross_irr",
    ]
    out = out[[c for c in cols if c in out.columns]]
    return out.style.format(fmt, na_rep="—")

with left:
    st.subheader("Left View")
    f_left = render_and_filter(ops_df, key_prefix="cmp_left")
    st.markdown("**Track Record**")
    # Fixed heights so charts below align across columns
    st.dataframe(_track_record_table(f_left, portfolio_header), use_container_width=True, height=420, key="cmp_tr_left")
    st.markdown("**Subtotals**")
    st.dataframe(_subtotals_table(f_left, portfolio_header), use_container_width=True, height=200, key="cmp_sub_left")
    st.markdown("**Value Creation (Portfolio)**")
    # Exclude deals missing required inputs for value creation visuals
    vc_left = f_left[f_left.get("vc_valid", True).astype(bool)]
    fig_l = _portfolio_vc_waterfall(vc_left)
    pie_l = _portfolio_vc_pie(vc_left)
    lc1, lc2 = st.columns([2, 1])
    if fig_l is not None:
        lc1.plotly_chart(fig_l, use_container_width=True, key="cmp_vc_left")
    if pie_l is not None:
        lc2.plotly_chart(pie_l, use_container_width=True, key="cmp_pie_left")

    # Deal Charts: Entry vs Exit (mirror Deal Charts page)
    st.markdown("**Deal Charts: Entry vs Exit**")
    # Prepare frame: unique x label and margins, sort by EBITDA change like Deal Charts
    x_label_col_left = f"{portfolio_header} (Fund)"
    if portfolio_header in f_left.columns and "fund_name" in f_left.columns:
        f_left[x_label_col_left] = f_left[portfolio_header].astype(str) + " — " + f_left["fund_name"].astype(str)
    else:
        f_left[x_label_col_left] = f_left.get(portfolio_header, pd.Series(dtype=str)).astype(str)
    with np.errstate(divide="ignore", invalid="ignore"):
        if {"entry_ebitda", "entry_revenue"}.issubset(f_left.columns):
            f_left["entry_margin_pct"] = pd.to_numeric(f_left["entry_ebitda"], errors="coerce") / pd.to_numeric(f_left["entry_revenue"], errors="coerce")
        if {"exit_ebitda", "exit_revenue"}.issubset(f_left.columns):
            f_left["exit_margin_pct"] = pd.to_numeric(f_left["exit_ebitda"], errors="coerce") / pd.to_numeric(f_left["exit_revenue"], errors="coerce")
    f_left["_sort_change_ebitda"] = pd.to_numeric(f_left.get("exit_ebitda"), errors="coerce") - pd.to_numeric(f_left.get("entry_ebitda"), errors="coerce")
    f_left_sorted = f_left.sort_values("_sort_change_ebitda", ascending=False, na_position="last")
    x_order_left = f_left_sorted[x_label_col_left].astype(str).tolist() if x_label_col_left in f_left_sorted.columns else []

    def grouped_bar(
        df_in: pd.DataFrame,
        y_entry: str,
        y_exit: str,
        y_format: str,
        title: str,
        x_order: List[str],
        x_label_col: str,
        key: str,
        exclude_outliers: bool = False,
        low_pct: float = 1.0,
        high_pct: float = 99.0,
        y_range: tuple | None = None,
    ):
        x_vals = df_in[x_label_col].astype(str).tolist() if x_label_col in df_in.columns else list(range(len(df_in)))
        y1 = pd.to_numeric(df_in.get(y_entry), errors="coerce")
        y2 = pd.to_numeric(df_in.get(y_exit), errors="coerce")
        df_plot = df_in.copy()
        df_plot["y1"] = y1
        df_plot["y2"] = y2
        if exclude_outliers:
            both = pd.concat([y1, y2]).dropna()
            if not both.empty and both.size > 5:
                q_low = both.quantile(low_pct / 100.0)
                q_high = both.quantile(high_pct / 100.0)
                mask = (df_plot["y1"].between(q_low, q_high)) & (df_plot["y2"].between(q_low, q_high))
                df_plot = df_plot[mask]
        fig = go.Figure()
        fund = df_plot.get("fund_name")
        moic = pd.to_numeric(df_plot.get("gross_moic"), errors="coerce")
        irr = pd.to_numeric(df_plot.get("gross_irr"), errors="coerce")
        status = df_plot.get("status")
        sector = df_plot.get("sector")
        custom = np.column_stack([
            fund.fillna("") if fund is not None else np.repeat("", len(df_plot)),
            moic.fillna(np.nan) if moic is not None else np.repeat(np.nan, len(df_plot)),
            irr.fillna(np.nan) if irr is not None else np.repeat(np.nan, len(df_plot)),
            status.fillna("") if status is not None else np.repeat("", len(df_plot)),
            sector.fillna("") if sector is not None else np.repeat("", len(df_plot)),
        ]) if len(df_plot) > 0 else np.empty((0,5))
        fig.add_bar(name="Entry", x=x_vals, y=df_plot["y1"], marker_color="#1f77b4")
        fig.add_bar(name="Exit", x=x_vals, y=df_plot["y2"], marker_color="#ff7f0e")
        fig.update_layout(
            barmode="group",
            title=title,
            xaxis_title=x_label_col,
            yaxis_title=y_entry.replace("entry_", "").replace("_", " ").title(),
            xaxis_tickangle=-45,
            legend_title_text="Point in Time",
            margin=dict(t=60, r=20, b=80, l=60),
            height=500,
        )
        if x_order:
            fig.update_xaxes(categoryorder="array", categoryarray=x_order)
        fig.update_yaxes(tickformat=y_format)
        if y_range is not None:
            try:
                fig.update_yaxes(range=list(y_range))
            except Exception:
                pass
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Fund: %{customdata[0]}<br>MOIC: %{customdata[1]:.1f}<br>IRR: %{customdata[2]:.1%}<br>Status: %{customdata[3]}<br>Sector: %{customdata[4]}<extra></extra>",
            customdata=custom,
        )
        st.plotly_chart(fig, use_container_width=True, key=key)

    # Outlier controls (left)
    ol1, ol2, ol3 = st.columns(3)
    excl_left = ol1.checkbox("Exclude outliers (pct)", value=True, key="cmp_dc_excl_left")
    low_left = ol2.number_input("Lower pct", min_value=0.0, max_value=50.0, value=1.0, step=0.5, key="cmp_dc_low_left")
    high_left = ol3.number_input("Upper pct", min_value=50.0, max_value=100.0, value=99.0, step=0.5, key="cmp_dc_high_left")

    grouped_bar(f_left_sorted, "entry_revenue", "exit_revenue", ",.1f", "Revenue: Entry vs Exit", x_order_left, x_label_col_left, key="cmp_dc_rev_left", exclude_outliers=excl_left, low_pct=low_left, high_pct=high_left)
    grouped_bar(f_left_sorted, "entry_ebitda", "exit_ebitda", ",.1f", "EBITDA: Entry vs Exit", x_order_left, x_label_col_left, key="cmp_dc_ebitda_left", exclude_outliers=excl_left, low_pct=low_left, high_pct=high_left)
    grouped_bar(f_left_sorted, "entry_margin_pct", "exit_margin_pct", ".1%", "EBITDA Margin: Entry vs Exit", x_order_left, x_label_col_left, key="cmp_dc_margin_left", exclude_outliers=excl_left, low_pct=low_left, high_pct=high_left)
    grouped_bar(f_left_sorted, "entry_tev_ebitda", "exit_tev_ebitda", ",.1f", "TEV/EBITDA: Entry vs Exit", x_order_left, x_label_col_left, key="cmp_dc_tevebitda_left", exclude_outliers=excl_left, low_pct=low_left, high_pct=high_left)
    grouped_bar(f_left_sorted, "entry_tev_revenue", "exit_tev_revenue", ",.1f", "TEV/Revenue: Entry vs Exit", x_order_left, x_label_col_left, key="cmp_dc_tevrev_left", exclude_outliers=excl_left, low_pct=low_left, high_pct=high_left)
    grouped_bar(f_left_sorted, "entry_leverage", "exit_leverage", ",.1f", "Leverage: Entry vs Exit", x_order_left, x_label_col_left, key="cmp_dc_lev_left", exclude_outliers=excl_left, low_pct=low_left, high_pct=high_left)
    grouped_bar(f_left_sorted, "entry_net_debt", "exit_net_debt", ",.1f", "Net Debt: Entry vs Exit", x_order_left, x_label_col_left, key="cmp_dc_nd_left", exclude_outliers=excl_left, low_pct=low_left, high_pct=high_left)
    grouped_bar(f_left_sorted, "entry_tev", "exit_tev", ",.1f", "TEV: Entry vs Exit", x_order_left, x_label_col_left, key="cmp_dc_tev_left", exclude_outliers=excl_left, low_pct=low_left, high_pct=high_left)

    # Summary stats (left): simple avg and weighted avg by invested
    st.markdown("**Averages (Left)**")
    def _stats_table(df_in: pd.DataFrame) -> pd.DataFrame:
        metrics = [
            ("Revenue", "entry_revenue", "exit_revenue"),
            ("EBITDA", "entry_ebitda", "exit_ebitda"),
            ("EBITDA Margin", "entry_margin_pct", "exit_margin_pct"),
            ("TEV/EBITDA", "entry_tev_ebitda", "exit_tev_ebitda"),
            ("TEV/Revenue", "entry_tev_revenue", "exit_tev_revenue"),
            ("Leverage", "entry_leverage", "exit_leverage"),
            ("Net Debt", "entry_net_debt", "exit_net_debt"),
            ("TEV", "entry_tev", "exit_tev"),
        ]
        rows = []
        w = pd.to_numeric(df_in.get("invested"), errors="coerce").clip(lower=0).fillna(0)
        for label, e_col, x_col in metrics:
            e = pd.to_numeric(df_in.get(e_col), errors="coerce")
            x = pd.to_numeric(df_in.get(x_col), errors="coerce")
            avg_e = e.mean(skipna=True)
            avg_x = x.mean(skipna=True)
            wa_e = float(np.average(e.fillna(0), weights=w)) if float(w.sum()) > 0 else np.nan
            wa_x = float(np.average(x.fillna(0), weights=w)) if float(w.sum()) > 0 else np.nan
            rows.append({"Metric": label, "Avg Entry": avg_e, "Avg Exit": avg_x, "WA Entry": wa_e, "WA Exit": wa_x})
        return pd.DataFrame(rows)
    stats_left = _stats_table(f_left_sorted)
    # Format: default numeric with 1 decimal; EBITDA Margin rows as percent with 1 decimal
    sleft = stats_left.set_index("Metric")
    sty_left = sleft.style.format("{:,.1f}")
    try:
        sty_left = sty_left.format(
            "{:.1%}",
            subset=pd.IndexSlice[["EBITDA Margin"], ["Avg Entry", "Avg Exit", "WA Entry", "WA Exit"]],
        )
    except Exception:
        pass
    st.dataframe(sty_left, use_container_width=True, key="cmp_dc_stats_left")

with right:
    st.subheader("Right View")
    f_right = render_and_filter(ops_df, key_prefix="cmp_right")
    st.markdown("**Track Record**")
    st.dataframe(_track_record_table(f_right, portfolio_header), use_container_width=True, height=420, key="cmp_tr_right")
    st.markdown("**Subtotals**")
    st.dataframe(_subtotals_table(f_right, portfolio_header), use_container_width=True, height=200, key="cmp_sub_right")
    st.markdown("**Value Creation (Portfolio)**")
    vc_right = f_right[f_right.get("vc_valid", True).astype(bool)]
    fig_r = _portfolio_vc_waterfall(vc_right)
    pie_r = _portfolio_vc_pie(vc_right)
    rc1, rc2 = st.columns([2, 1])
    if fig_r is not None:
        rc1.plotly_chart(fig_r, use_container_width=True, key="cmp_vc_right")
    if pie_r is not None:
        rc2.plotly_chart(pie_r, use_container_width=True, key="cmp_pie_right")

    # Deal Charts: Entry vs Exit (mirror Deal Charts page)
    st.markdown("**Deal Charts: Entry vs Exit**")
    x_label_col_right = f"{portfolio_header} (Fund)"
    if portfolio_header in f_right.columns and "fund_name" in f_right.columns:
        f_right[x_label_col_right] = f_right[portfolio_header].astype(str) + " — " + f_right["fund_name"].astype(str)
    else:
        f_right[x_label_col_right] = f_right.get(portfolio_header, pd.Series(dtype=str)).astype(str)
    with np.errstate(divide="ignore", invalid="ignore"):
        if {"entry_ebitda", "entry_revenue"}.issubset(f_right.columns):
            f_right["entry_margin_pct"] = pd.to_numeric(f_right["entry_ebitda"], errors="coerce") / pd.to_numeric(f_right["entry_revenue"], errors="coerce")
        if {"exit_ebitda", "exit_revenue"}.issubset(f_right.columns):
            f_right["exit_margin_pct"] = pd.to_numeric(f_right["exit_ebitda"], errors="coerce") / pd.to_numeric(f_right["exit_revenue"], errors="coerce")
    f_right["_sort_change_ebitda"] = pd.to_numeric(f_right.get("exit_ebitda"), errors="coerce") - pd.to_numeric(f_right.get("entry_ebitda"), errors="coerce")
    f_right_sorted = f_right.sort_values("_sort_change_ebitda", ascending=False, na_position="last")
    x_order_right = f_right_sorted[x_label_col_right].astype(str).tolist() if x_label_col_right in f_right_sorted.columns else []

    # Outlier controls (right)
    or1, or2, or3 = st.columns(3)
    excl_right = or1.checkbox("Exclude outliers (pct)", value=True, key="cmp_dc_excl_right")
    low_right = or2.number_input("Lower pct", min_value=0.0, max_value=50.0, value=1.0, step=0.5, key="cmp_dc_low_right")
    high_right = or3.number_input("Upper pct", min_value=50.0, max_value=100.0, value=99.0, step=0.5, key="cmp_dc_high_right")

    grouped_bar(f_right_sorted, "entry_revenue", "exit_revenue", ",.1f", "Revenue: Entry vs Exit", x_order_right, x_label_col_right, key="cmp_dc_rev_right", exclude_outliers=excl_right, low_pct=low_right, high_pct=high_right)
    grouped_bar(f_right_sorted, "entry_ebitda", "exit_ebitda", ",.1f", "EBITDA: Entry vs Exit", x_order_right, x_label_col_right, key="cmp_dc_ebitda_right", exclude_outliers=excl_right, low_pct=low_right, high_pct=high_right)
    grouped_bar(f_right_sorted, "entry_margin_pct", "exit_margin_pct", ".1%", "EBITDA Margin: Entry vs Exit", x_order_right, x_label_col_right, key="cmp_dc_margin_right", exclude_outliers=excl_right, low_pct=low_right, high_pct=high_right)
    grouped_bar(f_right_sorted, "entry_tev_ebitda", "exit_tev_ebitda", ",.1f", "TEV/EBITDA: Entry vs Exit", x_order_right, x_label_col_right, key="cmp_dc_tevebitda_right", exclude_outliers=excl_right, low_pct=low_right, high_pct=high_right)
    grouped_bar(f_right_sorted, "entry_tev_revenue", "exit_tev_revenue", ",.1f", "TEV/Revenue: Entry vs Exit", x_order_right, x_label_col_right, key="cmp_dc_tevrev_right", exclude_outliers=excl_right, low_pct=low_right, high_pct=high_right)
    grouped_bar(f_right_sorted, "entry_leverage", "exit_leverage", ",.1f", "Leverage: Entry vs Exit", x_order_right, x_label_col_right, key="cmp_dc_lev_right", exclude_outliers=excl_right, low_pct=low_right, high_pct=high_right)
    grouped_bar(f_right_sorted, "entry_net_debt", "exit_net_debt", ",.1f", "Net Debt: Entry vs Exit", x_order_right, x_label_col_right, key="cmp_dc_nd_right", exclude_outliers=excl_right, low_pct=low_right, high_pct=high_right)
    grouped_bar(f_right_sorted, "entry_tev", "exit_tev", ",.1f", "TEV: Entry vs Exit", x_order_right, x_label_col_right, key="cmp_dc_tev_right", exclude_outliers=excl_right, low_pct=low_right, high_pct=high_right)

    # Summary stats (right)
    st.markdown("**Averages (Right)**")
    stats_right = _stats_table(f_right_sorted)
    sright = stats_right.set_index("Metric")
    sty_right = sright.style.format("{:,.1f}")
    try:
        sty_right = sty_right.format(
            "{:.1%}",
            subset=pd.IndexSlice[["EBITDA Margin"], ["Avg Entry", "Avg Exit", "WA Entry", "WA Exit"]],
        )
    except Exception:
        pass
    st.dataframe(sty_right, use_container_width=True, key="cmp_dc_stats_right")


