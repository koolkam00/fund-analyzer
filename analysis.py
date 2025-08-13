from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


"""Utilities for parsing the Portfolio Metrics sheet and computing per-deal metrics."""


def parse_to_datetime(series: pd.Series) -> pd.Series:
    """Robust date parsing with common formats and Excel serials."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)
    # Try direct parse; coerce errors to NaT
    parsed = pd.to_datetime(series, errors="coerce")
    # Attempt Excel serial handling if many NaT remain and series is numeric
    if parsed.isna().mean() > 0.5 and pd.api.types.is_numeric_dtype(series):
        # Excel origin 1899-12-30
        origin = pd.Timestamp("1899-12-30")
        parsed = origin + pd.to_timedelta(series.fillna(0).astype(float), unit="D")
    return parsed


def compute_irr_from_flows(dates: List[pd.Timestamp], amounts: List[float]) -> Optional[float]:
    return None


def _xnpv(rate: float, dates: List[pd.Timestamp], amounts: List[float]) -> float:
    """Compute NPV for irregular cash flows (XNPV)."""
    if rate <= -1.0:
        return np.inf
    t0 = min(dates)
    npv = 0.0
    for dt, amt in zip(dates, amounts):
        days = (dt - t0).days
        npv += amt / ((1 + rate) ** (days / 365.2425))
    return npv


def _xirr(dates: List[pd.Timestamp], amounts: List[float], guess: float = 0.15) -> Optional[float]:
    """Solve for IRR given dated cash flows using Newton's method with backtracking."""
    # Newton-Raphson with derivative approximation
    rate = guess
    for _ in range(100):
        f = _xnpv(rate, dates, amounts)
        # Derivative via finite difference
        h = 1e-6
        f1 = _xnpv(rate + h, dates, amounts)
        d = (f1 - f) / h
        if abs(d) < 1e-12:
            break
        new_rate = rate - f / d
        # Backtracking if diverging
        if not np.isfinite(new_rate) or new_rate <= -0.999999:
            new_rate = (rate + max(-0.9, min(1.0, new_rate))) / 2
        if abs(new_rate - rate) < 1e-9:
            rate = new_rate
            break
        rate = new_rate
    if not np.isfinite(rate) or rate <= -0.999999:
        return None
    return float(rate)


def normalize_cashflows_from_cashflow_table(*args, **kwargs) -> pd.DataFrame:
    return pd.DataFrame()


def normalize_cashflows_from_summary(*args, **kwargs) -> pd.DataFrame:
    return pd.DataFrame()


def compute_deal_metrics(*args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(), pd.DataFrame()


def compute_portfolio_summary(deals_df: pd.DataFrame) -> Dict[str, float]:
    if deals_df.empty:
        return {
            "invested": 0.0,
            "distributions": 0.0,
            "current_nav": 0.0,
            "dpi": np.nan,
            "rvpi": np.nan,
            "tvpi": np.nan,
            "num_deals": 0,
            "realized_pct": 0.0,
            "loss_ratio": np.nan,
        }
    invested = float(deals_df["invested"].sum())
    distributions = float(deals_df["distributions"].sum())
    nav = float(deals_df["current_nav"].sum())
    invested_safe = invested if invested != 0 else np.nan
    tvpi = (distributions + nav) / invested_safe if invested_safe else np.nan
    dpi = distributions / invested_safe if invested_safe else np.nan
    rvpi = nav / invested_safe if invested_safe else np.nan
    num_deals = int(deals_df.shape[0])
    realized_pct = float((deals_df["status"] == "Realized").mean()) if "status" in deals_df.columns else np.nan
    loss_ratio = float((deals_df["tvpi"] < 1.0).mean()) if "tvpi" in deals_df.columns else np.nan
    return {
        "invested": invested,
        "distributions": distributions,
        "current_nav": nav,
        "dpi": dpi,
        "rvpi": rvpi,
        "tvpi": tvpi,
        "num_deals": num_deals,
        "realized_pct": realized_pct,
        "loss_ratio": loss_ratio,
    }


def filter_deals(
    deals_df: pd.DataFrame,
    sector: Optional[List[str]] = None,
    geography: Optional[List[str]] = None,
    status: Optional[List[str]] = None,
    vintage_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    out = deals_df.copy()
    if sector:
        out = out[out["sector"].isin(sector)]
    if geography:
        out = out[out["geography"].isin(geography)]
    if status:
        out = out[out["status"].isin(status)]
    if vintage_range and "vintage_year" in out.columns and out["vintage_year"].notna().any():
        lo, hi = vintage_range
        out = out[(out["vintage_year"] >= lo) & (out["vintage_year"] <= hi)]
    return out


def build_templates() -> Dict[str, pd.DataFrame]:
    # Cash flows template
    cashflows = pd.DataFrame(
        {
            "Deal": ["Alpha", "Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"],
            "Date": [
                "2018-03-01",
                "2019-01-15",
                "2022-10-30",
                "2019-06-10",
                "2023-09-01",
                "2020-02-20",
                "2024-12-31",
            ],
            "Amount": [-5000000, -1000000, 3500000, -7000000, 10500000, -4000000, 4500000],
            "Flow Type": [
                "Contribution",
                "Contribution",
                "Distribution",
                "Contribution",
                "Distribution",
                "Contribution",
                "NAV",
            ],
            "Sector": ["Healthcare", "Healthcare", "Healthcare", "Tech", "Tech", "Consumer", "Consumer"],
            "Geography": ["US", "US", "US", "EU", "EU", "US", "US"],
        }
    )

    # Deal summary template
    summary = pd.DataFrame(
        {
            "Deal": ["Alpha", "Beta", "Gamma"],
            "Investment Date": ["2018-03-01", "2019-06-10", "2020-02-20"],
            "Invested": [6000000, 7000000, 4000000],
            "Exit Date": ["2022-10-30", "2023-09-01", ""],
            "Proceeds": [3500000, 10500000, ""],
            "Current Value": [2500000, 0, 4500000],
            "Valuation Date": ["2024-12-31", "", "2024-12-31"],
            "Sector": ["Healthcare", "Tech", "Consumer"],
            "Geography": ["US", "EU", "US"],
        }
    )

    return {"cashflows_template": cashflows, "deal_summary_template": summary}


# ===== Operational & Valuation metrics detection =====

def _normalize_header(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
    )


def _find_first_matching(cols: List[str], patterns: List[str]) -> Optional[str]:
    for c in cols:
        lc = _normalize_header(c)
        if all(p in lc for p in patterns):
            return c
    return None


def _find_metric_pair(cols: List[str], metric_terms: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (entry_col, exit_col) based on header keywords."""
    entry_kw = ["entry", "at entry", "invest", "investment", "initial", "pre"]
    exit_kw = ["exit", "at exit", "realized", "sale", "current", "post"]

    entry_col = None
    exit_col = None

    for c in cols:
        lc = _normalize_header(c)
        if any(term in lc for term in metric_terms):
            if any(k in lc for k in entry_kw) and entry_col is None:
                entry_col = c
            if any(k in lc for k in exit_kw) and exit_col is None:
                exit_col = c

    # Fallback: if only one metric column found twice with explicit 'entry'/'exit' missing, try best guess by suffix numbers
    if entry_col is None or exit_col is None:
        candidates = [c for c in cols if any(t in _normalize_header(c) for t in metric_terms)]
        if len(candidates) >= 2 and (entry_col is None or exit_col is None):
            # pick first as entry, last as exit
            entry_col = entry_col or candidates[0]
            exit_col = exit_col or candidates[-1]
    return entry_col, exit_col


def extract_operational_and_valuation_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    # Removed heuristic extractor; using template-order extractor only.
    return pd.DataFrame(), {}


def add_growth_and_cagr(
    ops_df: pd.DataFrame, holding_years: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Compute growth % and CAGR for revenue, EBITDA, TEV, Net Debt."""
    if ops_df is None or ops_df.empty:
        return ops_df

    df = ops_df.copy()

    # Ensure date columns are proper datetimes and fill missing exit_date with current month-end
    if "invest_date" in df.columns:
        df["invest_date"] = parse_to_datetime(df["invest_date"])  # type: ignore[arg-type]
    if "exit_date" in df.columns:
        df["exit_date"] = parse_to_datetime(df["exit_date"])  # type: ignore[arg-type]
    from pandas.tseries.offsets import MonthEnd as _MonthEnd
    if "exit_date" in df.columns:
        df["exit_date"] = df["exit_date"].fillna(pd.Timestamp("now").normalize() + _MonthEnd(0))

    def growth(entry: pd.Series, exit_: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            return (exit_ - entry) / entry.replace({0: np.nan})

    def cagr(entry: pd.Series, exit_: pd.Series, years: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            return (exit_.replace({0: np.nan}) / entry.replace({0: np.nan})) ** (1 / years) - 1

    # Holding years priority: passed arg; else compute from dates if present
    years = None
    if holding_years is not None:
        years = pd.to_numeric(holding_years, errors="coerce")
    elif "invest_date" in df.columns and "exit_date" in df.columns:
        # Compute only when both are datetimes
        try:
            years = (df["exit_date"] - df["invest_date"]).dt.days / 365.2425
        except Exception:
            years = pd.Series([np.nan] * len(df))

    # Persist holding period in years
    if years is not None:
        df["holding_years"] = pd.to_numeric(years, errors="coerce")
    else:
        df["holding_years"] = np.nan

    # Revenue
    df["revenue_growth_pct"] = growth(df.get("entry_revenue"), df.get("exit_revenue"))
    # EBITDA: guard against negative or zero values (percent growth undefined)
    e_entry = pd.to_numeric(df.get("entry_ebitda"), errors="coerce")
    e_exit = pd.to_numeric(df.get("exit_ebitda"), errors="coerce")
    mask_pos = (e_entry > 0) & (e_exit > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ebitda_growth_pct"] = np.where(mask_pos, (e_exit - e_entry) / e_entry, np.nan)
    df["tev_growth_pct"] = growth(df.get("entry_tev"), df.get("exit_tev"))
    df["net_debt_change_pct"] = growth(df.get("entry_net_debt"), df.get("exit_net_debt"))

    if years is not None:
        yrs = pd.to_numeric(years, errors="coerce")
        # guard against non-positive years
        yrs = yrs.where(yrs > 0)
        df["revenue_cagr"] = cagr(df.get("entry_revenue"), df.get("exit_revenue"), yrs)
        # EBITDA CAGR only when both entry and exit are positive and years > 0
        df["ebitda_cagr"] = np.where(
            mask_pos & (yrs > 0),
            (e_exit.replace({0: np.nan}) / e_entry.replace({0: np.nan})) ** (1 / yrs) - 1,
            np.nan,
        )
        df["tev_cagr"] = cagr(df.get("entry_tev"), df.get("exit_tev"), yrs)
    else:
        df["revenue_cagr"] = np.nan
        df["ebitda_cagr"] = np.nan
        df["tev_cagr"] = np.nan

    return df


# ===== Fixed-order template based extraction =====

def _classify_header_to_role(header: str) -> Optional[str]:
    h = _normalize_header(header)
    # Exact alias mapping for Portfolio Metrics template
    exact_alias = {
        "portfolio company": "portfolio_company",
        "company name": "portfolio_company",
        "sector": "sector",
        "key vertical": "sector",
        "vertical": "sector",
        "vertical description": "sector",
        "region of majority operations": "geography",
        "company country": "geography",
        "country (hq)": "geography",
        "acquisition financial date": "invest_date",
        "acquisition/financial close": "invest_date",
        "date invested": "invest_date",
        "date investment": "invest_date",
        "date realized": "exit_date",
        "date final exit": "exit_date",
        "total invested cost ($mm)": "invested",
        "total invested cost ($m)": "invested",
        "total investment cost ($mm)": "invested",
        "total investment cost ($m)": "invested",
        "gross proceeds realized": "proceeds",
        "gross proceeds distributed to fund": "proceeds",
        "current or fair value ($mm)": "current_value",
        "current or fair value ($m)": "current_value",
        "current fair value ($mm)": "current_value",
        "current fair value ($m)": "current_value",
        "moic (gross)": "gross_moic",
        "irr (gross)": "gross_irr",
        # Entry metrics
        "entry revenue ($mm)": "entry_revenue",
        "revenue (entry) ($mm)": "entry_revenue",
        "entry ebitda ($mm)": "entry_ebitda",
        "ebitda (entry) ($mm)": "entry_ebitda",
        "entry enterprise value ($mm)": "entry_tev",
        "entry ev ($mm)": "entry_tev",
        "enterprise value (entry) ($mm)": "entry_tev",
        "entry net debt ($mm)": "entry_net_debt",
        "net debt (entry) ($mm)": "entry_net_debt",
        "entry ev/ebitda": "entry_tev_ebitda",
        # Current/Exit metrics
        "current/exit revenue ($mm)": "exit_revenue",
        "current revenue ($mm)": "exit_revenue",
        "exit revenue ($mm)": "exit_revenue",
        "revenue (exit/current) ($mm)": "exit_revenue",
        "current/exit ebitda ($mm)": "exit_ebitda",
        "current ebitda ($mm)": "exit_ebitda",
        "exit ebitda ($mm)": "exit_ebitda",
        "ebitda (exit/current) ($mm)": "exit_ebitda",
        "current/exit enterprise value ($mm)": "exit_tev",
        "current enterprise value ($mm)": "exit_tev",
        "exit enterprise value ($mm)": "exit_tev",
        "current/exit ev ($mm)": "exit_tev",
        "enterprise value (exit/current) ($mm)": "exit_tev",
        "current/exit net debt ($mm)": "exit_net_debt",
        "current net debt ($mm)": "exit_net_debt",
        "exit net debt ($mm)": "exit_net_debt",
        "net debt (exit/current) ($mm)": "exit_net_debt",
        "current/exit ev/ebitda": "exit_tev_ebitda",
        "multiple (entry)": "entry_tev_ebitda",
        "multiple (exit/current) ($mm)": "exit_tev_ebitda",
        "fund position status": "status",
        "kam vertical": "sector",
        "most recently available financial statements": "valuation_date",
    }
    if h in exact_alias:
        return exact_alias[h]
    # core identifiers
    if "portfolio company" in h:
        return "portfolio_company"
    if ("company" in h and "id" not in h) and "portfolio" not in h:
        return "portfolio_company"
    if "sector" in h or "industry" in h:
        return "sector"
    if "geograph" in h or "region" in h or "country" in h:
        return "geography"
    if "investment" in h and "date" in h or ("entry" in h and "date" in h):
        return "invest_date"
    if ("exit" in h and "date" in h) or ("realization" in h and "date" in h):
        return "exit_date"
    if ("valuation" in h and "date" in h):
        return "valuation_date"

    # cash / value fields
    if "invested" in h or ("investment" in h and ("amount" in h or "cost" in h or "capital" in h)):
        return "invested"
    if "proceeds" in h or ("realization" in h and ("value" in h or "proceeds" in h)):
        return "proceeds"
    if ("current" in h and ("value" in h or "nav" in h)) or ("nav" in h):
        return "current_value"

    # operating and valuation metrics
    is_entry = ("entry" in h) or ("invest" in h and "date" not in h) or ("initial" in h) or ("pre" in h)
    is_exit = ("exit" in h) or ("realiz" in h) or ("current" in h) or ("post" in h)
    if "revenue" in h or "sales" in h:
        return "entry_revenue" if is_entry else ("exit_revenue" if is_exit else None)
    if "ebitda" in h:
        return "entry_ebitda" if is_entry else ("exit_ebitda" if is_exit else None)
    if ("tev" in h) or ("enterprise" in h and "value" in h) or (h.strip() == "ev"):
        return "entry_tev" if is_entry else ("exit_tev" if is_exit else None)
    if ("net" in h and "debt" in h) or (h.replace(" ", "") in {"nd", "netdebt"}):
        return "entry_net_debt" if is_entry else ("exit_net_debt" if is_exit else None)
    if "moic" in h and "gross" in h or h.strip() == "moic":
        return "gross_moic"
    if "irr" in h and "gross" in h or h.strip() == "irr":
        return "gross_irr"
    return None


def extract_operational_by_template_order(
    df_uploaded: pd.DataFrame,
    template_headers: List[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Map uploaded df columns to canonical metrics using fixed column positions per provided template order."""
    if df_uploaded is None or df_uploaded.empty or not template_headers:
        return pd.DataFrame(), {}
    role_to_series: Dict[str, pd.Series] = {}
    mapping_used: Dict[str, str] = {}

    # Fixed position-to-role mapping (1-based positions per user's list)
    pos_to_role: Dict[int, str] = {
        1: "portfolio_company",
        2: "invest_date",
        3: "fund_name",  # Fund Name (GP)
        30: "equity_entry_total",  # Equity (Entry) ($MM)
        31: "kam_equity_entry",    # Kam Equity (Entry) ($MM)
        6: "geography",  # Country (HQ)
        7: "region",     # Region of majority operations (optional)
        8: "sector",     # Kam Vertical
        11: "investment_strategy",
        12: "instrument_type",
        14: "purchase_process",
        18: "exit_date",
        20: "status",    # Fund Position Status
        19: "exit_type",
        38: "valuation_date",  # Most Recently Available Financial Statements
        24: "invested",
        25: "proceeds",
        26: "current_value",
        28: "gross_moic",
        29: "gross_irr",
        48: "kam_ownership_exit_pct",  # Kam Equity Ownership Fund (Exit/Current)
        32: "entry_net_debt",
        33: "entry_tev",
        34: "entry_revenue",
        35: "entry_ebitda",
        36: "entry_tev_ebitda",
        40: "exit_net_debt",
        41: "exit_tev",
        42: "exit_revenue",
        43: "exit_ebitda",
        44: "exit_tev_ebitda",
    }

    numeric_roles = {
        "entry_revenue",
        "exit_revenue",
        "entry_ebitda",
        "exit_ebitda",
        "entry_tev",
        "exit_tev",
        "entry_net_debt",
        "exit_net_debt",
        "gross_moic",
        "gross_irr",
        "invested",
        "proceeds",
        "current_value",
        "entry_tev_ebitda",
        "exit_tev_ebitda",
        "equity_entry_total",
        "kam_equity_entry",
        "kam_ownership_exit_pct",
    }

    for pos, role in pos_to_role.items():
        idx = pos - 1
        if idx < 0 or idx >= df_uploaded.shape[1]:
            continue
        col_series = df_uploaded.iloc[:, idx]
        if role in numeric_roles:
            role_to_series[role] = pd.to_numeric(
                col_series.astype(str)
                .str.replace(r"[,$%]", "", regex=True)
                .str.replace(r"^\((.*)\)$", r"-\1", regex=True),
                errors="coerce",
            )
        elif role in {"invest_date", "exit_date", "valuation_date"}:
            role_to_series[role] = parse_to_datetime(col_series)
        else:
            role_to_series[role] = col_series.astype(str)
        mapping_used[role] = f"pos[{idx}]"

    # Build ops df with required identifier
    out = pd.DataFrame(
        {
            "portfolio_company": role_to_series.get(
                "portfolio_company",
                df_uploaded.iloc[:, 0].astype(str) if df_uploaded.shape[1] > 0 else pd.Series(dtype=str),
            )
        }
    )
    # Text attributes
    def add_text(name: str, role_key: str):
        if role_key in role_to_series:
            out[name] = role_to_series[role_key]
        else:
            out[name] = np.nan
    def add(name: str, role_key: str):
        if role_key in role_to_series:
            out[name] = role_to_series[role_key]
        else:
            out[name] = np.nan

    add_text("fund_name", "fund_name")
    add_text("investment_strategy", "investment_strategy")
    add_text("instrument_type", "instrument_type")
    add_text("purchase_process", "purchase_process")
    add_text("exit_type", "exit_type")
    add("sector", "sector")
    add("geography", "geography")
    add("invest_date", "invest_date")
    add("exit_date", "exit_date")
    # Portfolio metrics specific: allow status and valuation_date from headers
    add("status", "status")
    add("valuation_date", "valuation_date")
    add("entry_revenue", "entry_revenue")
    add("exit_revenue", "exit_revenue")
    add("entry_ebitda", "entry_ebitda")
    add("exit_ebitda", "exit_ebitda")
    add("entry_tev", "entry_tev")
    add("exit_tev", "exit_tev")
    add("entry_net_debt", "entry_net_debt")
    add("exit_net_debt", "exit_net_debt")
    add("gross_moic", "gross_moic")
    add("gross_irr", "gross_irr")
    add("invested", "invested")
    add("proceeds", "proceeds")
    add("current_value", "current_value")
    add("equity_entry_total", "equity_entry_total")
    add("kam_equity_entry", "kam_equity_entry")
    add("kam_ownership_exit_pct", "kam_ownership_exit_pct")
    # Equity fields are optional but included for completeness
    add("equity_entry", "equity_entry")
    add("equity_exit", "equity_exit")

    # Multiples
    with np.errstate(divide="ignore", invalid="ignore"):
        # Multiples defined only when EBITDA > 0
        out["entry_tev_ebitda"] = np.where(out["entry_ebitda"] > 0, out["entry_tev"] / out["entry_ebitda"], np.nan)
        out["exit_tev_ebitda"] = np.where(out["exit_ebitda"] > 0, out["exit_tev"] / out["exit_ebitda"], np.nan)
        # Leverage ratios and TEV/Revenue
        # Allow negative net debt; require positive EBITDA denominator
        out["entry_leverage"] = np.where(out["entry_ebitda"] > 0, out["entry_net_debt"] / out["entry_ebitda"], np.nan)
        out["exit_leverage"] = np.where(out["exit_ebitda"] > 0, out["exit_net_debt"] / out["exit_ebitda"], np.nan)
        out["entry_tev_revenue"] = out["entry_tev"] / out["entry_revenue"]
        out["exit_tev_revenue"] = out["exit_tev"] / out["exit_revenue"]

    return out, mapping_used


