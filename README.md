## Private Equity Fund Analyzer

Analyze private equity fund deals from an uploaded Excel/CSV or cash flows workbook. Map your columns using a fixed template, compute deal-level and portfolio metrics, run value creation and PME benchmarking, and visualize results across multiple pages.

### Highlights
- Upload Excel/CSV (multi-sheet support for Excel)
- Fixed template mapping for Portfolio Metrics (headers in row 2 supported); all fields 3–20 are parsed and available as filters (e.g., Fund Name, Fund Currency, Cross-Fund Investment, Country/Region, Sector/Vertical, Company Currency, Strategy, Instrument, Public/Private, Purchase Process/Type, Deal Role, Seller Type, Exit Type, Status)
- Derived metrics: Holding Period, revenue/EBITDA growth and CAGR, TEV/EBITDA, TEV/Revenue, leverage, Net Debt/TEV (LTV), value creation bridge (Revenue Growth, Margin Expansion, Multiple Change, Deleveraging)
- Cash flows (PME page): robust XIRR with multiple solvers, KS‑PME, index MOIC, and segmented summaries
- Consistent compact filters across pages (start empty; update on select)
- Built with Streamlit + Plotly; deployable to Streamlit Cloud

### Quickstart
1. Create and activate a virtual environment (recommended):
   - macOS/Linux:
     ```bash
python3 -m venv .venv && source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```bash
pip install --upgrade pip
pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
streamlit run app.py
   ```

### Data formats

1) Cash flows table (preferred for accurate IRR):

Required columns:
- Deal (string)
- Date (date)
- Amount (number; negative=contribution, positive=distribution). If your file uses positive for investments, you can flip the sign in the app.

Optional columns:
- Flow Type (Contribution/Distribution/NAV). If missing, type is inferred from sign
- Sector (string)
- Geography (string)

2) Deal summary (Portfolio Metrics):

Required columns:
- Deal (string)
- Investment Date (date)
- Invested (number)

Optional columns:
- Exit Date (date)
- Proceeds (number)
- Current Value (number)
- Valuation Date (date; defaults to today if current value is provided without a date)
- Sector (string)
- Geography (string)
- Plus the full template columns (3–20) for filtering

The Portfolio Metrics pages use the fixed-order mapping to compute growth, CAGRs, multiples, leverage, and value creation. PME uses uploaded cash flows directly.

### Pages

1. Track Record
   - Grouped by fund with expandable tables; per-deal details and fund totals
   - Columns include company, sector, status, ownership %, % of fund invested, invested, realized, NAV, realized/unrealized/total MOIC, gross IRR, invest/exit dates, holding years
   - “All Funds — Total” with subtotals by realization status
   - Quick nav to Company Detail

2. Value Creation
   - Deal-level and portfolio-level value creation bridge: Revenue Growth, Margin Expansion, Multiple Change, Deleveraging
   - Per-fund tables show: Portfolio Company, Status, Holding Years, Revenue Growth %, Revenue CAGR, EBITDA Growth %, EBITDA CAGR, EBITDA Margin Change %, Entry/Exit TEV, Entry/Exit Leverage
   - Portfolio waterfall and per-deal waterfall + attribution pie

3. Deal Charts
   - Grouped bar charts comparing entry vs exit metrics per deal (Revenue, EBITDA, EBITDA Margin, TEV/EBITDA, TEV/Revenue, Leverage)
   - Percentile outlier trimming; sorted by largest change in EBITDA

4. Scatter Plots
   - Investment Date on X vs selected metrics on Y; color by Fund; size by Invested
   - Metrics include Revenue/EBITDA/TEV/Net Debt (entry/exit), TEV/EBITDA, TEV/Revenue, Leverage, EBITDA margins, and LTV (Net Debt/TEV at entry/exit)
   - Outlier trimming; standardized hovers

5. Ownership Analysis
   - Entry/Exit ownership % vs MOIC/IRR with outlier trimming and standardized hovers

6. Capital Deployment & Realizations
   - Invested by year (stacked by sector) + WA Entry TEV/EBITDA and TEV/Revenue overlays
   - Realizations by Exit Year uses realized-year logic (proceeds/status and raw Exit Date) to avoid NAV-only placeholders
   - Cumulative Invested vs Proceeds; Invested by Sector

7. App
   - Main data upload and deal-level table with aggregations (Total, Weighted Avg, Average, Median, Max, Min)

8. Company Detail
   - KPIs: Invested, Proceeds, NAV, Total MOIC, Gross IRR
   - Additional KPIs: Status, Invest Date, Exit Date (if realized), Sector, Ownership %
   - New mapped fields as metrics: Fund Currency, Country/Region, Strategy, Instrument, Public/Private, Purchase Process/Type, Deal Role, Seller Type, Exit Type, Fund Name, Company Currency, Vertical Description
   - Tables: Entry/Exit Operating Metrics (with growth/CAGR/margins) and Valuation Multiples & Leverage
   - Value creation waterfall for the deal

9. PME Benchmarking (KS‑PME)
   - Upload cash flows (Date, Type, Value, Portfolio Company, Fund). Downloadable template includes Fund in Column E
   - Calculates KS‑PME per company/fund; By‑Fund tables; segmented summaries (Fully/Partially/Unrealized/Total) for KS‑PME, MOIC, IRR
   - KS‑PME bar by company (hover shows total MOIC, IRR); distribution with threshold line at 1.0
   - Portfolio Summary includes overall KS‑PME/MOIC/IRR and KS‑PME ≥ 1.0 vs < 1.0 breakout (counts and invested/realized/NAV)

### Notes
- IRR can be calculated robustly (multiple solvers) from cash flows; deal pages use metrics provided in the Portfolio Metrics file
- MOIC in PME: (Distributions + last NAV) / |Calls|
- Date parsing handles Excel serials and epoch mis-parses
- Realizations by Exit Year exclude NAV‑only placeholders (use raw Exit Date + proceeds/status)

### Troubleshooting
- If Excel sheets don’t appear: ensure the file is .xlsx and not password protected
- If dates aren’t recognized: confirm header row selection (1‑based) and date column formatting; the app handles common formats and Excel serials
- If IRR shows NaN on PME: ensure at least one negative call and one positive flow (dist or last NAV) per deal

### License
MIT

