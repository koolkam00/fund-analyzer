## Private Equity Fund Analyzer

Analyze private equity fund deals from an uploaded Excel/CSV. Map your columns, compute deal-level and portfolio metrics, and visualize results.

### Features
- Upload Excel/CSV (multi-sheet support for Excel)
- Flexible column mapping for two data shapes:
  - Cash flows table: Deal, Date, Amount (optional: Flow Type, Sector, Geography)
  - Deal summary: Deal, Investment Date, Invested, Exit Date, Proceeds, Current Value (optional: Sector, Geography)
- Metrics: IRR, MOIC/TVPI, DPI, RVPI, Holding Period, Vintage Year
- Portfolio KPIs and cohort analysis by vintage
- Visualizations: IRR/MOIC distributions, deployment by vintage, scatter IRR vs MOIC, realization bridge
- Filters: sector, geography, status, vintage range
- Download per-deal metrics and portfolio summary
- Built with Streamlit + Plotly

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

2) Deal summary:

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

The app converts deal summaries into synthetic cash flows for analysis.

### Notes
- IRR can be calculated as realized-only or including current NAV as an inflow at valuation date.
- For DPI, only realized distributions are counted. For RVPI, only current NAV is counted.
- MOIC is TVPI at the deal level: (Distributions + Current NAV) / Invested.

### Troubleshooting
- If Excel sheets don’t appear: ensure the file is .xlsx and not password protected.
- If dates aren’t recognized: check your date column formatting; the app contains a parser with common formats.
- If IRR shows NaN: ensure each deal has at least one negative and one positive cash flow (when using NAV-inclusive IRR, current value can serve as the positive flow).

### License
MIT

