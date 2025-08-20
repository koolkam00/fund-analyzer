from __future__ import annotations

import pandas as pd
import streamlit as st

from data_loader import ensure_workbook_loaded


st.set_page_config(page_title="Fund Manager Profile", layout="wide")

st.title("Fund Manager Profile")
st.caption("High-level overview of the GP/firm from the uploaded workbook.")

sheets, _, _, _ = ensure_workbook_loaded()
if not sheets:
    st.info("Upload the master workbook to view the manager profile.")
    st.stop()

# Identify the profile sheet by common names or position
sheet_names = list(sheets.keys())
mgr_sheet_name = None
for nm in sheet_names:
    low = str(nm).strip().lower()
    if low in {"fund manager profile", "manager profile", "fund manager"}:
        mgr_sheet_name = nm
        break
if mgr_sheet_name is None and len(sheet_names) > 4:
    mgr_sheet_name = sheet_names[4]

df_mgr = sheets.get(mgr_sheet_name, pd.DataFrame())
if df_mgr.empty:
    st.info("Manager profile sheet not found or is empty.")
    st.stop()

def _norm(s: str) -> str:
    return str(s).strip()

df = df_mgr.copy()
df.columns = [_norm(c) for c in df.columns]
row = df.iloc[0].fillna("")

# Pull key fields
firm = str(row.get("Firm Name", row.get("firm name", row.get("name", ""))))
hq = str(row.get("Headquarters", row.get("headquarters", "")))
aum = row.get("AUM ($MM)", row.get("aum ($mm)", ""))
strategy = str(row.get("Strategy Focus", row.get("strategy focus", "")))
region = str(row.get("Region Focus", row.get("region focus", "")))
founded = str(row.get("Year Founded", row.get("year founded", "")))
team = str(row.get("Team Size", row.get("team size", "")))
email = str(row.get("Contact Email", row.get("contact email", "")))
site = str(row.get("Website", row.get("website", "")))
desc = str(row.get("Description", row.get("description", "")))

# Header card
st.markdown(
    f"""
    <style>
    .firm-card {{
        border: 1px solid #e4e8f0;
        border-radius: 8px;
        padding: 14px 16px;
        background: #fafbfe;
        margin-bottom: 12px;
    }}
    .firm-title {{
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }}
    .firm-sub {{
        color: #57606a;
        margin: 0 0 6px 0;
    }}
    .firm-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 10px 16px;
        margin-top: 8px;
    }}
    .firm-item .label {{ color: #6e7781; font-size: 0.85rem; }}
    .firm-item .value {{ font-weight: 600; }}
    </style>
    <div class="firm-card">
      <div class="firm-title">{firm}</div>
      <div class="firm-sub">{hq}</div>
      <div class="firm-grid">
        <div class="firm-item"><div class="label">AUM</div><div class="value">{(f"${float(aum):,.1f}MM" if str(aum).strip() else "—")}</div></div>
        <div class="firm-item"><div class="label">Strategy</div><div class="value">{strategy or "—"}</div></div>
        <div class="firm-item"><div class="label">Region</div><div class="value">{region or "—"}</div></div>
        <div class="firm-item"><div class="label">Founded</div><div class="value">{founded or "—"}</div></div>
        <div class="firm-item"><div class="label">Team Size</div><div class="value">{team or "—"}</div></div>
        <div class="firm-item"><div class="label">Email</div><div class="value">{email or "—"}</div></div>
        <div class="firm-item"><div class="label">Website</div><div class="value">{site or "—"}</div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Description box
if desc and desc.strip():
    st.markdown("**Description**")
    st.write(desc)



