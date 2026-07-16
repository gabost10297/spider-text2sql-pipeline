"""Page 2 — browse Spider database schemas from tables.json."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_ROOT = Path(os.environ.get("SPIDER_DATA_DIR", "data")).resolve()
TABLES_JSON = DATA_ROOT / "spider" / "tables.json"
st.set_page_config(
    page_title="Schema Explorer",
    layout="wide",
)

st.title("Schema Explorer")


@st.cache_data(show_spinner="Loading tables.json…")
def load_tables() -> list[dict]:
    if not TABLES_JSON.exists():
        return []
    with TABLES_JSON.open(encoding="utf-8") as f:
        return json.load(f)


tables_data = load_tables()

if not tables_data:
    st.error("tables.json not found. Run the download script first.")
    st.stop()

db_map = {entry["db_id"]: entry for entry in tables_data}
all_dbs = sorted(db_map.keys())

selected_db = st.sidebar.selectbox("Database", all_dbs)
entry = db_map[selected_db]

# ── Header ────────────────────────────────────────────────────────────────────
st.subheader(f"`{selected_db}`")
col1, col2 = st.columns(2)
col1.metric("Tables", len(entry.get("table_names_original", [])))
col2.metric("Columns", len(entry.get("column_names_original", [])) - 1)  # -1 for the * sentinel

# ── Build per-table view ───────────────────────────────────────────────────────
table_names = entry.get("table_names_original", [])
col_names = entry.get("column_names_original", [])   # [table_idx, col_name]
col_types = entry.get("column_types", [])
pk_cols = set(entry.get("primary_keys", []))
fk_pairs = entry.get("foreign_keys", [])             # [[col_idx, ref_col_idx], ...]

fk_from: dict[int, str] = {}
for src, dst in fk_pairs:
    dst_table_idx, dst_col = col_names[dst]
    fk_from[src] = f"{table_names[dst_table_idx]}.{dst_col}"

for t_idx, t_name in enumerate(table_names):
    with st.expander(f"**{t_name}**", expanded=len(table_names) <= 6):
        rows = []
        for c_idx, (tbl_idx, col_name) in enumerate(col_names):
            if tbl_idx != t_idx:
                continue
            rows.append(
                {
                    "Column": col_name,
                    "Type": col_types[c_idx] if c_idx < len(col_types) else "",
                    "PK": "yes" if c_idx in pk_cols else "",
                    "FK ->": fk_from.get(c_idx, ""),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("Raw JSON entry"):
    st.json(entry)
