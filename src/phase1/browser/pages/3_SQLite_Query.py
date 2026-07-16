"""Page 3 — run ad-hoc SQL against any local Spider SQLite database."""

import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_ROOT = Path(os.environ.get("SPIDER_DATA_DIR", "data")).resolve()
DATABASES_DIR = DATA_ROOT / "databases"
st.set_page_config(
    page_title="SQLite Query",
    layout="wide",
)

st.title("SQLite Query Runner")


@st.cache_data(show_spinner=False)
def list_databases() -> list[str]:
    if not DATABASES_DIR.exists():
        return []
    return sorted(p.parent.name for p in DATABASES_DIR.glob("*/*.sqlite"))


@st.cache_data(show_spinner=False)
def get_table_names(db_id: str) -> list[str]:
    path = DATABASES_DIR / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


all_dbs = list_databases()

if not all_dbs:
    st.error("No SQLite databases found. Run the download script first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
selected_db = st.sidebar.selectbox("Database", all_dbs)
db_path = DATABASES_DIR / selected_db / f"{selected_db}.sqlite"

table_names = get_table_names(selected_db)
st.sidebar.markdown("**Tables**")
for t in table_names:
    st.sidebar.code(t, language=None)

# ── Quick-select shortcuts ────────────────────────────────────────────────────
st.caption(f"Connected to `{db_path}`")

shortcut_table = st.selectbox(
    "Quick preview table (fills query box)",
    ["— pick a table —"] + table_names,
    label_visibility="collapsed",
)

default_sql = (
    f'SELECT * FROM "{shortcut_table}" LIMIT 50;'
    if shortcut_table != "— pick a table —"
    else "SELECT * FROM sqlite_master WHERE type='table';"
)

# ── Query editor ──────────────────────────────────────────────────────────────
sql = st.text_area("SQL", value=default_sql, height=140)
row_limit = st.slider("Row limit", min_value=10, max_value=500, value=100, step=10)
run = st.button("Run", type="primary")

if run:
    try:
        conn = sqlite3.connect(db_path)
        query = sql.strip().rstrip(";")
        if "limit" not in query.lower():
            query = f"SELECT * FROM ({query}) AS _q LIMIT {row_limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        st.success(f"{len(df):,} rows returned")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, file_name=f"{selected_db}_result.csv", mime="text/csv")

    except Exception as exc:
        st.error(f"**Error:** {exc}")
