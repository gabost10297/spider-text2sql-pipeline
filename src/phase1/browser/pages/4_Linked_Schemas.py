"""Page 4 — Phase 2 schema linker text for each database."""

import os
from pathlib import Path

import streamlit as st

from src.phase2.schema_linker import extract_schema_by_db_id, list_db_ids

DATA_ROOT = Path(os.environ.get("SPIDER_DATA_DIR", "data")).resolve()
SCHEMAS_DIR = DATA_ROOT / "spider" / "schemas"

st.set_page_config(page_title="Linked Schemas", layout="wide")
st.title("Linked Schemas")
st.caption("Phase 2 — schema linker output used inside LLM prompts")

db_ids = list_db_ids()
if not db_ids:
    st.error("No SQLite databases found. Run the Phase 1 download script first.")
    st.stop()

cached = sorted(p.stem for p in SCHEMAS_DIR.glob("*.txt")) if SCHEMAS_DIR.exists() else []
st.sidebar.metric("Local databases", len(db_ids))
st.sidebar.metric("Cached schema files", len(cached))

selected_db = st.sidebar.selectbox("Database", db_ids)
source = st.sidebar.radio(
    "Source",
    ["Live SQLite extract", "Cached .txt file"],
    help="Cached files come from: python -m src.phase2.schema_linker --all --out data/spider/schemas",
)

cached_path = SCHEMAS_DIR / f"{selected_db}.txt"

try:
    if source.startswith("Cached"):
        if not cached_path.exists():
            st.warning(
                f"No cached file at `{cached_path}`. "
                "Falling back to live extract, or run the schema linker `--all --out` command."
            )
            schema_text = extract_schema_by_db_id(selected_db)
        else:
            schema_text = cached_path.read_text(encoding="utf-8")
    else:
        schema_text = extract_schema_by_db_id(selected_db)
except Exception as exc:
    st.error(f"Failed to load schema: {exc}")
    st.stop()

st.subheader(f"`{selected_db}`")
n_tables = schema_text.count("Table ")
n_fks = schema_text.count("foreign keys")
c1, c2 = st.columns(2)
c1.metric("Tables in prompt text", n_tables)
c2.metric("Tables with FKs", n_fks)

st.code(schema_text, language="text")
