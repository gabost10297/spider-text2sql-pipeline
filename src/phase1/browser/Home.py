"""Spider dataset browser — home page."""

import json
import os
from pathlib import Path

import streamlit as st

DATA_ROOT = Path(os.environ.get("SPIDER_DATA_DIR", "data")).resolve()
DOWNLOAD_MARKER = DATA_ROOT / "spider" / ".download_complete"
HF_EXPORT_DIR = DATA_ROOT / "spider" / "hf_export"
DATABASES_DIR = DATA_ROOT / "databases"
SCHEMAS_DIR = DATA_ROOT / "spider" / "schemas"
PROMPTS_DIR = DATA_ROOT / "spider" / "prompts"

st.set_page_config(
    page_title="Spider Browser",
    layout="wide",
)

st.title("Spider Dataset Browser")
st.caption("Local data & prompt explorer")

# ── Phase 1 status ────────────────────────────────────────────────────────────
st.subheader("Phase 1")
col1, col2, col3 = st.columns(3)

train_parquet = HF_EXPORT_DIR / "train.parquet"
val_parquet = HF_EXPORT_DIR / "validation.parquet"
n_dbs = sum(1 for _ in DATABASES_DIR.glob("*/*.sqlite")) if DATABASES_DIR.exists() else 0

with col1:
    if train_parquet.exists():
        import pandas as pd
        n_train = len(pd.read_parquet(train_parquet, columns=["db_id"]))
        st.metric("Train examples", f"{n_train:,}")
    else:
        st.metric("Train examples", "—")

with col2:
    if val_parquet.exists():
        import pandas as pd
        n_val = len(pd.read_parquet(val_parquet, columns=["db_id"]))
        st.metric("Validation examples", f"{n_val:,}")
    else:
        st.metric("Validation examples", "—")

with col3:
    st.metric("SQLite databases", n_dbs if n_dbs else "—")

if DOWNLOAD_MARKER.exists():
    meta = json.loads(DOWNLOAD_MARKER.read_text())
    st.success("Data downloaded")
    with st.expander("Download manifest"):
        st.json(meta)
else:
    st.warning(
        "Data not yet downloaded. Run:  \n"
        "```\ndocker compose run --rm phase1 python -m src.phase1.download_spider\n```"
    )

# ── Phase 2 status ────────────────────────────────────────────────────────────
st.subheader("Phase 2")
p1, p2, p3 = st.columns(3)

n_schemas = len(list(SCHEMAS_DIR.glob("*.txt"))) if SCHEMAS_DIR.exists() else 0
train_jsonl = PROMPTS_DIR / "train.jsonl"
val_jsonl = PROMPTS_DIR / "validation.jsonl"

with p1:
    st.metric("Cached schema files", n_schemas if n_schemas else "—")

with p2:
    if train_jsonl.exists():
        n = sum(1 for _ in train_jsonl.open(encoding="utf-8"))
        st.metric("Train prompts (JSONL)", f"{n:,}")
    else:
        st.metric("Train prompts (JSONL)", "—")

with p3:
    if val_jsonl.exists():
        n = sum(1 for _ in val_jsonl.open(encoding="utf-8"))
        st.metric("Validation prompts (JSONL)", f"{n:,}")
    else:
        st.metric("Validation prompts (JSONL)", "—")

if n_schemas == 0 and not train_jsonl.exists():
    st.info(
        "Phase 2 artifacts optional for browsing (prompts can be generated live).  \n"
        "Schema dump: `python -m src.phase2.schema_linker --all --out data/spider/schemas`  \n"
        "Prompt export: `python -m src.phase2.prompts --export-all`"
    )

st.divider()
st.markdown(
    """
### Pages

| Page | What it shows |
|------|---------------|
| **Splits Browser** | Page through train / validation Q+SQL pairs |
| **Schema Explorer** | Tables, columns, PKs, FKs from `tables.json` |
| **SQLite Query** | Run ad-hoc SQL against any local database |
| **Linked Schemas** | Schema-linker text (live or cached `.txt`) |
| **Prompt Viewer** | Llama-3 prompts (JSONL or generate live) |

Use the sidebar to navigate.
"""
)
