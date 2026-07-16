"""Spider dataset browser — home page."""

import json
import os
from pathlib import Path

import streamlit as st

DATA_ROOT = Path(os.environ.get("SPIDER_DATA_DIR", "data")).resolve()
DOWNLOAD_MARKER = DATA_ROOT / "spider" / ".download_complete"
HF_EXPORT_DIR = DATA_ROOT / "spider" / "hf_export"
DATABASES_DIR = DATA_ROOT / "databases"
st.set_page_config(
    page_title="Spider Browser",
    layout="wide",
)

st.title("Spider Dataset Browser")
st.caption("Raw local data explorer")

# ── Status cards ──────────────────────────────────────────────────────────────
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

# ── Download status ────────────────────────────────────────────────────────────
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

st.divider()
st.markdown(
    """
### Pages

| Page | What it shows |
|------|---------------|
| **Splits Browser** | Page through train / validation Q+SQL pairs, filter by DB or keyword |
| **Schema Explorer** | Browse tables, columns, and foreign keys for any of the 166 databases |
| **SQLite Query** | Run ad-hoc SQL against any local database and see live results |

Use the sidebar to navigate.
"""
)
