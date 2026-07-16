"""Page 1 — browse train / validation Q+SQL pairs."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_ROOT = Path(os.environ.get("SPIDER_DATA_DIR", "data")).resolve()
HF_EXPORT_DIR = DATA_ROOT / "spider" / "hf_export"
st.set_page_config(
    page_title="Splits Browser",
    layout="wide",
)

st.title("Splits Browser")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading parquet…")
def load_split(name: str) -> pd.DataFrame:
    path = HF_EXPORT_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


train_df = load_split("train")
val_df = load_split("validation")

if train_df.empty and val_df.empty:
    st.error("No parquet files found. Run the download script first.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
split_choice = st.sidebar.radio("Split", ["train", "validation"])
df = train_df if split_choice == "train" else val_df

all_dbs = sorted(df["db_id"].unique().tolist())
selected_db = st.sidebar.selectbox("Filter by db_id", ["(all)"] + all_dbs)
keyword = st.sidebar.text_input("Search question", placeholder="e.g. average age")
page_size = st.sidebar.select_slider("Rows per page", options=[10, 25, 50, 100], value=25)

# ── Filter ────────────────────────────────────────────────────────────────────
filtered = df.copy()
if selected_db != "(all)":
    filtered = filtered[filtered["db_id"] == selected_db]
if keyword.strip():
    filtered = filtered[filtered["question"].str.contains(keyword.strip(), case=False, na=False)]

total = len(filtered)
n_pages = max(1, (total - 1) // page_size + 1)
page = st.sidebar.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)

st.caption(f"**{total:,}** examples  ·  page {page}/{n_pages}")

start = (page - 1) * page_size
page_df = filtered.iloc[start : start + page_size].reset_index(drop=True)

# ── Display ───────────────────────────────────────────────────────────────────
for _, row in page_df.iterrows():
    with st.expander(f"[{row['db_id']}]  {row['question']}"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Question**")
            st.write(row["question"])
            st.markdown("**Database**")
            st.code(row["db_id"], language=None)
        with col2:
            st.markdown("**Golden SQL**")
            st.code(row["query"], language="sql")
