"""Page 5 — Phase 2 Llama-3 Instruct prompts / training records."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.phase2.prompts import generate_prompt_for_example

DATA_ROOT = Path(os.environ.get("SPIDER_DATA_DIR", "data")).resolve()
HF_EXPORT_DIR = DATA_ROOT / "spider" / "hf_export"
PROMPTS_DIR = DATA_ROOT / "spider" / "prompts"

st.set_page_config(page_title="Prompt Viewer", layout="wide")
st.title("Prompt Viewer")
st.caption("Llama-3 Instruct training / inference prompts")


@st.cache_data(show_spinner="Loading parquet…")
def load_parquet(split: str) -> pd.DataFrame:
    path = HF_EXPORT_DIR / f"{split}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading JSONL…")
def load_jsonl(split: str) -> pd.DataFrame:
    path = PROMPTS_DIR / f"{split}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


split = st.sidebar.radio("Split", ["train", "validation"])
mode = st.sidebar.radio(
    "Prompt source",
    ["Exported JSONL", "Generate live"],
    help="JSONL from: python -m src.phase2.prompts --export-all",
)
inference = st.sidebar.checkbox(
    "Inference mode (omit golden SQL)",
    value=False,
    disabled=mode == "Exported JSONL",
    help="Only applies when generating live. Exported JSONL is always training-style.",
)

if mode == "Exported JSONL":
    df = load_jsonl(split)
    if df.empty:
        st.warning(
            f"No file at `{PROMPTS_DIR / f'{split}.jsonl'}`. "
            "Run:  \n"
            "```\ndocker compose run --rm phase1 python -m src.phase2.prompts --export-all\n```\n"
            "Or switch to **Generate live**."
        )
        st.stop()
else:
    df = load_parquet(split)
    if df.empty:
        st.error("Parquet splits not found. Run the Phase 1 download script first.")
        st.stop()

all_dbs = sorted(df["db_id"].unique().tolist())
selected_db = st.sidebar.selectbox("Filter by db_id", ["(all)"] + all_dbs)
keyword = st.sidebar.text_input("Search question", placeholder="e.g. average age")
page_size = st.sidebar.select_slider("Rows per page", options=[5, 10, 25, 50], value=10)

filtered = df.copy()
if selected_db != "(all)":
    filtered = filtered[filtered["db_id"] == selected_db]
if keyword.strip():
    filtered = filtered[
        filtered["question"].str.contains(keyword.strip(), case=False, na=False)
    ]

total = len(filtered)
n_pages = max(1, (total - 1) // page_size + 1)
page = st.sidebar.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)

st.caption(f"**{total:,}** examples  ·  page {page}/{n_pages}  ·  source: {mode}")

start = (page - 1) * page_size
page_df = filtered.iloc[start : start + page_size].reset_index(drop=True)

for _, row in page_df.iterrows():
    with st.expander(f"[{row['db_id']}]  {row['question']}"):
        st.markdown("**Question**")
        st.write(row["question"])
        st.markdown("**Golden SQL**")
        st.code(row["query"], language="sql")

        if mode == "Exported JSONL" and "prompt" in row and pd.notna(row["prompt"]):
            prompt = row["prompt"]
        else:
            try:
                sql = None if inference else row["query"]
                prompt = generate_prompt_for_example(row["db_id"], row["question"], sql)
            except Exception as exc:
                st.error(f"Could not build prompt: {exc}")
                continue

        st.markdown("**Full prompt**")
        st.code(prompt, language="text")
