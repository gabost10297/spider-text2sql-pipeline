"""Llama-3 Instruct prompt templates for Text-to-SQL.

Training: schema + question + golden SQL.
Inference: schema + question, assistant turn left open after ``SQL: ``.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from src.phase1.paths import HF_EXPORT_DIR, SPIDER_ROOT
from src.phase2.schema_linker import extract_schema_by_db_id

SYSTEM_MESSAGE = (
    "You are a precise database administrator. Translate the user question "
    "into an executable SQL query using the schema provided."
)

PROMPTS_DIR = SPIDER_ROOT / "prompts"


@lru_cache(maxsize=256)
def get_schema(db_id: str) -> str:
    """Cached schema text for *db_id* (live SQLite extract)."""
    return extract_schema_by_db_id(db_id)


def generate_training_prompt(
    schema: str,
    question: str,
    sql: str | None = None,
) -> str:
    """Build a Llama-3 Instruct chat prompt.

    If *sql* is provided, the assistant turn is closed with ``<|eot_id|>``
    (training). Otherwise the prompt ends at ``SQL: `` (inference).
    """
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_MESSAGE}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Database Schema:\n{schema}\n"
        f"Question: {question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "SQL: "
    )
    if sql is not None:
        prompt += f"{sql.strip()}<|eot_id|>"
    return prompt


def generate_prompt_for_example(
    db_id: str,
    question: str,
    sql: str | None = None,
) -> str:
    """Resolve schema for *db_id* and format the full prompt."""
    return generate_training_prompt(get_schema(db_id), question, sql)


def _load_split(split: str):
    import pandas as pd

    path = HF_EXPORT_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: docker compose run --rm phase1 "
            "python -m src.phase1.download_spider"
        )
    return pd.read_parquet(path)


def export_split_prompts(split: str, out_path: Path, *, include_sql: bool = True) -> int:
    """Write one JSONL record per example: db_id, question, query, prompt."""
    df = _load_split(split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            sql = row.query if include_sql else None
            prompt = generate_prompt_for_example(row.db_id, row.question, sql)
            record = {
                "db_id": row.db_id,
                "question": row.question,
                "query": row.query,
                "prompt": prompt,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Llama-3 Instruct Text-to-SQL prompts"
    )
    parser.add_argument("--db-id", help="Database id for ad-hoc prompt")
    parser.add_argument("--question", help="Natural-language question")
    parser.add_argument("--sql", help="Golden SQL (training mode); omit for inference")
    parser.add_argument(
        "--from-parquet",
        metavar="SPLIT",
        choices=["train", "validation"],
        help="Load one example from a parquet split",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index when using --from-parquet (default: 0)",
    )
    parser.add_argument(
        "--export-split",
        metavar="SPLIT",
        choices=["train", "validation"],
        help="Export all prompts for a split to JSONL",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export train + validation prompts to data/spider/prompts/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path for --export-split (default: data/spider/prompts/{split}.jsonl)",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="With --from-parquet / --export-split / --export-all: omit golden SQL from the prompt",
    )
    args = parser.parse_args()

    if args.export_all:
        for split in ("train", "validation"):
            out = PROMPTS_DIR / f"{split}.jsonl"
            n = export_split_prompts(split, out, include_sql=not args.inference)
            print(f"Wrote {n:,} prompts → {out}")
        return

    if args.export_split:
        out = args.out or (PROMPTS_DIR / f"{args.export_split}.jsonl")
        n = export_split_prompts(
            args.export_split, out, include_sql=not args.inference
        )
        print(f"Wrote {n:,} prompts → {out}")
        return

    if args.from_parquet:
        df = _load_split(args.from_parquet)
        if args.index < 0 or args.index >= len(df):
            raise SystemExit(f"--index out of range (0..{len(df) - 1})")
        row = df.iloc[args.index]
        sql = None if args.inference else row["query"]
        text = generate_prompt_for_example(row["db_id"], row["question"], sql)
        print(text)
        return

    if args.db_id and args.question:
        text = generate_prompt_for_example(args.db_id, args.question, args.sql)
        print(text)
        return

    parser.print_help()
    print(
        "\nExamples:\n"
        "  python -m src.phase2.prompts --from-parquet train --index 0\n"
        "  python -m src.phase2.prompts --db-id concert_singer "
        '--question "How many singers do we have?"\n'
        "  python -m src.phase2.prompts --export-split train\n"
        "  python -m src.phase2.prompts --export-all\n"
    )


if __name__ == "__main__":
    main()
