"""Exploratory analysis of the Spider dataset (Phase 1.2–1.3)."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from src.phase1.paths import DATABASES_DIR, DOWNLOAD_MARKER, TABLES_JSON, db_sqlite_path

console = Console()
HF_DATASET_ID = "xlangai/spider"

# Core fields called out in the engineering blueprint
CORE_KEYS = ("question", "db_id", "query")


def _load_hf() -> dict:
    console.print(f"[bold]Loading[/bold] {HF_DATASET_ID} …")
    return load_dataset(HF_DATASET_ID)


def _print_split_overview(dataset) -> None:
    table = Table(title="Dataset splits")
    table.add_column("Split")
    table.add_column("Rows", justify="right")
    table.add_column("Features")
    for name, split in dataset.items():
        table.add_row(name, f"{len(split):,}", ", ".join(split.column_names))
    console.print(table)


def _print_sample_record(dataset) -> None:
    sample = dataset["train"][0]
    console.print(Panel.fit("[bold]Sample training record[/bold] (train[0])"))
    console.print(Pretty(dict(sample), expand_all=True))

    missing = [k for k in CORE_KEYS if k not in sample]
    if missing:
        console.print(f"[red]Missing expected keys:[/red] {missing}")
    else:
        schema = Table(title="Core schema (blueprint §1.3)")
        schema.add_column("Key")
        schema.add_column("Type")
        schema.add_column("Example / notes")
        schema.add_row("question", "str", str(sample["question"])[:80])
        schema.add_row("db_id", "str", str(sample["db_id"]))
        schema.add_row("query", "str", str(sample["query"])[:80])
        console.print(schema)


def _print_db_id_stats(dataset) -> None:
    train_ids = Counter(dataset["train"]["db_id"])
    val_key = "validation" if "validation" in dataset else "dev"
    val_ids = Counter(dataset[val_key]["db_id"]) if val_key in dataset else Counter()

    table = Table(title="Database coverage")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Unique db_id (train)", str(len(train_ids)))
    if val_ids:
        table.add_row(f"Unique db_id ({val_key})", str(len(val_ids)))
        overlap = len(set(train_ids) & set(val_ids))
        table.add_row("Train ∩ val db overlap", str(overlap))
    table.add_row("Most common train db", f"{train_ids.most_common(1)[0]}")
    console.print(table)


def _print_query_complexity(dataset, n: int = 5) -> None:
    df = dataset["train"].to_pandas()
    df["n_joins"] = df["query"].str.lower().str.count(r"\bjoin\b")
    df["has_group_by"] = df["query"].str.lower().str.contains(r"\bgroup\s+by\b", regex=True)
    df["has_subquery"] = df["query"].str.contains(r"\(\s*select", case=False, regex=True)

    stats = Table(title="Rough SQL complexity signals (train)")
    stats.add_column("Signal")
    stats.add_column("Count / rate", justify="right")
    stats.add_row("Rows with ≥1 JOIN", f"{(df['n_joins'] > 0).sum():,} ({(df['n_joins'] > 0).mean():.1%})")
    stats.add_row("Rows with GROUP BY", f"{df['has_group_by'].sum():,} ({df['has_group_by'].mean():.1%})")
    stats.add_row("Rows with subquery", f"{df['has_subquery'].sum():,} ({df['has_subquery'].mean():.1%})")
    console.print(stats)

    console.print(Panel.fit(f"[bold]{n} hardest-looking examples by JOIN count[/bold]"))
    hard = df.sort_values("n_joins", ascending=False).head(n)
    for _, row in hard.iterrows():
        console.print(
            f"[cyan]{row['db_id']}[/cyan] | joins={row['n_joins']}\n"
            f"  Q: {row['question']}\n"
            f"  SQL: {row['query']}\n"
        )


def _inspect_local_sqlite(db_id: str = "concert_singer") -> None:
    path = db_sqlite_path(db_id)
    if not path.exists():
        console.print(
            Panel(
                f"Local SQLite for [bold]{db_id}[/bold] not found at\n{path}\n\n"
                "Run: [bold]python -m src.phase1.download_spider[/bold]",
                title="Databases not downloaded yet",
                style="yellow",
            )
        )
        return

    console.print(Panel.fit(f"[bold]SQLite peek[/bold]: {path}"))
    conn = sqlite3.connect(path)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            conn,
        )["name"].tolist()
        console.print(f"Tables ({len(tables)}): {tables}")

        for table in tables[:3]:
            info = pd.read_sql_query(f"PRAGMA table_info('{table}')", conn)
            cols = ", ".join(f"{r['name']} ({r['type']})" for _, r in info.iterrows())
            n_rows = pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{table}"', conn)["n"].iloc[0]
            console.print(f"  • {table}: {n_rows} rows | {cols}")
    finally:
        conn.close()


def _tables_json_summary() -> None:
    if not TABLES_JSON.exists():
        console.print("[dim]tables.json not present yet (run download_spider).[/dim]")
        return
    with TABLES_JSON.open(encoding="utf-8") as f:
        tables = json.load(f)
    console.print(f"tables.json databases: {len(tables)}")
    sample = tables[0]
    console.print(
        f"Example db_id={sample.get('db_id')} | "
        f"tables={len(sample.get('table_names_original', []))} | "
        f"columns={len(sample.get('column_names_original', []))}"
    )


def explore() -> None:
    if DOWNLOAD_MARKER.exists():
        meta = json.loads(DOWNLOAD_MARKER.read_text(encoding="utf-8"))
        console.print(Panel(Pretty(meta), title="Download marker"))
    else:
        console.print(
            "[yellow]Tip:[/yellow] run [bold]python -m src.phase1.download_spider[/bold] "
            "to fetch SQLite databases before schema-linking work."
        )

    dataset = _load_hf()
    _print_split_overview(dataset)
    _print_sample_record(dataset)
    _print_db_id_stats(dataset)
    _print_query_complexity(dataset)
    _tables_json_summary()
    _inspect_local_sqlite("concert_singer")

    n_local = sum(1 for _ in Path(DATABASES_DIR).glob("*/*.sqlite")) if DATABASES_DIR.exists() else 0
    console.print(
        Panel.fit(
            f"Local SQLite DBs under {DATABASES_DIR}: [bold]{n_local}[/bold]\n"
            "Phase 1 complete when HF splits load and (optionally) databases are present.",
            title="Phase 1 status",
        )
    )


def main() -> None:
    explore()


if __name__ == "__main__":
    main()
