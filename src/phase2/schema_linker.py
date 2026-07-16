"""Automated schema linking: SQLite catalog → structural text for prompts.

Reads tables, columns, types, and foreign keys from a Spider SQLite file
and serialises them into a text block suitable for LLM prompts.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from src.phase1.paths import DATABASES_DIR, db_sqlite_path


def extract_db_schema(db_path: str | Path) -> str:

    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        schema_parts: list[str] = []
        for table in tables:
            # Quote identifiers so names with spaces / reserved words work
            cursor.execute(f'PRAGMA table_info("{table}")')
            columns_info = cursor.fetchall()
            # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
            cols = [f"{col[1]} ({col[2] or 'NULL'})" for col in columns_info]

            cursor.execute(f'PRAGMA foreign_key_list("{table}")')
            fks = cursor.fetchall()
            # PRAGMA foreign_key_list: (id, seq, table, from, to, on_update, on_delete, match)
            fk_strings = [
                f"{table}({fk[3]}) -> {fk[2]}({fk[4]})" for fk in fks
            ]

            table_schema = f"Table {table}: columns = [{', '.join(cols)}]"
            if fk_strings:
                table_schema += f" | foreign keys = [{', '.join(fk_strings)}]"
            schema_parts.append(table_schema)

        return "\n".join(schema_parts)
    finally:
        conn.close()


def extract_schema_by_db_id(db_id: str) -> str:
    """Resolve *db_id* under data/databases/ and extract its schema."""
    return extract_db_schema(db_sqlite_path(db_id))


def list_db_ids() -> list[str]:
    if not DATABASES_DIR.exists():
        return []
    return sorted(
        p.parent.name
        for p in DATABASES_DIR.glob("*/*.sqlite")
        if p.name == f"{p.parent.name}.sqlite"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Spider SQLite schemas into prompt text"
    )
    parser.add_argument(
        "db_id",
        nargs="?",
        help="Database id (e.g. concert_singer). Omit to list all ids.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every local database",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Output path. With a single db_id: write that file. "
            "With --all: treat as a directory and write one {db_id}.txt per database."
        ),
    )
    args = parser.parse_args()

    if args.all:
        ids = list_db_ids()
        if not ids:
            raise SystemExit(
                f"No databases under {DATABASES_DIR}. "
                "Run: docker compose run --rm phase1 python -m src.phase1.download_spider"
            )

        if args.out:
            out_dir = args.out
            out_dir.mkdir(parents=True, exist_ok=True)
            for db_id in ids:
                schema = extract_schema_by_db_id(db_id)
                path = out_dir / f"{db_id}.txt"
                path.write_text(schema, encoding="utf-8")
            print(f"Wrote {len(ids)} schemas → {out_dir}/")
        else:
            blocks = [
                f"=== {db_id} ===\n{extract_schema_by_db_id(db_id)}" for db_id in ids
            ]
            print("\n\n".join(blocks))
        return

    if args.db_id:
        text = extract_schema_by_db_id(args.db_id)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            print(f"Wrote schema → {args.out}")
        else:
            print(text)
        return

    ids = list_db_ids()
    print(f"{len(ids)} databases under {DATABASES_DIR}")
    for db_id in ids:
        print(f"  {db_id}")
    print("\nUsage: python -m src.phase2.schema_linker <db_id>")
    print("       python -m src.phase2.schema_linker <db_id> --out path/to/file.txt")
    print("       python -m src.phase2.schema_linker --all")
    print("       python -m src.phase2.schema_linker --all --out data/spider/schemas")


if __name__ == "__main__":
    main()
