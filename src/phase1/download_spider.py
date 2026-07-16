"""Download Spider question/SQL pairs (Hugging Face) and SQLite databases.

Layout after a successful run:

    data/
      spider/
        hf_export/          # train/validation parquet exports
        tables.json         # schema catalog from the official release
        .download_complete
      databases/
        {db_id}/{db_id}.sqlite
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table

from src.phase1.paths import (
    DATABASES_DIR,
    DOWNLOAD_MARKER,
    HF_EXPORT_DIR,
    SPIDER_ROOT,
    TABLES_JSON,
    ensure_data_dirs,
)

console = Console()

HF_DATASET_ID = "xlangai/spider"
HF_DB_REPO_ID = "HAL-9001/spider-databases"
HF_DB_FILENAME = "spider_data.zip"


def _export_hf_splits() -> dict[str, int]:
    console.print(f"[bold]Loading Hugging Face dataset[/bold]: {HF_DATASET_ID}")
    dataset = load_dataset(HF_DATASET_ID)

    counts: dict[str, int] = {}
    for split_name, split in dataset.items():
        out_path = HF_EXPORT_DIR / f"{split_name}.parquet"
        split.to_pandas().to_parquet(out_path, index=False)
        counts[split_name] = len(split)
        console.print(f"  • exported {split_name}: {len(split):,} rows → {out_path}")
    return counts


def _download_official_zip(dest_zip: Path) -> Path:
    """Download spider_data.zip from Hugging Face; return path to the local archive."""
    if dest_zip.exists() and dest_zip.stat().st_size > 1_000_000:
        console.print(f"[dim]Reusing existing archive[/dim]: {dest_zip}")
        return dest_zip

    console.print(
        f"[bold]Downloading Spider SQLite archive[/bold]: "
        f"{HF_DB_REPO_ID}/{HF_DB_FILENAME}"
    )
    console.print("[dim]This can take several minutes (~200MB).[/dim]")

    cached = Path(
        hf_hub_download(
            repo_id=HF_DB_REPO_ID,
            filename=HF_DB_FILENAME,
            repo_type="dataset",
            local_dir=str(SPIDER_ROOT),
        )
    )

    if cached.resolve() != dest_zip.resolve():
        shutil.copy2(cached, dest_zip)

    if not dest_zip.exists() or dest_zip.stat().st_size < 1_000_000:
        raise RuntimeError(
            "Spider archive download failed or looks incomplete. "
            f"Expected a large zip at {dest_zip}."
        )
    return dest_zip


def _is_macos_junk(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._")


def _find_database_root(extracted: Path) -> Path:
    """Locate the folder that contains per-db SQLite directories."""
    preferred = (
        extracted / "spider_data" / "database",
        extracted / "database",
    )
    for candidate in preferred:
        if candidate.is_dir():
            return candidate

    matches = [
        p
        for p in extracted.rglob("database")
        if p.is_dir() and not _is_macos_junk(p)
    ]
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find a 'database/' directory inside extracted archive under {extracted}"
    )


def _install_databases_and_tables(archive_zip: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="spider_unpack_") as tmp:
        tmp_path = Path(tmp)
        console.print("[bold]Extracting archive…[/bold]")
        with zipfile.ZipFile(archive_zip, "r") as zf:
            zf.extractall(tmp_path)

        db_root = _find_database_root(tmp_path)
        console.print(f"[dim]Using database root:[/dim] {db_root}")
        installed = 0

        for db_dir in sorted(p for p in db_root.iterdir() if p.is_dir() and not _is_macos_junk(p)):
            db_id = db_dir.name
            sqlite_files = [
                p
                for p in db_dir.glob("*.sqlite")
                if not _is_macos_junk(p) and p.stat().st_size > 1024
            ]
            if not sqlite_files:
                continue

            # Prefer {db_id}.sqlite when present
            source = next(
                (p for p in sqlite_files if p.name == f"{db_id}.sqlite"),
                sqlite_files[0],
            )
            target_dir = DATABASES_DIR / db_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{db_id}.sqlite"
            shutil.copy2(source, target)
            installed += 1

        tables_candidates = [
            p for p in tmp_path.rglob("tables.json") if not _is_macos_junk(p)
        ]
        # Prefer spider_data/tables.json over test_tables.json
        tables_candidates.sort(
            key=lambda p: (p.name != "tables.json", "test" in p.name, len(p.parts))
        )
        if tables_candidates:
            shutil.copy2(tables_candidates[0], TABLES_JSON)
            console.print(f"  • tables.json → {TABLES_JSON}")
        else:
            console.print("[yellow]Warning: tables.json not found in archive[/yellow]")

    return installed


def download_spider(*, force: bool = False) -> None:
    ensure_data_dirs()

    if DOWNLOAD_MARKER.exists() and not force:
        console.print(
            f"[green]Spider assets already present[/green] ({DOWNLOAD_MARKER}). "
            "Pass --force to re-download."
        )
        return

    counts = _export_hf_splits()

    archive_zip = _download_official_zip(SPIDER_ROOT / HF_DB_FILENAME)
    n_dbs = _install_databases_and_tables(archive_zip)

    meta = {
        "hf_dataset": HF_DATASET_ID,
        "hf_db_repo": HF_DB_REPO_ID,
        "hf_split_counts": counts,
        "n_sqlite_databases": n_dbs,
        "databases_dir": str(DATABASES_DIR),
        "tables_json": str(TABLES_JSON) if TABLES_JSON.exists() else None,
    }
    DOWNLOAD_MARKER.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = Table(title="Spider download complete")
    summary.add_column("Item")
    summary.add_column("Value")
    for split, n in counts.items():
        summary.add_row(f"HF split: {split}", f"{n:,}")
    summary.add_row("SQLite databases", str(n_dbs))
    summary.add_row("Databases path", str(DATABASES_DIR))
    console.print(summary)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download Spider dataset assets")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing exports/databases",
    )
    args = parser.parse_args()
    download_spider(force=args.force)


if __name__ == "__main__":
    main()
