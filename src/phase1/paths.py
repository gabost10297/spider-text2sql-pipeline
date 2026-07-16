"""Shared filesystem layout for Spider assets."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SPIDER_DATA_DIR", "data")).resolve()

SPIDER_ROOT = DATA_ROOT / "spider"
HF_EXPORT_DIR = SPIDER_ROOT / "hf_export"
DATABASES_DIR = DATA_ROOT / "databases"
TABLES_JSON = SPIDER_ROOT / "tables.json"
DOWNLOAD_MARKER = SPIDER_ROOT / ".download_complete"


def ensure_data_dirs() -> None:
    SPIDER_ROOT.mkdir(parents=True, exist_ok=True)
    HF_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATABASES_DIR.mkdir(parents=True, exist_ok=True)


def db_sqlite_path(db_id: str) -> Path:
    """Canonical path used later by FastAPI: data/databases/{db_id}/{db_id}.sqlite"""
    return DATABASES_DIR / db_id / f"{db_id}.sqlite"
