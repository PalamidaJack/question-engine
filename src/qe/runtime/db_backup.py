"""Database backup and restore using SQLite's online backup API.

Provides non-blocking backup of all QE SQLite databases while
the engine continues running. Supports backup to a directory
with timestamped filenames and restore from a backup.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Default database files to back up
_DEFAULT_DB_FILES = [
    "data/qe.db",
    "data/event_log.db",
    "data/dlq.db",
]


def backup_database(
    source_path: str,
    dest_path: str,
) -> dict[str, Any]:
    """Back up a single SQLite database using the online backup API.

    This uses sqlite3.Connection.backup() which creates a consistent
    snapshot even while other connections are writing to the database.

    Returns a dict with backup metadata.
    """
    start = time.monotonic()
    source_p = Path(source_path)

    if not source_p.exists():
        return {
            "source": source_path,
            "dest": dest_path,
            "status": "skipped",
            "reason": "source not found",
        }

    dest_p = Path(dest_path)
    dest_p.parent.mkdir(parents=True, exist_ok=True)

    source_conn = sqlite3.connect(source_path)
    try:
        dest_conn = sqlite3.connect(dest_path)
    except Exception:
        source_conn.close()
        raise

    try:
        source_conn.backup(dest_conn)
        elapsed = time.monotonic() - start
        size = dest_p.stat().st_size

        log.info(
            "db.backup source=%s dest=%s size=%d elapsed=%.2fs",
            source_path,
            dest_path,
            size,
            elapsed,
        )

        return {
            "source": source_path,
            "dest": dest_path,
            "status": "completed",
            "size_bytes": size,
            "elapsed_seconds": round(elapsed, 3),
        }
    finally:
        source_conn.close()
        dest_conn.close()


def backup_all(
    dest_dir: str = "data/backups",
    db_files: list[str] | None = None,
) -> dict[str, Any]:
    """Back up all QE databases to a timestamped directory.

    Returns a summary of all backup operations.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(dest_dir) / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    files = db_files or _DEFAULT_DB_FILES
    results: list[dict[str, Any]] = []

    for db_file in files:
        db_name = Path(db_file).name
        dest_path = str(backup_dir / db_name)
        result = backup_database(db_file, dest_path)
        results.append(result)

    completed = sum(1 for r in results if r["status"] == "completed")
    total_size = sum(r.get("size_bytes", 0) for r in results)

    log.info(
        "db.backup_all dir=%s completed=%d/%d size=%d",
        str(backup_dir),
        completed,
        len(results),
        total_size,
    )

    return {
        "backup_dir": str(backup_dir),
        "timestamp": timestamp,
        "databases": results,
        "completed": completed,
        "total": len(results),
        "total_size_bytes": total_size,
    }


def restore_database(
    backup_path: str,
    dest_path: str,
) -> dict[str, Any]:
    """Restore a database from a backup file.

    Copies the backup file to the destination path.
    The destination file is overwritten if it exists.
    """
    backup_p = Path(backup_path)
    if not backup_p.exists():
        return {
            "backup": backup_path,
            "dest": dest_path,
            "status": "failed",
            "reason": "backup file not found",
        }

    dest_p = Path(dest_path)
    dest_p.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    shutil.copy2(backup_path, dest_path)
    elapsed = time.monotonic() - start

    log.info(
        "db.restored backup=%s dest=%s elapsed=%.2fs",
        backup_path,
        dest_path,
        elapsed,
    )

    return {
        "backup": backup_path,
        "dest": dest_path,
        "status": "restored",
        "elapsed_seconds": round(elapsed, 3),
    }


def list_backups(backup_dir: str = "data/backups") -> list[dict[str, Any]]:
    """List available backups with metadata."""
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return []

    backups = []
    for entry in sorted(backup_path.iterdir(), reverse=True):
        if entry.is_dir():
            db_files = list(entry.glob("*.db"))
            total_size = sum(f.stat().st_size for f in db_files)
            backups.append({
                "name": entry.name,
                "path": str(entry),
                "databases": [f.name for f in db_files],
                "total_size_bytes": total_size,
            })

    return backups
