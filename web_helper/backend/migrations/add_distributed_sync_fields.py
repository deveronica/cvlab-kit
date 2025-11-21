#!/usr/bin/env python3
"""Migration: Add distributed sync fields to queue_experiments table.

This migration adds support for distributed experiment execution
by tracking sync status and remote file metadata.
"""

import sqlite3
import sys
from pathlib import Path


def migrate():
    """Apply migration."""
    db_path = Path("web_helper/state/db.sqlite")

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Please run web_helper at least once to create the database.")
        sys.exit(1)

    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(queue_experiments)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        columns_to_add = {
            "server_origin": "TEXT DEFAULT 'local'",
            "sync_status": "TEXT DEFAULT 'synced'",
            "last_sync_at": "DATETIME",
            "remote_mtime": "INTEGER",
            "recovery_checkpoint": "TEXT",  # JSON stored as TEXT
        }

        # Add missing columns
        for column_name, column_def in columns_to_add.items():
            if column_name not in existing_columns:
                sql = f"ALTER TABLE queue_experiments ADD COLUMN {column_name} {column_def}"
                print(f"Adding column: {column_name}")
                cursor.execute(sql)
            else:
                print(f"Column already exists: {column_name}")

        conn.commit()
        print("✓ Migration completed successfully")

    except Exception as e:
        conn.rollback()
        print(f"✗ Migration failed: {e}")
        sys.exit(1)

    finally:
        conn.close()


def rollback():
    """Rollback migration (remove added columns).

    Note: SQLite does not support DROP COLUMN directly.
    This requires recreating the table without those columns.
    """
    print("Rollback not implemented (SQLite limitation)")
    print("To rollback, manually recreate the database or remove added columns")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Database migration for distributed sync"
    )
    parser.add_argument("--rollback", action="store_true", help="Rollback migration")

    args = parser.parse_args()

    if args.rollback:
        rollback()
    else:
        migrate()
