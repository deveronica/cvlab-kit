#!/usr/bin/env python3
"""Migration: Rename MD5 columns to hash columns in queue_experiments table.

This migration renames:
- config_md5 → config_hash
- log_md5 → log_hash
- error_log_md5 → error_log_hash

Purpose: Transition from MD5 to xxhash3 for faster file change detection.
"""

import sqlite3
import sys
from pathlib import Path


def migrate():
    """Apply migration."""
    db_path = Path("web_helper/web_helper.db")

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Skipping migration (database will be created with new schema).")
        return

    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check current columns
        cursor.execute("PRAGMA table_info(queue_experiments)")
        columns = {row[1] for row in cursor.fetchall()}

        # Define column renames
        renames = [
            ("config_md5", "config_hash"),
            ("log_md5", "log_hash"),
            ("error_log_md5", "error_log_hash"),
        ]

        for old_name, new_name in renames:
            if old_name in columns and new_name not in columns:
                # SQLite 3.25.0+ supports ALTER TABLE RENAME COLUMN
                sql = f"ALTER TABLE queue_experiments RENAME COLUMN {old_name} TO {new_name}"
                print(f"Renaming column: {old_name} → {new_name}")
                cursor.execute(sql)
            elif new_name in columns:
                print(f"Column already renamed: {new_name}")
            elif old_name not in columns:
                print(f"Old column not found: {old_name} (may be new database)")

        conn.commit()
        print("Migration completed successfully")

    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
