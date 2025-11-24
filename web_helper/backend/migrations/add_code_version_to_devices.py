#!/usr/bin/env python3
"""Migration: Add code_version column to devices table.

This migration adds code version tracking for reproducibility.
The code_version field stores git hash, files hash, and uv.lock hash.
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
        cursor.execute("PRAGMA table_info(devices)")
        columns = {row[1] for row in cursor.fetchall()}

        if "code_version" not in columns:
            sql = "ALTER TABLE devices ADD COLUMN code_version TEXT"
            print("Adding column: code_version")
            cursor.execute(sql)
        else:
            print("Column already exists: code_version")

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
