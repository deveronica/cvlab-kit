#!/usr/bin/env python3
"""Migration: Add error_message field to queue_experiments table.

This migration adds support for storing error messages when jobs fail.
"""

import sqlite3
import sys
from pathlib import Path


def migrate():
    """Apply migration."""
    # Try new location first, fallback to old location
    db_path = Path("web_helper/state/web_helper.db")
    if not db_path.exists():
        db_path = Path("web_helper/web_helper.db")

    if not db_path.exists():
        print(f"Database not found")
        print("Please run web_helper at least once to create the database.")
        sys.exit(1)

    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(queue_experiments)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if "error_message" not in existing_columns:
            sql = "ALTER TABLE queue_experiments ADD COLUMN error_message TEXT"
            print("Adding column: error_message")
            cursor.execute(sql)
            conn.commit()
            print("✓ Migration completed successfully")
        else:
            print("Column already exists: error_message")
            print("✓ No migration needed")

    except Exception as e:
        conn.rollback()
        print(f"✗ Migration failed: {e}")
        sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
