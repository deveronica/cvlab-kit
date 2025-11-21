"""Add ProcessState table for daemon process management.

This migration creates the process_states table to enable:
- SSH-independent daemon process management
- Process lifecycle tracking (start, stop, status)
- Recovery after system restart
"""

from sqlalchemy import create_engine, inspect

from web_helper.backend.daemon.models import ProcessState
from web_helper.backend.models.database import Base, engine


def upgrade():
    """Create process_states table if it doesn't exist."""
    inspector = inspect(engine)

    if "process_states" not in inspector.get_table_names():
        ProcessState.__table__.create(engine)
        print("✅ Created process_states table")
    else:
        print("⚠️  process_states table already exists, skipping")


def downgrade():
    """Drop process_states table."""
    inspector = inspect(engine)

    if "process_states" in inspector.get_table_names():
        ProcessState.__table__.drop(engine)
        print("✅ Dropped process_states table")
    else:
        print("⚠️  process_states table does not exist, skipping")


if __name__ == "__main__":
    print("🔧 Running migration: add_process_state_table")
    upgrade()
