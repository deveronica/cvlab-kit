"""Database configuration and connection management."""

import logging
import shutil
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Database configuration
STATE_DIR = Path("web_helper/state")
SQLITE_DATABASE_URL = f"sqlite:///{STATE_DIR}/web_helper.db"
engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Database dependency for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_session():
    """Context manager for database session (for non-FastAPI code).

    Usage:
        with get_session() as session:
            session.query(...)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def migrate_state_files():
    """Migrate state files from old location to new state/ folder.

    Moves:
    - web_helper/web_helper.db → web_helper/state/web_helper.db
    - web_helper/queue_state.json → web_helper/state/queue_state.json
    - web_helper/daemon_state.json → web_helper/state/daemon_state.json
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    old_files = [
        ("web_helper/web_helper.db", STATE_DIR / "web_helper.db"),
        ("web_helper/queue_state.json", STATE_DIR / "queue_state.json"),
        ("web_helper/daemon_state.json", STATE_DIR / "daemon_state.json"),
    ]

    for old_path, new_path in old_files:
        old = Path(old_path)
        if old.exists() and not new_path.exists():
            logger.info(f"Migrating {old} → {new_path}")
            shutil.move(str(old), str(new_path))


def init_database():
    """Initialize database tables."""
    # Migrate old state files to new location
    migrate_state_files()

    # Import all models to ensure they're registered with Base
    # Note: ProcessState moved to JSON file, no longer in DB

    Base.metadata.create_all(bind=engine)
