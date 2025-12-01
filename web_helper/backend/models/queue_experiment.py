"""QueueExperiment model for tracking experiment execution history."""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String

from .database import Base


class QueueExperiment(Base):
    """Database model for queue experiment execution history."""

    __tablename__ = "queue_experiments"

    id = Column(Integer, primary_key=True, index=True)
    experiment_uid = Column(String, unique=True, index=True)  # e.g., "20251007_7e7e"

    # Basic info
    name = Column(String)
    project = Column(String, index=True)
    status = Column(
        String, default="unknown"
    )  # pending, running, completed, failed, cancelled

    # Timestamps
    created_at = Column(DateTime)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # File paths
    config_path = Column(String)  # Path to config.yaml
    log_path = Column(String)  # Path to terminal_log.log
    error_log_path = Column(String)  # Path to terminal_err.log

    # Execution details
    assigned_device = Column(String, nullable=True)
    priority = Column(String, default="normal")
    pid = Column(Integer, nullable=True)  # Process ID for running jobs
    exit_code = Column(Integer, nullable=True)
    error_message = Column(String, nullable=True)  # Error message if failed

    # File checksums for change detection (xxhash3)
    config_hash = Column(String, nullable=True)
    log_hash = Column(String, nullable=True)
    error_log_hash = Column(String, nullable=True)

    # Additional metadata (renamed to avoid SQLAlchemy reserved word)
    meta = Column(JSON, nullable=True)

    # Distributed execution support
    server_origin = Column(String, default="local")  # "local" | "remote-{host_id}"
    sync_status = Column(String, default="synced")  # "synced" | "syncing" | "outdated"
    last_sync_at = Column(DateTime, nullable=True)
    remote_mtime = Column(
        Integer, nullable=True
    )  # Remote file modification time (epoch)
    recovery_checkpoint = Column(JSON, nullable=True)  # {epoch: 5, step: 1000, ...}

    # Last update timestamp
    last_indexed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
