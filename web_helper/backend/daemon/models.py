"""Database models for daemon process management."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, JSON, String

from web_helper.backend.models.database import Base


class ProcessState(Base):
    """Daemon process state for SSH-independent execution.

    This table tracks all daemon processes (backend, frontend, middleend)
    to enable:
    - Process management (start, stop, status)
    - SSH session independence
    - Recovery after system restart
    """

    __tablename__ = "process_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    process_type = Column(String, nullable=False)  # "backend", "frontend", "middleend"
    pid = Column(Integer, nullable=False)
    status = Column(String, nullable=False)  # "running", "stopped"
    host = Column(String, nullable=False)
    port = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    stopped_at = Column(DateTime, nullable=True)
    config = Column(JSON, nullable=True)  # Restart configuration
    log_file = Column(String, nullable=True)  # Path to log file

    def __repr__(self):
        return f"<ProcessState(type={self.process_type}, pid={self.pid}, status={self.status})>"
