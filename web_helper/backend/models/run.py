"""Run model for experiment tracking."""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String, UniqueConstraint

from .database import Base


class Run(Base):
    """Database model for experiment runs."""

    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True)
    project = Column(String, index=True)
    run_name = Column(
        String, index=True
    )  # User-specified run identifier (supports overwrite)

    status = Column(String, default="unknown")
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    config_path = Column(String)
    metrics_path = Column(String)
    checkpoint_path = Column(String)
    total_steps = Column(Integer, default=0)
    hyperparameters = Column(JSON)  # Hyperparameters from config
    final_metrics = Column(JSON)  # Last step metrics
    max_metrics = Column(JSON)  # Maximum value for each metric across all steps
    min_metrics = Column(JSON)  # Minimum value for each metric across all steps
    mean_metrics = Column(JSON)  # Mean value for each metric across all steps
    median_metrics = Column(JSON)  # Median value for each metric across all steps
    last_updated = Column(DateTime, default=datetime.utcnow)

    # File change detection (mtime + size for idempotent reindexing)
    file_fingerprint = Column(String)  # Format: "mtime_size" for metrics file

    # User annotations
    notes = Column(String(2000), default="")  # User notes for this run
    tags = Column(JSON, default=list)  # List of tag strings

    # Composite unique constraint: same run_name can exist in different projects
    __table_args__ = (
        UniqueConstraint("project", "run_name", name="uix_project_run_name"),
    )
