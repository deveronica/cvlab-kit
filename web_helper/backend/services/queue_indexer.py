"""Queue experiment indexer service.

Scans queue_logs/ directory and maintains a database of all experiments
with xxhash3 checksums for fast change detection.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import xxhash
import yaml
from sqlalchemy.orm import Session

from ..models import QueueExperiment
from ..models.database import SessionLocal

logger = logging.getLogger(__name__)

# Experiment UID pattern: YYYYMMDD_hash4
EXPERIMENT_UID_PATTERN = re.compile(r"^\d{8}_[a-f0-9]{4}$")


def calculate_file_hash(file_path: Path) -> Optional[str]:
    """Calculate xxhash3 checksum of a file for fast change detection."""
    if not file_path.exists():
        return None

    try:
        with open(file_path, "rb") as f:
            return xxhash.xxh3_64(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        return None


def detect_status_smart(
    experiment_dir: Path, db_record: Optional[QueueExperiment] = None
) -> str:
    """Detect experiment status using DB + file-based approach.

    Args:
        experiment_dir: Path to experiment directory
        db_record: Existing database record (if any)

    Returns:
        Status string: running, completed, failed, crashed, pending, expired
    """
    log_path = experiment_dir / "terminal_log.log"
    error_log_path = experiment_dir / "terminal_err.log"
    config_path = experiment_dir / "config.yaml"

    # 1. Check DB record first (Queue Manager managed jobs)
    if db_record and db_record.status == "running":
        # Verify process is still alive
        if db_record.pid and psutil.pid_exists(db_record.pid):
            return "running"
        else:
            # Process died without proper cleanup → crashed
            logger.warning(f"Process {db_record.pid} died for {experiment_dir.name}")
            return "crashed"

    # 2. Check log file modification time (within 1 minute = likely running)
    if log_path.exists() and log_path.stat().st_size > 0:
        log_mtime = log_path.stat().st_mtime
        if time.time() - log_mtime < 60:  # 1분 = 60초
            return "running"

    # 3. Check error log exists and has content (even if terminal_log.log is empty/missing)
    if error_log_path.exists() and error_log_path.stat().st_size > 0:
        try:
            with open(error_log_path) as f:
                error_content = f.read()
                if (
                    "error" in error_content.lower()
                    or "exception" in error_content.lower()
                ):
                    return "failed"
                else:
                    return "completed"
        except Exception as e:
            logger.error(f"Failed to read error log {error_log_path}: {e}")
            return "unknown"

    # 4. Check normal log exists and has content
    if log_path.exists() and log_path.stat().st_size > 0:
        return "completed"

    # 5. Check if config exists but no logs and 8 hours have passed → expired
    if config_path.exists():
        config_mtime = config_path.stat().st_mtime
        hours_since_creation = (time.time() - config_mtime) / 3600

        # No log files exist but config is older than 8 hours
        if hours_since_creation >= 8:
            if (
                not log_path.exists()
                and not error_log_path.exists()
                or (not log_path.exists() or log_path.stat().st_size == 0)
                and (not error_log_path.exists() or error_log_path.stat().st_size == 0)
            ):
                return "expired"

    # 6. Default: pending
    return "pending"


def extract_experiment_info(
    experiment_dir: Path, db_record: Optional[QueueExperiment] = None
) -> Optional[Dict[str, Any]]:
    """Extract experiment information from directory.

    Args:
        experiment_dir: Path to experiment directory
        db_record: Existing database record for status detection

    Returns:
        Experiment info dict, or None if invalid UID format
    """
    experiment_uid = experiment_dir.name

    # Validate experiment UID format
    if not EXPERIMENT_UID_PATTERN.match(experiment_uid):
        logger.warning(
            f"Skipping invalid experiment UID format: {experiment_uid} "
            f"(expected format: YYYYMMDD_hash4)"
        )
        return None

    config_path = experiment_dir / "config.yaml"
    log_path = experiment_dir / "terminal_log.log"
    error_log_path = experiment_dir / "terminal_err.log"

    # Use config.yaml creation time for created_at, fall back to directory ctime
    if config_path.exists():
        created_at = datetime.fromtimestamp(config_path.stat().st_ctime)
    else:
        created_at = datetime.fromtimestamp(experiment_dir.stat().st_ctime)

    info = {
        "experiment_uid": experiment_uid,
        "name": experiment_uid,  # Default name
        "project": None,
        "status": "unknown",
        "created_at": created_at,
        "config_path": str(config_path) if config_path.exists() else None,
        "log_path": str(log_path) if log_path.exists() else None,
        "error_log_path": str(error_log_path) if error_log_path.exists() else None,
        "config_hash": calculate_file_hash(config_path),
        "log_hash": calculate_file_hash(log_path),
        "error_log_hash": calculate_file_hash(error_log_path),
    }

    # Extract project and name from config.yaml
    if config_path.exists():
        try:
            with open(config_path) as f:
                # Use FullLoader to support Python object tags (like !!python/tuple)
                config = yaml.load(f, Loader=yaml.FullLoader)
                if config:
                    info["project"] = config.get(
                        "project", config.get("log_dir", "").split("/")[-1]
                    )
                    info["name"] = config.get("experiment_name", experiment_uid)
        except Exception as e:
            logger.warning(f"Failed to parse config for {experiment_uid}: {e}")

    # Smart status detection (DB + file-based)
    info["status"] = detect_status_smart(experiment_dir, db_record)

    # Infer started_at from log file creation/modification time
    if log_path.exists() and log_path.stat().st_size > 0:
        info["started_at"] = datetime.fromtimestamp(log_path.stat().st_mtime)
    elif error_log_path.exists() and error_log_path.stat().st_size > 0:
        info["started_at"] = datetime.fromtimestamp(error_log_path.stat().st_mtime)

    # Infer completed_at for terminal states (completed, failed, expired, crashed)
    if info["status"] in ["completed", "failed", "expired", "crashed"]:
        # Use the most recent log file modification time as completion time
        log_times = []
        if log_path.exists():
            log_times.append(log_path.stat().st_mtime)
        if error_log_path.exists():
            log_times.append(error_log_path.stat().st_mtime)

        if log_times:
            info["completed_at"] = datetime.fromtimestamp(max(log_times))
        elif config_path.exists():
            # Fallback: use config creation time for expired jobs with no logs
            info["completed_at"] = datetime.fromtimestamp(config_path.stat().st_mtime)

    return info


def index_queue_experiments(db: Session) -> Dict[str, int]:
    """Index all experiments in queue_logs/ directory.

    Returns:
        Statistics about the indexing operation
    """
    queue_logs_dir = Path("web_helper/queue_logs")

    if not queue_logs_dir.exists():
        logger.warning(f"Queue logs directory not found: {queue_logs_dir}")
        return {"scanned": 0, "added": 0, "updated": 0, "unchanged": 0}

    stats = {"scanned": 0, "added": 0, "updated": 0, "unchanged": 0}

    # Scan all experiment directories
    for experiment_dir in sorted(queue_logs_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue

        stats["scanned"] += 1
        experiment_uid = experiment_dir.name

        # Check if experiment exists in database
        existing = (
            db.query(QueueExperiment)
            .filter(QueueExperiment.experiment_uid == experiment_uid)
            .first()
        )

        # Extract experiment info (pass existing record for smart status detection)
        info = extract_experiment_info(experiment_dir, db_record=existing)

        # Skip invalid experiments
        if info is None:
            continue

        if existing:
            # Check if any files have changed using hash or time-based fields changed
            needs_update = False

            # Hash-based file change detection (xxhash3)
            if existing.config_hash != info["config_hash"]:
                needs_update = True
            if existing.log_hash != info["log_hash"]:
                needs_update = True
            if existing.error_log_hash != info["error_log_hash"]:
                needs_update = True

            # Time-based field changes (status, created_at)
            # Status can change over time (e.g., pending -> expired after 8 hours)
            if existing.status != info["status"]:
                needs_update = True
            # created_at can change if we fix the calculation method
            if existing.created_at != info["created_at"]:
                needs_update = True

            if needs_update:
                # Update existing experiment
                for key, value in info.items():
                    if key != "experiment_uid":
                        setattr(existing, key, value)
                existing.last_indexed = datetime.utcnow()
                stats["updated"] += 1
                logger.info(f"Updated experiment: {experiment_uid}")
            else:
                stats["unchanged"] += 1
        else:
            # Create new experiment entry
            new_experiment = QueueExperiment(**info)
            db.add(new_experiment)
            stats["added"] += 1
            logger.info(f"Added new experiment: {experiment_uid}")

    db.commit()
    logger.info(f"Queue indexing complete: {stats}")
    return stats


def index_queue_experiments_startup():
    """Run queue experiment indexing on application startup."""
    logger.info("Starting queue experiment indexing...")

    db = SessionLocal()
    try:
        stats = index_queue_experiments(db)
        logger.info(f"Queue experiment indexing complete: {stats}")
    except Exception as e:
        logger.error(f"Failed to index queue experiments: {e}", exc_info=True)
    finally:
        db.close()


def reindex_single_experiment(experiment_uid: str, db: Session) -> bool:
    """Reindex a single experiment.

    Args:
        experiment_uid: The experiment UID to reindex
        db: Database session

    Returns:
        True if successful, False otherwise
    """
    queue_logs_dir = Path("web_helper/queue_logs")
    experiment_dir = queue_logs_dir / experiment_uid

    if not experiment_dir.exists():
        logger.warning(f"Experiment directory not found: {experiment_dir}")
        return False

    try:
        existing = (
            db.query(QueueExperiment)
            .filter(QueueExperiment.experiment_uid == experiment_uid)
            .first()
        )

        info = extract_experiment_info(experiment_dir, db_record=existing)

        if existing:
            for key, value in info.items():
                if key != "experiment_uid":
                    setattr(existing, key, value)
            existing.last_indexed = datetime.utcnow()
        else:
            new_experiment = QueueExperiment(**info)
            db.add(new_experiment)

        db.commit()
        logger.info(f"Reindexed experiment: {experiment_uid}")
        return True
    except Exception as e:
        logger.error(f"Failed to reindex experiment {experiment_uid}: {e}")
        db.rollback()
        return False


def check_pending_experiments(db: Session) -> Dict[str, int]:
    """Lightweight check for time-based status transitions.

    Only checks experiments that are currently 'pending' and updates
    their status if they should transition to 'expired' (8+ hours old).

    This is more efficient than full reindexing as it:
    - Only queries pending experiments
    - Skips hash calculation
    - Only updates status field

    Returns:
        Statistics about status transitions
    """
    stats = {"checked": 0, "transitioned": 0}

    try:
        # Query only pending experiments
        pending_experiments = (
            db.query(QueueExperiment).filter(QueueExperiment.status == "pending").all()
        )

        stats["checked"] = len(pending_experiments)

        for experiment in pending_experiments:
            queue_logs_dir = Path("web_helper/queue_logs")
            experiment_dir = queue_logs_dir / experiment.experiment_uid

            if not experiment_dir.exists():
                continue

            # Re-detect status (lightweight, no hash calculation)
            new_status = detect_status_smart(experiment_dir, db_record=experiment)

            # Check if status changed
            if new_status != experiment.status:
                logger.info(
                    f"Status transition for {experiment.experiment_uid}: "
                    f"{experiment.status} → {new_status}"
                )
                experiment.status = new_status
                experiment.last_indexed = datetime.utcnow()

                # Update completed_at for newly expired experiments
                if new_status == "expired":
                    config_path = experiment_dir / "config.yaml"
                    if config_path.exists():
                        experiment.completed_at = datetime.fromtimestamp(
                            config_path.stat().st_mtime
                        )

                stats["transitioned"] += 1

        db.commit()

        if stats["transitioned"] > 0:
            logger.info(f"Pending check complete: {stats}")

    except Exception as e:
        logger.error(f"Failed to check pending experiments: {e}", exc_info=True)
        db.rollback()

    return stats


async def periodic_full_reindex(interval_seconds: int = 600):
    """Periodic full reindex task (default: every 10 minutes).

    Provides integrity guarantee by running full hash-based reindexing
    to catch any missed Watchdog events or file changes.

    Args:
        interval_seconds: Seconds between reindex runs (default: 600 = 10 minutes)
    """
    logger.info(f"Starting periodic full reindex (interval: {interval_seconds}s)")

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            logger.debug("Running periodic full reindex...")
            db = SessionLocal()
            try:
                stats = index_queue_experiments(db)
                if stats["updated"] > 0 or stats["added"] > 0:
                    logger.info(f"Periodic full reindex complete: {stats}")
            finally:
                db.close()

        except asyncio.CancelledError:
            logger.info("Periodic full reindex task cancelled")
            break
        except Exception as e:
            logger.error(f"Periodic full reindex failed: {e}", exc_info=True)


async def periodic_status_check(interval_seconds: int = 60):
    """Periodic status check task (default: every 1 minute).

    Handles time-based status transitions (e.g., pending → expired)
    without requiring file modifications to trigger updates.

    Args:
        interval_seconds: Seconds between status checks (default: 60 = 1 minute)
    """
    logger.info(f"Starting periodic status check (interval: {interval_seconds}s)")

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            logger.debug("Running periodic status check...")
            db = SessionLocal()
            try:
                stats = check_pending_experiments(db)
                # Only log if there were actual transitions
                if stats["transitioned"] > 0:
                    logger.info(f"Periodic status check complete: {stats}")
            finally:
                db.close()

        except asyncio.CancelledError:
            logger.info("Periodic status check task cancelled")
            break
        except Exception as e:
            logger.error(f"Periodic status check failed: {e}", exc_info=True)


class PeriodicIndexer:
    """Manages periodic indexing tasks for queue experiments."""

    def __init__(self):
        self.status_check_task: Optional[asyncio.Task] = None
        self.full_reindex_task: Optional[asyncio.Task] = None

    async def start(
        self, status_check_interval: int = 60, full_reindex_interval: int = 600
    ):
        """Start periodic indexing tasks.

        Args:
            status_check_interval: Seconds between status checks (default: 60)
            full_reindex_interval: Seconds between full reindex (default: 600)
        """
        logger.info("Starting periodic indexer tasks...")

        # Start status check task (1 minute)
        self.status_check_task = asyncio.create_task(
            periodic_status_check(status_check_interval)
        )

        # Start full reindex task (10 minutes)
        self.full_reindex_task = asyncio.create_task(
            periodic_full_reindex(full_reindex_interval)
        )

        logger.info("✅ Periodic indexer tasks started")

    async def stop(self):
        """Stop all periodic indexing tasks."""
        logger.info("Stopping periodic indexer tasks...")

        tasks = []
        if self.status_check_task:
            self.status_check_task.cancel()
            tasks.append(self.status_check_task)

        if self.full_reindex_task:
            self.full_reindex_task.cancel()
            tasks.append(self.full_reindex_task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("✅ Periodic indexer tasks stopped")


# Global instance for app.py integration
periodic_indexer = PeriodicIndexer()
