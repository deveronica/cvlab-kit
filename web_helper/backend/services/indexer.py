"""Log file indexing service for CVLab-Kit."""

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from sqlalchemy.orm import Session

from ..models import Run, get_db
from .event_manager import event_manager

logger = logging.getLogger(__name__)


class LogIndexer:
    """Indexes log files from the logs directory into database."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self._running = False

    async def scan_and_index(self, force: bool = False) -> Dict[str, Any]:
        """Perform initial scan and index all log files."""
        if not self.logs_dir.exists():
            logger.warning(f"Logs directory {self.logs_dir} does not exist")
            return {"projects": 0, "runs": 0, "files": 0}

        logger.info(f"Starting log indexing scan in {self.logs_dir}")

        stats = {"projects": 0, "runs": 0, "files": 0}

        # Get database session
        with next(get_db()) as db:
            # If force reindex, delete all existing runs first
            # This ensures legacy "pending" runs without metrics are removed
            if force:
                deleted_count = db.query(Run).delete()
                db.commit()
                logger.info(
                    f"Force reindex: deleted {deleted_count} existing run records"
                )

            # Find all project directories
            for project_dir in self.logs_dir.iterdir():
                if not project_dir.is_dir() or project_dir.name.startswith("."):
                    continue

                project_name = project_dir.name
                logger.info(f"Processing project: {project_name}")

                project_stats = await self._index_project(
                    db, project_name, project_dir, force
                )

                if project_stats["runs"] > 0:
                    stats["projects"] += 1
                    stats["runs"] += project_stats["runs"]
                    stats["files"] += project_stats["files"]

            # Commit all changes
            db.commit()

        logger.info(f"Indexing complete: {stats}")

        # Send indexing complete event
        await event_manager.broadcast(
            {
                "type": "indexing_complete",
                "stats": stats,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return stats

    async def _index_project(
        self, db: Session, project_name: str, project_dir: Path, force: bool = False
    ) -> Dict[str, int]:
        """Index all runs in a project directory according to CVLab-Kit structure."""
        stats = {"runs": 0, "files": 0}

        # Group files by run_uid
        runs_data = {}

        # Scan all files in the project directory
        file_extensions = {".yaml", ".csv", ".pt", ".pth", ".json"}
        for file_path in project_dir.iterdir():
            if not file_path.is_file():
                continue

            if file_path.suffix not in file_extensions:
                continue

            # Skip temporary or hidden files
            if file_path.name.startswith(".") or file_path.name.endswith(".tmp"):
                continue

            # Parse run info using consolidated parser
            from ..utils.file_parsers import parse_run_info

            run_info = parse_run_info(file_path)
            if run_info is None:
                # Skip non-run files (JSON, etc.) or files we can't parse
                continue

            run_name = run_info.get("run_name")

            if not run_name or len(run_name.strip()) == 0:
                logger.debug(f"Could not parse run_name from {file_path}")
                continue

            if run_name not in runs_data:
                runs_data[run_name] = {
                    "files": [],
                    "config_file": None,
                    "metrics_file": None,
                    "checkpoint_file": None,
                    "output_config_file": None,
                }

            runs_data[run_name]["files"].append(file_path)

            # Categorize files based on CVLab-Kit conventions
            file_type = run_info.get("type")
            if file_type == "config":
                runs_data[run_name]["config_file"] = file_path
            elif file_type == "metrics":
                runs_data[run_name]["metrics_file"] = file_path
            elif file_type == "checkpoint":
                runs_data[run_name]["checkpoint_file"] = file_path
            elif file_type == "output_config":
                runs_data[run_name]["output_config_file"] = file_path

        logger.info(f"Found {len(runs_data)} unique runs in project {project_name}")

        # Create or update run records (only if metrics file exists)
        for run_name, run_data in runs_data.items():
            try:
                # CRITICAL: Skip runs without metrics file
                # Run represents experiment RESULTS, not just config
                metrics_file = run_data.get("metrics_file")
                if not metrics_file or not metrics_file.exists():
                    logger.debug(
                        f"Skipping run {run_name}: no metrics file (config-only runs don't create Run records)"
                    )
                    continue

                logger.info(f"Attempting to index run: {run_name}")
                run = await self._create_or_update_run(
                    db, project_name, run_name, run_data, force
                )
                if run:
                    logger.info(f"Successfully indexed run: {run_name}")
                    stats["runs"] += 1
                    stats["files"] += len(run_data["files"])
                else:
                    logger.warning(f"Run {run_name} returned None (skipped or failed)")
            except Exception as e:
                import traceback

                logger.error(f"Error indexing run {run_name}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")

        return stats

    def _get_file_fingerprint(self, file_path: Path) -> str:
        """Get file fingerprint using mtime + size for fast change detection."""
        try:
            stat = file_path.stat()
            # Format: "mtime_size" - both will change if file is modified
            return f"{stat.st_mtime:.6f}_{stat.st_size}"
        except Exception as e:
            logger.debug(f"Could not get fingerprint for {file_path}: {e}")
            return ""

    async def _create_or_update_run(
        self, db: Session, project: str, run_name: str, run_data: Dict, force: bool
    ) -> Optional[Run]:
        """Create or update a run record with enhanced CVLab-Kit support.

        This method implements idempotent reindexing with mtime+size change detection:
        - If (project, run_name) exists in DB:
          - Check metrics file fingerprint (mtime + size)
          - If unchanged → SKIP (no parsing, no DB update)
          - If changed → UPDATE (parse and overwrite)
        - If not exists → INSERT (new run)

        Args:
            force: If True, skip fingerprint check and always update
        """
        logger.info(f"[DEBUG] _create_or_update_run called for {project}/{run_name}")
        metrics_file = run_data.get("metrics_file")
        logger.info(f"[DEBUG] metrics_file: {metrics_file}")

        # Calculate current file fingerprint
        current_fingerprint = ""
        if metrics_file and metrics_file.exists():
            current_fingerprint = self._get_file_fingerprint(metrics_file)
        logger.info(f"[DEBUG] current_fingerprint: {current_fingerprint}")

        # Check if run already exists
        try:
            existing_run = (
                db.query(Run)
                .filter(Run.project == project, Run.run_name == run_name)
                .first()
            )
            logger.info(f"[DEBUG] existing_run: {existing_run}")

            if existing_run:
                # Check if metrics are empty (failed parsing) - always reparse
                has_empty_metrics = (
                    not existing_run.final_metrics
                    or len(existing_run.final_metrics) == 0
                )

                # Check if file has changed (unless force=True or metrics are empty)
                if (
                    not force
                    and not has_empty_metrics
                    and existing_run.file_fingerprint == current_fingerprint
                ):
                    logger.debug(
                        f"Run {run_name} unchanged (fingerprint match), skipping update"
                    )
                    return existing_run  # Return existing run without update

                reason = (
                    "force=True"
                    if force
                    else "file changed"
                    if existing_run.file_fingerprint != current_fingerprint
                    else "empty metrics"
                )
                logger.debug(f"Run {run_name} exists, updating ({reason})")
            else:
                logger.info(f"[DEBUG] Creating new run for {run_name}")
        except Exception as e:
            import traceback

            logger.error(
                f"[CRITICAL ERROR] Exception in _create_or_update_run DB query for {project}/{run_name}: {e}"
            )
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

        # Parse config files if available (prefer output config over input config)
        config_data = {}
        config_file = run_data.get("output_config_file") or run_data.get("config_file")
        if config_file:
            try:
                with open(config_file) as f:
                    # Use FullLoader like config.py to handle Python-specific tags
                    config_data = yaml.load(f, Loader=yaml.FullLoader)
                logger.debug(f"Parsed config for {run_name}: {len(config_data)} keys")
            except Exception as e:
                logger.warning(f"Could not parse config {config_file}: {e}")

        # Parse metrics if available
        metrics_info = {}
        metrics_file = run_data.get("metrics_file")
        if metrics_file and metrics_file.exists():
            try:
                file_size = metrics_file.stat().st_size
                if file_size > 0:
                    metrics_info = self._parse_metrics_file(metrics_file)
                    logger.debug(
                        f"Parsed metrics for {run_name}: {metrics_info.get('total_steps', 0)} steps"
                    )
                else:
                    logger.debug(f"Metrics file for {run_name} is empty")
            except Exception as e:
                logger.warning(f"Could not parse metrics {metrics_file}: {e}")

        # Determine run status based on available files and content
        status = self._determine_run_status(run_data, metrics_info)

        # Extract timing information from file timestamps
        started_at, finished_at = self._extract_timing_info(run_data, status)

        # Create or update run
        try:
            if existing_run:
                # UPDATE: Overwrite existing run with new data
                existing_run.status = status
                existing_run.config_path = (
                    str(config_file) if config_file else existing_run.config_path
                )
                existing_run.metrics_path = (
                    str(metrics_file) if metrics_file else existing_run.metrics_path
                )
                existing_run.checkpoint_path = (
                    str(run_data["checkpoint_file"])
                    if run_data.get("checkpoint_file")
                    else existing_run.checkpoint_path
                )
                existing_run.total_steps = metrics_info.get(
                    "total_steps", existing_run.total_steps or 0
                )
                existing_run.hyperparameters = (
                    config_data if config_data else existing_run.hyperparameters or {}
                )
                existing_run.final_metrics = metrics_info.get(
                    "final_metrics", existing_run.final_metrics or {}
                )
                existing_run.max_metrics = metrics_info.get(
                    "max_metrics", existing_run.max_metrics or {}
                )
                existing_run.min_metrics = metrics_info.get(
                    "min_metrics", existing_run.min_metrics or {}
                )
                existing_run.mean_metrics = metrics_info.get(
                    "mean_metrics", existing_run.mean_metrics or {}
                )
                existing_run.median_metrics = metrics_info.get(
                    "median_metrics", existing_run.median_metrics or {}
                )
                existing_run.file_fingerprint = (
                    current_fingerprint  # Update fingerprint
                )
                existing_run.last_updated = datetime.now()
                if started_at:
                    existing_run.started_at = started_at
                if finished_at:
                    existing_run.finished_at = finished_at
                run = existing_run
                logger.debug(
                    f"Updated run: {project}/{run_name} ({status}, {metrics_info.get('total_steps', 0)} steps)"
                )
            else:
                # INSERT: Create new run
                run = Run(
                    project=project,
                    run_name=run_name,
                    status=status,
                    config_path=str(config_file) if config_file else None,
                    metrics_path=str(metrics_file) if metrics_file else None,
                    checkpoint_path=str(run_data["checkpoint_file"])
                    if run_data.get("checkpoint_file")
                    else None,
                    total_steps=metrics_info.get("total_steps", 0),
                    hyperparameters=config_data if config_data else {},
                    final_metrics=metrics_info.get("final_metrics", {}),
                    max_metrics=metrics_info.get("max_metrics", {}),
                    min_metrics=metrics_info.get("min_metrics", {}),
                    mean_metrics=metrics_info.get("mean_metrics", {}),
                    median_metrics=metrics_info.get("median_metrics", {}),
                    file_fingerprint=current_fingerprint,  # Store initial fingerprint
                    started_at=started_at,
                    finished_at=finished_at,
                )
                db.add(run)
                logger.debug(
                    f"Created run: {project}/{run_name} ({status}, {metrics_info.get('total_steps', 0)} steps)"
                )

            return run
        except Exception as e:
            import traceback

            logger.error(
                f"[CRITICAL ERROR] Exception in _create_or_update_run for {project}/{run_name}: {e}"
            )
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _determine_run_status(self, run_data: Dict, metrics_info: Dict) -> str:
        """Determine run status based on YAML config epochs and CSV progress.

        NOTE: This method assumes metrics file exists (validated in _index_project).

        Status determination logic:
        1. Check if YAML config has explicit 'status' field (from cvlabkit)
        2. Compare CSV last step/epoch with YAML target epochs
        3. Check CSV file staleness for running experiments (terminated if stale)
        4. Fallback to file-based heuristics if epoch info unavailable

        Possible statuses: running, completed, failed, unknown
        (NOT pending - pending is Queue/Experiment concept only)
        """
        metrics_file = run_data.get("metrics_file")
        config_file = run_data.get("config_file") or run_data.get("output_config_file")

        # Defensive check: metrics file should always exist at this point
        # (filtered in _index_project), but handle gracefully
        if not metrics_file or not metrics_file.exists():
            logger.warning(
                "_determine_run_status called without metrics file - this should not happen"
            )
            return "unknown"

        # Empty metrics file = failed
        if metrics_file.stat().st_size == 0:
            return "failed"

        # Try to read status from YAML config (if cvlabkit writes it)
        yaml_status = None
        target_epochs = None

        if config_file and config_file.exists():
            try:
                with open(config_file) as f:
                    # Use FullLoader like config.py to handle Python-specific tags
                    config = yaml.load(f, Loader=yaml.FullLoader)

                    if config:
                        # Check for explicit status field
                        yaml_status = config.get("status")
                        # Get target epochs from config
                        target_epochs = (
                            config.get("epochs")
                            or config.get("num_epochs")
                            or config.get("max_epochs")
                        )
            except Exception as e:
                logger.debug(f"Could not parse YAML config for status: {e}")

        # If YAML has explicit status, use it (except for 'running' - we verify with epochs and staleness)
        if yaml_status and yaml_status not in ["running", "pending"]:
            return yaml_status

        # Epoch-based status determination
        total_steps = metrics_info.get("total_steps", 0)
        last_epoch = metrics_info.get("last_epoch", 0)

        # Debug logging
        run_name = run_data.get("run_name", "unknown")
        logger.info(
            f"[STATUS DEBUG] {run_name}: target_epochs={target_epochs}, last_epoch={last_epoch}, total_steps={total_steps}, yaml_status={yaml_status}"
        )

        if total_steps > 0:
            # If we have target epochs and last epoch info, compare them
            if target_epochs and last_epoch:
                logger.info(
                    f"[STATUS DEBUG] {run_name}: Checking last_epoch ({last_epoch}) >= target_epochs ({target_epochs})"
                )
                if last_epoch >= target_epochs:
                    logger.info(
                        f"[STATUS DEBUG] {run_name}: COMPLETED (epochs matched)"
                    )
                    return "completed"
                elif last_epoch > 0:
                    # Has progress but not finished - check if stale
                    # If CSV hasn't been modified in 1 hour, experiment was likely terminated
                    try:
                        last_modified = datetime.fromtimestamp(
                            metrics_file.stat().st_mtime
                        )
                        time_since_update = datetime.now() - last_modified
                        STALENESS_THRESHOLD = timedelta(hours=1)

                        if time_since_update > STALENESS_THRESHOLD:
                            # Experiment hasn't updated in over 1 hour but didn't complete
                            return "terminated"
                    except Exception as e:
                        logger.debug(f"Could not check CSV staleness: {e}")

                    return "running"

            # If we have target epochs but use 'step' column instead of 'epoch'
            # Assume step == epoch for most cases
            if target_epochs and total_steps >= target_epochs:
                logger.info(
                    f"[STATUS DEBUG] {run_name}: COMPLETED (total_steps {total_steps} >= target_epochs {target_epochs})"
                )
                return "completed"
            elif target_epochs and total_steps > 0:
                # Check staleness for incomplete runs
                logger.info(
                    f"[STATUS DEBUG] {run_name}: Incomplete run - checking staleness"
                )
                try:
                    last_modified = datetime.fromtimestamp(metrics_file.stat().st_mtime)
                    time_since_update = datetime.now() - last_modified
                    STALENESS_THRESHOLD = timedelta(hours=1)

                    if time_since_update > STALENESS_THRESHOLD:
                        logger.info(
                            f"[STATUS DEBUG] {run_name}: TERMINATED (stale, last_modified={last_modified}, time_since={time_since_update})"
                        )
                        return "terminated"
                except Exception as e:
                    logger.debug(f"Could not check CSV staleness: {e}")

                logger.info(f"[STATUS DEBUG] {run_name}: RUNNING (recent file)")
                return "running"

            # Fallback heuristics
            # If substantial progress (>10 steps/epochs), likely running or completed
            logger.info(
                f"[STATUS DEBUG] {run_name}: Entering fallback heuristics (total_steps={total_steps})"
            )
            if total_steps > 10:
                # Check staleness
                try:
                    last_modified = datetime.fromtimestamp(metrics_file.stat().st_mtime)
                    time_since_update = datetime.now() - last_modified
                    STALENESS_THRESHOLD = timedelta(hours=1)

                    if time_since_update > STALENESS_THRESHOLD:
                        # If YAML says running but file is stale, mark as terminated
                        if yaml_status == "running":
                            return "terminated"
                        # Otherwise assume completed if we have a checkpoint
                        checkpoint_file = run_data.get("checkpoint_file")
                        if checkpoint_file and checkpoint_file.exists():
                            return "completed"
                        return "terminated"
                except Exception as e:
                    logger.debug(f"Could not check CSV staleness: {e}")

                # If YAML explicitly says running and file is recent, keep it
                if yaml_status == "running":
                    return "running"
                # Otherwise, assume completed if we have a checkpoint
                checkpoint_file = run_data.get("checkpoint_file")
                if checkpoint_file and checkpoint_file.exists():
                    return "completed"
                return "running"
            else:
                # Check if few steps but stale
                try:
                    last_modified = datetime.fromtimestamp(metrics_file.stat().st_mtime)
                    time_since_update = datetime.now() - last_modified
                    STALENESS_THRESHOLD = timedelta(hours=1)

                    if time_since_update > STALENESS_THRESHOLD:
                        return "terminated"
                except Exception as e:
                    logger.debug(f"Could not check CSV staleness: {e}")

                return "running"  # Few steps, likely still running
        else:
            return "failed"  # Has metrics file but no data

    def _extract_timing_info(self, run_data: Dict, status: str):
        """Extract start/end times from file timestamps.

        Args:
            run_data: Dictionary containing file paths
            status: Run status ('running', 'completed', 'failed', 'terminated', etc.)

        Returns:
            Tuple of (started_at, finished_at)

        Logic:
            - started_at: Always use config file ctime (creation time, immutable)
            - finished_at:
                * running: None (experiment still in progress)
                * completed/failed/terminated: metrics file mtime (last update time)
        """
        started_at = None
        finished_at = None

        try:
            # Use config file creation time as start time
            config_file = run_data.get("config_file") or run_data.get(
                "output_config_file"
            )
            if config_file and config_file.exists():
                started_at = datetime.fromtimestamp(config_file.stat().st_ctime)

            # Use metrics file modification time as end time (only for non-running experiments)
            metrics_file = run_data.get("metrics_file")
            if metrics_file and metrics_file.exists():
                # For running experiments, finished_at should be None
                # For completed/failed/terminated experiments, use file mtime as end time
                # Note: We use mtime even for empty files (failed runs that didn't write data)
                if status != "running":
                    finished_at = datetime.fromtimestamp(metrics_file.stat().st_mtime)

        except Exception as e:
            logger.debug(f"Could not extract timing info: {e}")

        # Fix invalid timestamps: finished_at must be >= started_at
        if finished_at and started_at and finished_at < started_at:
            logger.debug(
                f"Invalid timestamps detected: finished_at ({finished_at}) < started_at ({started_at}). "
                f"Setting finished_at = started_at to fix filesystem timing issue."
            )
            finished_at = started_at

        return started_at, finished_at

    # Note: _parse_run_info method removed - now using consolidated parser from utils.file_parsers

    def _parse_metrics_file(self, metrics_file: Path) -> Dict[str, Any]:
        """Parse metrics from CSV file, computing final/max/min/mean statistics."""
        info = {
            "total_steps": 0,
            "final_metrics": {},
            "max_metrics": {},
            "min_metrics": {},
            "mean_metrics": {},
            "median_metrics": {},
            "last_epoch": 0,
        }

        try:
            with open(metrics_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if rows:
                    info["total_steps"] = len(rows)

                    # Initialize tracking for min/max/mean
                    metric_values = {}  # {metric_name: [values]}

                    # Process all rows to collect values
                    for row in rows:
                        for key, value in row.items():
                            # Skip non-numeric columns (step, epoch are for tracking only)
                            if key in ["step", "epoch"]:
                                continue

                            try:
                                numeric_value = float(value)
                                if key not in metric_values:
                                    metric_values[key] = []
                                metric_values[key].append(numeric_value)
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                pass

                    # Get final metrics (last row)
                    final_row = rows[-1]
                    for key, value in final_row.items():
                        # Skip non-metric columns (step, epoch are for tracking only)
                        if key in ["step", "epoch"]:
                            continue
                        try:
                            info["final_metrics"][key] = float(value)
                        except (ValueError, TypeError):
                            info["final_metrics"][key] = value

                    # Calculate min/max/mean/median for each metric
                    for metric_name, values in metric_values.items():
                        if len(values) > 0:
                            info["max_metrics"][metric_name] = max(values)
                            info["min_metrics"][metric_name] = min(values)
                            info["mean_metrics"][metric_name] = sum(values) / len(
                                values
                            )

                            # Calculate median
                            sorted_values = sorted(values)
                            n = len(sorted_values)
                            if n % 2 == 0:
                                info["median_metrics"][metric_name] = (
                                    sorted_values[n // 2 - 1] + sorted_values[n // 2]
                                ) / 2
                            else:
                                info["median_metrics"][metric_name] = sorted_values[
                                    n // 2
                                ]

                    # Extract last epoch/step for status determination
                    if "epoch" in final_row:
                        try:
                            info["last_epoch"] = int(float(final_row["epoch"]))
                        except (ValueError, TypeError):
                            pass
                    elif "step" in final_row:
                        try:
                            info["last_epoch"] = int(float(final_row["step"]))
                        except (ValueError, TypeError):
                            pass

        except Exception as e:
            logger.warning(f"Error parsing metrics file {metrics_file}: {e}")

        return info

    async def reindex_project(self, project_name: str) -> Dict[str, Any]:
        """Reindex a specific project."""
        project_dir = self.logs_dir / project_name
        if not project_dir.exists():
            return {"error": f"Project {project_name} not found"}

        with next(get_db()) as db:
            # Remove existing runs for this project
            db.query(Run).filter(Run.project == project_name).delete()
            db.commit()

            # Reindex
            stats = await self._index_project(db, project_name, project_dir, force=True)
            db.commit()

        return {"project": project_name, "stats": stats}


# Global indexer instance
log_indexer = LogIndexer()
