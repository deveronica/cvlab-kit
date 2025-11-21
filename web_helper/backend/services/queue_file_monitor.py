"""File system monitoring for queue_logs directory using watchdog."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .event_manager import event_manager

logger = logging.getLogger(__name__)


class QueueFileHandler(FileSystemEventHandler):
    """Handles file system events for queue_logs directory.

    Uses time-based debouncing to prevent duplicate processing while
    allowing legitimate re-creation events after TTL expiration.
    """

    def __init__(self, queue_monitor, debounce_ttl: int = 5):
        self.queue_monitor = queue_monitor
        # Dict[event_key, timestamp] for time-based debouncing
        self.processed_events: Dict[str, float] = {}
        # Debounce TTL in seconds (default: 5 seconds)
        self.debounce_ttl = debounce_ttl

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            # New experiment directory created
            self._schedule_reindex(Path(event.src_path).name, "directory_created")
            return

        file_path = Path(event.src_path)
        if self._should_process_file(file_path):
            self._schedule_reindex(file_path.parent.name, "file_created")

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if self._should_process_file(file_path):
            self._schedule_reindex(file_path.parent.name, "file_modified")

    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory:
            # Experiment directory deleted
            self._schedule_cleanup(Path(event.src_path).name)
            return

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should trigger reindexing."""
        # Only process specific files
        if file_path.name not in {
            "config.yaml",
            "terminal_log.log",
            "terminal_err.log",
        }:
            return False

        # Skip temporary files
        if file_path.name.startswith(".") or file_path.name.endswith(".tmp"):
            return False

        return True

    def _schedule_reindex(self, experiment_uid: str, event_type: str):
        """Schedule reindexing for a specific experiment with time-based debouncing.

        Uses TTL-based debouncing (default 5 seconds) to:
        - Prevent duplicate processing of rapid successive events
        - Allow legitimate file re-creation events after TTL expires
        - Auto-cleanup old entries to prevent memory leaks

        Args:
            experiment_uid: Experiment identifier
            event_type: Type of event (file_created, file_modified, etc.)
        """
        try:
            current_time = time.time()
            event_key = f"{experiment_uid}:{event_type}"

            # Time-based debouncing: check if event was recently processed
            if event_key in self.processed_events:
                last_processed_time = self.processed_events[event_key]
                time_since_last = current_time - last_processed_time

                # Skip if within debounce TTL (5 seconds)
                if time_since_last < self.debounce_ttl:
                    logger.debug(
                        f"Debouncing {event_key} (last processed {time_since_last:.1f}s ago)"
                    )
                    return

            # Update timestamp for this event
            self.processed_events[event_key] = current_time

            # Cleanup expired entries (older than TTL * 2)
            # This prevents memory leaks while keeping recent events
            cleanup_threshold = current_time - (self.debounce_ttl * 2)
            expired_keys = [
                k for k, v in self.processed_events.items() if v < cleanup_threshold
            ]
            for k in expired_keys:
                del self.processed_events[k]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired debounce entries")

            # Schedule async task
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._reindex_async(experiment_uid, event_type))
            except RuntimeError:
                # No event loop running, use threading
                import threading

                def run_in_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            self._reindex_async(experiment_uid, event_type)
                        )
                    finally:
                        loop.close()

                threading.Thread(target=run_in_thread, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to schedule reindex for {experiment_uid}: {e}")

    def _schedule_cleanup(self, experiment_uid: str):
        """Schedule cleanup for deleted experiment."""
        try:
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._cleanup_async(experiment_uid))
            except RuntimeError:
                import threading

                def run_in_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._cleanup_async(experiment_uid))
                    finally:
                        loop.close()

                threading.Thread(target=run_in_thread, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to schedule cleanup for {experiment_uid}: {e}")

    async def _reindex_async(self, experiment_uid: str, event_type: str):
        """Reindex a single experiment asynchronously."""
        try:
            logger.info(f"Reindexing experiment {experiment_uid} due to {event_type}")

            from ..models.database import SessionLocal
            from ..services.queue_indexer import reindex_single_experiment

            db = SessionLocal()
            try:
                success = reindex_single_experiment(experiment_uid, db)
                if success:
                    # Broadcast update via SSE
                    await event_manager.send_queue_update(
                        {
                            "event_type": event_type,
                            "experiment_uid": experiment_uid,
                            "action": "reindexed",
                        }
                    )
                    logger.info(f"Successfully reindexed {experiment_uid}")
                else:
                    logger.warning(f"Failed to reindex {experiment_uid}")
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error reindexing {experiment_uid}: {e}", exc_info=True)

    async def _cleanup_async(self, experiment_uid: str):
        """Remove deleted experiment from database."""
        try:
            logger.info(f"Cleaning up deleted experiment {experiment_uid}")

            from ..models.database import SessionLocal
            from ..models.queue_experiment import QueueExperiment

            db = SessionLocal()
            try:
                # Delete from database
                experiment = (
                    db.query(QueueExperiment)
                    .filter(QueueExperiment.experiment_uid == experiment_uid)
                    .first()
                )

                if experiment:
                    db.delete(experiment)
                    db.commit()

                    # Broadcast deletion via SSE
                    await event_manager.send_queue_update(
                        {
                            "event_type": "deleted",
                            "experiment_uid": experiment_uid,
                            "action": "removed",
                        }
                    )
                    logger.info(f"Successfully cleaned up {experiment_uid}")
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error cleaning up {experiment_uid}: {e}", exc_info=True)


class QueueFileMonitor:
    """Monitors file system changes in queue_logs directory."""

    def __init__(self, queue_logs_dir: str = "web_helper/queue_logs"):
        self.queue_logs_dir = Path(queue_logs_dir)
        self.observer = Observer()
        self.handler = QueueFileHandler(self)
        self._running = False

    async def start(self):
        """Start file monitoring."""
        if self._running:
            logger.warning("Queue file monitor already running")
            return

        # Create queue_logs directory if it doesn't exist
        self.queue_logs_dir.mkdir(parents=True, exist_ok=True)

        # Start watching
        self.observer.schedule(self.handler, str(self.queue_logs_dir), recursive=True)
        self.observer.start()
        self._running = True

        logger.info(f"Queue file monitor started for directory: {self.queue_logs_dir}")

        # Send initial scan complete event
        await event_manager.broadcast(
            {
                "type": "queue_monitor_started",
                "message": f"Monitoring {self.queue_logs_dir} for changes",
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

    async def stop(self):
        """Stop file monitoring."""
        if not self._running:
            return

        self.observer.stop()
        self.observer.join()
        self._running = False

        logger.info("Queue file monitor stopped")

    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running and self.observer.is_alive()


# Global queue file monitor instance
queue_file_monitor = QueueFileMonitor()
