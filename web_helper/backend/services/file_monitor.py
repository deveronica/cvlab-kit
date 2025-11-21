"""File system monitoring using watchdog for automatic indexing."""

import asyncio
import logging
from pathlib import Path
from typing import Set

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .event_manager import event_manager

logger = logging.getLogger(__name__)


class CVLabKitFileHandler(FileSystemEventHandler):
    """Handles file system events for CVLab-Kit logs directory."""

    def __init__(self, file_monitor):
        self.file_monitor = file_monitor
        self.processed_files: Set[str] = set()

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if self._should_process_file(file_path):
            self._schedule_async_task(file_path, "created")

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if self._should_process_file(file_path):
            self._schedule_async_task(file_path, "modified")

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        # Only process .yaml, .csv, .pt, .pth files in logs directory
        if file_path.suffix not in {".yaml", ".csv", ".pt", ".pth"}:
            return False

        # Skip temporary files
        if file_path.name.startswith(".") or file_path.name.endswith(".tmp"):
            return False

        return True

    def _schedule_async_task(self, file_path: Path, event_type: str):
        """Schedule async task safely, handling event loop issues."""
        try:
            # Check if there's already an event loop running
            try:
                loop = asyncio.get_running_loop()
                # If loop is running, schedule the coroutine
                asyncio.create_task(self._process_file_async(file_path, event_type))
            except RuntimeError:
                # No event loop running, use threading to run in new loop
                import threading

                def run_in_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            self._process_file_async(file_path, event_type)
                        )
                    finally:
                        loop.close()

                threading.Thread(target=run_in_thread, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to schedule async task for {file_path}: {e}")

    async def _process_file_async(self, file_path: Path, event_type: str):
        """Process file asynchronously."""
        try:
            # Avoid duplicate processing
            file_key = f"{file_path}:{event_type}"
            if file_key in self.processed_files:
                return

            self.processed_files.add(file_key)

            # Clean up old entries to prevent memory leak
            if len(self.processed_files) > 1000:
                self.processed_files.clear()

            logger.info(f"Processing {event_type} file: {file_path}")

            # Extract project and run information using consolidated parser
            from ..utils.file_parsers import parse_project_from_path, parse_run_info

            project = parse_project_from_path(file_path)
            if project:
                # Parse run info from filename
                run_info = parse_run_info(file_path)

                # Only broadcast if we successfully parsed run info
                if run_info:
                    await event_manager.send_run_update(
                        {
                            "event_type": event_type,
                            "project": project,
                            "file_path": str(file_path),
                            "file_type": file_path.suffix,
                            "run_info": run_info,
                        }
                    )

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")


class FileMonitor:
    """Monitors file system changes in CVLab-Kit logs directory."""

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.observer = Observer()
        self.handler = CVLabKitFileHandler(self)
        self._running = False

    async def start(self):
        """Start file monitoring."""
        if self._running:
            logger.warning("File monitor already running")
            return

        # Create logs directory if it doesn't exist
        self.logs_dir.mkdir(exist_ok=True)

        # Start watching
        self.observer.schedule(self.handler, str(self.logs_dir), recursive=True)
        self.observer.start()
        self._running = True

        logger.info(f"File monitor started for directory: {self.logs_dir}")

        # Send initial scan complete event
        await event_manager.broadcast(
            {
                "type": "file_monitor_started",
                "message": f"Monitoring {self.logs_dir} for changes",
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

        logger.info("File monitor stopped")

    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running and self.observer.is_alive()


# Global file monitor instance
file_monitor = FileMonitor()
