"""Log synchronizer for distributed experiment execution.

Uses xxhash3 for fast file integrity verification during sync.
Supports delta sync for append-only files and full sync with hash verification.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import aiofiles
import httpx
import xxhash
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


def calculate_content_hash(content: bytes) -> str:
    """Calculate xxhash3 of content for integrity verification."""
    return xxhash.xxh3_64(content).hexdigest()


def calculate_file_hash(file_path: Path) -> Optional[str]:
    """Calculate xxhash3 of a file."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "rb") as f:
            return xxhash.xxh3_64(f.read()).hexdigest()
    except Exception:
        return None


class LogSynchronizer:
    """Synchronizes experiment logs from client to server.

    Monitors local log directory and uploads changes in real-time.
    Supports delta sync for CSV files and full sync for YAML/PT files.
    """

    def __init__(self, server_url: str, workspace: Path):
        """Initialize log synchronizer.

        Args:
            server_url: Web helper server URL
            workspace: Local workspace directory (e.g., logs_server-name/)
        """
        self.server_url = server_url.rstrip("/")
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.sync_state_file = self.workspace / ".sync_state.json"
        self.sync_state = self._load_sync_state()

        self.observer = Observer()
        self.http_client = httpx.AsyncClient(timeout=30.0)

        self._running = False

    def _load_sync_state(self) -> Dict:
        """Load synchronization state from disk."""
        if self.sync_state_file.exists():
            try:
                with open(self.sync_state_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load sync state: {e}")

        return {
            "server_url": self.server_url,
            "last_heartbeat": None,
            "active_experiments": {},
        }

    def _save_sync_state(self):
        """Save synchronization state to disk."""
        try:
            with open(self.sync_state_file, "w") as f:
                json.dump(self.sync_state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sync state: {e}")

    def start_sync(self, experiment_uid: str, project: str, run_name: str):
        """Start syncing a new experiment.

        Args:
            experiment_uid: Unique experiment identifier
            project: Project name
            run_name: Run name
        """
        if experiment_uid in self.sync_state["active_experiments"]:
            logger.warning(f"Experiment {experiment_uid} already syncing")
            return

        self.sync_state["active_experiments"][experiment_uid] = {
            "experiment_uid": experiment_uid,
            "project": project,
            "run_name": run_name,
            "started_at": datetime.utcnow().isoformat(),
            "files": {},
            "last_sync": None,
            "sync_status": "syncing",
        }
        self._save_sync_state()

        # Start watchdog for both Experiment and Run directories
        # 1. Experiment logs (terminal output)
        experiment_dir = self.workspace / "experiments" / experiment_uid
        experiment_dir.mkdir(parents=True, exist_ok=True)

        experiment_handler = ExperimentSyncHandler(
            self, experiment_uid, project, run_name, sync_type="experiment"
        )
        self.observer.schedule(experiment_handler, str(experiment_dir), recursive=False)

        # 2. Run logs (cvlabkit output)
        run_dir = self.workspace / "runs" / project
        run_dir.mkdir(parents=True, exist_ok=True)

        run_handler = ExperimentSyncHandler(
            self, experiment_uid, project, run_name, sync_type="run"
        )
        self.observer.schedule(run_handler, str(run_dir), recursive=False)

        if not self._running:
            self.observer.start()
            self._running = True

        logger.info(f"Started syncing experiment {experiment_uid} (Experiment + Run)")

    async def sync_file_change(
        self,
        experiment_uid: str,
        file_path: Path,
        event_type: str,
        sync_type: str = "run",
    ):
        """Sync a file change to server.

        Args:
            experiment_uid: Experiment identifier
            file_path: Changed file path
            event_type: "created" or "modified"
            sync_type: "experiment" (terminal logs) or "run" (cvlabkit output)
        """
        try:
            exp_state = self.sync_state["active_experiments"].get(experiment_uid)
            if not exp_state:
                logger.warning(f"Experiment {experiment_uid} not tracked")
                return

            file_name = file_path.name
            file_info = exp_state["files"].get(file_name, {})

            # Check if file actually changed (mtime + size based)
            current_stat = file_path.stat()
            current_mtime = int(current_stat.st_mtime)
            current_size = current_stat.st_size

            if (
                file_info.get("mtime") == current_mtime
                and file_info.get("size") == current_size
            ):
                # No actual change
                return

            # Determine sync strategy
            if file_path.suffix in {".csv", ".log"}:
                # Delta sync for append-only files (CSV, log)
                await self._sync_csv_delta(
                    experiment_uid,
                    file_path,
                    file_name,
                    file_info,
                    current_size,
                    sync_type,
                )
            else:
                # Full sync for YAML/PT (non-append files)
                await self._sync_full_file(
                    experiment_uid, file_path, file_name, sync_type
                )

            # Update file info
            exp_state["files"][file_name] = {
                "mtime": current_mtime,
                "size": current_size,
                "synced_at": datetime.utcnow().isoformat(),
                "last_offset": current_size if file_path.suffix == ".csv" else None,
            }
            exp_state["last_sync"] = datetime.utcnow().isoformat()
            self._save_sync_state()

        except Exception as e:
            logger.error(f"Failed to sync {file_path}: {e}")

    async def _sync_csv_delta(
        self,
        experiment_uid: str,
        file_path: Path,
        file_name: str,
        file_info: Dict,
        current_size: int,
        sync_type: str = "run",
    ):
        """Sync CSV delta (incremental).

        Args:
            experiment_uid: Experiment ID
            file_path: CSV file path
            file_name: File name
            file_info: Current file info from sync state
            current_size: Current file size
            sync_type: "experiment" or "run"
        """
        last_offset = file_info.get("last_offset", 0)

        if current_size <= last_offset:
            # File truncated or no new data
            return

        # Read delta
        async with aiofiles.open(file_path, "rb") as f:
            await f.seek(last_offset)
            delta_content = await f.read()

        if not delta_content:
            return

        # Upload delta with hash verification
        content_hash = calculate_content_hash(delta_content)
        files = {"delta": (file_name, delta_content, "application/octet-stream")}
        headers = {"X-Content-Hash": content_hash}

        if sync_type == "experiment":
            # Experiment terminal logs → web_helper/queue_logs/{exp_uid}/
            url = f"{self.server_url}/api/sync/experiment/{experiment_uid}/{file_name}"
        else:
            # Run cvlabkit output → logs/{project}/
            url = f"{self.server_url}/api/sync/run/{experiment_uid}/{file_name}"

        response = await self.http_client.post(url, files=files, headers=headers)

        if response.status_code == 200:
            logger.debug(
                f"Delta synced ({sync_type}): {file_name} ({len(delta_content)} bytes, offset: {last_offset})"
            )
        else:
            logger.error(
                f"Failed to sync delta: {response.status_code} {response.text}"
            )
            raise Exception(f"Delta sync failed: {response.status_code}")

    async def _sync_full_file(
        self,
        experiment_uid: str,
        file_path: Path,
        file_name: str,
        sync_type: str = "run",
    ):
        """Sync entire file (YAML/PT).

        Args:
            experiment_uid: Experiment ID
            file_path: File path
            file_name: File name
            sync_type: "experiment" or "run"
        """
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()

        # Upload with hash verification
        content_hash = calculate_content_hash(content)
        files = {"file": (file_name, content, "application/octet-stream")}
        headers = {"X-Content-Hash": content_hash}

        if sync_type == "experiment":
            # Experiment metadata → web_helper/queue_logs/{exp_uid}/
            url = f"{self.server_url}/api/sync/experiment/{experiment_uid}/{file_name}"
        else:
            # Run results → logs/{project}/
            url = f"{self.server_url}/api/sync/run/{experiment_uid}/{file_name}"

        response = await self.http_client.post(url, files=files, headers=headers)

        if response.status_code == 200:
            logger.debug(
                f"Full file synced ({sync_type}): {file_name} ({len(content)} bytes)"
            )
        else:
            logger.error(
                f"Failed to sync full file: {response.status_code} {response.text}"
            )
            raise Exception(f"Full file sync failed: {response.status_code}")

    async def recover_from_disconnection(self, experiment_uid: str):
        """Recover sync after network disconnection.

        Uses hash-based comparison to detect and sync missing data.
        Handles both Experiment files (terminal logs) and Run files (cvlabkit output).

        Args:
            experiment_uid: Experiment to recover
        """
        try:
            logger.info(f"Recovering sync for {experiment_uid}")

            # Query server state
            response = await self.http_client.get(
                f"{self.server_url}/api/sync/status/{experiment_uid}"
            )

            if response.status_code != 200:
                logger.error(f"Failed to query server state: {response.status_code}")
                return

            server_state = response.json()["data"]
            server_files = server_state.get("files", {})
            server_experiment_files = server_state.get("experiment_files", {})

            # Compare with local state
            exp_state = self.sync_state["active_experiments"].get(experiment_uid)
            if not exp_state:
                logger.warning(f"Experiment {experiment_uid} not tracked locally")
                return

            project = exp_state["project"]
            run_name = exp_state["run_name"]
            synced_count = 0

            # 1. Recover Experiment files (terminal logs)
            experiment_dir = self.workspace / "experiments" / experiment_uid
            if experiment_dir.exists():
                for file_path in experiment_dir.glob("*.log"):
                    file_name = file_path.name
                    server_info = server_experiment_files.get(file_name, {})

                    # Hash-based comparison
                    local_hash = calculate_file_hash(file_path)
                    server_hash = server_info.get("hash")

                    if local_hash and local_hash != server_hash:
                        # Server needs update - use delta sync for logs
                        server_size = server_info.get("size", 0)
                        local_size = file_path.stat().st_size

                        if local_size > server_size:
                            await self._sync_csv_delta(
                                experiment_uid,
                                file_path,
                                file_name,
                                {"last_offset": server_size},
                                local_size,
                                sync_type="experiment",
                            )
                            synced_count += 1
                            logger.debug(f"Recovered experiment file: {file_name}")

            # 2. Recover Run files (cvlabkit output)
            run_dir = self.workspace / "runs" / project
            if run_dir.exists():
                for file_path in run_dir.glob(f"{run_name}.*"):
                    if not file_path.is_file():
                        continue

                    file_name = file_path.name
                    server_info = server_files.get(file_name, {})

                    # Hash-based comparison
                    local_hash = calculate_file_hash(file_path)
                    server_hash = server_info.get("hash")

                    if local_hash and local_hash != server_hash:
                        # Server needs update
                        if file_path.suffix in {".csv", ".log"}:
                            # Delta sync for append-only files
                            server_size = server_info.get("size", 0)
                            local_size = file_path.stat().st_size

                            if local_size > server_size:
                                await self._sync_csv_delta(
                                    experiment_uid,
                                    file_path,
                                    file_name,
                                    {"last_offset": server_size},
                                    local_size,
                                    sync_type="run",
                                )
                                synced_count += 1
                        else:
                            # Full sync for YAML/PT
                            await self._sync_full_file(
                                experiment_uid, file_path, file_name, sync_type="run"
                            )
                            synced_count += 1

                        logger.debug(f"Recovered run file: {file_name}")

            if synced_count > 0:
                logger.info(f"Recovery complete for {experiment_uid}: {synced_count} files synced")
            else:
                logger.info(f"Recovery complete for {experiment_uid}: already in sync")

        except Exception as e:
            logger.error(f"Failed to recover sync: {e}")

    async def final_sync(self, experiment_uid: str):
        """Perform final sync when experiment completes.

        Ensures all files are fully synced before cleanup.

        Args:
            experiment_uid: Completed experiment ID
        """
        try:
            logger.info(f"Final sync for {experiment_uid}")

            exp_state = self.sync_state["active_experiments"].get(experiment_uid)
            if not exp_state:
                return

            project = exp_state["project"]
            run_name = exp_state.get("run_name", "")

            # 1. Sync experiment terminal logs (workspace/experiments/{exp_uid}/)
            experiment_dir = self.workspace / "experiments" / experiment_uid
            if experiment_dir.exists():
                for file_path in experiment_dir.glob("*"):
                    if file_path.is_file():
                        await self.sync_file_change(
                            experiment_uid, file_path, "modified", sync_type="experiment"
                        )

            # 2. Sync run output files (workspace/runs/{project}/)
            run_dir = self.workspace / "runs" / project
            if run_dir.exists():
                for file_path in run_dir.glob(f"{run_name}*"):
                    if file_path.is_file():
                        await self.sync_file_change(
                            experiment_uid, file_path, "modified", sync_type="run"
                        )

            exp_state["sync_status"] = "completed"
            self._save_sync_state()

            logger.info(f"Final sync complete for {experiment_uid}")

        except Exception as e:
            logger.error(f"Failed final sync: {e}")

    def stop(self):
        """Stop synchronizer."""
        if self._running:
            self.observer.stop()
            self.observer.join()
            self._running = False

    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


class ExperimentSyncHandler(FileSystemEventHandler):
    """Watchdog handler for experiment file changes."""

    def __init__(
        self,
        synchronizer: LogSynchronizer,
        experiment_uid: str,
        project: str,
        run_name: str,
        sync_type: str = "run",
    ):
        super().__init__()
        self.synchronizer = synchronizer
        self.experiment_uid = experiment_uid
        self.project = project
        self.run_name = run_name
        self.sync_type = sync_type  # "experiment" or "run"

    def on_created(self, event):
        """Handle file creation."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only process files matching run_name pattern
        if not self._should_sync_file(file_path):
            return

        asyncio.create_task(
            self.synchronizer.sync_file_change(
                self.experiment_uid, file_path, "created", self.sync_type
            )
        )

    def on_modified(self, event):
        """Handle file modification."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if not self._should_sync_file(file_path):
            return

        asyncio.create_task(
            self.synchronizer.sync_file_change(
                self.experiment_uid, file_path, "modified", self.sync_type
            )
        )

    def _should_sync_file(self, file_path: Path) -> bool:
        """Check if file should be synced."""
        # Skip hidden/temp files
        if file_path.name.startswith("."):
            return False

        if self.sync_type == "experiment":
            # Experiment: terminal logs only
            if file_path.suffix != ".log":
                return False
            # Accept terminal_log.log and terminal_err.log
            if file_path.name not in {"terminal_log.log", "terminal_err.log"}:
                return False
        else:
            # Run: cvlabkit output (CSV, YAML, PT)
            if file_path.suffix not in {".csv", ".yaml", ".yml", ".pt", ".pth"}:
                return False
            # Check if file matches run_name pattern
            if not file_path.name.startswith(self.run_name):
                return False

        return True
