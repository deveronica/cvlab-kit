#!/usr/bin/env python3
"""Device Agent for distributed CVLab-Kit execution.

This agent runs on remote GPU servers and:
1. Sends heartbeats with system stats
2. Polls for new experiments to execute
3. Runs experiments locally
4. Syncs logs back to server in real-time
"""

import argparse
import asyncio
import logging
import os
import re
import signal
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import httpx
import psutil
import yaml

try:
    import pynvml

    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    pynvml = None

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .log_synchronizer import LogSynchronizer
from .component_manager import ComponentManager

# Code version tracking for reproducibility
try:
    from web_helper.backend.services.hash_utils import get_code_version

    CODE_VERSION_AVAILABLE = True
except ImportError:
    CODE_VERSION_AVAILABLE = False
    get_code_version = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DeviceAgent:
    """Client agent for distributed experiment execution."""

    def __init__(
        self,
        server_url: str,
        host_id: Optional[str] = None,
        heartbeat_interval: int = 10,
        poll_interval: int = 5,
        api_key: Optional[str] = None,
        connect_timeout: float = 10.0,
        request_timeout: float = 30.0,
        max_jobs: int = 1,
    ):
        """Initialize device agent.

        Args:
            server_url: Web helper server URL
            host_id: Custom host identifier (default: hostname)
            heartbeat_interval: Heartbeat interval in seconds
            poll_interval: Job polling interval in seconds
            api_key: API key for authentication (or set CVLABKIT_API_KEY env var)
            connect_timeout: Connection timeout in seconds (default: 10)
            request_timeout: Total request timeout in seconds (default: 30)
            max_jobs: Maximum concurrent jobs to run (default: 1)
        """
        self.server_url = server_url.rstrip("/")
        self.host_id = host_id or socket.gethostname()
        self.heartbeat_interval = heartbeat_interval
        self.poll_interval = poll_interval
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.max_jobs = max_jobs

        # API key for authentication
        self.api_key = api_key or os.environ.get("CVLABKIT_API_KEY")

        # Workspace for logs
        server_name = self._sanitize_server_name(server_url)
        self.workspace = Path(f"logs_{server_name}")
        self.workspace.mkdir(parents=True, exist_ok=True)

        # HTTP client with optional auth headers and configurable timeout
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            logger.info("🔐 API key authentication enabled")

        timeout = httpx.Timeout(
            connect=self.connect_timeout,
            read=self.request_timeout,
            write=self.request_timeout,
            pool=self.request_timeout,
        )
        self.http_client = httpx.AsyncClient(timeout=timeout, headers=headers)

        # Log synchronizer
        self.synchronizer = LogSynchronizer(server_url, self.workspace)

        # Component manager for version-controlled component sync
        self.component_manager = ComponentManager(
            server_url=server_url,
            base_path=Path.cwd(),
            api_key=self.api_key,
        )

        # Active jobs
        self.active_jobs: Dict[str, Dict] = {}

        # Running flag
        self.running = False

        # Backoff configuration for reconnection
        self.base_backoff = 1.0  # Initial backoff in seconds
        self.max_backoff = 60.0  # Maximum backoff in seconds
        self.backoff_factor = 2.0  # Exponential factor
        self.current_backoff = self.base_backoff
        self.consecutive_failures = 0

        # Initialize NVIDIA if available
        if NVIDIA_ML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                logger.info("NVIDIA ML initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA ML: {e}")

        logger.info(f"Device agent initialized: {self.host_id} → {self.server_url}")

    @staticmethod
    def _sanitize_server_name(url: str) -> str:
        """Sanitize server URL for directory name."""
        # Extract hostname from URL
        match = re.search(r"://([^:/]+)", url)
        if match:
            hostname = match.group(1)
            return re.sub(r"[^a-zA-Z0-9_-]", "_", hostname)
        return "default"

    async def start(self):
        """Start the device agent."""
        self.running = True
        logger.info("Device agent starting...")

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._job_polling_loop()),
            asyncio.create_task(self._monitor_active_jobs()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Device agent tasks cancelled")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the device agent."""
        logger.info("Stopping device agent...")
        self.running = False

        # Stop synchronizer
        self.synchronizer.stop()
        await self.synchronizer.close()

        # Close component manager
        self.component_manager.close()

        # Close HTTP client
        await self.http_client.aclose()

        # Cleanup NVIDIA
        if NVIDIA_ML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        logger.info("Device agent stopped")

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, stopping...")
        self.running = False

    def _reset_backoff(self):
        """Reset backoff to initial values after successful connection."""
        was_disconnected = self.consecutive_failures > 0
        self.current_backoff = self.base_backoff
        self.consecutive_failures = 0
        return was_disconnected

    def _calculate_backoff(self) -> float:
        """Calculate next backoff duration with exponential increase."""
        self.consecutive_failures += 1
        backoff = min(
            self.base_backoff * (self.backoff_factor ** (self.consecutive_failures - 1)),
            self.max_backoff,
        )
        self.current_backoff = backoff
        return backoff

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to server with exponential backoff on failures."""
        while self.running:
            try:
                success = await self._send_heartbeat()
                if success:
                    was_disconnected = self._reset_backoff()

                    # Trigger recovery for active experiments after reconnection
                    if was_disconnected and self.active_jobs:
                        logger.info("Connection restored, triggering sync recovery...")
                        await self._trigger_recovery_for_active_jobs()

                    await asyncio.sleep(self.heartbeat_interval)
                else:
                    backoff = self._calculate_backoff()
                    logger.warning(
                        f"Heartbeat failed, retrying in {backoff:.1f}s "
                        f"(attempt {self.consecutive_failures})"
                    )
                    await asyncio.sleep(backoff)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                backoff = self._calculate_backoff()
                logger.info(f"Retrying in {backoff:.1f}s...")
                await asyncio.sleep(backoff)

    async def _trigger_recovery_for_active_jobs(self):
        """Trigger sync recovery for all active experiments after reconnection."""
        for experiment_uid in list(self.active_jobs.keys()):
            try:
                await self.synchronizer.recover_from_disconnection(experiment_uid)
            except Exception as e:
                logger.error(f"Failed to recover sync for {experiment_uid}: {e}")

    async def _send_heartbeat(self) -> bool:
        """Send single heartbeat with system stats.

        Returns:
            True if heartbeat was sent successfully, False otherwise.
        """
        try:
            stats = self._collect_system_stats()

            response = await self.http_client.post(
                f"{self.server_url}/api/devices/heartbeat", json=stats
            )

            if response.status_code == 200:
                logger.debug(f"Heartbeat sent: {self.host_id}")
                return True
            else:
                logger.warning(
                    f"Heartbeat failed: {response.status_code} {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            return False

    def _collect_system_stats(self) -> Dict:
        """Collect system statistics."""
        stats = {
            "host_id": self.host_id,
            "cpu_util": psutil.cpu_percent(interval=1),
            "memory_used": psutil.virtual_memory().used / (1024**3),  # bytes to GB
            "memory_total": psutil.virtual_memory().total / (1024**3),  # bytes to GB
            "disk_free": psutil.disk_usage("/").free / (1024**3),  # bytes to GB
        }

        # GPU stats (NVIDIA)
        if NVIDIA_ML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                stats["gpu_count"] = device_count

                if device_count > 0:
                    # Use first GPU for summary stats
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    stats["gpu_util"] = util.gpu
                    stats["vram_used"] = mem_info.used / (1024**3)  # bytes to GB
                    stats["vram_total"] = mem_info.total / (1024**3)  # bytes to GB

                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        stats["gpu_temperature"] = temp
                    except Exception:
                        pass

                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle)
                        stats["gpu_power_usage"] = power / 1000.0  # mW to W
                    except Exception:
                        pass

                    # Detailed info for all GPUs
                    gpus = []
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        name = pynvml.nvmlDeviceGetName(handle)

                        gpu_info = {
                            "id": i,  # Frontend expects "id"
                            "name": name if isinstance(name, str) else name.decode(),
                            "util": util.gpu,  # Frontend expects "util"
                            "vram_used": mem_info.used / (1024**3),  # Frontend expects "vram_used"
                            "vram_total": mem_info.total / (1024**3),  # Frontend expects "vram_total"
                        }

                        # Add temperature if available
                        try:
                            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                            gpu_info["temperature"] = temp
                        except Exception:
                            pass

                        # Add power usage if available
                        try:
                            power = pynvml.nvmlDeviceGetPowerUsage(handle)
                            gpu_info["power_usage"] = power / 1000.0  # mW to W
                        except Exception:
                            pass

                        gpus.append(gpu_info)

                    stats["gpus"] = gpus

            except Exception as e:
                logger.debug(f"Failed to collect GPU stats: {e}")

        # PyTorch version
        if TORCH_AVAILABLE:
            stats["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                stats["cuda_version"] = torch.version.cuda

        # Code version for reproducibility tracking
        if CODE_VERSION_AVAILABLE:
            try:
                code_version = get_code_version()
                stats["code_version"] = code_version
            except Exception as e:
                logger.debug(f"Failed to collect code version: {e}")

        # Active jobs for server-side validation
        if self.active_jobs:
            stats["active_jobs"] = list(self.active_jobs.keys())

        return stats

    async def _job_polling_loop(self):
        """Poll for new jobs to execute."""
        while self.running:
            try:
                # Check if we have capacity for more jobs
                if len(self.active_jobs) >= self.max_jobs:
                    await asyncio.sleep(self.poll_interval)
                    continue

                # Poll for next job
                job = await self._poll_next_job()

                if job:
                    logger.info(
                        f"Starting job (active: {len(self.active_jobs) + 1}/{self.max_jobs})"
                    )
                    asyncio.create_task(self._execute_job(job))

            except Exception as e:
                logger.error(f"Job polling failed: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _poll_next_job(self) -> Optional[Dict]:
        """Poll server for next job assigned to this device.

        Returns:
            Job dict if available, None otherwise
        """
        try:
            response = await self.http_client.get(
                f"{self.server_url}/api/queue/next_job",
                params={"host_id": self.host_id},
            )

            if response.status_code == 200:
                data = response.json()
                # Check for actual job, not just the wrapper {"job": None}
                if data.get("data") and data["data"].get("job"):
                    job = data["data"]["job"]
                    # Validate job has required fields
                    if not job.get("experiment_uid"):
                        logger.warning(f"Job missing experiment_uid: {job}")
                        return None
                    return job
            elif response.status_code != 404:
                logger.warning(f"Poll failed: {response.status_code}")

        except Exception as e:
            logger.debug(f"Poll error: {e}")

        return None

    async def _execute_job(self, job: Dict):
        """Execute a job locally.

        Args:
            job: Job specification from server
        """
        experiment_uid = job.get("experiment_uid")
        if not experiment_uid:
            logger.error(f"Job missing experiment_uid: {job}")
            return
        logger.info(f"Executing job: {experiment_uid} ({job.get('name', 'unnamed')})")

        try:
            # 1. Download config
            config_path = await self._download_config(
                job["config_path"], experiment_uid
            )

            # 2. Sync components from version store
            sync_result = self.component_manager.sync_from_version_store(config_path)
            if sync_result["failed"]:
                logger.warning(
                    f"Failed to sync some components: {sync_result['failed']}"
                )
            if sync_result["synced"]:
                logger.info(f"Synced components: {sync_result['synced']}")

            # Save experiment manifest for reproducibility
            if sync_result["component_hashes"]:
                self.component_manager.save_experiment_manifest(
                    experiment_uid, sync_result["component_hashes"]
                )

            # 3. Setup directory structure (Experiment vs Run separation)
            project = job["project"]
            run_name = job.get("meta", {}).get("run_name", experiment_uid)

            # Experiment logs (terminal output for process management)
            experiment_dir = self.workspace / "experiments" / experiment_uid
            experiment_dir.mkdir(parents=True, exist_ok=True)

            stdout_log = experiment_dir / "terminal_log.log"
            stderr_log = experiment_dir / "terminal_err.log"

            # Run logs (cvlabkit output for result analysis)
            run_dir = self.workspace / "runs" / project
            run_dir.mkdir(parents=True, exist_ok=True)

            # 4. Start synchronization (both Experiment and Run)
            self.synchronizer.start_sync(experiment_uid, project, run_name)

            # 5. Override config device if GPU-specific assignment
            assigned_device = job.get("assigned_device")
            if assigned_device and ":" in assigned_device:
                # Virtual device format: "gnode-3:0" → GPU 0
                try:
                    gpu_id = int(assigned_device.split(":")[-1])

                    # Override config device field
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)

                    if config is None:
                        config = {}

                    config['device'] = gpu_id

                    with open(config_path, 'w') as f:
                        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

                    logger.info(f"Overrode config device to GPU {gpu_id}")
                except Exception as e:
                    logger.error(f"Failed to override config device: {e}")

            # 6. Execute cvlabkit
            env = os.environ.copy()
            # cvlabkit writes to runs/ directory
            env["CVLAB_LOG_DIR"] = str(run_dir)

            process = await asyncio.create_subprocess_exec(
                "uv",
                "run",
                "main.py",
                "--config",
                str(config_path),
                "--fast",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=Path.cwd(),
            )

            # Track active job
            self.active_jobs[experiment_uid] = {
                "process": process,
                "job": job,
                "started_at": datetime.utcnow(),
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
            }

            # Stream output to log files
            asyncio.create_task(
                self._stream_output(
                    process.stdout, stdout_log, experiment_uid, "stdout"
                )
            )
            asyncio.create_task(
                self._stream_output(
                    process.stderr, stderr_log, experiment_uid, "stderr"
                )
            )

            # Wait for completion
            return_code = await process.wait()

            logger.info(f"Job {experiment_uid} finished with code {return_code}")

            # 6. Final sync
            await self.synchronizer.final_sync(experiment_uid)

            # 7. Report completion to server
            if return_code == 0:
                await self._report_job_completion(experiment_uid, success=True)
            else:
                await self._report_job_completion(
                    experiment_uid, success=False, error=f"Exit code: {return_code}"
                )

            # 8. Cleanup
            del self.active_jobs[experiment_uid]

        except Exception as e:
            logger.error(f"Failed to execute job {experiment_uid}: {e}")
            if experiment_uid in self.active_jobs:
                del self.active_jobs[experiment_uid]
            await self._report_job_completion(experiment_uid, success=False, error=str(e))

    async def _download_config(self, config_path: str, experiment_uid: str) -> Path:
        """Download config file from server.

        Args:
            config_path: Path to config on server
            experiment_uid: Experiment ID

        Returns:
            Local path to downloaded config
        """
        response = await self.http_client.get(
            f"{self.server_url}/api/configs/raw", params={"path": config_path}
        )

        if response.status_code != 200:
            raise Exception(f"Failed to download config: {response.status_code}")

        # Save config locally
        local_config = self.workspace / f"{experiment_uid}_config.yaml"
        with open(local_config, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded config: {config_path} → {local_config}")
        return local_config

    async def _stream_output(
        self, stream, log_file: Path, experiment_uid: str, stream_name: str
    ):
        """Stream subprocess output to log file.

        Args:
            stream: Subprocess stream (stdout/stderr)
            log_file: Log file path
            experiment_uid: Experiment ID
            stream_name: "stdout" or "stderr"
        """
        try:
            with open(log_file, "w") as f:
                while True:
                    line = await stream.readline()
                    if not line:
                        break

                    decoded = line.decode("utf-8", errors="replace")
                    f.write(decoded)
                    f.flush()

        except Exception as e:
            logger.error(f"Error streaming {stream_name} for {experiment_uid}: {e}")

    async def _report_job_completion(
        self, experiment_uid: str, success: bool, error: Optional[str] = None
    ):
        """Report job completion to server."""
        try:
            response = await self.http_client.post(
                f"{self.server_url}/api/queue/complete_job",
                json={
                    "experiment_uid": experiment_uid,
                    "success": success,
                    "error_message": error,
                },
            )
            if response.status_code == 200:
                logger.info(f"Reported job {experiment_uid} completion: {'success' if success else 'failed'}")
            else:
                logger.warning(f"Failed to report job completion: {response.status_code}")
        except Exception as e:
            logger.error(f"Error reporting job completion: {e}")

    async def _monitor_active_jobs(self):
        """Monitor active jobs and handle failures."""
        while self.running:
            try:
                for experiment_uid, job_info in list(self.active_jobs.items()):
                    process = job_info["process"]

                    # Check if process is still running
                    if process.returncode is not None:
                        # Process finished, will be cleaned up by execute_job
                        continue

                    # Could add health checks here (e.g., timeout detection)

            except Exception as e:
                logger.error(f"Error monitoring jobs: {e}")

            await asyncio.sleep(30)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CVLab-Kit Device Agent for distributed execution"
    )
    parser.add_argument(
        "--server",
        required=True,
        help="Web helper server URL (e.g., https://lab-server:8000)",
    )
    parser.add_argument("--host-id", help="Custom host identifier (default: hostname)")
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=10,
        help="Heartbeat interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Job polling interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CVLABKIT_API_KEY"),
        help="API key for authentication (or set CVLABKIT_API_KEY env var)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help="Maximum concurrent jobs to run (default: 1)",
    )

    args = parser.parse_args()

    agent = DeviceAgent(
        server_url=args.server,
        host_id=args.host_id,
        heartbeat_interval=args.heartbeat_interval,
        poll_interval=args.poll_interval,
        api_key=args.api_key,
        connect_timeout=args.connect_timeout,
        request_timeout=args.request_timeout,
        max_jobs=args.max_jobs,
    )

    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
