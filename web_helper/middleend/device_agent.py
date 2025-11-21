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
    ):
        """Initialize device agent.

        Args:
            server_url: Web helper server URL
            host_id: Custom host identifier (default: hostname)
            heartbeat_interval: Heartbeat interval in seconds
            poll_interval: Job polling interval in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.host_id = host_id or socket.gethostname()
        self.heartbeat_interval = heartbeat_interval
        self.poll_interval = poll_interval

        # Workspace for logs
        server_name = self._sanitize_server_name(server_url)
        self.workspace = Path(f"logs_{server_name}")
        self.workspace.mkdir(parents=True, exist_ok=True)

        # HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Log synchronizer
        self.synchronizer = LogSynchronizer(server_url, self.workspace)

        # Active jobs
        self.active_jobs: Dict[str, Dict] = {}

        # Running flag
        self.running = False

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

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to server."""
        while self.running:
            try:
                await self._send_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")

            await asyncio.sleep(self.heartbeat_interval)

    async def _send_heartbeat(self):
        """Send single heartbeat with system stats."""
        try:
            stats = self._collect_system_stats()

            response = await self.http_client.post(
                f"{self.server_url}/api/devices/heartbeat", json=stats
            )

            if response.status_code == 200:
                logger.debug(f"Heartbeat sent: {self.host_id}")
            else:
                logger.warning(
                    f"Heartbeat failed: {response.status_code} {response.text}"
                )

        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")

    def _collect_system_stats(self) -> Dict:
        """Collect system statistics."""
        stats = {
            "host_id": self.host_id,
            "cpu_util": psutil.cpu_percent(interval=1),
            "memory_used": psutil.virtual_memory().used,
            "memory_total": psutil.virtual_memory().total,
            "disk_free": psutil.disk_usage("/").free,
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
                    stats["vram_used"] = mem_info.used
                    stats["vram_total"] = mem_info.total

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
                            "index": i,
                            "name": name if isinstance(name, str) else name.decode(),
                            "utilization": util.gpu,
                            "memory_used": mem_info.used,
                            "memory_total": mem_info.total,
                        }
                        gpus.append(gpu_info)

                    stats["gpus"] = gpus

            except Exception as e:
                logger.debug(f"Failed to collect GPU stats: {e}")

        # PyTorch version
        if TORCH_AVAILABLE:
            stats["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                stats["cuda_version"] = torch.version.cuda

        return stats

    async def _job_polling_loop(self):
        """Poll for new jobs to execute."""
        while self.running:
            try:
                # Check if we have capacity
                if len(self.active_jobs) >= 1:
                    # Only run one job at a time for now
                    await asyncio.sleep(self.poll_interval)
                    continue

                # Poll for next job
                job = await self._poll_next_job()

                if job:
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
                if data.get("data"):
                    return data["data"]
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
        experiment_uid = job["experiment_uid"]
        logger.info(f"Executing job: {experiment_uid} ({job.get('name', 'unnamed')})")

        try:
            # 1. Download config
            config_path = await self._download_config(
                job["config_path"], experiment_uid
            )

            # 2. Setup directory structure (Experiment vs Run separation)
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

            # 3. Start synchronization (both Experiment and Run)
            self.synchronizer.start_sync(experiment_uid, project, run_name)

            # 4. Execute cvlabkit
            env = os.environ.copy()
            # cvlabkit writes to runs/ directory
            env["CVLAB_LOG_DIR"] = str(run_dir)

            # Override CUDA_VISIBLE_DEVICES if specified
            assigned_device = job.get("assigned_device")
            if assigned_device and assigned_device.startswith("cuda:"):
                gpu_id = assigned_device.split(":")[-1]
                env["CUDA_VISIBLE_DEVICES"] = gpu_id

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

            # 5. Final sync
            await self.synchronizer.final_sync(experiment_uid)

            # 6. Cleanup
            del self.active_jobs[experiment_uid]

        except Exception as e:
            logger.error(f"Failed to execute job {experiment_uid}: {e}")
            if experiment_uid in self.active_jobs:
                del self.active_jobs[experiment_uid]

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
            async with asyncio.open_file(log_file, "w") as f:
                while True:
                    line = await stream.readline()
                    if not line:
                        break

                    decoded = line.decode("utf-8", errors="replace")
                    await f.write(decoded)
                    await f.flush()

        except Exception as e:
            logger.error(f"Error streaming {stream_name} for {experiment_uid}: {e}")

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

    args = parser.parse_args()

    agent = DeviceAgent(
        server_url=args.server,
        host_id=args.host_id,
        heartbeat_interval=args.heartbeat_interval,
        poll_interval=args.poll_interval,
    )

    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
