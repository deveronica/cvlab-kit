"""Heartbeat client agent for middleend (GPU worker)."""

import logging
import socket
import time
from typing import Any, Optional

import psutil
import requests

# Optional nvidia-ml-py for better GPU monitoring
try:
    import pynvml

    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    pynvml = None


class ClientAgent:
    """Client agent for heartbeat monitoring according to spec."""

    def __init__(
        self,
        web_helper_url: str = "http://localhost:8000",
        heartbeat_interval: int = 10,
        host_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize client agent.

        Args:
            web_helper_url: URL of web-helper API
            heartbeat_interval: Heartbeat interval in seconds
            host_id: Custom host identifier (defaults to hostname)
            api_key: API key for authentication (or set CVLABKIT_API_KEY env var)
        """
        import os

        self.web_helper_url = web_helper_url.rstrip("/")
        self.heartbeat_interval = heartbeat_interval
        self.host_id = host_id or socket.gethostname()
        self.running = False

        # API key for authentication
        self.api_key = api_key or os.environ.get("CVLABKIT_API_KEY")
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # Backoff configuration for reconnection
        self.base_backoff = 1.0  # Initial backoff in seconds
        self.max_backoff = 60.0  # Maximum backoff in seconds
        self.backoff_factor = 2.0  # Exponential factor
        self.consecutive_failures = 0

        logging.info(f"Client agent initialized for host {self.host_id}")

    def _reset_backoff(self):
        """Reset backoff to initial values after successful connection."""
        self.consecutive_failures = 0

    def _calculate_backoff(self) -> float:
        """Calculate next backoff duration with exponential increase."""
        self.consecutive_failures += 1
        backoff = min(
            self.base_backoff * (self.backoff_factor ** (self.consecutive_failures - 1)),
            self.max_backoff,
        )
        return backoff

    def start(self):
        """Start the heartbeat monitoring loop with exponential backoff."""
        self.running = True
        logging.info("Starting heartbeat monitoring...")

        while self.running:
            try:
                success = self._send_heartbeat()
                if success:
                    self._reset_backoff()
                    time.sleep(self.heartbeat_interval)
                else:
                    backoff = self._calculate_backoff()
                    logging.warning(
                        f"Heartbeat failed, retrying in {backoff:.1f}s "
                        f"(attempt {self.consecutive_failures})"
                    )
                    time.sleep(backoff)
            except KeyboardInterrupt:
                logging.info("Heartbeat monitoring stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in heartbeat loop: {e}")
                backoff = self._calculate_backoff()
                logging.info(f"Retrying in {backoff:.1f}s...")
                time.sleep(backoff)

    def stop(self):
        """Stop the heartbeat monitoring."""
        self.running = False
        logging.info("Heartbeat monitoring stopped")

    def _send_heartbeat(self) -> bool:
        """Send a single heartbeat to the web-helper.

        Returns:
            True if heartbeat was sent successfully, False otherwise.
        """
        try:
            # Collect system stats
            stats = self._collect_system_stats()

            # Send to web-helper
            response = requests.post(
                f"{self.web_helper_url}/api/devices/heartbeat",
                json=stats,
                headers=self.headers,
                timeout=5,
            )

            if response.status_code == 200:
                logging.debug(f"Heartbeat sent successfully for {self.host_id}")
                return True
            else:
                logging.warning(f"Heartbeat failed with status {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send heartbeat: {e}")
            return False
        except Exception as e:
            logging.error(f"Error collecting system stats: {e}")
            return False

    def _collect_system_stats(self) -> dict[str, Any]:
        """Collect system statistics for heartbeat.

        Returns:
            Dictionary with system stats according to spec format:
            {host_id, gpu_util, vram, cpu, disk, torch/cuda, ts}

            Multi-GPU Support:
            - gpu_util: Average utilization across all GPUs (backward compatible)
            - vram_used/vram_total: Sum across all GPUs (backward compatible)
            - gpu_count: Number of GPUs detected
            - gpus: Detailed array of individual GPU stats
        """
        stats = {
            "host_id": self.host_id,
            "cpu_util": None,
            "memory_used": None,
            "memory_total": None,
            "disk_free": None,
            "gpu_util": None,
            "vram_total": None,
            "vram_used": None,
            "gpu_count": 0,
            "gpus": None,
            "torch_version": None,
            "cuda_version": None,
        }

        # CPU utilization
        try:
            stats["cpu_util"] = psutil.cpu_percent(interval=1)
        except Exception as e:
            logging.debug(f"Could not get CPU stats: {e}")

        # RAM usage
        try:
            memory = psutil.virtual_memory()
            stats["memory_used"] = memory.used / (1024**3)  # GB
            stats["memory_total"] = memory.total / (1024**3)  # GB
        except Exception as e:
            logging.debug(f"Could not get memory stats: {e}")

        # Disk space
        try:
            disk_usage = psutil.disk_usage("/")
            stats["disk_free"] = disk_usage.free / (1024**3)  # GB
        except Exception as e:
            logging.debug(f"Could not get disk stats: {e}")

        # Multi-GPU support - try nvidia-ml-py first for accurate stats
        detailed_gpus = self.get_detailed_gpu_info()
        if detailed_gpus:
            # Convert detailed GPU info to heartbeat format
            gpus_array = []
            total_vram_mb = 0.0
            total_vram_used_mb = 0.0
            total_util = 0.0

            for gpu in detailed_gpus:
                gpu_data = {
                    "id": gpu["index"],
                    "name": gpu["name"],
                    "util": gpu["utilization_gpu"],
                    "vram_used": gpu["memory_used"] / 1024,  # Convert MB to GB
                    "vram_total": gpu["memory_total"] / 1024,  # Convert MB to GB
                    "temperature": gpu["temperature"],
                    "power_usage": gpu["power_usage"],
                }
                gpus_array.append(gpu_data)

                # Accumulate for aggregated values
                total_vram_mb += gpu["memory_total"]
                total_vram_used_mb += gpu["memory_used"]
                total_util += gpu["utilization_gpu"]

            # Set aggregated values (backward compatible)
            stats["gpu_count"] = len(detailed_gpus)
            stats["gpu_util"] = total_util / len(detailed_gpus)  # Average
            stats["vram_total"] = total_vram_mb / 1024  # Convert MB to GB (sum)
            stats["vram_used"] = total_vram_used_mb / 1024  # Convert MB to GB (sum)
            stats["gpus"] = gpus_array

            # Get PyTorch version if available
            try:
                import torch

                stats["torch_version"] = torch.__version__
                if torch.cuda.is_available():
                    stats["cuda_version"] = torch.version.cuda
            except ImportError:
                pass
        else:
            # Fallback to PyTorch-based stats (single GPU - CUDA or MPS)
            try:
                import torch

                stats["torch_version"] = torch.__version__

                # Try CUDA first
                if torch.cuda.is_available():
                    stats["cuda_version"] = torch.version.cuda

                    # Get GPU stats for primary device
                    device = torch.device("cuda:0")
                    torch.cuda.synchronize(device)

                    # Memory info
                    device_props = torch.cuda.get_device_properties(device)

                    vram_total_mb = device_props.total_memory / (1024**2)  # MB
                    vram_used_mb = torch.cuda.memory_allocated(device) / (1024**2)  # MB

                    stats["vram_total"] = vram_total_mb / 1024  # GB
                    stats["vram_used"] = vram_used_mb / 1024  # GB

                    # Estimate GPU utilization (simplified)
                    if vram_total_mb > 0:
                        memory_util = (vram_used_mb / vram_total_mb) * 100
                        stats["gpu_util"] = min(100.0, memory_util)
                    else:
                        stats["gpu_util"] = 0.0

                    stats["gpu_count"] = 1

                # Try Apple MPS (Metal Performance Shaders)
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    try:
                        # Get MPS device info
                        device = torch.device("mps")

                        # Get memory info (limited API)
                        if hasattr(torch.mps, "current_allocated_memory"):
                            vram_used_bytes = torch.mps.current_allocated_memory()
                            vram_used_gb = vram_used_bytes / (1024**3)
                            stats["vram_used"] = vram_used_gb

                            # Apple doesn't expose total VRAM easily
                            # Use system memory as rough estimate for unified memory
                            stats["vram_total"] = stats.get("memory_total", 0)

                            if stats["vram_total"] > 0:
                                stats["gpu_util"] = (
                                    vram_used_gb / stats["vram_total"]
                                ) * 100

                            stats["gpu_count"] = 1

                            # Create basic GPU info for Apple Silicon
                            stats["gpus"] = [
                                {
                                    "id": 0,
                                    "name": "Apple Silicon GPU",
                                    "util": stats["gpu_util"]
                                    if stats["gpu_util"]
                                    else 0.0,
                                    "vram_used": vram_used_gb,
                                    "vram_total": stats["vram_total"],
                                    "temperature": None,
                                    "power_usage": None,
                                }
                            ]
                    except Exception as e:
                        logging.debug(f"Could not get MPS GPU stats: {e}")

            except ImportError:
                logging.debug("PyTorch not available for GPU stats")
            except Exception as e:
                logging.debug(f"Could not get GPU stats: {e}")

        return stats

    def _get_nvidia_gpu_stats(self) -> dict[str, Any]:
        """Get detailed GPU statistics using nvidia-ml-py if available."""
        if not NVIDIA_ML_AVAILABLE:
            return {}

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                return {}

            # Get stats for the first GPU (primary device)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_total = memory_info.total / (1024**2)  # MB
            vram_used = memory_info.used / (1024**2)  # MB

            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu

            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temperature = None

            # Power usage
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
            except Exception:
                power_usage = None

            # Get PyTorch/CUDA version info
            stats = {
                "gpu_util": float(gpu_util),
                "vram_total": float(vram_total),
                "vram_used": float(vram_used),
                "gpu_temperature": temperature,
                "gpu_power_usage": power_usage,
            }

            # Try to get PyTorch version if available
            try:
                import torch

                stats["torch_version"] = torch.__version__
                if torch.cuda.is_available():
                    stats["cuda_version"] = torch.version.cuda
            except ImportError:
                pass

            pynvml.nvmlShutdown()
            return stats

        except Exception as e:
            logging.debug(f"Error getting NVIDIA GPU stats: {e}")
            return {}

    def get_detailed_gpu_info(self) -> list[dict[str, Any]]:
        """Get detailed information for all GPUs."""
        if not NVIDIA_ML_AVAILABLE:
            return []

        gpus = []
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Basic info
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Additional stats
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temperature = None

                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    power_limit = (
                        pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1]
                        / 1000.0
                    )
                except Exception:
                    power_usage = None
                    power_limit = None

                gpu_info = {
                    "index": i,
                    "name": name,
                    "memory_total": memory_info.total / (1024**2),  # MB
                    "memory_used": memory_info.used / (1024**2),  # MB
                    "memory_free": memory_info.free / (1024**2),  # MB
                    "utilization_gpu": utilization.gpu,
                    "utilization_memory": utilization.memory,
                    "temperature": temperature,
                    "power_usage": power_usage,
                    "power_limit": power_limit,
                }

                gpus.append(gpu_info)

            pynvml.nvmlShutdown()

        except Exception as e:
            logging.error(f"Error getting detailed GPU info: {e}")

        return gpus

    def test_connection(self) -> bool:
        """Test connection to web-helper.

        Returns:
            True if connection successful
        """
        try:
            response = requests.get(
                f"{self.web_helper_url}/api/devices",
                headers=self.headers,
                timeout=5,
            )
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Connection test failed: {e}")
            return False

    def send_single_heartbeat(self) -> bool:
        """Send a single heartbeat for testing.

        Returns:
            True if heartbeat sent successfully
        """
        try:
            self._send_heartbeat()
            return True
        except Exception as e:
            logging.error(f"Failed to send test heartbeat: {e}")
            return False
