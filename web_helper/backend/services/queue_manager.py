"""Advanced queue management service"""

import asyncio
import json
import logging
import os
import queue
import random
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import psutil
import yaml
import xxhash

from ..models import Device
from ..models.database import SessionLocal
from ..models.queue import (
    JobPriority,
    JobStatus,
    JobSubmission,
    QueueConfiguration,
    QueueJob,
)
from ..models.queue_experiment import QueueExperiment
from .event_manager import event_manager

logger = logging.getLogger(__name__)


def generate_experiment_uid() -> str:
    """Generate unique experiment UID in format {YYYYMMDD}_{hash4}"""
    date_str = datetime.now().strftime("%Y%m%d")
    random_str = str(random.randint(0, 999999))
    hash_hex = xxhash.xxh3_64(random_str.encode()).hexdigest()
    return f"{date_str}_{hash_hex[:4]}"


def override_config_device(config_path: str, device_id: str) -> None:
    """Override device field in config YAML.

    Args:
        config_path: Path to config YAML file
        device_id: Device ID (e.g., "gnode-3:0" or "gnode-3")
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}")
            return

        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        # Extract GPU ID from device_id
        if ":" in device_id:
            # Virtual device format: "gnode-3:0" → GPU 0
            gpu_id = int(device_id.split(":")[-1])
            config['device'] = gpu_id
            logger.info(f"Overriding config device to GPU {gpu_id} for {config_path}")
        else:
            # Single device format: keep as is or use all GPUs
            # Don't override if not specific GPU assignment
            logger.debug(f"No GPU override needed for single device: {device_id}")
            return

        # Save config
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    except Exception as e:
        logger.error(f"Failed to override config device: {e}", exc_info=True)


class QueueManager:
    """Advanced queue management system"""

    def __init__(self, config: Optional[QueueConfiguration] = None):
        self.config = config or QueueConfiguration()
        self._jobs: Dict[str, QueueJob] = {}
        self._job_queue = queue.PriorityQueue()
        self._running_jobs: Set[str] = set()
        self._device_assignments: Dict[
            str, List[str]
        ] = {}  # device_id -> [experiment_uids]
        self._lock = threading.RLock()

        # Background worker thread
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._started = False

        # Load persistent state
        self._load_state()

    def start(self):
        """Start the queue manager"""
        if self._started:
            return

        self._started = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Queue manager started")

    def stop(self):
        """Stop the queue manager"""
        if not self._started:
            return

        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        self._save_state()
        self._started = False
        logger.info("Queue manager stopped")

    def submit_job(self, submission: JobSubmission) -> QueueJob:
        """Submit a new job to the queue"""
        # job_id removed - using experiment_uid

        # Use provided experiment_uid or generate new one
        experiment_uid = submission.experiment_uid or generate_experiment_uid()

        # Note: run_uid removed - cvlabkit handles run_name from config YAML

        job = QueueJob(
            # job_id field removed
            experiment_uid=experiment_uid,
            name=submission.name,
            project=submission.project,
            config_path=submission.config_path,
            priority=submission.priority,
            requirements=submission.requirements,
            tags=submission.tags,
            environment_vars=submission.environment_vars,
            metadata=submission.metadata,
            command=self._build_command(submission.config_path),
            working_directory=str(Path.cwd()),
        )

        with self._lock:
            self._jobs[experiment_uid] = job

            # Add to priority queue (lower number = higher priority)
            priority_value = self._get_priority_value(job.priority)
            self._job_queue.put(
                (priority_value, job.created_at.timestamp(), experiment_uid)
            )

            job.status = JobStatus.QUEUED
            job.queued_at = datetime.now()

        logger.info(f"Experiment {experiment_uid} ({job.name}) submitted to queue")
        self._save_state()

        # Create experiment in DB for Queue-Results consistency
        self._create_experiment_in_db(job)

        # Broadcast job submission
        self._broadcast_job_update_sync(job, "job_submitted")

        return job

    def get_job(self, experiment_uid: str) -> Optional[QueueJob]:
        """Get job by ID"""
        return self._jobs.get(experiment_uid)

    def get_next_job_for_device(self, host_id: str) -> Optional[QueueJob]:
        """Get next queued job for a remote device.

        This is used by remote workers to poll for jobs to execute.
        Returns the highest priority queued job and marks it as ASSIGNED.
        Worker must call confirm_job_started() with PID to mark as RUNNING.
        """
        with self._lock:
            # Find queued jobs sorted by priority
            queued_jobs = [
                job for job in self._jobs.values()
                if job.status == JobStatus.QUEUED
            ]

            if not queued_jobs:
                return None

            # Sort by priority and queue time
            queued_jobs.sort(
                key=lambda j: (self._get_priority_value(j.priority), j.queued_at.timestamp())
            )

            # Get the first job
            job = queued_jobs[0]

            # Mark as ASSIGNED (not RUNNING yet - wait for PID confirmation)
            job.status = JobStatus.ASSIGNED
            job.assigned_device = host_id

            # Track assignment time for timeout detection
            if not job.metadata:
                job.metadata = {}
            job.metadata["assigned_at"] = datetime.now().isoformat()

            # Track assignment
            self._device_assignments[host_id] = job.experiment_uid

            # Update DB for Queue-Results consistency
            # Use started_at for assignment time (will be overwritten when confirmed)
            self._update_experiment_status_in_db(
                job.experiment_uid, "assigned",
                assigned_device=host_id,
                started_at=datetime.now(),  # For timeout detection
            )

            # Remove from queue
            new_queue = []
            while not self._job_queue.empty():
                try:
                    item = self._job_queue.get_nowait()
                    if item[2] != job.experiment_uid:
                        new_queue.append(item)
                except:
                    break
            for item in new_queue:
                self._job_queue.put(item)

            self._save_state()
            self._broadcast_job_update_sync(job)

            logger.info(f"Job {job.experiment_uid} assigned to device {host_id}")
            return job

    def confirm_job_started(self, experiment_uid: str, pid: int) -> bool:
        """Confirm job has started with a PID.

        Called by worker after successfully spawning the process.
        Only then is the job marked as RUNNING.
        """
        with self._lock:
            job = self._jobs.get(experiment_uid)
            if not job:
                logger.warning(f"Job {experiment_uid} not found for start confirmation")
                return False

            if job.status != JobStatus.ASSIGNED:
                logger.warning(f"Job {experiment_uid} not in ASSIGNED state: {job.status}")
                return False

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

            # Store PID in metadata for monitoring
            if not job.metadata:
                job.metadata = {}
            job.metadata["pid"] = pid

            # Track as running
            self._running_jobs[experiment_uid] = job

            # Update DB
            self._update_experiment_status_in_db(
                experiment_uid, "running",
                started_at=job.started_at,
            )

            self._save_state()
            self._broadcast_job_update_sync(job)

            logger.info(f"Job {experiment_uid} confirmed running with PID {pid}")
            return True

    def complete_remote_job(self, experiment_uid: str, success: bool, error_message: Optional[str] = None):
        """Mark a remote job as completed.

        Called by remote workers when job execution finishes.
        """
        with self._lock:
            job = self._jobs.get(experiment_uid)
            if not job:
                logger.warning(f"Job {experiment_uid} not found for completion")
                return

            # Set completion time
            job.completed_at = datetime.now()

            if success:
                job.status = JobStatus.COMPLETED
            else:
                job.status = JobStatus.FAILED
                if error_message:
                    if not job.metadata:
                        job.metadata = {}
                    job.metadata["error"] = error_message

            # Free the device
            if job.assigned_device and job.assigned_device in self._device_assignments:
                del self._device_assignments[job.assigned_device]

            # Remove from running jobs
            if experiment_uid in self._running_jobs:
                del self._running_jobs[experiment_uid]

            self._save_state()
            self._broadcast_job_update_sync(job)

            logger.info(f"Remote job {experiment_uid} completed: {'success' if success else 'failed'}")

        # Update database for Queue-Results consistency (outside lock)
        status_str = "completed" if success else "failed"
        self._update_experiment_status_in_db(
            experiment_uid, status_str,
            completed_at=job.completed_at,
            error_message=error_message if not success else None
        )

    def list_jobs(
        self, status: Optional[JobStatus] = None, project: Optional[str] = None
    ) -> List[QueueJob]:
        """List jobs with optional filtering"""
        jobs = list(self._jobs.values())

        if status:
            jobs = [job for job in jobs if job.status == status]

        if project:
            jobs = [job for job in jobs if job.project == project]

        # Check if config.yaml exists for each job and add to metadata
        for job in jobs:
            config_path = (
                Path("web_helper/queue_logs") / job.experiment_uid / "config.yaml"
            )
            if not job.metadata:
                job.metadata = {}
            job.metadata["has_config"] = config_path.exists()

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def cancel_job(self, experiment_uid: str) -> bool:
        """Cancel a job"""
        job = self._jobs.get(experiment_uid)
        if not job:
            return False

        with self._lock:
            if job.status == JobStatus.RUNNING:
                # Try to terminate the process
                self._terminate_job(experiment_uid)

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            self._free_device(experiment_uid)

        # Update database for Queue-Results consistency
        self._update_experiment_status_in_db(
            experiment_uid, "cancelled", job.completed_at
        )

        logger.info(f"Experiment {experiment_uid} cancelled")
        self._save_state()

        # Broadcast job cancellation
        self._broadcast_job_update_sync(job, "job_cancelled")

        return True

    def _worker_loop(self):
        """Main worker loop for processing jobs"""
        while not self._stop_event.is_set():
            try:
                self._process_queue()
                self._monitor_running_jobs()
                self._stop_event.wait(5.0)
            except Exception as e:
                logger.error(f"Error in queue worker loop: {e}", exc_info=True)
                self._stop_event.wait(10.0)

    def _process_queue(self):
        """Process queued jobs and start them if resources are available"""
        if len(self._running_jobs) >= self.config.max_concurrent_jobs:
            return

        available_devices = self._get_available_devices()
        if not available_devices:
            return

        while (
            not self._job_queue.empty()
            and len(self._running_jobs) < self.config.max_concurrent_jobs
        ):
            try:
                _, _, experiment_uid = self._job_queue.get_nowait()
                job = self._jobs.get(experiment_uid)
                if not job or job.status != JobStatus.QUEUED:
                    continue

                device = self._find_suitable_device(job, available_devices)
                if not device:
                    self._job_queue.put(
                        (
                            self._get_priority_value(job.priority),
                            job.queued_at.timestamp(),
                            experiment_uid,
                        )
                    )
                    break

                if self._start_job(job, device):
                    available_devices.remove(device)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing queue: {e}", exc_info=True)

    def _start_job(self, job: QueueJob, device: str) -> bool:
        """Start a job on the specified device as a detached process."""
        if os.name == "nt":
            logger.error("Detached process execution is not supported on Windows.")
            return False

        try:
            env = os.environ.copy()
            if device.startswith("cuda:"):
                env["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
            elif device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            if job.environment_vars:
                env.update(job.environment_vars)

            # Separate Experiment and Run paths (CLAUDE.md principle)
            run_name = (
                job.metadata.get("run_name", job.experiment_uid)
                if job.metadata
                else job.experiment_uid
            )

            # 1. Experiment logs (terminal output for process management)
            queue_log_dir = Path("web_helper/queue_logs") / job.experiment_uid
            queue_log_dir.mkdir(parents=True, exist_ok=True)

            stdout_file = queue_log_dir / "terminal_log.log"
            stderr_file = queue_log_dir / "terminal_err.log"

            # 2. Run logs (cvlabkit output for result analysis)
            # cvlabkit writes directly to logs/ via default behavior
            run_dir = Path("logs") / job.project
            run_dir.mkdir(parents=True, exist_ok=True)

            # 3. Override config device before execution
            override_config_device(job.config_path, device)

            command = (
                f"nohup uv run main.py --config {job.config_path} --fast "
                f"> {stdout_file} 2> {stderr_file} & echo $!"
            )

            process = subprocess.Popen(
                command,
                cwd=job.working_directory,
                env=env,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )

            stdout, stderr = process.communicate()
            pid_str = stdout.strip().decode()
            if not pid_str.isdigit():
                raise RuntimeError(
                    f"Could not get PID for job {job.experiment_uid}. Stderr: {stderr.decode()}"
                )

            pid = int(pid_str)

            if not job.metadata:
                job.metadata = {}
            job.metadata["stdout_log"] = str(stdout_file)
            job.metadata["stderr_log"] = str(stderr_file)
            job.metadata["pid"] = pid

            with self._lock:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                job.assigned_device = device
                self._running_jobs.add(job.experiment_uid)
                if device not in self._device_assignments:
                    self._device_assignments[device] = []
                self._device_assignments[device].append(job.experiment_uid)

            logger.info(
                f"Started job {job.experiment_uid} ({job.name}) on device {device} with PID {pid}"
            )

            # Update QueueExperiment DB with PID and running status
            db = SessionLocal()
            try:
                db_experiment = (
                    db.query(QueueExperiment)
                    .filter(QueueExperiment.experiment_uid == job.experiment_uid)
                    .first()
                )
                if db_experiment:
                    db_experiment.pid = pid
                    db_experiment.status = "running"
                    db_experiment.started_at = job.started_at
                    db_experiment.assigned_device = device
                    db.commit()
                    logger.debug(f"Updated DB with PID {pid} for {job.experiment_uid}")
            except Exception as e:
                logger.error(
                    f"Failed to update DB with PID for {job.experiment_uid}: {e}"
                )
                db.rollback()
            finally:
                db.close()

            self._save_state()
            self._broadcast_job_update_sync(job, "job_started")

            threading.Thread(
                target=self._monitor_job, args=(job.experiment_uid,), daemon=True
            ).start()
            return True

        except Exception as e:
            logger.error(
                f"Failed to start job {job.experiment_uid}: {e}", exc_info=True
            )
            with self._lock:
                job.status = JobStatus.FAILED
                job.error_message = f"Failed to start: {str(e)}"
                job.completed_at = datetime.now()

            # Update database for Queue-Results consistency
            self._update_experiment_status_in_db(
                job.experiment_uid, "failed", completed_at=job.completed_at
            )
            self._broadcast_job_update_sync(job, "job_failed")
            self._save_state()
            return False

    def _monitor_job(self, experiment_uid: str):
        """Monitor a specific job's execution using its PID."""
        job = self._jobs.get(experiment_uid)
        if not job:
            return

        pid = job.metadata.get("pid")
        if not pid:
            logger.error(f"No PID found for monitoring job {experiment_uid}")
            return

        try:
            time.sleep(2)

            while job.status == JobStatus.RUNNING:
                if not psutil.pid_exists(pid):
                    logger.info(
                        f"Process with PID {pid} for job {experiment_uid} no longer exists."
                    )
                    with self._lock:
                        if job.status == JobStatus.RUNNING:
                            job.status = JobStatus.COMPLETED
                            job.progress = 1.0
                            job.completed_at = datetime.now()
                            if not job.metadata:
                                job.metadata = {}
                            job.metadata["completion_status"] = (
                                "assumed_completed_by_monitor"
                            )
                            self._free_device(experiment_uid)

                    # Update database for Queue-Results consistency
                    self._update_experiment_status_in_db(
                        experiment_uid, "completed", job.completed_at
                    )

                    self._broadcast_job_update_sync(job, "job_completed")
                    self._save_state()
                    break

                self._update_job_progress(job)
                time.sleep(10)

        except Exception as e:
            logger.error(f"Error monitoring job {experiment_uid}: {e}", exc_info=True)
            with self._lock:
                job.status = JobStatus.FAILED
                job.error_message = f"Monitoring error: {str(e)}"
                job.completed_at = datetime.now()
                self._free_device(experiment_uid)

            # Update database for Queue-Results consistency
            self._update_experiment_status_in_db(
                experiment_uid, "failed", job.completed_at
            )

            self._broadcast_job_update_sync(job, "job_failed")

    def _monitor_running_jobs(self):
        pass

    def _terminate_job(self, experiment_uid: str):
        """Terminate a running job using its PID."""
        job = self._jobs.get(experiment_uid)
        if not job or not job.metadata or "pid" not in job.metadata:
            logger.warning(f"Cannot terminate job {experiment_uid}: no PID found")
            return

        pid = job.metadata["pid"]
        try:
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                for child in process.children(recursive=True):
                    child.terminate()
                process.terminate()

                gone, alive = psutil.wait_procs([process], timeout=3)
                if alive:
                    for p in alive:
                        p.kill()
                logger.info(f"Terminated job {experiment_uid} (PID {pid})")
            else:
                logger.info(
                    f"Process for job {experiment_uid} (PID {pid}) already terminated."
                )
        except psutil.NoSuchProcess:
            logger.info(
                f"Process for job {experiment_uid} (PID {pid}) already terminated."
            )
        except Exception as e:
            logger.error(f"Error terminating job {experiment_uid}: {e}", exc_info=True)

    def _free_device(self, experiment_uid: str):
        """Free the device assigned to a job"""
        with self._lock:
            self._running_jobs.discard(experiment_uid)
            job = self._jobs.get(experiment_uid)
            if job and job.assigned_device:
                device_jobs = self._device_assignments.get(job.assigned_device, [])
                if experiment_uid in device_jobs:
                    device_jobs.remove(experiment_uid)

    def _create_experiment_in_db(self, job: QueueJob):
        """Create experiment entry in database for Queue-Results consistency."""
        db = SessionLocal()
        try:
            existing = (
                db.query(QueueExperiment)
                .filter(QueueExperiment.experiment_uid == job.experiment_uid)
                .first()
            )
            if existing:
                existing.status = job.status.value
                existing.name = job.name
                existing.project = job.project
                existing.config_path = job.config_path
                db.commit()
            else:
                db_experiment = QueueExperiment(
                    experiment_uid=job.experiment_uid,
                    name=job.name,
                    project=job.project,
                    status=job.status.value,
                    config_path=job.config_path,
                    created_at=job.created_at,
                )
                db.add(db_experiment)
                db.commit()
            logger.debug(f"Created/updated DB entry for {job.experiment_uid}")
        except Exception as e:
            logger.error(f"Failed to create DB entry for {job.experiment_uid}: {e}")
            db.rollback()
        finally:
            db.close()

    def _update_experiment_status_in_db(
        self,
        experiment_uid: str,
        status: str,
        completed_at: Optional[datetime] = None,
        started_at: Optional[datetime] = None,
        assigned_device: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Update experiment status in database for Queue-Results consistency."""
        db = SessionLocal()
        try:
            db_experiment = (
                db.query(QueueExperiment)
                .filter(QueueExperiment.experiment_uid == experiment_uid)
                .first()
            )

            if db_experiment:
                db_experiment.status = status
                if completed_at:
                    db_experiment.completed_at = completed_at
                if started_at:
                    db_experiment.started_at = started_at
                if assigned_device:
                    db_experiment.assigned_device = assigned_device
                if error_message:
                    db_experiment.error_message = error_message
                db.commit()
                logger.debug(f"Updated DB status for {experiment_uid}: {status}")
            else:
                logger.warning(
                    f"Experiment {experiment_uid} not found in DB for status update"
                )
        except Exception as e:
            logger.error(f"Failed to update DB status for {experiment_uid}: {e}")
            db.rollback()
        finally:
            db.close()

    def _save_state(self):
        """Save queue state to disk"""
        try:
            state_file = Path("web_helper/state/queue_state.json")
            state_file.parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                serializable_jobs = {}
                for experiment_uid, job in self._jobs.items():
                    job_dict = job.model_dump(exclude_none=True)
                    for field in [
                        "created_at",
                        "queued_at",
                        "started_at",
                        "completed_at",
                    ]:
                        if job_dict.get(field):
                            job_dict[field] = job_dict[field].isoformat()
                    serializable_jobs[experiment_uid] = job_dict

                state = {
                    "jobs": serializable_jobs,
                }

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save queue state: {e}", exc_info=True)

    def _load_state(self):
        """Load queue state and recover running jobs."""
        state_file = Path("web_helper/state/queue_state.json")
        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                state = json.load(f)

            for experiment_uid, job_dict in state.get("jobs", {}).items():
                for field in ["created_at", "queued_at", "started_at", "completed_at"]:
                    if job_dict.get(field):
                        job_dict[field] = datetime.fromisoformat(job_dict[field])

                job = QueueJob(**job_dict)
                self._jobs[experiment_uid] = job

                if job.status == JobStatus.RUNNING:
                    pid = job.metadata.get("pid")
                    if pid and psutil.pid_exists(pid):
                        logger.info(
                            f"Recovering running job {experiment_uid} with PID {pid}"
                        )
                        self._running_jobs.add(experiment_uid)
                        if job.assigned_device:
                            if job.assigned_device not in self._device_assignments:
                                self._device_assignments[job.assigned_device] = []
                            self._device_assignments[job.assigned_device].append(
                                experiment_uid
                            )

                        threading.Thread(
                            target=self._monitor_job,
                            args=(job.experiment_uid,),
                            daemon=True,
                        ).start()
                    else:
                        logger.warning(
                            f"Job {experiment_uid} was in running state but PID {pid} not found. Marking as failed."
                        )
                        job.status = JobStatus.FAILED
                        job.error_message = (
                            "Process terminated while web_helper was offline."
                        )
                        job.completed_at = datetime.now()

                        # Update database for Queue-Results consistency on restart
                        self._update_experiment_status_in_db(
                            experiment_uid, "failed", job.completed_at
                        )
                        # Note: broadcast may not reach clients during startup, but included for consistency
                        self._broadcast_job_update_sync(job, "job_failed")

                elif job.status == JobStatus.QUEUED:
                    priority_value = self._get_priority_value(job.priority)
                    self._job_queue.put(
                        (priority_value, job.created_at.timestamp(), experiment_uid)
                    )

            logger.info(f"Loaded and processed {len(self._jobs)} jobs from state file")

        except Exception as e:
            logger.error(f"Failed to load queue state: {e}", exc_info=True)

    def _get_available_devices(self) -> List[str]:
        """Get list of available and healthy devices.

        For Multi-GPU devices, each GPU is treated as a virtual device.
        Returns device IDs in format:
        - Single GPU: "host_id" (e.g., "gnode-1")
        - Multi-GPU: "host_id:gpu_id" (e.g., "gnode-3:0", "gnode-3:1")

        Only returns devices that:
        1. Have sent a heartbeat within the last 3 seconds (healthy)
        2. Are not currently assigned to a job

        Returns:
            List of device IDs (including virtual devices) available for job assignment
        """
        db = SessionLocal()
        try:
            devices = db.query(Device).all()

            available = []
            now = datetime.utcnow()

            for device in devices:
                # Check if device has recent heartbeat (within 3 seconds)
                if not device.last_heartbeat:
                    continue

                time_diff = (now - device.last_heartbeat).total_seconds()
                if time_diff > 3:
                    # Device is stale or disconnected
                    continue

                # Device is healthy - expand to virtual devices if Multi-GPU
                if device.gpu_count and device.gpu_count > 1:
                    # Multi-GPU: treat each GPU as a virtual device
                    for gpu_id in range(device.gpu_count):
                        virtual_device_id = f"{device.host_id}:{gpu_id}"

                        # Check if this virtual device is available
                        assigned_jobs = self._device_assignments.get(virtual_device_id, [])
                        active_jobs = [
                            experiment_uid
                            for experiment_uid in assigned_jobs
                            if experiment_uid in self._jobs
                            and self._jobs[experiment_uid].status == JobStatus.RUNNING
                        ]

                        if len(active_jobs) < 1:  # Only one job per GPU
                            available.append(virtual_device_id)
                else:
                    # Single GPU or CPU-only: use host_id directly
                    device_id = device.host_id
                    assigned_jobs = self._device_assignments.get(device_id, [])

                    # Filter out completed/cancelled jobs
                    active_jobs = [
                        experiment_uid
                        for experiment_uid in assigned_jobs
                        if experiment_uid in self._jobs
                        and self._jobs[experiment_uid].status == JobStatus.RUNNING
                    ]

                    if len(active_jobs) < 1:  # Only one job per device
                        available.append(device_id)

            if not available:
                logger.debug("No healthy devices available for job dispatch")

            return available

        except Exception as e:
            logger.error(f"Error getting available devices: {e}", exc_info=True)
            return []
        finally:
            db.close()

    def _find_suitable_device(
        self, job: QueueJob, available_devices: List[str]
    ) -> Optional[str]:
        return available_devices[0] if available_devices else None

    def _get_priority_value(self, priority: JobPriority) -> int:
        return {
            JobPriority.URGENT: 0,
            JobPriority.HIGH: 1,
            JobPriority.NORMAL: 2,
            JobPriority.LOW: 3,
        }.get(priority, 2)

    def _build_command(self, config_path: str) -> str:
        return f"uv run main.py --config {config_path} --fast"

    def _broadcast_job_update_sync(
        self, job: QueueJob, event_type: str = "status_change"
    ):
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we are, schedule the coroutine to run in the background
            loop.create_task(self._broadcast_job_update(job, event_type))
        except RuntimeError:
            # No running event loop, so we can use asyncio.run()
            asyncio.run(self._broadcast_job_update(job, event_type))

    async def _broadcast_job_update(
        self, job: QueueJob, event_type: str = "status_change"
    ):
        try:
            job_dict = job.model_dump(exclude_none=True)
            # Convert datetime objects to ISO format strings for JSON serialization
            for field in ["created_at", "queued_at", "started_at", "completed_at"]:
                if job_dict.get(field):
                    job_dict[field] = job_dict[field].isoformat()
            await event_manager.send_queue_update(job_dict)
        except Exception as e:
            logger.error(
                f"Failed to broadcast job update for {job.experiment_uid}: {e}"
            )

    def _update_job_progress(self, job: QueueJob):
        pass  # Simplified


# Global queue manager instance
_queue_manager: Optional[QueueManager] = None


def get_queue_manager() -> QueueManager:
    """Get the global queue manager instance"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
        _queue_manager.start()
    return _queue_manager


def shutdown_queue_manager():
    """Shutdown the global queue manager"""
    global _queue_manager
    if _queue_manager:
        _queue_manager.stop()
        _queue_manager = None
