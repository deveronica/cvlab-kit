"""Advanced queue management API endpoints."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..models import QueueExperiment, get_db
from ..models.queue import (
    JobPriority,
    JobStatus,
    JobSubmission,
    QueueJob,
)
from ..services.queue_manager import get_queue_manager
from ..utils import error_response, success_response

router = APIRouter(prefix="/queue")


@router.get("/")
@router.get("")
async def queue_root():
    """Queue API root endpoint."""
    return success_response(
        {
            "message": "Queue API",
            "endpoints": {
                "list": "/queue/list",
                "stats": "/queue/stats",
                "submit": "/queue/submit",
                "job": "/queue/job/{job_id}",
            },
        }
    )


class JobResponse(BaseModel):
    """Job response model"""

    job: QueueJob


class JobListResponse(BaseModel):
    """Job list response model"""

    jobs: List[QueueJob]
    total: int


@router.get("/list", response_model=JobListResponse)
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    project: Optional[str] = Query(None, description="Filter by project"),
    limit: int = Query(
        50, ge=1, le=1000, description="Maximum number of jobs to return"
    ),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
):
    """List jobs with optional filtering and pagination."""
    try:
        queue_manager = get_queue_manager()
        jobs = queue_manager.list_jobs(status=status, project=project)

        # Apply pagination
        total = len(jobs)
        paginated_jobs = jobs[offset : offset + limit]

        return JobListResponse(jobs=paginated_jobs, total=total)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to list jobs: {str(e)}",
            ),
        )


@router.get("/stats")
async def get_queue_stats():
    """Get queue statistics."""
    try:
        queue_manager = get_queue_manager()
        stats = queue_manager.get_queue_stats()
        return success_response(stats.model_dump())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get queue stats: {str(e)}",
            ),
        )


@router.get("/config/{experiment_uid}")
async def get_queue_config(experiment_uid: str):
    """Get config file for a queued experiment.

    Used by remote workers to download config before executing.
    """
    config_path = Path("web_helper/queue_logs") / experiment_uid / "config.yaml"

    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail=error_response(
                title="Not Found",
                status=404,
                detail=f"Config not found for experiment {experiment_uid}",
            ),
        )

    try:
        content = config_path.read_text(encoding="utf-8")
        return PlainTextResponse(content, media_type="text/yaml")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to read config: {str(e)}",
            ),
        )


@router.get("/next_job")
async def get_next_job(host_id: str = Query(..., description="Device host ID")):
    """Get the next queued job for a remote device.

    This endpoint is used by remote workers to poll for jobs.
    Returns the highest priority queued job and marks it as assigned.
    """
    try:
        queue_manager = get_queue_manager()
        job = queue_manager.get_next_job_for_device(host_id)

        if not job:
            return success_response({"job": None})

        return success_response({
            "job": {
                "experiment_uid": job.experiment_uid,
                "config_path": job.config_path,
                "project": job.project,
                "priority": job.priority.value,
                "metadata": job.metadata,
            }
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get next job: {str(e)}",
            ),
        )


class JobStartedRequest(BaseModel):
    """Request model for job start confirmation"""
    experiment_uid: str = Field(..., description="Experiment UID")
    pid: int = Field(..., description="Process ID of running job")


@router.post("/confirm_started")
async def confirm_job_started(request: JobStartedRequest):
    """Confirm job has started with a PID.

    Called by worker after successfully spawning the process.
    Only then is the job marked as RUNNING.
    """
    try:
        queue_manager = get_queue_manager()
        success = queue_manager.confirm_job_started(request.experiment_uid, request.pid)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=error_response(
                    title="Bad Request",
                    status=400,
                    detail=f"Failed to confirm job start for {request.experiment_uid}",
                ),
            )

        return success_response({"message": "Job start confirmed", "experiment_uid": request.experiment_uid})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to confirm job start: {str(e)}",
            ),
        )


class JobCompletionRequest(BaseModel):
    """Request model for job completion"""
    experiment_uid: str = Field(..., description="Experiment UID")
    success: bool = Field(..., description="Whether job succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")


@router.post("/complete_job")
async def complete_job(request: JobCompletionRequest):
    """Mark a remote job as completed.

    Called by remote workers when job execution finishes.
    """
    try:
        queue_manager = get_queue_manager()
        queue_manager.complete_remote_job(
            request.experiment_uid,
            request.success,
            request.error_message
        )
        return success_response({"message": "Job completion recorded"})

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to complete job: {str(e)}",
            ),
        )


@router.post("/submit", response_model=JobResponse)
async def submit_job(submission: JobSubmission):
    """Submit a new job to the queue."""
    try:
        # Validate config file exists
        config_path = Path(submission.config_path)
        if not config_path.exists():
            raise HTTPException(
                status_code=400,
                detail=error_response(
                    title="Bad Request",
                    status=400,
                    detail=f"Configuration file not found: {submission.config_path}",
                ),
            )

        queue_manager = get_queue_manager()
        job = queue_manager.submit_job(submission)

        return JobResponse(job=job)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to submit job: {str(e)}",
            ),
        )


class DirectJobSubmission(BaseModel):
    """Direct job submission with inline YAML config"""

    config: str = Field(..., description="YAML configuration content")
    device: str = Field(default="any", description="Target device ID or 'any'")
    priority: str = Field(default="normal", description="Job priority")
    name: Optional[str] = Field(
        None, description="Job name (auto-generated if not provided)"
    )
    project: Optional[str] = Field(
        None, description="Project name (extracted from config if not provided)"
    )


@router.post("/add", response_model=JobResponse)
async def add_job(submission: DirectJobSubmission):
    """Add a new job to the queue with inline YAML config."""
    import uuid
    from datetime import datetime

    import yaml

    try:
        # Parse YAML using cvlabkit's Config class to support Python tags
        try:
            # Use cvlabkit's YAML loader which supports python/tuple and other tags
            config_data = yaml.load(submission.config, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            raise HTTPException(
                status_code=400,
                detail=error_response(
                    title="Bad Request",
                    status=400,
                    detail=f"Invalid YAML configuration: {str(e)}",
                ),
            )

        # Extract project name from config or use provided name
        project_name = submission.project or config_data.get(
            "project", "default_project"
        )
        job_name = (
            submission.name
            or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        # Generate experiment_uid in same format as queue_manager
        from ..services.queue_manager import generate_experiment_uid

        experiment_uid = generate_experiment_uid()

        # Create experiment directory in queue_logs
        experiment_dir = Path("web_helper/queue_logs") / experiment_uid
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Standardized config filename
        config_path = experiment_dir / "config.yaml"

        # Write YAML content to file
        with open(config_path, "w") as f:
            f.write(submission.config)

        # Convert priority string to enum
        try:
            priority_enum = JobPriority(submission.priority.lower())
        except ValueError:
            priority_enum = JobPriority.NORMAL

        # Create JobSubmission for the existing submit logic
        job_submission = JobSubmission(
            name=job_name,
            project=project_name,
            config_path=str(config_path),
            priority=priority_enum,
            experiment_uid=experiment_uid,  # Use the experiment_uid we generated
            metadata={
                "device_preference": submission.device,
                "submitted_from": "execute_tab",
                "original_config_inline": True,
            },
        )

        # Submit the job using existing logic
        queue_manager = get_queue_manager()
        job = queue_manager.submit_job(job_submission)

        return JobResponse(job=job)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to add job: {str(e)}",
            ),
        )


@router.get("/experiment/{experiment_uid}", response_model=JobResponse)
async def get_job(experiment_uid: str):
    """Get experiment details by UID."""
    try:
        queue_manager = get_queue_manager()
        job = queue_manager.get_job(experiment_uid)

        if not job:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Experiment not found: {experiment_uid}",
                ),
            )

        return JobResponse(job=job)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get experiment: {str(e)}",
            ),
        )


@router.post("/experiment/{experiment_uid}/cancel")
async def cancel_job(experiment_uid: str):
    """Cancel an experiment."""
    try:
        queue_manager = get_queue_manager()
        success = queue_manager.cancel_job(experiment_uid)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Experiment not found or cannot be cancelled: {experiment_uid}",
                ),
            )

        return success_response(
            {"message": f"Experiment {experiment_uid} cancelled successfully"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to cancel job: {str(e)}",
            ),
        )


@router.post("/job/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a running job."""
    try:
        queue_manager = get_queue_manager()
        success = queue_manager.pause_job(job_id)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=error_response(
                    title="Bad Request",
                    status=400,
                    detail=f"Job not found or cannot be paused: {job_id}",
                ),
            )

        return success_response({"message": f"Job {job_id} paused successfully"})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to pause job: {str(e)}",
            ),
        )


@router.post("/job/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused job."""
    try:
        queue_manager = get_queue_manager()
        success = queue_manager.resume_job(job_id)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=error_response(
                    title="Bad Request",
                    status=400,
                    detail=f"Job not found or cannot be resumed: {job_id}",
                ),
            )

        return success_response({"message": f"Job {job_id} resumed successfully"})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to resume job: {str(e)}",
            ),
        )


@router.post("/job/{job_id}/priority")
async def set_job_priority(job_id: str, priority: JobPriority):
    """Set job priority."""
    try:
        queue_manager = get_queue_manager()
        success = queue_manager.set_job_priority(job_id, priority)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found", status=404, detail=f"Job not found: {job_id}"
                ),
            )

        return success_response(
            {"message": f"Job {job_id} priority set to {priority.value}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to set job priority: {str(e)}",
            ),
        )


@router.post("/cleanup")
async def cleanup_old_jobs(
    hours: float = Query(168.0, ge=1, description="Age in hours"),
):
    """Clean up old completed jobs."""
    try:
        queue_manager = get_queue_manager()
        cleaned_count = queue_manager.cleanup_old_jobs(hours)

        return success_response(
            {
                "message": f"Cleaned up {cleaned_count} old jobs",
                "cleaned_count": cleaned_count,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to cleanup jobs: {str(e)}",
            ),
        )


@router.post("/reindex")
async def reindex():
    """Reindex runs from filesystem."""
    try:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return success_response(
                {"message": "No logs directory found", "indexed": 0}
            )

        indexed_count = 0

        # Simple file scan for demonstration
        for yaml_file in logs_dir.glob("**/*.yaml"):
            if "config" in yaml_file.name:
                continue
            indexed_count += 1

        return success_response(
            {"message": f"Reindexed {indexed_count} files", "indexed": indexed_count}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to reindex: {str(e)}",
            ),
        )


@router.get("/job/{job_id}/log")
async def get_job_log(job_id: str):
    """Get the complete log content for a specific job."""
    try:
        queue_manager = get_queue_manager()
        job = queue_manager.get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found", status=404, detail=f"Job not found: {job_id}"
                ),
            )

        # Get log file path from experiment_uid
        log_path = (
            Path("web_helper/queue_logs") / job.experiment_uid / "terminal_log.log"
        )

        if not log_path.exists():
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Log file not found for job: {job_id}",
                ),
            )

        content = log_path.read_text(encoding="utf-8")
        return PlainTextResponse(content)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get job log: {str(e)}",
            ),
        )


@router.get("/job/{job_id}/config")
async def get_job_config(job_id: str):
    """Get the config YAML content for a specific job."""
    try:
        queue_manager = get_queue_manager()
        job = queue_manager.get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found", status=404, detail=f"Job not found: {job_id}"
                ),
            )

        # Get config file path from experiment_uid
        config_path = Path("web_helper/queue_logs") / job.experiment_uid / "config.yaml"

        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Config file not found for job: {job_id}",
                ),
            )

        content = config_path.read_text(encoding="utf-8")
        return success_response({"content": content})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get job config: {str(e)}",
            ),
        )


@router.get("/job/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Get combined stdout and stderr logs for a specific job."""
    try:
        queue_manager = get_queue_manager()
        job = queue_manager.get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found", status=404, detail=f"Job not found: {job_id}"
                ),
            )

        # Get log file paths from experiment_uid
        log_dir = Path("web_helper/queue_logs") / job.experiment_uid
        stdout_path = log_dir / "terminal_log.log"
        stderr_path = log_dir / "terminal_err.log"

        combined_logs = []

        # Read stdout
        if stdout_path.exists():
            try:
                stdout_content = stdout_path.read_text(encoding="utf-8")
                if stdout_content.strip():
                    combined_logs.append("=== STDOUT ===")
                    combined_logs.append(stdout_content)
            except Exception as e:
                combined_logs.append(f"=== STDOUT (Error reading) ===\n{str(e)}")

        # Read stderr
        if stderr_path.exists():
            try:
                stderr_content = stderr_path.read_text(encoding="utf-8")
                if stderr_content.strip():
                    combined_logs.append("\n=== STDERR ===")
                    combined_logs.append(stderr_content)
            except Exception as e:
                combined_logs.append(f"\n=== STDERR (Error reading) ===\n{str(e)}")

        if not combined_logs:
            return success_response({"content": "No logs available yet."})

        return success_response({"content": "\n".join(combined_logs)})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get job logs: {str(e)}",
            ),
        )


@router.websocket("/job/{job_id}/ws/logs")
async def websocket_job_log_stream(websocket: WebSocket, job_id: str):
    """WebSocket endpoint to stream logs for a specific job in real-time."""
    await websocket.accept()

    try:
        queue_manager = get_queue_manager()
        job = queue_manager.get_job(job_id)

        if not job:
            await websocket.send_text(f"ERROR: Job not found: {job_id}")
            await websocket.close()
            return

        # Get log file path from experiment_uid
        log_path = (
            Path("web_helper/queue_logs") / job.experiment_uid / "terminal_log.log"
        )

        # Wait for log file to be created if job is pending/queued
        max_wait = 30  # seconds
        wait_count = 0
        while not log_path.exists() and wait_count < max_wait:
            await asyncio.sleep(1)
            wait_count += 1
            # Refresh job status
            job = queue_manager.get_job(job_id)
            if not job or job.status in ["failed", "cancelled"]:
                await websocket.send_text(
                    f"ERROR: Job {job.status or 'disappeared'} before log file created"
                )
                await websocket.close()
                return

        if not log_path.exists():
            await websocket.send_text(
                f"ERROR: Log file not found after waiting: {log_path}"
            )
            await websocket.close()
            return

        await websocket.send_text(f"--- CONNECTED TO LOG STREAM FOR JOB {job_id} ---")

        # Stream log file content
        with open(log_path, encoding="utf-8") as log_file:
            # Read existing content first
            existing_content = log_file.read()
            if existing_content:
                for line in existing_content.splitlines():
                    await websocket.send_text(line)

            # Then stream new lines as they appear
            while True:
                line = log_file.readline()
                if not line:
                    # Check job status
                    job = queue_manager.get_job(job_id)
                    if not job:
                        await websocket.send_text("--- JOB REMOVED FROM QUEUE ---")
                        break
                    if job.status in ["completed", "failed", "cancelled"]:
                        await websocket.send_text(f"--- JOB {job.status.upper()} ---")
                        break
                    await asyncio.sleep(0.1)  # Wait for new lines
                    continue
                await websocket.send_text(line.rstrip())

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from log stream for job {job_id}")
    except Exception as e:
        try:
            await websocket.send_text(
                f"ERROR: An error occurred while streaming logs: {e}"
            )
        except Exception as ex:
            logger.error(f"Failed to send error message to WebSocket: {ex}")
    finally:
        try:
            await websocket.close()
        except Exception as ex:
            logger.debug(f"Error closing WebSocket (already closed?): {ex}")


# ============================================================================
# Queue Experiments API - File-based experiment history
# ============================================================================


@router.get("/experiments")
async def list_queue_experiments(
    db: Session = Depends(get_db),
    status: Optional[str] = Query(None, description="Filter by status"),
    project: Optional[str] = Query(None, description="Filter by project"),
    limit: int = Query(
        1000, ge=1, le=10000, description="Maximum number of experiments to return"
    ),
    offset: int = Query(0, ge=0, description="Number of experiments to skip"),
):
    """List all queue experiments from queue_logs/ directory.

    This endpoint provides access to all experiment execution history,
    indexed from the queue_logs/ directory with hash-based change detection.
    """
    try:
        # Build query
        query = db.query(QueueExperiment)

        # Apply filters
        if status:
            query = query.filter(QueueExperiment.status == status)
        if project:
            query = query.filter(QueueExperiment.project == project)

        # Get total count before pagination
        total = query.count()

        # Apply sorting (newest first) and pagination
        experiments = (
            query.order_by(QueueExperiment.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        # Convert to dict format
        experiment_list = []
        for exp in experiments:
            # Check if config file exists
            config_exists = False
            if exp.config_path:
                config_exists = Path(exp.config_path).exists()

            experiment_list.append(
                {
                    "experiment_uid": exp.experiment_uid,
                    "name": exp.name,
                    "project": exp.project,
                    "status": exp.status,
                    "created_at": exp.created_at.isoformat()
                    if exp.created_at
                    else None,
                    "started_at": exp.started_at.isoformat()
                    if exp.started_at
                    else None,
                    "completed_at": exp.completed_at.isoformat()
                    if exp.completed_at
                    else None,
                    "config_path": exp.config_path,
                    "log_path": exp.log_path,
                    "error_log_path": exp.error_log_path,
                    "assigned_device": exp.assigned_device,
                    "priority": exp.priority,
                    "exit_code": exp.exit_code,
                    "error_message": exp.error_message,
                    "last_indexed": exp.last_indexed.isoformat()
                    if exp.last_indexed
                    else None,
                    "metadata": {
                        "has_config": config_exists,
                        **(exp.meta if exp.meta else {}),
                    },
                }
            )

        return success_response(
            {
                "experiments": experiment_list,
                "total": total,
                "limit": limit,
                "offset": offset,
            },
            {"message": "Queue experiments retrieved successfully"},
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to list queue experiments: {str(e)}",
            ),
        )


@router.post("/experiments/reindex")
async def reindex_queue_experiments(db: Session = Depends(get_db)):
    """Trigger reindexing of all queue experiments."""
    try:
        from ..services.queue_indexer import index_queue_experiments

        stats = index_queue_experiments(db)

        return success_response(
            stats, {"message": "Queue experiments reindexed successfully"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to reindex queue experiments: {str(e)}",
            ),
        )


@router.get("/experiments/{experiment_uid}/config")
async def get_experiment_config(experiment_uid: str):
    """Get config.yaml content for a queue experiment."""
    try:
        config_path = Path("web_helper/queue_logs") / experiment_uid / "config.yaml"

        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Config file not found for experiment: {experiment_uid}",
                ),
            )

        content = config_path.read_text(encoding="utf-8")
        return success_response({"content": content})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get experiment config: {str(e)}",
            ),
        )


@router.get("/experiments/{experiment_uid}/logs")
async def get_experiment_logs(experiment_uid: str):
    """Get combined stdout and stderr logs for a queue experiment."""
    try:
        log_dir = Path("web_helper/queue_logs") / experiment_uid
        stdout_path = log_dir / "terminal_log.log"
        stderr_path = log_dir / "terminal_err.log"

        # Check if files exist
        stdout_exists = stdout_path.exists()
        stderr_exists = stderr_path.exists()

        if not stdout_exists and not stderr_exists:
            return success_response(
                {
                    "content": "Log files not created yet (experiment may not have started)"
                }
            )

        combined_logs = []

        # Read stdout
        if stdout_exists:
            try:
                stdout_content = stdout_path.read_text(encoding="utf-8")
                if stdout_content.strip():
                    combined_logs.append("=== STDOUT ===")
                    combined_logs.append(stdout_content)
            except Exception as e:
                combined_logs.append(f"=== STDOUT (Error reading) ===\n{str(e)}")

        # Read stderr
        if stderr_exists:
            try:
                stderr_content = stderr_path.read_text(encoding="utf-8")
                if stderr_content.strip():
                    combined_logs.append("\n=== STDERR ===")
                    combined_logs.append(stderr_content)
            except Exception as e:
                combined_logs.append(f"\n=== STDERR (Error reading) ===\n{str(e)}")

        if not combined_logs:
            return success_response(
                {"content": "Experiment started but no output generated yet"}
            )

        return success_response({"content": "\n".join(combined_logs)})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get experiment logs: {str(e)}",
            ),
        )


@router.get("/experiments/{experiment_uid}/stderr")
async def get_experiment_stderr(experiment_uid: str):
    """Get stderr logs only for a queue experiment."""
    try:
        log_dir = Path("web_helper/queue_logs") / experiment_uid
        stderr_path = log_dir / "terminal_err.log"

        if not stderr_path.exists():
            return success_response(
                {
                    "content": "Error log file not created yet (experiment may not have started)"
                }
            )

        try:
            stderr_content = stderr_path.read_text(encoding="utf-8")
            if not stderr_content.strip():
                return success_response(
                    {"content": "No errors occurred during execution ✓"}
                )

            return success_response({"content": stderr_content})
        except Exception as e:
            return success_response({"content": f"Error reading stderr log: {str(e)}"})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get experiment stderr: {str(e)}",
            ),
        )


@router.get("/monitor/status")
async def get_monitor_status():
    """Get queue file monitor status (debug endpoint)."""
    from ..services.queue_file_monitor import queue_file_monitor

    return success_response(
        {
            "is_running": queue_file_monitor.is_running(),
            "queue_logs_dir": str(queue_file_monitor.queue_logs_dir),
            "observer_alive": queue_file_monitor.observer.is_alive()
            if queue_file_monitor._running
            else False,
        }
    )
