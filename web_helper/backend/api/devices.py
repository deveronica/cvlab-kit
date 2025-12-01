"""Device management API endpoints."""

import logging
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..models import Device, get_db
from ..models.queue_experiment import QueueExperiment
from ..services.event_manager import event_manager
from ..utils import error_response, success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/devices")


async def _validate_running_jobs(db: Session, host_id: str, active_jobs: List[str]):
    """Validate that jobs marked as running on this device are actually running.

    If a job is marked as running on this device but not in active_jobs,
    it means the worker isn't running it anymore (crashed, etc).
    """
    try:
        # Find jobs marked as running on this device (including virtual devices like gnode-3:0)
        running_jobs = (
            db.query(QueueExperiment)
            .filter(
                QueueExperiment.status == "running",
                QueueExperiment.assigned_device.like(f"{host_id}%")
            )
            .all()
        )

        for job in running_jobs:
            if job.experiment_uid not in active_jobs:
                logger.warning(
                    f"Job {job.experiment_uid} marked as running on {host_id} "
                    f"but not in worker's active_jobs. Marking as crashed."
                )
                job.status = "crashed"
                job.completed_at = datetime.utcnow()

        db.commit()
    except Exception as e:
        logger.error(f"Failed to validate running jobs for {host_id}: {e}")
        db.rollback()


@router.get("/")
@router.get("")
async def get_devices(db: Session = Depends(get_db)):
    """Get all devices with their status."""
    try:
        devices = db.query(Device).all()
        now = datetime.utcnow()

        device_list = []
        for device in devices:
            # Calculate real-time status based on last heartbeat
            # Heartbeat interval is 10s, so:
            # - healthy: <= 30s (3x heartbeat interval - allows for network delays)
            # - stale: 30s - 60s (intermediate warning state)
            # - disconnected: > 60s (clearly offline)
            status = "disconnected"
            if device.last_heartbeat:
                time_since_heartbeat = (now - device.last_heartbeat).total_seconds()
                if time_since_heartbeat <= 30:
                    status = "healthy"
                elif time_since_heartbeat <= 60:
                    status = "stale"
                else:
                    status = "disconnected"

            device_list.append(
                {
                    "host_id": device.host_id,
                    "gpu_util": device.gpu_util,
                    "vram_used": device.vram_used,
                    "vram_total": device.vram_total,
                    "gpu_temperature": device.gpu_temperature,
                    "gpu_power_usage": device.gpu_power_usage,
                    "gpu_count": device.gpu_count or 0,
                    "gpus": device.gpus_detail,
                    "cpu_util": device.cpu_util,
                    "memory_used": device.memory_used,
                    "memory_total": device.memory_total,
                    "disk_free": device.disk_free,
                    "torch_version": device.torch_version,
                    "cuda_version": device.cuda_version,
                    "status": status,
                    "last_heartbeat": (
                        device.last_heartbeat.isoformat() + "Z"
                        if device.last_heartbeat
                        else None
                    ),
                }
            )

        return success_response(
            device_list, {"message": "Devices retrieved successfully"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to fetch devices: {str(e)}",
            ),
        )


@router.post("/heartbeat")
async def receive_heartbeat(heartbeat_data: Dict, db: Session = Depends(get_db)):
    """Receive heartbeat from client agents."""
    try:
        host_id = heartbeat_data.get("host_id")
        if not host_id:
            raise HTTPException(
                status_code=400,
                detail=error_response(
                    title="Bad Request", status=400, detail="host_id is required"
                ),
            )

        # Find or create device
        device = db.query(Device).filter(Device.host_id == host_id).first()
        if not device:
            device = Device(host_id=host_id)
            db.add(device)

        # Update device with heartbeat data
        device.gpu_util = heartbeat_data.get("gpu_util")
        device.vram_used = heartbeat_data.get("vram_used")
        device.vram_total = heartbeat_data.get("vram_total")
        device.gpu_temperature = heartbeat_data.get("gpu_temperature")
        device.gpu_power_usage = heartbeat_data.get("gpu_power_usage")
        device.gpu_count = heartbeat_data.get("gpu_count", 0)
        device.gpus_detail = heartbeat_data.get("gpus")
        device.cpu_util = heartbeat_data.get("cpu_util")
        device.memory_used = heartbeat_data.get("memory_used")
        device.memory_total = heartbeat_data.get("memory_total")
        device.disk_free = heartbeat_data.get("disk_free")
        device.torch_version = heartbeat_data.get("torch_version")
        device.cuda_version = heartbeat_data.get("cuda_version")
        device.code_version = heartbeat_data.get("code_version")  # Reproducibility tracking
        device.active_jobs = heartbeat_data.get("active_jobs")  # Jobs running on this device
        device.status = "online"  # Store raw status, compute display status on read
        device.last_heartbeat = datetime.utcnow()

        db.commit()

        # Validate running jobs - mark stale jobs as crashed
        await _validate_running_jobs(db, host_id, device.active_jobs or [])

        # Broadcast device update via SSE (with real-time status)
        await event_manager.send_device_update(
            {
                "host_id": host_id,
                "gpu_util": device.gpu_util,
                "vram_used": device.vram_used,
                "vram_total": device.vram_total,
                "gpu_temperature": device.gpu_temperature,
                "gpu_power_usage": device.gpu_power_usage,
                "gpu_count": device.gpu_count,
                "gpus": device.gpus_detail,
                "cpu_util": device.cpu_util,
                "memory_used": device.memory_used,
                "memory_total": device.memory_total,
                "disk_free": device.disk_free,
                "torch_version": device.torch_version,
                "cuda_version": device.cuda_version,
                "code_version": device.code_version,
                "status": "healthy",  # Fresh heartbeat is always healthy
                "last_heartbeat": device.last_heartbeat.isoformat() + "Z"
                if device.last_heartbeat
                else None,
            }
        )

        return success_response(
            {"message": "Heartbeat received", "host_id": host_id},
            {"timestamp": datetime.utcnow().isoformat() + "Z"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to process heartbeat: {str(e)}",
            ),
        )


@router.delete("/{host_id}")
async def delete_device(host_id: str, db: Session = Depends(get_db)):
    """Delete a device. Only disconnected devices can be deleted."""
    try:
        device = db.query(Device).filter(Device.host_id == host_id).first()
        if not device:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Device '{host_id}' not found",
                ),
            )

        # Check if device is disconnected (> 60s since last heartbeat)
        now = datetime.utcnow()
        if device.last_heartbeat:
            time_since_heartbeat = (now - device.last_heartbeat).total_seconds()
            if time_since_heartbeat <= 60:
                raise HTTPException(
                    status_code=400,
                    detail=error_response(
                        title="Bad Request",
                        status=400,
                        detail="Only disconnected devices can be deleted. Wait for the device to disconnect first.",
                    ),
                )

        db.delete(device)
        db.commit()

        # Broadcast device removal via SSE
        await event_manager.broadcast(
            {
                "type": "device_removed",
                "data": {"host_id": host_id},
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

        return success_response(
            {"message": f"Device '{host_id}' deleted successfully"},
            {"host_id": host_id},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to delete device: {str(e)}",
            ),
        )
