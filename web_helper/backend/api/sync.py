"""Distributed log synchronization API endpoints."""

import logging
from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..models import get_db
from ..models.queue_experiment import QueueExperiment
from ..services.event_manager import event_manager
from ..utils import error_response, success_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sync")


@router.post("/experiment/{experiment_uid}/{file_name}")
async def upload_experiment_file(
    experiment_uid: str,
    file_name: str,
    delta: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload Experiment file (terminal logs).

    Experiment files are for process management (Execute tab).
    Written to web_helper/queue_logs/{exp_uid}/

    Args:
        experiment_uid: Unique experiment identifier
        file_name: terminal_log.log or terminal_err.log
        delta: Binary content to append
    """
    try:
        exp = (
            db.query(QueueExperiment)
            .filter(QueueExperiment.experiment_uid == experiment_uid)
            .first()
        )

        if not exp:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Experiment Not Found",
                    status=404,
                    detail=f"Experiment {experiment_uid} not found",
                ),
            )

        # Experiment logs → web_helper/queue_logs/
        file_path = Path("web_helper/queue_logs") / experiment_uid / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Append or write
        content = await delta.read()
        mode = "ab" if file_path.suffix == ".log" else "wb"
        async with aiofiles.open(file_path, mode) as f:
            await f.write(content)

        exp.sync_status = "synced"
        exp.last_sync_at = datetime.utcnow()
        db.commit()

        logger.info(
            f"Experiment file synced: {experiment_uid}/{file_name} ({len(content)} bytes)"
        )

        await event_manager.send_run_update(
            {
                "event_type": "experiment_sync",
                "experiment_uid": experiment_uid,
                "file_name": file_name,
                "bytes_synced": len(content),
            }
        )

        return success_response(
            {"message": "Experiment file synced", "bytes_synced": len(content)},
            {"timestamp": datetime.utcnow().isoformat() + "Z"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload experiment file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to sync experiment file: {str(e)}",
            ),
        )


@router.post("/run/{experiment_uid}/{file_name}")
async def upload_run_file(
    experiment_uid: str,
    file_name: str,
    delta: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload Run file (cvlabkit output).

    Run files are for result analysis (Projects tab).
    Written to logs/{project}/

    Args:
        experiment_uid: Unique experiment identifier
        file_name: run_name.csv, run_name.yaml, run_name.pt
        delta: Binary content (append for CSV, replace for others)
    """
    try:
        exp = (
            db.query(QueueExperiment)
            .filter(QueueExperiment.experiment_uid == experiment_uid)
            .first()
        )

        if not exp:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Experiment Not Found",
                    status=404,
                    detail=f"Experiment {experiment_uid} not found",
                ),
            )

        # Run results → logs/
        file_path = Path("logs") / exp.project / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV: append, Others: replace
        content = await delta.read()
        mode = "ab" if file_path.suffix == ".csv" else "wb"
        async with aiofiles.open(file_path, mode) as f:
            await f.write(content)

        exp.sync_status = "synced"
        exp.last_sync_at = datetime.utcnow()
        exp.remote_mtime = int(file_path.stat().st_mtime)
        db.commit()

        logger.info(
            f"Run file synced: {experiment_uid}/{file_name} ({len(content)} bytes)"
        )

        await event_manager.send_run_update(
            {
                "event_type": "run_sync",
                "experiment_uid": experiment_uid,
                "project": exp.project,
                "file_name": file_name,
                "sync_status": "synced",
                "bytes_synced": len(content),
            }
        )

        return success_response(
            {"message": "Run file synced", "bytes_synced": len(content)},
            {"timestamp": datetime.utcnow().isoformat() + "Z"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload run file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to sync run file: {str(e)}",
            ),
        )


# Backward compatibility (deprecated)
@router.post("/delta/{experiment_uid}/{file_name}")
async def upload_delta(
    experiment_uid: str,
    file_name: str,
    delta: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """[DEPRECATED] Use /experiment or /run endpoints instead."""
    logger.warning(f"Deprecated /delta endpoint used for {experiment_uid}/{file_name}")
    # Default to run sync for backward compatibility
    return await upload_run_file(experiment_uid, file_name, delta, db)


@router.post("/full/{experiment_uid}/{file_name}")
async def upload_full_file(
    experiment_uid: str,
    file_name: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """[DEPRECATED] Use /experiment or /run endpoints instead."""
    logger.warning(f"Deprecated /full endpoint used for {experiment_uid}/{file_name}")
    # Default to run sync for backward compatibility
    return await upload_run_file(experiment_uid, file_name, file, db)


@router.get("/status/{experiment_uid}")
async def get_sync_status(experiment_uid: str, db: Session = Depends(get_db)):
    """Query current sync status for reconnection recovery.

    Returns file metadata (mtime, size) for all files associated with
    the experiment. Used by clients to detect missing deltas after
    network disconnection.

    Args:
        experiment_uid: Unique experiment identifier

    Returns:
        Sync status and file metadata
    """
    try:
        exp = (
            db.query(QueueExperiment)
            .filter(QueueExperiment.experiment_uid == experiment_uid)
            .first()
        )

        if not exp:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Experiment Not Found",
                    status=404,
                    detail=f"Experiment {experiment_uid} not found",
                ),
            )

        # Scan log directory for all related files
        log_dir = Path("logs") / exp.project
        files = {}

        if log_dir.exists():
            # Find all files matching the experiment's run pattern
            # Assuming run_name is derived from experiment_uid or stored in meta
            run_name_pattern = exp.meta.get("run_name", "*") if exp.meta else "*"

            for file_path in log_dir.glob(f"{run_name_pattern}.*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    files[file_path.name] = {
                        "mtime": int(stat.st_mtime),
                        "size": stat.st_size,
                        "synced_at": (
                            exp.last_sync_at.isoformat() + "Z"
                            if exp.last_sync_at
                            else None
                        ),
                    }

        return success_response(
            {
                "experiment_uid": experiment_uid,
                "project": exp.project,
                "sync_status": exp.sync_status,
                "last_sync_at": (
                    exp.last_sync_at.isoformat() + "Z" if exp.last_sync_at else None
                ),
                "server_origin": exp.server_origin,
                "files": files,
            },
            {"timestamp": datetime.utcnow().isoformat() + "Z"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get sync status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get sync status: {str(e)}",
            ),
        )


@router.post("/checkpoint/{experiment_uid}")
async def update_checkpoint(
    experiment_uid: str,
    checkpoint: dict,
    db: Session = Depends(get_db),
):
    """Update recovery checkpoint for experiment.

    Stores current training state (epoch, step, metrics) to enable
    resumption after network disconnection or crash.

    Args:
        experiment_uid: Unique experiment identifier
        checkpoint: Recovery data (e.g., {"epoch": 5, "step": 1000})
    """
    try:
        exp = (
            db.query(QueueExperiment)
            .filter(QueueExperiment.experiment_uid == experiment_uid)
            .first()
        )

        if not exp:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Experiment Not Found",
                    status=404,
                    detail=f"Experiment {experiment_uid} not found",
                ),
            )

        exp.recovery_checkpoint = checkpoint
        db.commit()

        logger.debug(f"Checkpoint updated for {experiment_uid}: {checkpoint}")

        return success_response(
            {"message": "Checkpoint updated", "checkpoint": checkpoint}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update checkpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to update checkpoint: {str(e)}",
            ),
        )
