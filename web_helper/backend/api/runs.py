"""Run-specific API endpoints for config and logs access."""

import csv
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..models import Run, get_db
from ..utils import error_response, success_response

router = APIRouter(prefix="/runs")


# Pydantic models for request validation
class UpdateNotesRequest(BaseModel):
    """Request model for updating run notes."""

    notes: str = Field(..., max_length=2000, description="User notes for the run")


class UpdateTagsRequest(BaseModel):
    """Request model for updating run tags."""

    tags: List[str] = Field(..., description="List of tags for the run")


def find_run_files(project: str, run_name: str):
    """Find config and log files for a specific run."""
    logs_dir = Path("logs") / project
    queue_logs_dir = Path("web_helper/queue_logs")

    # Look for files with exact run_name match
    config_files = []
    log_files = []

    # Search in logs/ directory (original location)
    if logs_dir.exists():
        # Look for config files - prioritize _config.yaml (full config) over .yaml (metadata)
        for pattern in [f"{run_name}_config.yaml", f"{run_name}.yaml"]:
            config_file = logs_dir / pattern
            if config_file.exists():
                config_files.append(config_file)

        # Look for CSV files
        for pattern in [f"{run_name}.csv"]:
            log_file = logs_dir / pattern
            if log_file.exists():
                log_files.append(log_file)

    # Search in web_helper/queue_logs/ directory (new experiment location)
    if queue_logs_dir.exists():
        for experiment_dir in queue_logs_dir.iterdir():
            if experiment_dir.is_dir():
                # Look for all YAML files in experiment directories
                for yaml_file in experiment_dir.glob("*.yaml"):
                    try:
                        with open(yaml_file, encoding="utf-8") as f:
                            content = f.read()
                            if (
                                run_name in content
                                or run_name in experiment_dir.name
                                or run_name in yaml_file.name
                            ):
                                if yaml_file not in config_files:
                                    config_files.append(yaml_file)
                    except Exception:
                        # If we can't read the file, skip it
                        pass

                # Look for execution logs that might contain run_name
                for log_pattern in ["*.out.log", "*.err.log", "*.log", "*.csv"]:
                    for log_file in experiment_dir.glob(log_pattern):
                        if run_name in log_file.name or run_name in experiment_dir.name:
                            if log_file not in log_files:
                                log_files.append(log_file)

        # Also check direct files in queue_logs root (for backward compatibility)
        for yaml_file in queue_logs_dir.glob("*.yaml"):
            if yaml_file.is_file():
                try:
                    with open(yaml_file, encoding="utf-8") as f:
                        content = f.read()
                        if run_name in content or run_name in yaml_file.name:
                            if yaml_file not in config_files:
                                config_files.append(yaml_file)
                except Exception:
                    pass

        for log_pattern in ["*.out.log", "*.err.log", "*.log", "*.csv"]:
            for log_file in queue_logs_dir.glob(log_pattern):
                if log_file.is_file() and run_name in log_file.name:
                    if log_file not in log_files:
                        log_files.append(log_file)

    return config_files, log_files


@router.get("/{project}/{run_name}/config")
async def get_run_config(project: str, run_name: str, db: Session = Depends(get_db)):
    """Get configuration file content for a specific run."""
    try:
        # Check if run exists in database
        run = (
            db.query(Run)
            .filter(Run.project == project, Run.run_name == run_name)
            .first()
        )

        if not run:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Run not found: {project}/{run_name}",
                ),
            )

        # Try to find config files
        config_files, _ = find_run_files(project, run_name)

        if not config_files:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Configuration file not found for run: {project}/{run_name}",
                ),
            )

        # Use the first found config file
        config_file = config_files[0]

        # Read config file content
        try:
            with open(config_file, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=error_response(
                    title="Internal Server Error",
                    status=500,
                    detail=f"Failed to read config file: {str(e)}",
                ),
            )

        return success_response(
            {
                "content": content,
                "file_path": str(config_file),
                "file_size": config_file.stat().st_size,
                "last_modified": config_file.stat().st_mtime,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get run config: {str(e)}",
            ),
        )


@router.get("/{project}/{run_name}/logs")
async def get_run_logs(project: str, run_name: str, db: Session = Depends(get_db)):
    """Get log content for a specific run."""
    try:
        # Check if run exists in database
        run = (
            db.query(Run)
            .filter(Run.project == project, Run.run_name == run_name)
            .first()
        )

        if not run:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Run not found: {project}/{run_name}",
                ),
            )

        # Try to find log files (CSV and potential text logs)
        _, log_files = find_run_files(project, run_name)

        # Also look for .out, .log, .txt files
        logs_dir = Path("logs") / project
        if logs_dir.exists():
            for pattern in ["*.out", "*.log", "*.txt"]:
                for file_path in logs_dir.glob(pattern):
                    if run_name in file_path.name:
                        log_files.append(file_path)

        if not log_files:
            # Return empty logs instead of error to match user expectation
            return success_response(
                {
                    "content": "No log files found for this run.",
                    "files": [],
                    "message": f"No log files found for run: {project}/{run_name}",
                }
            )

        # Read all log files and combine them
        combined_logs = []
        file_info = []

        for log_file in log_files:
            try:
                with open(log_file, encoding="utf-8") as f:
                    content = f.read()
                combined_logs.append(f"=== {log_file.name} ===\n{content}\n")
                file_info.append(
                    {
                        "name": log_file.name,
                        "path": str(log_file),
                        "size": log_file.stat().st_size,
                        "last_modified": log_file.stat().st_mtime,
                    }
                )
            except UnicodeDecodeError:
                # Handle binary files or encoding issues
                combined_logs.append(
                    f"=== {log_file.name} ===\n[Binary file or encoding error]\n"
                )
                file_info.append(
                    {
                        "name": log_file.name,
                        "path": str(log_file),
                        "size": log_file.stat().st_size,
                        "last_modified": log_file.stat().st_mtime,
                        "error": "Could not read file (binary or encoding issue)",
                    }
                )
            except Exception as e:
                combined_logs.append(
                    f"=== {log_file.name} ===\n[Error reading file: {str(e)}]\n"
                )
                file_info.append(
                    {"name": log_file.name, "path": str(log_file), "error": str(e)}
                )

        return success_response(
            {
                "content": "\n".join(combined_logs),
                "files": file_info,
                "total_files": len(log_files),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get run logs: {str(e)}",
            ),
        )


@router.get("/{project}/{run_name}")
async def get_run_details(project: str, run_name: str, db: Session = Depends(get_db)):
    """Get complete run details including config and log file paths."""
    try:
        # Check if run exists in database
        run = (
            db.query(Run)
            .filter(Run.project == project, Run.run_name == run_name)
            .first()
        )

        if not run:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Run not found: {project}/{run_name}",
                ),
            )

        # Find associated files
        config_files, log_files = find_run_files(project, run_name)

        # Build response
        response_data = {
            "run_name": run.run_name,
            "project": run.project,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "config_path": run.config_path,
            "metrics_path": run.metrics_path,
            "total_steps": run.total_steps,
            "final_metrics": run.final_metrics or {},
            "available_files": {
                "config_files": [str(f) for f in config_files],
                "log_files": [str(f) for f in log_files],
            },
        }

        return success_response(response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get run details: {str(e)}",
            ),
        )


@router.get("/{project}/{run_name}/metrics")
async def get_run_metrics(
    project: str,
    run_name: str,
    downsample: Optional[int] = Query(None, description="Downsample to N points"),
    db: Session = Depends(get_db),
):
    """Get step-wise metrics from CSV file for a specific run.

    Returns all step data from the CSV file. Use downsample parameter to reduce data points.
    """
    try:
        # Check if run exists in database
        run = (
            db.query(Run)
            .filter(Run.project == project, Run.run_name == run_name)
            .first()
        )

        if not run:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Run not found: {project}/{run_name}",
                ),
            )

        # Find CSV file
        logs_dir = Path("logs") / project
        csv_file = None

        # Try exact match first
        potential_csv = logs_dir / f"{run_name}.csv"
        if potential_csv.exists():
            csv_file = potential_csv
        else:
            # Search for CSV files containing run_name
            if logs_dir.exists():
                for file_path in logs_dir.glob("*.csv"):
                    if run_name in file_path.name:
                        csv_file = file_path
                        break

        if not csv_file:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"CSV file not found for run: {project}/{run_name}",
                ),
            )

        # Read CSV file
        try:
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)

            # Convert numeric values
            for row in data:
                for key, value in row.items():
                    try:
                        # Try to convert to float
                        row[key] = float(value)
                    except (ValueError, TypeError):
                        # Keep as string if not numeric
                        pass

            # Apply downsampling if requested
            if downsample and downsample > 0 and len(data) > downsample:
                # Simple uniform downsampling
                step = len(data) / downsample
                indices = [int(i * step) for i in range(downsample)]
                data = [data[i] for i in indices]

            return success_response(
                {
                    "data": data,
                    "total_steps": len(data),
                    "file_path": str(csv_file),
                    "columns": list(data[0].keys()) if data else [],
                }
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=error_response(
                    title="Internal Server Error",
                    status=500,
                    detail=f"Failed to read CSV file: {str(e)}",
                ),
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get run metrics: {str(e)}",
            ),
        )


@router.patch("/{project}/{run_name}/notes")
async def update_run_notes(
    project: str,
    run_name: str,
    request: UpdateNotesRequest,
    db: Session = Depends(get_db),
):
    """Update notes for a specific run."""
    try:
        # Find the run
        run = (
            db.query(Run)
            .filter(Run.project == project, Run.run_name == run_name)
            .first()
        )

        if not run:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Run not found: {project}/{run_name}",
                ),
            )

        # Update notes
        run.notes = request.notes
        run.last_updated = datetime.utcnow()
        db.commit()
        db.refresh(run)

        return success_response(
            {
                "run_name": run.run_name,
                "project": run.project,
                "notes": run.notes,
                "last_updated": run.last_updated.isoformat(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to update notes: {str(e)}",
            ),
        )


@router.patch("/{project}/{run_name}/tags")
async def update_run_tags(
    project: str,
    run_name: str,
    request: UpdateTagsRequest,
    db: Session = Depends(get_db),
):
    """Update tags for a specific run."""
    try:
        # Find the run
        run = (
            db.query(Run)
            .filter(Run.project == project, Run.run_name == run_name)
            .first()
        )

        if not run:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Run not found: {project}/{run_name}",
                ),
            )

        # Update tags
        run.tags = request.tags
        run.last_updated = datetime.utcnow()
        db.commit()
        db.refresh(run)

        return success_response(
            {
                "run_name": run.run_name,
                "project": run.project,
                "tags": run.tags,
                "last_updated": run.last_updated.isoformat(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to update tags: {str(e)}",
            ),
        )


@router.get("/{project}/tags")
async def get_project_tags(project: str, db: Session = Depends(get_db)):
    """Get all unique tags used in a project for autocomplete."""
    try:
        # Get all runs in the project
        runs = db.query(Run).filter(Run.project == project).all()

        # Collect all unique tags
        all_tags = set()
        for run in runs:
            if run.tags:
                all_tags.update(run.tags)

        return success_response({"project": project, "tags": sorted(list(all_tags))})

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to get project tags: {str(e)}",
            ),
        )


@router.get("/{project}/{run_name}/artifacts")
async def list_run_artifacts(
    project: str, run_name: str, db: Session = Depends(get_db)
):
    """List all available artifacts for a specific run."""
    try:
        # Check if run exists
        run = (
            db.query(Run)
            .filter(Run.project == project, Run.run_name == run_name)
            .first()
        )

        if not run:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Run not found: {project}/{run_name}",
                ),
            )

        artifacts = []
        logs_dir = Path("logs") / project

        # Helper function to add artifact info
        def add_artifact(file_path: Path, artifact_type: str):
            if file_path and file_path.exists():
                artifacts.append(
                    {
                        "name": file_path.name,
                        "type": artifact_type,
                        "size": file_path.stat().st_size,
                        "path": str(file_path),
                        "last_modified": file_path.stat().st_mtime,
                    }
                )

        # Add config file
        if run.config_path:
            add_artifact(Path(run.config_path), "config")

        # Add metrics CSV
        if run.metrics_path:
            add_artifact(Path(run.metrics_path), "metrics")

        # Add checkpoint file
        if run.checkpoint_path:
            add_artifact(Path(run.checkpoint_path), "checkpoint")

        # Search for additional files in logs directory
        if logs_dir.exists():
            # Look for checkpoints (.pt, .pth, .ckpt)
            for ext in ["*.pt", "*.pth", "*.ckpt"]:
                for file_path in logs_dir.glob(ext):
                    if run_name in file_path.name:
                        add_artifact(file_path, "checkpoint")

            # Look for log files
            for ext in ["*.log", "*.out", "*.err"]:
                for file_path in logs_dir.glob(ext):
                    if run_name in file_path.name:
                        add_artifact(file_path, "log")

        # Remove duplicates based on path
        seen_paths = set()
        unique_artifacts = []
        for artifact in artifacts:
            if artifact["path"] not in seen_paths:
                seen_paths.add(artifact["path"])
                unique_artifacts.append(artifact)

        return success_response(
            {
                "run_name": run_name,
                "project": project,
                "artifacts": unique_artifacts,
                "total_count": len(unique_artifacts),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to list artifacts: {str(e)}",
            ),
        )


@router.get("/{project}/{run_name}/artifacts/download")
async def download_run_artifact(
    project: str,
    run_name: str,
    file_path: str = Query(..., description="Path to the artifact file"),
    db: Session = Depends(get_db),
):
    """Download a specific artifact file."""
    try:
        # Check if run exists
        run = (
            db.query(Run)
            .filter(Run.project == project, Run.run_name == run_name)
            .first()
        )

        if not run:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Run not found: {project}/{run_name}",
                ),
            )

        # Validate file path (security check)
        artifact_path = Path(file_path)
        if not artifact_path.exists():
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Artifact file not found: {file_path}",
                ),
            )

        # Ensure file is within logs directory (security)
        logs_dir = Path("logs").resolve()
        try:
            artifact_path_resolved = artifact_path.resolve()
            if not str(artifact_path_resolved).startswith(str(logs_dir)):
                raise HTTPException(
                    status_code=403,
                    detail=error_response(
                        title="Forbidden",
                        status=403,
                        detail="Access denied: File must be within logs directory",
                    ),
                )
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=error_response(
                    title="Bad Request", status=400, detail="Invalid file path"
                ),
            )

        # Return file for download
        return FileResponse(
            path=str(artifact_path),
            filename=artifact_path.name,
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to download artifact: {str(e)}",
            ),
        )
