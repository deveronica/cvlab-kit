"""Project management API endpoints."""

import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..models import Run, get_db
from ..services.indexer import log_indexer
from ..utils import error_response, success_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects")


def extract_hyperparameters_from_config(config_path: str) -> dict:
    """Extract hyperparameters from YAML config file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return {}

        with open(config_file) as f:
            # Use FullLoader like config.py to handle Python-specific tags
            config_data = yaml.load(f, Loader=yaml.FullLoader)

        if not isinstance(config_data, dict):
            return {}

        # Return all config data as hyperparameters (including nested structures)
        # Filter out non-serializable types and internal keys
        def sanitize_value(value):
            """Convert complex types to JSON-serializable ones."""
            if value is None:
                return None
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [sanitize_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            else:
                # Convert other types to string representation
                return str(value)

        hyperparams = {}
        for key, value in config_data.items():
            # Skip internal/system keys
            if key.startswith("_"):
                continue
            hyperparams[key] = sanitize_value(value)

        return hyperparams

    except Exception as e:
        logger.warning(f"Error extracting hyperparameters from {config_path}: {e}")
        return {}


@router.get("/")
@router.get("")
async def get_projects(db: Session = Depends(get_db)):
    """Get all projects with their runs."""
    try:
        runs = db.query(Run).all()
        projects = {}
        project_earliest_time = {}  # Track earliest started_at per project

        for run in runs:
            if run.project not in projects:
                projects[run.project] = []
                project_earliest_time[run.project] = None

            projects[run.project].append(
                {
                    "run_name": run.run_name,
                    "status": run.status,
                    "started_at": run.started_at.isoformat()
                    if run.started_at
                    else None,
                    "finished_at": run.finished_at.isoformat()
                    if run.finished_at
                    else None,
                    "config_path": run.config_path,
                }
            )

            # Track earliest started_at for this project
            if run.started_at:
                if (
                    project_earliest_time[run.project] is None
                    or run.started_at < project_earliest_time[run.project]
                ):
                    project_earliest_time[run.project] = run.started_at

        project_list = [{"name": name, "runs": runs} for name, runs in projects.items()]

        # Sort by earliest run time (project creation time), newest first
        project_list.sort(
            key=lambda p: project_earliest_time.get(p["name"]) or "", reverse=True
        )

        return success_response(
            project_list, {"message": "Projects retrieved successfully"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to fetch projects: {str(e)}",
            ),
        )


@router.get("/{project_name}/experiments")
async def get_project_experiments(project_name: str, db: Session = Depends(get_db)):
    """Get detailed experiment data for a project with hyperparameters."""
    try:
        runs = db.query(Run).filter(Run.project == project_name).all()

        if not runs:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found",
                    status=404,
                    detail=f"Project '{project_name}' not found",
                ),
            )

        experiments = []
        for run in runs:
            # Extract hyperparameters from config file
            hyperparams = {}
            if run.config_path and Path(run.config_path).exists():
                hyperparams = extract_hyperparameters_from_config(run.config_path)

            # Get metrics from run
            final_metrics = run.final_metrics or {}
            max_metrics = run.max_metrics or {}
            min_metrics = run.min_metrics or {}
            mean_metrics = run.mean_metrics or {}

            experiment = {
                "run_name": run.run_name,
                "status": run.status,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "last_updated": run.last_updated.isoformat()
                if run.last_updated
                else None,
                "config_path": run.config_path,
                "metrics_path": run.metrics_path,
                "total_steps": run.total_steps or 0,
                "hyperparameters": hyperparams,
                "final_metrics": final_metrics,
                "max_metrics": max_metrics,
                "min_metrics": min_metrics,
                "mean_metrics": mean_metrics,
                "notes": run.notes or "",
                "tags": run.tags or [],
            }
            experiments.append(experiment)

        return success_response(
            {
                "project": project_name,
                "experiment_count": len(experiments),
                "experiments": experiments,
            },
            {
                "message": f"Experiments for project '{project_name}' retrieved successfully"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to fetch experiments for project '{project_name}': {str(e)}",
            ),
        )


@router.post("/reindex")
async def reindex_projects():
    """Trigger reindexing of all projects."""
    try:
        stats = await log_indexer.scan_and_index(force=True)
        return success_response(stats, {"message": "Reindexing completed successfully"})

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to reindex projects: {str(e)}",
            ),
        )


@router.post("/reindex/{project_name}")
async def reindex_project(project_name: str):
    """Trigger reindexing of a specific project."""
    try:
        result = await log_indexer.reindex_project(project_name)

        if "error" in result:
            raise HTTPException(
                status_code=404,
                detail=error_response(
                    title="Not Found", status=404, detail=result["error"]
                ),
            )

        return success_response(
            result, {"message": f"Project {project_name} reindexed successfully"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=error_response(
                title="Internal Server Error",
                status=500,
                detail=f"Failed to reindex project {project_name}: {str(e)}",
            ),
        )
