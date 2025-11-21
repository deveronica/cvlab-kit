"""API endpoints for fetching experiment metrics."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..models import Run, get_db
from ..utils.responses import success_response

router = APIRouter(prefix="/metrics", tags=["metrics"])

LOGS_DIR = Path("logs")


@router.get("/{project}/{run_name}")
async def get_metrics(
    project: str,
    run_name: str,
    db: Session = Depends(get_db),
    downsample: Optional[int] = Query(
        None, description="Downsample to N points for performance"
    ),
):
    """Get metrics for a specific run with optional downsampling."""
    run = db.query(Run).filter(Run.run_name == run_name, Run.project == project).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")

    # Try multiple path patterns based on CVLab-Kit structure
    possible_paths = [
        # Primary: logs/<project>/<run_name>.csv (CLAUDE.md spec)
        LOGS_DIR / project / f"{run_name}.csv",
        # Backup: Use stored metrics_path from run record
        Path(run.metrics_path) if run.metrics_path else None,
        # Fallback: logs/<run_name>.csv (flat structure)
        LOGS_DIR / f"{run_name}.csv",
        # Alternative: logs/<project>/<run_name>_metrics.csv
        LOGS_DIR / project / f"{run_name}_metrics.csv",
    ]

    metrics_path = None
    for path in possible_paths:
        if path and path.exists():
            metrics_path = path
            break

    # If no direct match, scan project directory for any CSV containing run_name
    if not metrics_path:
        project_dir = LOGS_DIR / project
        if project_dir.exists():
            for csv_file in project_dir.glob("*.csv"):
                if run_name in csv_file.name:
                    metrics_path = csv_file
                    break

    if not metrics_path:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics file not found for run {run_name} in project {project}",
        )

    try:
        # Read and parse CSV data
        metrics_data = _read_metrics_file(metrics_path, downsample)

        return success_response(
            {
                "data": metrics_data,
                "metadata": {
                    "file_path": str(metrics_path),
                    "total_points": len(metrics_data),
                    "downsampled": downsample is not None,
                    "columns": list(metrics_data[0].keys()) if metrics_data else [],
                },
            },
            {"message": "Metrics retrieved successfully"},
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read metrics file: {str(e)}"
        )


def _read_metrics_file(
    file_path: Path, downsample: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Read and parse metrics CSV file with optional downsampling."""
    metrics_data = []

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        # Detect if file is empty
        first_char = csvfile.read(1)
        if not first_char:
            return []

        csvfile.seek(0)
        reader = csv.DictReader(csvfile)

        # Read all rows
        all_rows = list(reader)

        # Apply downsampling if requested
        if downsample and len(all_rows) > downsample:
            # Simple uniform sampling
            step = len(all_rows) / downsample
            indices = [int(i * step) for i in range(downsample)]
            # Always include the last row
            if indices[-1] != len(all_rows) - 1:
                indices[-1] = len(all_rows) - 1
            sampled_rows = [all_rows[i] for i in indices]
        else:
            sampled_rows = all_rows

        # Convert values to appropriate types
        for row in sampled_rows:
            converted_row = {}
            for key, value in row.items():
                converted_row[key] = _convert_value(value)
            metrics_data.append(converted_row)

    return metrics_data


def _convert_value(value: str) -> Any:
    """Convert string values to appropriate types (int, float, or keep as string)."""
    if not value or value.strip() == "":
        return None

    value = value.strip()

    # Try int first
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Keep as string
    return value


@router.get("/compare")
async def compare_metrics(
    run_names: str = Query(..., description="Comma-separated list of run names"),
    projects: Optional[str] = Query(
        None, description="Comma-separated list of projects (same order as runs)"
    ),
    columns: Optional[str] = Query(
        None, description="Comma-separated list of columns to include"
    ),
    db: Session = Depends(get_db),
):
    """Compare metrics across multiple runs."""
    run_name_list = [name.strip() for name in run_names.split(",")]
    project_list = [p.strip() for p in projects.split(",")] if projects else []
    column_list = [c.strip() for c in columns.split(",")] if columns else None

    if project_list and len(project_list) != len(run_name_list):
        raise HTTPException(
            status_code=400, detail="Number of projects must match number of run names"
        )

    results = {}

    for i, run_name in enumerate(run_name_list):
        project = project_list[i] if project_list else None

        # Find run in database
        query = db.query(Run).filter(Run.run_name == run_name)
        if project:
            query = query.filter(Run.project == project)

        run = query.first()
        if not run:
            continue

        try:
            # Get metrics for this run
            possible_paths = [
                LOGS_DIR / run.project / f"{run_name}.csv",
                Path(run.metrics_path) if run.metrics_path else None,
                LOGS_DIR / f"{run_name}.csv",
            ]

            metrics_path = None
            for path in possible_paths:
                if path and path.exists():
                    metrics_path = path
                    break

            if metrics_path:
                metrics_data = _read_metrics_file(metrics_path)

                # Filter columns if specified
                if column_list and metrics_data:
                    filtered_data = []
                    for row in metrics_data:
                        filtered_row = {
                            k: v for k, v in row.items() if k in column_list
                        }
                        filtered_data.append(filtered_row)
                    metrics_data = filtered_data

                results[f"{run.project}/{run_name}"] = {
                    "data": metrics_data,
                    "metadata": {
                        "project": run.project,
                        "run_name": run_name,
                        "status": run.status,
                        "total_points": len(metrics_data),
                    },
                }

        except Exception as e:
            # Skip failed runs but log the error
            results[f"{run.project if run else 'unknown'}/{run_name}"] = {
                "error": str(e)
            }

    return success_response(
        results, {"message": f"Retrieved metrics for {len(results)} runs"}
    )


@router.get("/summary/{project}")
async def get_project_metrics_summary(project: str, db: Session = Depends(get_db)):
    """Get metrics summary for all runs in a project."""
    runs = db.query(Run).filter(Run.project == project).all()

    if not runs:
        raise HTTPException(
            status_code=404, detail=f"No runs found for project {project}"
        )

    summary = {"project": project, "total_runs": len(runs), "runs": []}

    for run in runs:
        run_summary = {
            "run_name": run.run_name,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "total_steps": run.total_steps,
            "final_metrics": run.final_metrics or {},
        }

        summary["runs"].append(run_summary)

    return success_response(
        summary, {"message": "Project metrics summary retrieved successfully"}
    )


@router.get("/statistics/{project}/{run_name}")
async def get_metric_statistics(
    project: str,
    run_name: str,
    metric_name: str = Query(..., description="Name of the metric column to analyze"),
    db: Session = Depends(get_db),
):
    """Calculate statistics for a specific metric across all steps.

    Returns:
        - count: Total number of steps
        - min: Minimum value
        - max: Maximum value
        - mean: Average value
        - std: Standard deviation
        - median: Median value
        - q25: 25th percentile
        - q75: 75th percentile
        - best: Best value (based on metric direction preference)
        - best_step: Step at which best value occurred
        - latest: Most recent value
        - trend: Trend indicator (improving/degrading/stable)
    """
    run = db.query(Run).filter(Run.run_name == run_name, Run.project == project).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")

    # Find metrics CSV file
    possible_paths = [
        LOGS_DIR / project / f"{run_name}.csv",
        Path(run.metrics_path) if run.metrics_path else None,
        LOGS_DIR / f"{run_name}.csv",
        LOGS_DIR / project / f"{run_name}_metrics.csv",
    ]

    metrics_path = None
    for path in possible_paths:
        if path and path.exists():
            metrics_path = path
            break

    if not metrics_path:
        # Scan project directory
        project_dir = LOGS_DIR / project
        if project_dir.exists():
            for csv_file in project_dir.glob("*.csv"):
                if run_name in csv_file.name:
                    metrics_path = csv_file
                    break

    if not metrics_path:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics file not found for run {run_name} in project {project}",
        )

    try:
        # Check if file is empty before reading
        if metrics_path.stat().st_size == 0:
            raise HTTPException(
                status_code=400,
                detail="This run has no metrics data available. The run may have failed or not completed.",
            )

        # Read CSV and extract metric column
        try:
            df = pd.read_csv(metrics_path)
        except pd.errors.EmptyDataError:
            raise HTTPException(
                status_code=400,
                detail="This run has no metrics data available. The CSV file is empty.",
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Unable to read metrics file: {str(e)}"
            )

        if len(df) == 0:
            raise HTTPException(
                status_code=400,
                detail="This run has no metrics data available. The CSV file contains no data rows.",
            )

        if metric_name not in df.columns:
            available_columns = [col for col in df.columns if col != "step"]
            raise HTTPException(
                status_code=400,
                detail=f"Metric '{metric_name}' not found. Available metrics: {', '.join(available_columns)}",
            )

        # Extract metric values (skip NaN)
        values = df[metric_name].dropna()

        if len(values) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No valid values found for metric '{metric_name}'. All values are missing or NaN.",
            )

        # Calculate statistics

        stats = {
            "metric_name": metric_name,
            "count": int(len(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
        }

        # Determine "best" value based on metric name heuristics
        # Assume metrics with "loss", "error" in name are lower-is-better
        # Otherwise, higher-is-better (e.g., accuracy, f1_score)
        metric_lower = metric_name.lower()
        is_lower_better = any(
            keyword in metric_lower for keyword in ["loss", "error", "distance"]
        )

        if is_lower_better:
            best_value = stats["min"]
            best_idx = values.idxmin()
        else:
            best_value = stats["max"]
            best_idx = values.idxmax()

        stats["best"] = float(best_value)
        stats["best_step"] = (
            int(df.loc[best_idx, "step"]) if "step" in df.columns else int(best_idx)
        )
        stats["latest"] = float(values.iloc[-1])
        stats["is_lower_better"] = is_lower_better

        # Calculate trend (comparing first 10% vs last 10%)
        split_point = max(1, len(values) // 10)
        early_mean = values.iloc[:split_point].mean()
        late_mean = values.iloc[-split_point:].mean()

        improvement_threshold = 0.01  # 1% improvement threshold
        if is_lower_better:
            improvement = (early_mean - late_mean) / (early_mean + 1e-10)
        else:
            improvement = (late_mean - early_mean) / (early_mean + 1e-10)

        if improvement > improvement_threshold:
            trend = "improving"
        elif improvement < -improvement_threshold:
            trend = "degrading"
        else:
            trend = "stable"

        stats["trend"] = trend
        stats["improvement_pct"] = float(improvement * 100)

        # Add value series for visualization (sampled to max 100 points)
        max_points = 100
        if len(values) > max_points:
            step_size = len(values) // max_points
            sampled_indices = list(range(0, len(values), step_size))
            if sampled_indices[-1] != len(values) - 1:
                sampled_indices.append(len(values) - 1)
        else:
            sampled_indices = list(range(len(values)))

        stats["series"] = [
            {
                "step": int(df.loc[idx, "step"]) if "step" in df.columns else int(idx),
                "value": float(values.iloc[idx]),
            }
            for idx in sampled_indices
        ]

        return success_response(
            stats, {"message": f"Statistics for {metric_name} calculated successfully"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate statistics: {str(e)}"
        )
