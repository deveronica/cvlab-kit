"""API endpoints for outlier detection in experiment runs."""

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..models.database import get_db
from ..models.run import Run
from ..services.outlier_detector import OutlierDetector, OutlierMethod

router = APIRouter(prefix="/outliers")


class OutlierDetectionRequest(BaseModel):
    """Request model for outlier detection."""

    project: str = Field(..., description="Project name")
    run_names: Optional[List[str]] = Field(
        None, description="Specific run names to analyze (None = all)"
    )
    hyperparam_columns: Optional[List[str]] = Field(
        None, description="Hyperparameter columns to analyze"
    )
    metric_columns: Optional[List[str]] = Field(
        None, description="Metric columns to analyze"
    )
    method: OutlierMethod = Field(
        "iqr", description="Detection method (iqr, zscore, modified_zscore)"
    )
    threshold: float = Field(
        1.5, description="Detection threshold (method-specific)", ge=0.1, le=10.0
    )


class OutlierDetectionResponse(BaseModel):
    """Response model for outlier detection."""

    project: str
    total_runs: int
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    method: str
    threshold: float


@router.post("/detect", response_model=OutlierDetectionResponse)
async def detect_outliers(request: OutlierDetectionRequest) -> OutlierDetectionResponse:
    """Detect outliers in experiment runs.

    This endpoint analyzes hyperparameters and metrics across runs
    to identify anomalous values using statistical methods.

    Args:
        request: Outlier detection parameters

    Returns:
        Outlier detection results for each column

    Raises:
        HTTPException: If project not found or insufficient data
    """
    db: Session = next(get_db())

    try:
        # Query runs from database
        query = db.query(Run).filter(Run.project == request.project)

        if request.run_names:
            query = query.filter(Run.run_name.in_(request.run_names))

        runs = query.all()

        if not runs:
            raise HTTPException(
                status_code=404, detail=f"No runs found for project: {request.project}"
            )

        # Convert to dictionaries
        runs_data = [
            {
                "run_name": run.run_name,
                "hyperparameters": run.hyperparameters or {},
                "metrics": {"final": run.final_metrics or {}},
            }
            for run in runs
        ]

        # Initialize detector
        detector = OutlierDetector(
            method=request.method, threshold=request.threshold, min_samples=3
        )

        # Determine columns to analyze
        hyperparam_columns = request.hyperparam_columns
        if hyperparam_columns is None:
            # Auto-detect numeric hyperparameter columns
            hyperparam_columns = _extract_numeric_columns(
                runs_data, column_type="hyperparam"
            )

        metric_columns = request.metric_columns
        if metric_columns is None:
            # Auto-detect numeric metric columns
            metric_columns = _extract_numeric_columns(runs_data, column_type="metric")

        # Analyze hyperparameters
        hp_results = {}
        if hyperparam_columns:
            hp_results = detector.analyze_multiple_columns(
                runs_data, hyperparam_columns, "hyperparam"
            )

        # Analyze metrics
        metric_results = {}
        if metric_columns:
            metric_results = detector.analyze_multiple_columns(
                runs_data, metric_columns, "metric"
            )

        return OutlierDetectionResponse(
            project=request.project,
            total_runs=len(runs),
            hyperparameters=hp_results,
            metrics=metric_results,
            method=request.method,
            threshold=request.threshold,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outlier detection failed: {e}")
    finally:
        db.close()


@router.get("/{project}/summary")
async def get_outlier_summary(
    project: str,
    method: OutlierMethod = Query("iqr", description="Detection method"),
    threshold: float = Query(1.5, description="Detection threshold", ge=0.1, le=10.0),
) -> Dict[str, Any]:
    """Get outlier summary for a project.

    This is a simplified endpoint that returns only the summary
    without detailed per-column results.

    Args:
        project: Project name
        method: Detection method
        threshold: Detection threshold

    Returns:
        Summary of outlier detection across all numeric columns
    """
    db: Session = next(get_db())

    try:
        runs = db.query(Run).filter(Run.project == project).all()

        if not runs:
            raise HTTPException(
                status_code=404, detail=f"No runs found for project: {project}"
            )

        runs_data = [
            {
                "run_name": run.run_name,
                "hyperparameters": run.hyperparameters or {},
                "metrics": {"final": run.final_metrics or {}},
            }
            for run in runs
        ]

        # Auto-detect columns
        hyperparam_columns = _extract_numeric_columns(runs_data, "hyperparam")
        metric_columns = _extract_numeric_columns(runs_data, "metric")

        detector = OutlierDetector(method=method, threshold=threshold)

        # Analyze all columns
        all_outlier_runs = set()
        outlier_counts = {}
        column_outliers = {}
        cell_outliers = {}  # {(run_name, column): True}

        for col in hyperparam_columns:
            run_names = [r["run_name"] for r in runs_data]
            values = [
                float(r["hyperparameters"].get(col, float("nan"))) for r in runs_data
            ]
            result = detector.detect_outliers(values, run_names)
            if result["outlier_runs"]:
                column_name = f"hyperparam.{col}"
                column_outliers[column_name] = len(result["outlier_runs"])
                all_outlier_runs.update(result["outlier_runs"])
                for run_name in result["outlier_runs"]:
                    outlier_counts[run_name] = outlier_counts.get(run_name, 0) + 1
                    cell_outliers[f"{run_name}|{column_name}"] = True

        for col in metric_columns:
            run_names = [r["run_name"] for r in runs_data]
            values = [
                float(r["metrics"]["final"].get(col, float("nan"))) for r in runs_data
            ]
            result = detector.detect_outliers(values, run_names)
            if result["outlier_runs"]:
                column_name = f"metric.{col}"
                column_outliers[column_name] = len(result["outlier_runs"])
                all_outlier_runs.update(result["outlier_runs"])
                for run_name in result["outlier_runs"]:
                    outlier_counts[run_name] = outlier_counts.get(run_name, 0) + 1
                    cell_outliers[f"{run_name}|{column_name}"] = True

        # Sort by outlier count
        top_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        return {
            "project": project,
            "total_runs": len(runs),
            "total_outlier_runs": len(all_outlier_runs),
            "outlier_percentage": (
                len(all_outlier_runs) / len(runs) * 100 if runs else 0
            ),
            "top_outlier_runs": [
                {"run_name": name, "outlier_count": count}
                for name, count in top_outliers
            ],
            "columns_with_outliers": column_outliers,
            "cell_outliers": cell_outliers,  # NEW: {run_name|column: true}
            "method": method,
            "threshold": threshold,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get outlier summary: {e}"
        )
    finally:
        db.close()


def _extract_numeric_columns(
    runs_data: List[Dict[str, Any]], column_type: Literal["hyperparam", "metric"]
) -> List[str]:
    """Extract numeric column names from run data.

    Args:
        runs_data: List of run dictionaries
        column_type: Type of columns to extract

    Returns:
        List of numeric column names
    """
    if not runs_data:
        return []

    numeric_columns = set()

    for run in runs_data:
        if column_type == "hyperparam":
            data = run.get("hyperparameters", {})
        else:  # metric
            data = run.get("metrics", {}).get("final", {})

        for key, value in data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_columns.add(key)

    return sorted(numeric_columns)
