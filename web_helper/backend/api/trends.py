"""API endpoints for trend analysis of experiment runs."""

from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..models.database import get_db
from ..models.run import Run
from ..services.trend_analyzer import TrendAnalyzer

router = APIRouter(prefix="/trends")


class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis."""

    project: str = Field(..., description="Project name")
    metrics: Optional[List[str]] = Field(
        None, description="Metrics to analyze (None = all)"
    )
    minimize_metrics: Optional[List[str]] = Field(
        None,
        description="Metrics where lower is better (default: loss, error, mse, mae, rmse)",
    )
    significance_level: float = Field(
        0.05, description="P-value threshold for significance", ge=0.001, le=0.1
    )


class TrendAnalysisResponse(BaseModel):
    """Response model for trend analysis."""

    project: str
    total_runs: int
    trends: Dict[str, Any]
    summary: Dict[str, Any]
    total_metrics: int
    analyzed_metrics: int


@router.post("/analyze", response_model=TrendAnalysisResponse)
async def analyze_trends(request: TrendAnalysisRequest) -> TrendAnalysisResponse:
    """Analyze performance trends for experiment metrics.

    This endpoint analyzes time-series data from experiments to identify
    trends, detect improvement/degradation patterns, and predict future performance.

    Args:
        request: Trend analysis parameters

    Returns:
        Trend analysis results for each metric

    Raises:
        HTTPException: If project not found or insufficient data
    """
    db: Session = next(get_db())

    try:
        # Query runs from database
        runs = db.query(Run).filter(Run.project == request.project).all()

        if not runs:
            raise HTTPException(
                status_code=404, detail=f"No runs found for project: {request.project}"
            )

        if len(runs) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient runs for trend analysis (need at least 3, got {len(runs)})",
            )

        # Convert to dictionaries
        runs_data = [
            {
                "run_name": run.run_name,
                "started_at": run.started_at,
                "metrics": {"final": run.final_metrics or {}},
                "final_metrics": run.final_metrics or {},
            }
            for run in runs
            if run.started_at  # Only include runs with timestamps
        ]

        if len(runs_data) < 3:
            raise HTTPException(
                status_code=400,
                detail="Insufficient runs with timestamps for trend analysis",
            )

        # Initialize analyzer
        analyzer = TrendAnalyzer(significance_level=request.significance_level)

        # Determine metrics to analyze
        metrics = request.metrics
        if metrics is None:
            # Auto-detect numeric metrics
            all_metrics = set()
            for run in runs_data:
                final_metrics = run.get("final_metrics") or run.get("metrics", {}).get(
                    "final", {}
                )
                for key, value in final_metrics.items():
                    # Convert numpy types to Python types and check
                    if isinstance(value, (int, float)) and not isinstance(
                        value, (bool, np.bool_)
                    ):
                        all_metrics.add(key)
            metrics = sorted(all_metrics)

        # Analyze trends
        result = analyzer.analyze_multiple_metrics(
            runs_data, metrics, request.minimize_metrics
        )

        return TrendAnalysisResponse(
            project=request.project,
            total_runs=len(runs),
            trends=result["trends"],
            summary=result["summary"],
            total_metrics=result["total_metrics"],
            analyzed_metrics=result["analyzed_metrics"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {e}")
    finally:
        db.close()


@router.get("/{project}/summary")
async def get_trend_summary(
    project: str,
    significance_level: float = Query(
        0.05, description="P-value threshold", ge=0.001, le=0.1
    ),
) -> Dict[str, Any]:
    """Get trend summary for a project.

    This is a simplified endpoint that returns only the summary
    without detailed per-metric results.

    Args:
        project: Project name
        significance_level: Statistical significance threshold

    Returns:
        Summary of trends across all metrics
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
                "started_at": run.started_at,
                "metrics": {"final": run.final_metrics or {}},
                "final_metrics": run.final_metrics or {},
            }
            for run in runs
            if run.started_at
        ]

        if len(runs_data) < 3:
            return {
                "project": project,
                "total_runs": len(runs),
                "message": "Insufficient runs for trend analysis (need at least 3 with timestamps)",
                "summary": {
                    "improving": [],
                    "degrading": [],
                    "stable": [],
                    "significant_trends": [],
                },
            }

        # Auto-detect metrics
        all_metrics = set()
        for run in runs_data:
            final_metrics = run.get("final_metrics") or run.get("metrics", {}).get(
                "final", {}
            )
            for key, value in final_metrics.items():
                # Convert numpy types to Python types and check
                if isinstance(value, (int, float)) and not isinstance(
                    value, (bool, np.bool_)
                ):
                    all_metrics.add(key)

        metrics = sorted(all_metrics)

        analyzer = TrendAnalyzer(significance_level=significance_level)
        result = analyzer.analyze_multiple_metrics(runs_data, metrics)

        # Calculate overall health score
        total = len(result["trends"])
        improving = len(result["summary"]["improving"])
        degrading = len(result["summary"]["degrading"])

        if total > 0:
            health_score = (improving - degrading) / total * 100
        else:
            health_score = 0

        return {
            "project": project,
            "total_runs": len(runs),
            "analyzed_metrics": result["analyzed_metrics"],
            "summary": result["summary"],
            "health_score": health_score,
            "significance_level": significance_level,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trend summary: {e}")
    finally:
        db.close()


@router.get("/{project}/{metric}/details")
async def get_metric_trend_details(
    project: str,
    metric: str,
    significance_level: float = Query(0.05, ge=0.001, le=0.1),
) -> Dict[str, Any]:
    """Get detailed trend analysis for a specific metric.

    Args:
        project: Project name
        metric: Metric name
        significance_level: Statistical significance threshold

    Returns:
        Detailed trend analysis with predictions
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
                "started_at": run.started_at,
                "metrics": {"final": run.final_metrics or {}},
                "final_metrics": run.final_metrics or {},
            }
            for run in runs
            if run.started_at
        ]

        analyzer = TrendAnalyzer(significance_level=significance_level)
        result = analyzer.analyze_multiple_metrics(runs_data, [metric])

        if metric not in result["trends"]:
            raise HTTPException(
                status_code=404,
                detail=f"Metric '{metric}' not found or has insufficient data",
            )

        return {
            "project": project,
            "metric": metric,
            "trend": result["trends"][metric],
            "runs_analyzed": len(runs_data),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get metric trend details: {e}"
        )
    finally:
        db.close()
