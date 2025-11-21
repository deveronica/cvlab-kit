"""API endpoints for best run recommendations."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..models.database import get_db
from ..models.run import Run
from ..services.run_recommender import RecommendationStrategy, RunRecommender

router = APIRouter(prefix="/recommendations")


def extract_hyperparameters_from_config(config_path: str) -> dict:
    """Extract hyperparameters from YAML config file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return {}

        with open(config_file) as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)

        if not isinstance(config_data, dict):
            return {}

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
                return str(value)

        hyperparams = {}
        for key, value in config_data.items():
            if key.startswith("_"):
                continue
            hyperparams[key] = sanitize_value(value)

        return hyperparams

    except Exception:
        return {}


class RecommendationRequest(BaseModel):
    """Request model for run recommendations."""

    project: str = Field(..., description="Project name")
    objectives: Optional[List[str]] = Field(
        None, description="Metrics to optimize (None = auto-select)"
    )
    strategy: RecommendationStrategy = Field(
        "pareto", description="Recommendation strategy (pareto, weighted, rank)"
    )
    weights: Optional[List[float]] = Field(
        None, description="Weights for weighted strategy (must match objectives length)"
    )
    minimize: Optional[List[bool]] = Field(
        None,
        description="For each objective, True if lower is better (None = auto-detect)",
    )
    top_k: int = Field(5, description="Number of recommendations", ge=1, le=20)


class RecommendationResponse(BaseModel):
    """Response model for run recommendations."""

    recommendations: List[Dict[str, Any]]
    strategy: str
    total_runs: int
    objectives: List[str]
    minimize: Optional[List[bool]]
    pareto_optimal_count: Optional[int] = None
    weights: Optional[List[float]] = None


@router.post("/find", response_model=RecommendationResponse)
async def find_best_runs(request: RecommendationRequest) -> RecommendationResponse:
    """Find best experiment runs using multi-objective optimization.

    This endpoint identifies optimal runs based on multiple metrics,
    finding runs that represent the best trade-offs (Pareto frontier).

    Args:
        request: Recommendation parameters

    Returns:
        List of recommended runs with scores/rankings

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

        # Convert to dictionaries with hyperparameters from config
        runs_data = []
        for run in runs:
            hyperparams = {}
            if run.config_path and Path(run.config_path).exists():
                hyperparams = extract_hyperparameters_from_config(run.config_path)

            runs_data.append(
                {
                    "run_name": run.run_name,
                    "metrics": {"final": run.final_metrics or {}},
                    "final_metrics": run.final_metrics or {},
                    "hyperparameters": hyperparams,
                }
            )

        # Auto-select objectives if not specified
        objectives = request.objectives
        if objectives is None:
            # Find most common metrics
            metric_counts = {}
            for run in runs_data:
                final_metrics = run.get("final_metrics") or run.get("metrics", {}).get(
                    "final", {}
                )
                for metric, value in final_metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metric_counts[metric] = metric_counts.get(metric, 0) + 1

            if not metric_counts:
                raise HTTPException(
                    status_code=400,
                    detail="No numeric metrics found in runs",
                )

            # Select top 2-3 most common metrics
            objectives = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)
            objectives = [obj for obj, _ in objectives[:3]]

        # Validate weights
        if request.weights and len(request.weights) != len(objectives):
            raise HTTPException(
                status_code=400,
                detail=f"Weights length ({len(request.weights)}) must match objectives length ({len(objectives)})",
            )

        # Validate minimize
        if request.minimize and len(request.minimize) != len(objectives):
            raise HTTPException(
                status_code=400,
                detail=f"Minimize length ({len(request.minimize)}) must match objectives length ({len(objectives)})",
            )

        # Initialize recommender
        recommender = RunRecommender(strategy=request.strategy)

        # Get recommendations
        result = recommender.recommend_best_runs(
            runs_data,
            objectives,
            strategy=request.strategy,
            weights=request.weights,
            minimize=request.minimize,
            top_k=request.top_k,
        )

        return RecommendationResponse(
            recommendations=result["recommendations"],
            strategy=result["strategy"],
            total_runs=result["total_runs"],
            objectives=result.get("objectives", objectives),
            minimize=result.get("minimize"),
            pareto_optimal_count=result.get("pareto_optimal_count"),
            weights=result.get("weights"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find best runs: {e}")
    finally:
        db.close()


@router.get("/{project}/pareto")
async def get_pareto_frontier(
    project: str,
    objectives: List[str] = Query(..., description="Metrics to optimize"),
) -> Dict[str, Any]:
    """Get Pareto-optimal runs for a project.

    This is a simplified endpoint specifically for Pareto optimization.

    Args:
        project: Project name
        objectives: Metrics to optimize (comma-separated)

    Returns:
        Pareto-optimal runs
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
                "metrics": {"final": run.final_metrics or {}},
                "final_metrics": run.final_metrics or {},
            }
            for run in runs
        ]

        recommender = RunRecommender(strategy="pareto")
        pareto_runs = recommender.find_pareto_frontier(runs_data, objectives)

        pareto_optimal = [r for r in pareto_runs if r["is_pareto_optimal"]]

        return {
            "project": project,
            "objectives": objectives,
            "pareto_optimal_runs": pareto_optimal,
            "total_pareto_optimal": len(pareto_optimal),
            "total_runs": len(runs),
            "pareto_percentage": (len(pareto_optimal) / len(runs) * 100 if runs else 0),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get Pareto frontier: {e}"
        )
    finally:
        db.close()


@router.get("/{project}/summary")
async def get_recommendation_summary(
    project: str,
    top_k: int = Query(3, description="Number of recommendations", ge=1, le=10),
) -> Dict[str, Any]:
    """Get recommendation summary for a project.

    This endpoint provides a quick overview of the best runs
    using auto-selected objectives.

    Args:
        project: Project name
        top_k: Number of recommendations

    Returns:
        Quick recommendation summary
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
                "metrics": {"final": run.final_metrics or {}},
                "final_metrics": run.final_metrics or {},
            }
            for run in runs
        ]

        # Auto-select objectives
        metric_counts = {}
        for run in runs_data:
            final_metrics = run.get("final_metrics") or run.get("metrics", {}).get(
                "final", {}
            )
            for metric, value in final_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1

        if not metric_counts:
            return {
                "project": project,
                "total_runs": len(runs),
                "message": "No numeric metrics found",
                "recommendations": [],
            }

        objectives = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)
        objectives = [obj for obj, _ in objectives[:3]]

        # Get Pareto recommendations
        recommender = RunRecommender(strategy="pareto")
        result = recommender.recommend_best_runs(runs_data, objectives, top_k=top_k)

        return {
            "project": project,
            "total_runs": len(runs),
            "auto_selected_objectives": objectives,
            "pareto_optimal_count": result.get("pareto_optimal_count", 0),
            "top_recommendations": result["recommendations"][:top_k],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get recommendation summary: {e}"
        )
    finally:
        db.close()
