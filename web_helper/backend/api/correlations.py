"""API endpoints for hyperparameter-metric correlation analysis."""

from pathlib import Path
from typing import Any, List

import numpy as np
import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from scipy import stats
from sqlalchemy import select

from ..models.database import get_db
from ..models.run import Run

router = APIRouter(prefix="/correlations")


class CorrelationResult(BaseModel):
    """Single hyperparameter-metric correlation result."""

    hyperparam_name: str
    metric_name: str
    correlation: float
    p_value: float
    method: str  # 'pearson', 'spearman', 'point_biserial'
    sample_size: int
    hyperparam_type: str  # 'numeric', 'categorical', 'boolean'


class CorrelationAnalysisResponse(BaseModel):
    """Response containing correlation analysis results."""

    project: str
    metric: str
    correlations: List[CorrelationResult]
    total_runs: int
    min_sample_threshold: int = 10


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


def get_hyperparam_type(values: List[Any]) -> str:
    """Determine hyperparameter type from values."""
    non_none_values = [v for v in values if v is not None]
    if not non_none_values:
        return "unknown"

    # Check if all values are numeric
    try:
        float_values = [float(v) for v in non_none_values]
        # Check if values have variation
        if len(set(float_values)) > 1:
            return "numeric"
        return "constant"
    except (ValueError, TypeError):
        pass

    # Check if boolean
    unique_values = set(non_none_values)
    if unique_values.issubset({True, False, "true", "false", "True", "False"}):
        return "boolean"

    # Otherwise categorical
    if len(unique_values) <= 10:  # Reasonable threshold for categories
        return "categorical"

    return "unknown"


def calculate_correlation(
    hyperparam_values: List[Any], metric_values: List[float], hyperparam_type: str
) -> tuple[float, float, str]:
    """Calculate correlation between hyperparameter and metric.

    Returns: (correlation, p_value, method)
    """
    # Filter out None values
    valid_pairs = [
        (h, m)
        for h, m in zip(hyperparam_values, metric_values)
        if h is not None and m is not None and not np.isnan(m)
    ]

    if len(valid_pairs) < 10:  # Minimum sample size for statistical significance
        return 0.0, 1.0, "insufficient_data"

    hyperparams, metrics = zip(*valid_pairs)

    if hyperparam_type == "numeric":
        # Convert to numeric
        try:
            h_numeric = np.array([float(h) for h in hyperparams])
            m_numeric = np.array(metrics)

            # Check for variation
            if np.std(h_numeric) == 0 or np.std(m_numeric) == 0:
                return 0.0, 1.0, "no_variation"

            # Use Spearman for robustness against outliers
            corr, p_value = stats.spearmanr(h_numeric, m_numeric)

            if np.isnan(corr):
                return 0.0, 1.0, "calculation_error"

            return float(corr), float(p_value), "spearman"
        except Exception:
            return 0.0, 1.0, "error"

    elif hyperparam_type == "boolean":
        # Convert to binary (0/1)
        try:
            h_binary = []
            for h in hyperparams:
                if h in [True, "true", "True", 1]:
                    h_binary.append(1)
                elif h in [False, "false", "False", 0]:
                    h_binary.append(0)
                else:
                    h_binary.append(int(bool(h)))

            h_binary = np.array(h_binary)
            m_numeric = np.array(metrics)

            # Use point-biserial correlation for binary vs continuous
            corr, p_value = stats.pointbiserialr(h_binary, m_numeric)

            if np.isnan(corr):
                return 0.0, 1.0, "calculation_error"

            return float(corr), float(p_value), "point_biserial"
        except Exception:
            return 0.0, 1.0, "error"

    elif hyperparam_type == "categorical":
        # For categorical, use ANOVA F-statistic converted to correlation-like metric
        try:
            # Group metrics by category
            groups = {}
            for h, m in zip(hyperparams, metrics):
                if h not in groups:
                    groups[h] = []
                groups[h].append(m)

            # Need at least 2 groups
            if len(groups) < 2:
                return 0.0, 1.0, "insufficient_groups"

            # Perform one-way ANOVA
            group_values = [np.array(v) for v in groups.values()]
            f_stat, p_value = stats.f_oneway(*group_values)

            if np.isnan(f_stat):
                return 0.0, 1.0, "calculation_error"

            # Convert F-statistic to correlation coefficient approximation
            # Using eta-squared (effect size)
            df_between = len(groups) - 1
            df_within = len(metrics) - len(groups)
            eta_squared = (f_stat * df_between) / (f_stat * df_between + df_within)
            correlation = np.sqrt(eta_squared) if eta_squared >= 0 else 0.0

            return float(correlation), float(p_value), "anova"
        except Exception:
            return 0.0, 1.0, "error"

    return 0.0, 1.0, "unknown_type"


@router.get("/{project}", response_model=CorrelationAnalysisResponse)
async def get_hyperparam_correlations(
    project: str,
    metric: str = Query(..., description="Metric name to correlate against"),
    use_max: bool = Query(
        True, description="Use max metric values (True) or min (False)"
    ),
    min_samples: int = Query(10, description="Minimum sample size for correlation"),
):
    """Calculate correlations between hyperparameters and a target metric.

    Returns the top correlations ranked by absolute correlation value.
    """
    db = next(get_db())

    try:
        # Get all runs for the project
        statement = select(Run).where(Run.project == project)
        runs = db.execute(statement).scalars().all()

        if len(runs) < min_samples:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: only {len(runs)} runs found (minimum {min_samples} required)",
            )

        # Extract hyperparameters and metrics
        run_data = []
        for run in runs:
            # Get metric value
            metrics_source = "max_metrics" if use_max else "min_metrics"
            metrics = getattr(run, metrics_source, {}) or {}

            metric_value = metrics.get(metric)
            if metric_value is None or not isinstance(metric_value, (int, float)):
                continue

            # Get hyperparameters from config
            hyperparams = extract_hyperparameters_from_config(run.config_path or "")

            run_data.append(
                {"metric_value": float(metric_value), "hyperparams": hyperparams}
            )

        if len(run_data) < min_samples:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient valid data: only {len(run_data)} runs with metric '{metric}' (minimum {min_samples} required)",
            )

        # Collect all hyperparameter names
        all_hyperparam_names = set()
        for rd in run_data:
            all_hyperparam_names.update(rd["hyperparams"].keys())

        # Calculate correlations for each hyperparameter
        correlations = []
        for hyperparam_name in sorted(all_hyperparam_names):
            # Extract values
            hyperparam_values = []
            metric_values = []

            for rd in run_data:
                h_value = rd["hyperparams"].get(hyperparam_name)
                m_value = rd["metric_value"]
                hyperparam_values.append(h_value)
                metric_values.append(m_value)

            # Determine type
            hyperparam_type = get_hyperparam_type(hyperparam_values)

            # Skip constants and unknowns
            if hyperparam_type in ["constant", "unknown"]:
                continue

            # Calculate correlation
            correlation, p_value, method = calculate_correlation(
                hyperparam_values, metric_values, hyperparam_type
            )

            # Skip if insufficient data or error
            if method in [
                "insufficient_data",
                "no_variation",
                "error",
                "calculation_error",
            ]:
                continue

            correlations.append(
                CorrelationResult(
                    hyperparam_name=hyperparam_name,
                    metric_name=metric,
                    correlation=correlation,
                    p_value=p_value,
                    method=method,
                    sample_size=len([v for v in hyperparam_values if v is not None]),
                    hyperparam_type=hyperparam_type,
                )
            )

        # Sort by absolute correlation (strongest first)
        correlations.sort(key=lambda x: abs(x.correlation), reverse=True)

        return CorrelationAnalysisResponse(
            project=project,
            metric=metric,
            correlations=correlations,
            total_runs=len(runs),
            min_sample_threshold=min_samples,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculating correlations: {str(e)}"
        )
