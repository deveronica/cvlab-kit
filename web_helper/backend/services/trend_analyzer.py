"""Trend analysis service for identifying performance patterns over time.

This service analyzes time-series data from experiments to identify trends,
detect improvement/degradation patterns, and predict future performance.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from scipy import stats

TrendDirection = Literal["improving", "degrading", "stable"]
TrendStrength = Literal["strong", "moderate", "weak", "none"]


class TrendAnalyzer:
    """Analyzes performance trends across experiment runs over time."""

    def __init__(
        self,
        significance_level: float = 0.05,
        stable_threshold: float = 0.01,
    ):
        """Initialize trend analyzer.

        Args:
            significance_level: P-value threshold for statistical significance
            stable_threshold: Slope threshold for considering trend as stable
        """
        self.significance_level = significance_level
        self.stable_threshold = stable_threshold

    def analyze_metric_trend(
        self,
        timestamps: List[datetime],
        values: List[float],
        metric_name: str,
        minimize: bool = False,
    ) -> Dict[str, Any]:
        """Analyze trend for a single metric over time.

        Args:
            timestamps: Run timestamps (datetime objects)
            values: Metric values corresponding to each timestamp
            metric_name: Name of the metric being analyzed
            minimize: True if lower values are better (e.g., loss)

        Returns:
            Dictionary containing:
                - direction: 'improving', 'degrading', or 'stable'
                - strength: 'strong', 'moderate', 'weak', or 'none'
                - slope: Linear regression slope
                - r_squared: Coefficient of determination
                - p_value: Statistical significance
                - prediction: Predicted next value
                - confidence_interval: 95% confidence interval
        """
        if len(timestamps) < 3:
            return {
                "metric": metric_name,
                "direction": "stable",
                "strength": "none",
                "message": "Insufficient data (need at least 3 points)",
            }

        # Convert timestamps to numeric (seconds since first run)
        first_time = min(timestamps)
        x = np.array([(t - first_time).total_seconds() for t in timestamps])
        y = np.array(values)

        # Remove NaN values
        valid_mask = ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) < 3:
            return {
                "metric": metric_name,
                "direction": "stable",
                "strength": "none",
                "message": "Too many NaN values",
            }

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
        r_squared = r_value**2

        # Determine trend direction
        if abs(slope) < self.stable_threshold:
            direction = "stable"
        elif minimize:
            # For metrics to minimize (e.g., loss)
            direction = "improving" if slope < 0 else "degrading"
        else:
            # For metrics to maximize (e.g., accuracy)
            direction = "improving" if slope > 0 else "degrading"

        # Determine trend strength based on R² and p-value
        if p_value > self.significance_level:
            strength = "none"  # Not statistically significant
        elif r_squared > 0.7:
            strength = "strong"
        elif r_squared > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        # Predict next value (extrapolate)
        last_x = x_valid[-1]
        avg_interval = np.mean(np.diff(x_valid)) if len(x_valid) > 1 else 0
        next_x = last_x + avg_interval
        prediction = slope * next_x + intercept

        # Calculate 95% confidence interval
        n = len(x_valid)
        t_val = stats.t.ppf(0.975, n - 2)  # 95% confidence
        se_prediction = std_err * np.sqrt(
            1
            + 1 / n
            + (next_x - np.mean(x_valid)) ** 2
            / np.sum((x_valid - np.mean(x_valid)) ** 2)
        )
        ci_lower = prediction - t_val * se_prediction
        ci_upper = prediction + t_val * se_prediction

        # Calculate improvement rate (percentage change per unit time)
        mean_y = np.mean(y_valid)
        improvement_rate = (slope / mean_y * 100) if mean_y != 0 else 0

        return {
            "metric": metric_name,
            "direction": direction,
            "strength": strength,
            "slope": float(slope),
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "is_significant": bool(p_value < self.significance_level),
            "prediction": {
                "value": float(prediction),
                "confidence_interval": [float(ci_lower), float(ci_upper)],
                "confidence_level": 0.95,
            },
            "improvement_rate": float(improvement_rate),
            "data_points": int(n),
            "minimize": bool(minimize),
        }

    def analyze_multiple_metrics(
        self,
        runs: List[Dict[str, Any]],
        metric_names: List[str],
        minimize_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze trends for multiple metrics.

        Args:
            runs: List of run dictionaries with 'started_at' and 'metrics'
            metric_names: Metrics to analyze
            minimize_metrics: Metrics where lower is better (default: ['loss', 'error'])

        Returns:
            Dictionary with trend analysis for each metric
        """
        if minimize_metrics is None:
            minimize_metrics = ["loss", "error", "mse", "mae", "rmse"]

        # Sort runs by timestamp
        sorted_runs = sorted(
            runs,
            key=lambda r: datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
            if isinstance(r["started_at"], str)
            else r["started_at"],
        )

        results = {}
        summary = {
            "improving": [],
            "degrading": [],
            "stable": [],
            "significant_trends": [],
        }

        for metric in metric_names:
            # Extract timestamps and values
            timestamps = []
            values = []

            for run in sorted_runs:
                if not run.get("started_at"):
                    continue

                # Parse timestamp
                if isinstance(run["started_at"], str):
                    ts = datetime.fromisoformat(
                        run["started_at"].replace("Z", "+00:00")
                    )
                else:
                    ts = run["started_at"]

                # Get metric value
                metric_val = None
                if run.get("metrics") and run["metrics"].get("final"):
                    metric_val = run["metrics"]["final"].get(metric)
                elif run.get("final_metrics"):
                    metric_val = run["final_metrics"].get(metric)

                if metric_val is not None and isinstance(metric_val, (int, float)):
                    timestamps.append(ts)
                    values.append(float(metric_val))

            if len(timestamps) < 3:
                continue

            # Determine if this metric should be minimized
            minimize = any(pattern in metric.lower() for pattern in minimize_metrics)

            # Analyze trend
            trend = self.analyze_metric_trend(timestamps, values, metric, minimize)
            results[metric] = trend

            # Update summary
            if trend["direction"] == "improving":
                summary["improving"].append(metric)
            elif trend["direction"] == "degrading":
                summary["degrading"].append(metric)
            else:
                summary["stable"].append(metric)

            if trend.get("is_significant", False):
                summary["significant_trends"].append(
                    {
                        "metric": metric,
                        "direction": trend["direction"],
                        "strength": trend["strength"],
                        "improvement_rate": trend["improvement_rate"],
                    }
                )

        return {
            "trends": results,
            "summary": summary,
            "total_metrics": len(metric_names),
            "analyzed_metrics": len(results),
        }

    def compare_trend_periods(
        self,
        runs: List[Dict[str, Any]],
        metric: str,
        split_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """Compare trends between early and recent runs.

        Args:
            runs: List of run dictionaries
            metric: Metric to analyze
            split_ratio: Where to split (0.5 = middle)

        Returns:
            Comparison of early vs recent trends
        """
        # Sort by timestamp
        sorted_runs = sorted(
            runs,
            key=lambda r: datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
            if isinstance(r["started_at"], str)
            else r["started_at"],
        )

        split_idx = int(len(sorted_runs) * split_ratio)
        early_runs = sorted_runs[:split_idx]
        recent_runs = sorted_runs[split_idx:]

        # Analyze both periods
        early_analysis = self.analyze_multiple_metrics(early_runs, [metric])
        recent_analysis = self.analyze_multiple_metrics(recent_runs, [metric])

        early_trend = early_analysis["trends"].get(metric, {})
        recent_trend = recent_analysis["trends"].get(metric, {})

        # Compare
        acceleration = "none"
        if early_trend and recent_trend:
            early_slope = early_trend.get("slope", 0)
            recent_slope = recent_trend.get("slope", 0)

            if abs(recent_slope) > abs(early_slope) * 1.5:
                if recent_slope > 0 and early_slope > 0:
                    acceleration = "accelerating_improvement"
                elif recent_slope < 0 and early_slope < 0:
                    acceleration = "accelerating_degradation"
            elif abs(recent_slope) < abs(early_slope) * 0.5:
                acceleration = "decelerating"

        return {
            "metric": metric,
            "early_period": {
                "runs": len(early_runs),
                "trend": early_trend,
            },
            "recent_period": {
                "runs": len(recent_runs),
                "trend": recent_trend,
            },
            "acceleration": acceleration,
            "split_ratio": split_ratio,
        }


def analyze_project_trends(
    runs: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    minimize_metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convenience function to analyze trends for a project.

    Args:
        runs: List of run dictionaries
        metrics: Metrics to analyze (None = all numeric metrics)
        minimize_metrics: Metrics where lower is better

    Returns:
        Trend analysis results
    """
    analyzer = TrendAnalyzer()

    # Auto-detect metrics if not specified
    if metrics is None:
        all_metrics = set()
        for run in runs:
            if run.get("metrics") and run["metrics"].get("final"):
                all_metrics.update(run["metrics"]["final"].keys())
            elif run.get("final_metrics"):
                all_metrics.update(run["final_metrics"].keys())
        metrics = sorted(all_metrics)

    return analyzer.analyze_multiple_metrics(runs, metrics, minimize_metrics)
