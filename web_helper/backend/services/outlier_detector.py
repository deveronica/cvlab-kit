"""Outlier detection service for identifying anomalous experiment runs.

This service provides multiple statistical methods for detecting outliers
in hyperparameters and metrics across experiment runs.
"""

from typing import Any, Dict, List, Literal, Optional

import numpy as np

OutlierMethod = Literal["iqr", "zscore", "modified_zscore", "isolation_forest"]


class OutlierDetector:
    """Detects outliers in experiment data using various statistical methods."""

    def __init__(
        self,
        method: OutlierMethod = "iqr",
        threshold: float = 1.5,
        min_samples: int = 3,
    ):
        """Initialize outlier detector.

        Args:
            method: Detection method ('iqr', 'zscore', 'modified_zscore', 'isolation_forest')
            threshold: Threshold for outlier detection (method-specific)
                - IQR: multiplier for IQR range (default: 1.5)
                - Z-score: number of standard deviations (default: 3.0)
                - Modified Z-score: MAD-based threshold (default: 3.5)
            min_samples: Minimum number of samples required for detection
        """
        self.method = method
        self.threshold = threshold
        self.min_samples = min_samples

    def detect_outliers(
        self, values: List[float], run_names: List[str]
    ) -> Dict[str, Any]:
        """Detect outliers in a list of values.

        Args:
            values: List of numeric values to analyze
            run_names: Corresponding run names for each value

        Returns:
            Dictionary containing:
                - outlier_indices: List of indices where outliers were detected
                - outlier_runs: List of run names identified as outliers
                - scores: Outlier scores for each value
                - statistics: Statistical summary (mean, std, bounds, etc.)
                - method: Detection method used
        """
        if len(values) < self.min_samples:
            return {
                "outlier_indices": [],
                "outlier_runs": [],
                "scores": [],
                "statistics": {},
                "method": self.method,
                "message": f"Insufficient samples (need {self.min_samples}, got {len(values)})",
            }

        values_array = np.array(values, dtype=float)

        # Remove NaN values
        valid_mask = ~np.isnan(values_array)
        valid_values = values_array[valid_mask]
        valid_runs = [run_names[i] for i, valid in enumerate(valid_mask) if valid]

        if len(valid_values) < self.min_samples:
            return {
                "outlier_indices": [],
                "outlier_runs": [],
                "scores": [],
                "statistics": {},
                "method": self.method,
                "message": "Too many NaN values",
            }

        # Select detection method
        if self.method == "iqr":
            result = self._detect_iqr(valid_values, valid_runs)
        elif self.method == "zscore":
            result = self._detect_zscore(valid_values, valid_runs)
        elif self.method == "modified_zscore":
            result = self._detect_modified_zscore(valid_values, valid_runs)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")

        return result

    def _detect_iqr(self, values: np.ndarray, run_names: List[str]) -> Dict[str, Any]:
        """Detect outliers using Interquartile Range (IQR) method.

        Outliers are values that fall below Q1 - threshold * IQR
        or above Q3 + threshold * IQR.
        """
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr

        # Identify outliers
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_runs = [run_names[i] for i in outlier_indices]

        # Calculate outlier scores (distance from bounds in IQR units)
        scores = []
        for value in values:
            if value < lower_bound:
                score = (lower_bound - value) / iqr
            elif value > upper_bound:
                score = (value - upper_bound) / iqr
            else:
                score = 0.0
            scores.append(float(score))

        return {
            "outlier_indices": outlier_indices,
            "outlier_runs": outlier_runs,
            "scores": scores,
            "statistics": {
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
            },
            "method": "iqr",
            "threshold": self.threshold,
        }

    def _detect_zscore(
        self, values: np.ndarray, run_names: List[str]
    ) -> Dict[str, Any]:
        """Detect outliers using Z-score method.

        Outliers are values with |z-score| > threshold.
        """
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return {
                "outlier_indices": [],
                "outlier_runs": [],
                "scores": [0.0] * len(values),
                "statistics": {"mean": float(mean), "std": 0.0},
                "method": "zscore",
                "message": "Zero standard deviation",
            }

        z_scores = np.abs((values - mean) / std)
        outlier_mask = z_scores > self.threshold
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_runs = [run_names[i] for i in outlier_indices]

        return {
            "outlier_indices": outlier_indices,
            "outlier_runs": outlier_runs,
            "scores": z_scores.tolist(),
            "statistics": {
                "mean": float(mean),
                "std": float(std),
                "threshold": float(self.threshold),
            },
            "method": "zscore",
            "threshold": self.threshold,
        }

    def _detect_modified_zscore(
        self, values: np.ndarray, run_names: List[str]
    ) -> Dict[str, Any]:
        """Detect outliers using Modified Z-score method (MAD-based).

        More robust than standard Z-score as it uses median and MAD
        (Median Absolute Deviation) instead of mean and std.
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad == 0:
            return {
                "outlier_indices": [],
                "outlier_runs": [],
                "scores": [0.0] * len(values),
                "statistics": {"median": float(median), "mad": 0.0},
                "method": "modified_zscore",
                "message": "Zero MAD",
            }

        modified_z_scores = 0.6745 * (values - median) / mad
        outlier_mask = np.abs(modified_z_scores) > self.threshold
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_runs = [run_names[i] for i in outlier_indices]

        return {
            "outlier_indices": outlier_indices,
            "outlier_runs": outlier_runs,
            "scores": np.abs(modified_z_scores).tolist(),
            "statistics": {
                "median": float(median),
                "mad": float(mad),
                "threshold": float(self.threshold),
            },
            "method": "modified_zscore",
            "threshold": self.threshold,
        }

    def analyze_multiple_columns(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        column_type: Literal["hyperparam", "metric"],
    ) -> Dict[str, Any]:
        """Analyze multiple columns for outliers.

        Args:
            data: List of run data dictionaries
            columns: Column names to analyze
            column_type: Type of columns ('hyperparam' or 'metric')

        Returns:
            Dictionary mapping column names to outlier detection results
        """
        results = {}
        run_names = [run["run_name"] for run in data]

        for col in columns:
            # Extract values for this column
            values = []
            for run in data:
                if column_type == "hyperparam":
                    value = run.get("hyperparameters", {}).get(col)
                else:  # metric
                    value = run.get("metrics", {}).get("final", {}).get(col)

                if value is not None and isinstance(value, (int, float)):
                    values.append(float(value))
                else:
                    values.append(np.nan)

            # Detect outliers for this column
            result = self.detect_outliers(values, run_names)
            results[col] = result

        # Calculate overall outlier summary
        all_outlier_runs = set()
        for result in results.values():
            all_outlier_runs.update(result.get("outlier_runs", []))

        outlier_counts = {}
        for run_name in all_outlier_runs:
            count = sum(
                1
                for result in results.values()
                if run_name in result.get("outlier_runs", [])
            )
            outlier_counts[run_name] = count

        return {
            "column_results": results,
            "summary": {
                "total_outlier_runs": len(all_outlier_runs),
                "outlier_runs": list(all_outlier_runs),
                "outlier_counts": outlier_counts,
                "method": self.method,
                "threshold": self.threshold,
            },
        }


def detect_outliers_in_runs(
    runs: List[Dict[str, Any]],
    hyperparam_columns: Optional[List[str]] = None,
    metric_columns: Optional[List[str]] = None,
    method: OutlierMethod = "iqr",
    threshold: float = 1.5,
) -> Dict[str, Any]:
    """Convenience function to detect outliers in experiment runs.

    Args:
        runs: List of run data dictionaries
        hyperparam_columns: Hyperparameter columns to analyze (None = all numeric)
        metric_columns: Metric columns to analyze (None = all numeric)
        method: Detection method
        threshold: Detection threshold

    Returns:
        Outlier detection results for all specified columns
    """
    detector = OutlierDetector(method=method, threshold=threshold)

    results = {"hyperparameters": {}, "metrics": {}}

    # Analyze hyperparameters
    if hyperparam_columns:
        hp_results = detector.analyze_multiple_columns(
            runs, hyperparam_columns, "hyperparam"
        )
        results["hyperparameters"] = hp_results

    # Analyze metrics
    if metric_columns:
        metric_results = detector.analyze_multiple_columns(
            runs, metric_columns, "metric"
        )
        results["metrics"] = metric_results

    return results
