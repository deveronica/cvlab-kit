"""Best run recommendation service using Pareto optimality.

This service identifies optimal experiment runs based on multiple objectives,
finding runs that represent the best trade-offs (Pareto frontier).
"""

from typing import Any, Dict, List, Literal, Optional

import numpy as np

RecommendationStrategy = Literal["pareto", "weighted", "rank"]


class RunRecommender:
    """Recommends best experiment runs using multi-objective optimization."""

    def __init__(self, strategy: RecommendationStrategy = "pareto"):
        """Initialize run recommender.

        Args:
            strategy: Recommendation strategy
                - 'pareto': Pareto-optimal runs (non-dominated solutions)
                - 'weighted': Weighted sum of normalized objectives
                - 'rank': Rank-based selection
        """
        self.strategy = strategy

    def find_pareto_frontier(
        self,
        runs: List[Dict[str, Any]],
        objectives: List[str],
        minimize: Optional[List[bool]] = None,
    ) -> List[Dict[str, Any]]:
        """Find Pareto-optimal runs (non-dominated solutions).

        A run is Pareto-optimal if no other run is better in all objectives.

        Args:
            runs: List of run dictionaries
            objectives: Metric names to optimize
            minimize: For each objective, True if lower is better

        Returns:
            List of Pareto-optimal runs with dominance info
        """
        if minimize is None:
            # Auto-detect: minimize if metric name contains these keywords
            minimize_keywords = ["loss", "error", "mse", "mae", "rmse", "time", "size"]
            minimize = [
                any(kw in obj.lower() for kw in minimize_keywords) for obj in objectives
            ]

        # Extract objective values
        runs_with_values = []
        for run in runs:
            values = []
            valid = True

            for obj in objectives:
                val = None
                if run.get("metrics") and run["metrics"].get("final"):
                    val = run["metrics"]["final"].get(obj)
                elif run.get("final_metrics"):
                    val = run["final_metrics"].get(obj)

                if val is None or not isinstance(val, (int, float)) or np.isnan(val):
                    valid = False
                    break

                values.append(float(val))

            if valid:
                runs_with_values.append(
                    {
                        "run": run,
                        "values": values,
                    }
                )

        if len(runs_with_values) == 0:
            return []

        # Convert to numpy array
        values_array = np.array([r["values"] for r in runs_with_values])

        # Flip signs for minimization objectives
        for i, should_minimize in enumerate(minimize):
            if should_minimize:
                values_array[:, i] = -values_array[:, i]

        # Find Pareto frontier
        n = len(runs_with_values)
        is_pareto = np.ones(n, dtype=bool)

        for i in range(n):
            if not is_pareto[i]:
                continue

            # Compare with all other runs
            for j in range(i + 1, n):
                if not is_pareto[j]:
                    continue

                # Check dominance
                i_dominates_j = np.all(values_array[i] >= values_array[j]) and np.any(
                    values_array[i] > values_array[j]
                )
                j_dominates_i = np.all(values_array[j] >= values_array[i]) and np.any(
                    values_array[j] > values_array[i]
                )

                if i_dominates_j:
                    is_pareto[j] = False
                elif j_dominates_i:
                    is_pareto[i] = False
                    break

        # Count dominating runs for each run
        pareto_runs = []
        for idx, r in enumerate(runs_with_values):
            dominated_by = 0
            dominates = 0

            for j in range(n):
                if j == idx:
                    continue

                i_dominates_j = np.all(values_array[idx] >= values_array[j]) and np.any(
                    values_array[idx] > values_array[j]
                )
                j_dominates_i = np.all(values_array[j] >= values_array[idx]) and np.any(
                    values_array[j] > values_array[idx]
                )

                if j_dominates_i:
                    dominated_by += 1
                elif i_dominates_j:
                    dominates += 1

            result = {
                "run_name": r["run"]["run_name"],
                "is_pareto_optimal": bool(is_pareto[idx]),
                "dominated_by_count": int(dominated_by),
                "dominates_count": int(dominates),
                "objective_values": {
                    obj: r["values"][i] for i, obj in enumerate(objectives)
                },
                "rank": int(dominated_by + 1),  # Pareto rank
            }
            pareto_runs.append(result)

        # Sort by rank (Pareto-optimal first)
        pareto_runs.sort(key=lambda x: (x["rank"], -x["dominates_count"]))

        return pareto_runs

    def recommend_by_weights(
        self,
        runs: List[Dict[str, Any]],
        objectives: List[str],
        weights: Optional[List[float]] = None,
        minimize: Optional[List[bool]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recommend runs using weighted sum of normalized objectives.

        Args:
            runs: List of run dictionaries
            objectives: Metric names to optimize
            weights: Weight for each objective (default: equal weights)
            minimize: For each objective, True if lower is better
            top_k: Number of top runs to return

        Returns:
            Top-k runs sorted by weighted score
        """
        if weights is None:
            weights = [1.0 / len(objectives)] * len(objectives)

        if minimize is None:
            minimize_keywords = ["loss", "error", "mse", "mae", "rmse", "time", "size"]
            minimize = [
                any(kw in obj.lower() for kw in minimize_keywords) for obj in objectives
            ]

        # Extract and normalize values
        runs_with_values = []
        all_values = {obj: [] for obj in objectives}

        for run in runs:
            values = {}
            valid = True

            for obj in objectives:
                val = None
                if run.get("metrics") and run["metrics"].get("final"):
                    val = run["metrics"]["final"].get(obj)
                elif run.get("final_metrics"):
                    val = run["final_metrics"].get(obj)

                if val is None or not isinstance(val, (int, float)) or np.isnan(val):
                    valid = False
                    break

                values[obj] = float(val)
                all_values[obj].append(float(val))

            if valid:
                runs_with_values.append(
                    {
                        "run": run,
                        "values": values,
                    }
                )

        if len(runs_with_values) == 0:
            return []

        # Normalize to [0, 1]
        normalized_runs = []
        for r in runs_with_values:
            normalized = {}
            for i, obj in enumerate(objectives):
                vals = all_values[obj]
                min_val = min(vals)
                max_val = max(vals)

                if max_val - min_val < 1e-10:
                    norm_val = 0.5
                else:
                    norm_val = (r["values"][obj] - min_val) / (max_val - min_val)

                # Flip for minimization
                if minimize[i]:
                    norm_val = 1.0 - norm_val

                normalized[obj] = norm_val

            # Calculate weighted score
            score = sum(
                normalized[obj] * weights[i] for i, obj in enumerate(objectives)
            )

            normalized_runs.append(
                {
                    "run_name": r["run"]["run_name"],
                    "score": float(score),
                    "objective_values": r["values"],
                    "normalized_values": normalized,
                }
            )

        # Sort by score descending
        normalized_runs.sort(key=lambda x: x["score"], reverse=True)

        return normalized_runs[:top_k]

    def recommend_best_runs(
        self,
        runs: List[Dict[str, Any]],
        objectives: List[str],
        strategy: Optional[RecommendationStrategy] = None,
        weights: Optional[List[float]] = None,
        minimize: Optional[List[bool]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Recommend best runs based on multiple objectives.

        Args:
            runs: List of run dictionaries
            objectives: Metrics to optimize
            strategy: Recommendation strategy (None = use self.strategy)
            weights: Weights for weighted strategy
            minimize: For each objective, True if lower is better
            top_k: Number of recommendations

        Returns:
            Dictionary with recommendations and analysis
        """
        if strategy is None:
            strategy = self.strategy

        if len(runs) == 0:
            return {
                "recommendations": [],
                "strategy": strategy,
                "message": "No runs available",
            }

        if len(objectives) == 0:
            return {
                "recommendations": [],
                "strategy": strategy,
                "message": "No objectives specified",
            }

        if strategy == "pareto":
            pareto_runs = self.find_pareto_frontier(runs, objectives, minimize)

            # Take top-k Pareto-optimal or near-optimal runs
            recommendations = pareto_runs[:top_k]

            return {
                "recommendations": recommendations,
                "strategy": "pareto",
                "total_runs": len(runs),
                "pareto_optimal_count": sum(
                    1 for r in pareto_runs if r["is_pareto_optimal"]
                ),
                "objectives": objectives,
                "minimize": minimize,
            }

        elif strategy == "weighted":
            recommendations = self.recommend_by_weights(
                runs, objectives, weights, minimize, top_k
            )

            return {
                "recommendations": recommendations,
                "strategy": "weighted",
                "total_runs": len(runs),
                "objectives": objectives,
                "weights": weights or [1.0 / len(objectives)] * len(objectives),
                "minimize": minimize,
            }

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def recommend_best_runs_for_project(
    runs: List[Dict[str, Any]],
    objectives: Optional[List[str]] = None,
    strategy: RecommendationStrategy = "pareto",
    top_k: int = 5,
) -> Dict[str, Any]:
    """Convenience function to recommend best runs for a project.

    Args:
        runs: List of run dictionaries
        objectives: Metrics to optimize (None = auto-select key metrics)
        strategy: Recommendation strategy
        top_k: Number of recommendations

    Returns:
        Recommendation results
    """
    # Auto-select objectives if not specified
    if objectives is None:
        # Find most common metrics
        metric_counts = {}
        for run in runs:
            if run.get("metrics") and run["metrics"].get("final"):
                for metric in run["metrics"]["final"].keys():
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1
            elif run.get("final_metrics"):
                for metric in run["final_metrics"].keys():
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1

        # Select top 2-3 most common metrics
        objectives = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)
        objectives = [obj for obj, _ in objectives[:3]]

    recommender = RunRecommender(strategy=strategy)
    return recommender.recommend_best_runs(runs, objectives, top_k=top_k)
