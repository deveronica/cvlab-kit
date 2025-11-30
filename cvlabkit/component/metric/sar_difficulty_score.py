"""Training-free SAR image difficulty metric.

Based on physical image indicators (ECR, TV, Entropy, GLCM, ENL).
Uses domain knowledge and XGBoost-validated weights (ρ = 0.46).

Reference: docs/20251115_sar_difficulty_metric_design.md
"""

import numpy as np
import torch

from cvlabkit.component.base import Metric
from cvlabkit.component.metric.physical_indicators import PhysicalIndicators


class SARDifficultyScore(Metric):
    """Training-free difficulty metric for SAR images.

    Computes difficulty score from physical indicators without requiring
    model training (avoids circular dependency in curriculum learning).

    Args:
        weight_strategy: Weight selection method:
            - 'equal': All indicators weighted equally (1/7)
            - 'xgboost': Feature importance from XGBoost validation (default)
            - dict: Custom weights {indicator_name: weight}

    Example:
        >>> difficulty_metric = SARDifficultyScore(weight_strategy='xgboost')
        >>> score = difficulty_metric(image)  # Higher = more difficult
        >>> scores = difficulty_metric.batch_compute(images)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Physical indicators calculator
        self.indicators = PhysicalIndicators(cfg)

        # Weight strategy
        weight_strategy = cfg.get("weight_strategy", "xgboost")

        if weight_strategy == "equal":
            # All indicators equally weighted
            self.weights = {
                "ecr": 1 / 7,
                "tv": 1 / 7,
                "kurtosis": 1 / 7,
                "shannon_entropy": 1 / 7,
                "glcm_contrast": 1 / 7,
                "glcm_entropy": 1 / 7,
                "enl": 1 / 7,
            }
        elif weight_strategy == "fisher":
            # Fisher Separability Criterion (training-free, class-based)
            # See: outputs/mstar_correlation_v2/fisher_weights_20251115_213857/
            self.weights = {
                "ecr": 0.059268,
                "tv": 0.212757,
                "kurtosis": 0.111432,
                "shannon_entropy": 0.188688,
                "glcm_contrast": 0.171590,
                "glcm_entropy": 0.184704,
                "enl": 0.071561,
            }
        elif weight_strategy == "xgboost":
            # From early epochs XGBoost analysis (validated ρ = 0.46)
            # See: outputs/mstar_correlation_v2/early_epochs_analysis_20251115_211533/
            self.weights = {
                "ecr": 0.113,
                "tv": 0.139,
                "kurtosis": 0.149,
                "shannon_entropy": 0.152,
                "glcm_contrast": 0.137,
                "glcm_entropy": 0.153,
                "enl": 0.156,
            }
        elif isinstance(weight_strategy, dict):
            # Custom weights
            self.weights = weight_strategy
        else:
            raise ValueError(f"Unknown weight_strategy: {weight_strategy}")

        # Normalization parameters (from MSTAR epochs 0-10)
        # See: outputs/mstar_correlation_v2/polynomial_formula_20251115_211814/
        self.mean = {
            "ecr": 0.697683,
            "tv": 5894.310279,
            "kurtosis": 39.904791,
            "shannon_entropy": 5.916306,
            "glcm_contrast": 548.930053,
            "glcm_entropy": 11.214768,
            "enl": 1.614446,
        }
        self.std = {
            "ecr": 0.644195,
            "tv": 3024.751877,
            "kurtosis": 32.557467,
            "shannon_entropy": 0.775157,
            "glcm_contrast": 521.660444,
            "glcm_entropy": 1.406019,
            "enl": 0.598112,
        }

    def forward(self, images, targets=None):
        """Compute difficulty scores for batch of images.

        Args:
            images: Batch of images (B, C, H, W) or list of PIL Images
            targets: Ignored (metric is training-free)

        Returns:
            dict with 'difficulty_scores': tensor of shape (B,)
        """
        if isinstance(images, (list, tuple)):
            # List of PIL Images
            scores = [self._compute_single(img) for img in images]
            return {"difficulty_scores": torch.tensor(scores)}

        # Batch tensor
        batch_size = images.shape[0]
        scores = []

        for i in range(batch_size):
            img = images[i]  # (C, H, W)
            score = self._compute_single(img)
            scores.append(score)

        return {"difficulty_scores": torch.tensor(scores)}

    def _compute_single(self, image):
        """Compute difficulty score for a single image.

        Args:
            image: torch.Tensor (C, H, W) or (H, W) or PIL.Image

        Returns:
            float: Difficulty score (higher = more difficult)
        """
        # Compute physical indicators
        indicator_dict = self.indicators._compute_single(image)

        # Normalize and weight
        difficulty = 0.0

        for name, value in indicator_dict.items():
            # Handle inf/nan
            if not np.isfinite(value):
                # Replace with mean (neutral contribution)
                value = self.mean[name]

            # Normalize: (x - μ) / σ
            normalized = (value - self.mean[name]) / self.std[name]

            # Weight and accumulate
            difficulty += self.weights[name] * normalized

        return float(difficulty)

    def batch_compute(self, images):
        """Convenience method for computing difficulty scores.

        Args:
            images: List of PIL Images or torch.Tensor (B, C, H, W)

        Returns:
            list of float: Difficulty scores
        """
        result = self.forward(images)
        return result["difficulty_scores"].tolist()

    def get_sorted_indices(self, images):
        """Get indices sorted by difficulty (easy → hard).

        Useful for curriculum learning data ordering.

        Args:
            images: List of PIL Images or torch.Tensor

        Returns:
            numpy.ndarray: Indices sorted by difficulty (ascending)
        """
        scores = self.batch_compute(images)
        return np.argsort(scores)

    def get_difficulty_stats(self, images):
        """Get difficulty distribution statistics.

        Args:
            images: List of PIL Images or torch.Tensor

        Returns:
            dict: Statistics (mean, std, min, max, percentiles)
        """
        scores = self.batch_compute(images)
        scores = np.array(scores)

        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "p25": float(np.percentile(scores, 25)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "p99": float(np.percentile(scores, 99)),
        }

    def visualize_distribution(self, images, save_path=None):
        """Visualize difficulty score distribution.

        Args:
            images: List of PIL Images or torch.Tensor
            save_path: Optional path to save plot

        Returns:
            matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        scores = self.batch_compute(images)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(scores, bins=50, alpha=0.7, edgecolor="black")
        ax1.set_xlabel("Difficulty Score", fontweight="bold")
        ax1.set_ylabel("Frequency", fontweight="bold")
        ax1.set_title("SAR Difficulty Score Distribution", fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Add statistics
        stats = self.get_difficulty_stats(images)
        stats_text = f"Mean: {stats['mean']:.3f}\n"
        stats_text += f"Std: {stats['std']:.3f}\n"
        stats_text += f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]"
        ax1.text(
            0.98,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Cumulative distribution
        sorted_scores = np.sort(scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

        ax2.plot(sorted_scores, cumulative, linewidth=2)
        ax2.set_xlabel("Difficulty Score", fontweight="bold")
        ax2.set_ylabel("Cumulative Probability", fontweight="bold")
        ax2.set_title("Cumulative Distribution Function", fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Mark percentiles
        for p in [25, 50, 75]:
            score = stats[f"p{p}"] if p != 50 else stats["median"]
            ax2.axvline(score, color="r", linestyle="--", alpha=0.5, linewidth=1)
            ax2.text(score, 0.02, f"P{p}", rotation=90, verticalalignment="bottom")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
