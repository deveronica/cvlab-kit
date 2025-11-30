"""Extended Physical Indicators Metric Component for Advanced SAR Image Analysis.

This metric extends PhysicalIndicators with additional SAR-specific quality metrics
based on recent research in speckle noise analysis, edge preservation, and texture features.
"""

from typing import Dict, List

import numpy as np
import torch
from scipy import stats
from scipy.ndimage import generic_filter, laplace, sobel
from skimage.feature import canny, graycomatrix, graycoprops

from cvlabkit.component.metric.physical_indicators import PhysicalIndicators


class ExtendedPhysicalIndicators(PhysicalIndicators):
    """Extended physical indicators including advanced SAR quality metrics.

    Additional Indicators:
    - Coefficient of Variation (CV): Normalized dispersion measure
    - Sobel Gradient Mean/Std: Edge strength statistics
    - Laplacian Variance: Focus/sharpness measure
    - GLCM Extended: Homogeneity, Correlation, Energy, Dissimilarity
    - Skewness: Distribution asymmetry
    - Local Variance Mean: Texture complexity
    - Edge Density: Ratio of edge pixels
    - High Frequency Content: FFT-based frequency analysis
    - Gradient Variance: Gradient distribution measure
    """

    def __init__(self, cfg):
        """Initialize extended physical indicators.

        Args:
            cfg: Configuration with additional parameters:
                - local_window_size (int): Window size for local variance (default: 7)
                - canny_sigma (float): Gaussian sigma for Canny edge detection (default: 1.0)
                - canny_low_threshold (float): Canny low threshold (default: 0.1)
                - canny_high_threshold (float): Canny high threshold (default: 0.2)
                - fft_hf_threshold (float): FFT high-frequency cutoff ratio (default: 0.5)
        """
        super().__init__(cfg)
        self.local_window_size = cfg.get("local_window_size", 7)
        self.canny_sigma = cfg.get("canny_sigma", 1.0)
        self.canny_low_threshold = cfg.get("canny_low_threshold", 0.1)
        self.canny_high_threshold = cfg.get("canny_high_threshold", 0.2)
        self.fft_hf_threshold = cfg.get("fft_hf_threshold", 0.5)

    def update(
        self, images: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Updates metric state with extended indicators.

        Args:
            images: Input images [B, C, H, W]
            preds: Model predictions (logits) [B, num_classes]
            targets: Ground truth labels [B]
        """
        # Convert to numpy
        images_np = images.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        batch_size = images_np.shape[0]

        for i in range(batch_size):
            # Track sample index
            current_idx = self.sample_counter
            self.sample_counter += 1

            # Get grayscale image
            img = images_np[i]
            if img.shape[0] == 3:  # RGB -> grayscale
                img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            elif img.shape[0] == 1:
                img = img[0]
            else:
                raise ValueError(f"Unexpected number of channels: {img.shape[0]}")

            # Compute original indicators (from parent class)
            ecr = self._compute_ecr(img)
            tv = self._compute_tv(img)
            kurtosis = self._compute_kurtosis(img)
            entropy = self._compute_shannon_entropy(img)
            glcm_contrast = self._compute_glcm_contrast(img)
            glcm_entropy = self._compute_glcm_entropy(img)
            enl = self._compute_enl(img)

            # Compute extended indicators
            cv = self._compute_coefficient_variation(img)
            sobel_mean, sobel_std = self._compute_sobel_stats(img)
            laplacian_var = self._compute_laplacian_variance(img)
            glcm_homo = self._compute_glcm_homogeneity(img)
            glcm_corr = self._compute_glcm_correlation(img)
            glcm_energy = self._compute_glcm_energy(img)
            glcm_dissim = self._compute_glcm_dissimilarity(img)
            skewness = self._compute_skewness(img)
            local_var_mean = self._compute_local_variance_mean(img)
            edge_density = self._compute_edge_density(img)
            hf_content = self._compute_high_frequency_content(img)
            grad_var = self._compute_gradient_variance(img)

            # Compute prediction metrics
            pred_entropy = self._compute_prediction_entropy(preds_np[i])
            ce_loss = self._compute_cross_entropy_loss(preds_np[i], targets_np[i])

            # Store original indicators
            self.ecr_values.append(ecr)
            self.tv_values.append(tv)
            self.kurtosis_values.append(kurtosis)
            self.entropy_values.append(entropy)
            self.glcm_contrast_values.append(glcm_contrast)
            self.glcm_entropy_values.append(glcm_entropy)
            self.enl_values.append(enl)
            self.pred_entropy_values.append(pred_entropy)
            self.ce_loss_values.append(ce_loss)
            self.targets_list.append(targets_np[i])

            # Store extended indicators
            self.cv_values.append(cv)
            self.sobel_mean_values.append(sobel_mean)
            self.sobel_std_values.append(sobel_std)
            self.laplacian_var_values.append(laplacian_var)
            self.glcm_homo_values.append(glcm_homo)
            self.glcm_corr_values.append(glcm_corr)
            self.glcm_energy_values.append(glcm_energy)
            self.glcm_dissim_values.append(glcm_dissim)
            self.skewness_values.append(skewness)
            self.local_var_mean_values.append(local_var_mean)
            self.edge_density_values.append(edge_density)
            self.hf_content_values.append(hf_content)
            self.grad_var_values.append(grad_var)

            # Store prediction correctness
            pred_class = np.argmax(preds_np[i])
            self.correct_list.append(int(pred_class == targets_np[i]))

            # Optionally save logits and indices
            if self.save_logits:
                self.logits_list.append(preds_np[i].tolist())
            if self.save_indices:
                self.indices_list.append(current_idx)

    def _compute_coefficient_variation(self, img: np.ndarray) -> float:
        """Compute Coefficient of Variation (CV).

        CV = σ / μ

        Normalized measure of dispersion. Used in SAR for speckle assessment.
        """
        mean = np.mean(img)
        std = np.std(img)

        if abs(mean) < 1e-10:
            return float("inf")

        return float(std / abs(mean))

    def _compute_sobel_stats(self, img: np.ndarray) -> tuple:
        """Compute Sobel gradient magnitude statistics.

        Returns:
            (mean, std) of gradient magnitudes
        """
        grad_x = sobel(img, axis=1)
        grad_y = sobel(img, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return float(np.mean(grad_magnitude)), float(np.std(grad_magnitude))

    def _compute_laplacian_variance(self, img: np.ndarray) -> float:
        """Compute variance of Laplacian.

        Measures image focus/sharpness. Higher values indicate sharper images.
        """
        lap = laplace(img)
        return float(np.var(lap))

    def _compute_glcm_homogeneity(self, img: np.ndarray) -> float:
        """Compute GLCM Homogeneity.

        Measures closeness of distribution of elements to GLCM diagonal.
        """
        img_normalized = (
            (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
        ).astype(np.uint8)

        glcm = graycomatrix(
            img_normalized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True,
        )

        homogeneity = graycoprops(glcm, "homogeneity")
        return float(np.mean(homogeneity))

    def _compute_glcm_correlation(self, img: np.ndarray) -> float:
        """Compute GLCM Correlation.

        Measures linear dependency of gray levels in co-occurrence matrix.
        """
        img_normalized = (
            (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
        ).astype(np.uint8)

        glcm = graycomatrix(
            img_normalized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True,
        )

        correlation = graycoprops(glcm, "correlation")
        return float(np.mean(correlation))

    def _compute_glcm_energy(self, img: np.ndarray) -> float:
        """Compute GLCM Energy (Angular Second Moment).

        Measures uniformity of texture. Higher for homogeneous images.
        """
        img_normalized = (
            (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
        ).astype(np.uint8)

        glcm = graycomatrix(
            img_normalized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True,
        )

        energy = graycoprops(glcm, "energy")
        return float(np.mean(energy))

    def _compute_glcm_dissimilarity(self, img: np.ndarray) -> float:
        """Compute GLCM Dissimilarity.

        Measures variation of gray level pairs. Similar to contrast but linear weighting.
        """
        img_normalized = (
            (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
        ).astype(np.uint8)

        glcm = graycomatrix(
            img_normalized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True,
        )

        dissimilarity = graycoprops(glcm, "dissimilarity")
        return float(np.mean(dissimilarity))

    def _compute_skewness(self, img: np.ndarray) -> float:
        """Compute skewness of pixel distribution.

        Measures asymmetry of distribution.
        """
        flat_img = img.flatten()
        return float(stats.skew(flat_img))

    def _compute_local_variance_mean(self, img: np.ndarray) -> float:
        """Compute mean of local variances.

        Measures local texture complexity using sliding window.
        """

        def local_var(values):
            return np.var(values)

        # Compute local variance using generic filter
        local_vars = generic_filter(img, local_var, size=self.local_window_size)

        return float(np.mean(local_vars))

    def _compute_edge_density(self, img: np.ndarray) -> float:
        """Compute edge density using Canny edge detector.

        Returns ratio of edge pixels to total pixels.
        """
        # Normalize image to [0, 1]
        img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-10)

        # Detect edges
        edges = canny(
            img_normalized,
            sigma=self.canny_sigma,
            low_threshold=self.canny_low_threshold,
            high_threshold=self.canny_high_threshold,
        )

        # Compute edge density
        edge_density = np.sum(edges) / edges.size

        return float(edge_density)

    def _compute_high_frequency_content(self, img: np.ndarray) -> float:
        """Compute high-frequency content ratio using FFT.

        Returns ratio of high-frequency energy to total energy.
        """
        # Compute 2D FFT
        fft = np.fft.fft2(img)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        # Get center coordinates
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2

        # Define high-frequency region (outer region)
        radius = int(min(h, w) * self.fft_hf_threshold / 2)

        # Create circular mask for low-frequency (center)
        y, x = np.ogrid[:h, :w]
        mask_lf = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
        mask_hf = ~mask_lf

        # Compute energies
        energy_lf = np.sum(magnitude[mask_lf] ** 2)
        energy_hf = np.sum(magnitude[mask_hf] ** 2)
        total_energy = energy_lf + energy_hf

        if total_energy < 1e-10:
            return 0.0

        return float(energy_hf / total_energy)

    def _compute_gradient_variance(self, img: np.ndarray) -> float:
        """Compute variance of gradient magnitudes.

        Measures variability in edge strengths.
        """
        grad_x = sobel(img, axis=1)
        grad_y = sobel(img, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return float(np.var(grad_magnitude))

    def compute(self) -> Dict[str, any]:
        """Compute final statistics including extended indicators.

        Returns:
            Dictionary with all indicators, statistics, and correlations.
        """
        if len(self.ecr_values) == 0:
            return {"num_samples": 0, "warning": "No samples processed"}

        # Get base results from parent class
        result = super().compute()

        # Add extended indicator raw values
        result.update(
            {
                "coefficient_variation": self.cv_values,
                "sobel_mean": self.sobel_mean_values,
                "sobel_std": self.sobel_std_values,
                "laplacian_variance": self.laplacian_var_values,
                "glcm_homogeneity": self.glcm_homo_values,
                "glcm_correlation": self.glcm_corr_values,
                "glcm_energy": self.glcm_energy_values,
                "glcm_dissimilarity": self.glcm_dissim_values,
                "skewness": self.skewness_values,
                "local_variance_mean": self.local_var_mean_values,
                "edge_density": self.edge_density_values,
                "high_frequency_content": self.hf_content_values,
                "gradient_variance": self.grad_var_values,
            }
        )

        # Convert to numpy for correlation calculation
        cv = np.array(self.cv_values)
        sobel_mean = np.array(self.sobel_mean_values)
        sobel_std = np.array(self.sobel_std_values)
        lap_var = np.array(self.laplacian_var_values)
        glcm_homo = np.array(self.glcm_homo_values)
        glcm_corr = np.array(self.glcm_corr_values)
        glcm_energy = np.array(self.glcm_energy_values)
        glcm_dissim = np.array(self.glcm_dissim_values)
        skewness = np.array(self.skewness_values)
        local_var = np.array(self.local_var_mean_values)
        edge_dens = np.array(self.edge_density_values)
        hf_content = np.array(self.hf_content_values)
        grad_var = np.array(self.grad_var_values)

        pred_entropy = np.array(self.pred_entropy_values)
        ce_loss = np.array(self.ce_loss_values)

        # Add extended correlations with prediction entropy
        result.update(
            {
                "corr_cv_pred_entropy": float(stats.spearmanr(cv, pred_entropy)[0]),
                "corr_sobel_mean_pred_entropy": float(
                    stats.spearmanr(sobel_mean, pred_entropy)[0]
                ),
                "corr_sobel_std_pred_entropy": float(
                    stats.spearmanr(sobel_std, pred_entropy)[0]
                ),
                "corr_laplacian_var_pred_entropy": float(
                    stats.spearmanr(lap_var, pred_entropy)[0]
                ),
                "corr_glcm_homo_pred_entropy": float(
                    stats.spearmanr(glcm_homo, pred_entropy)[0]
                ),
                "corr_glcm_corr_pred_entropy": float(
                    stats.spearmanr(glcm_corr, pred_entropy)[0]
                ),
                "corr_glcm_energy_pred_entropy": float(
                    stats.spearmanr(glcm_energy, pred_entropy)[0]
                ),
                "corr_glcm_dissim_pred_entropy": float(
                    stats.spearmanr(glcm_dissim, pred_entropy)[0]
                ),
                "corr_skewness_pred_entropy": float(
                    stats.spearmanr(skewness, pred_entropy)[0]
                ),
                "corr_local_var_pred_entropy": float(
                    stats.spearmanr(local_var, pred_entropy)[0]
                ),
                "corr_edge_density_pred_entropy": float(
                    stats.spearmanr(edge_dens, pred_entropy)[0]
                ),
                "corr_hf_content_pred_entropy": float(
                    stats.spearmanr(hf_content, pred_entropy)[0]
                ),
                "corr_grad_var_pred_entropy": float(
                    stats.spearmanr(grad_var, pred_entropy)[0]
                ),
            }
        )

        # Add extended correlations with cross entropy loss
        result.update(
            {
                "corr_cv_ce_loss": float(stats.spearmanr(cv, ce_loss)[0]),
                "corr_sobel_mean_ce_loss": float(
                    stats.spearmanr(sobel_mean, ce_loss)[0]
                ),
                "corr_sobel_std_ce_loss": float(stats.spearmanr(sobel_std, ce_loss)[0]),
                "corr_laplacian_var_ce_loss": float(
                    stats.spearmanr(lap_var, ce_loss)[0]
                ),
                "corr_glcm_homo_ce_loss": float(stats.spearmanr(glcm_homo, ce_loss)[0]),
                "corr_glcm_corr_ce_loss": float(stats.spearmanr(glcm_corr, ce_loss)[0]),
                "corr_glcm_energy_ce_loss": float(
                    stats.spearmanr(glcm_energy, ce_loss)[0]
                ),
                "corr_glcm_dissim_ce_loss": float(
                    stats.spearmanr(glcm_dissim, ce_loss)[0]
                ),
                "corr_skewness_ce_loss": float(stats.spearmanr(skewness, ce_loss)[0]),
                "corr_local_var_ce_loss": float(stats.spearmanr(local_var, ce_loss)[0]),
                "corr_edge_density_ce_loss": float(
                    stats.spearmanr(edge_dens, ce_loss)[0]
                ),
                "corr_hf_content_ce_loss": float(
                    stats.spearmanr(hf_content, ce_loss)[0]
                ),
                "corr_grad_var_ce_loss": float(stats.spearmanr(grad_var, ce_loss)[0]),
            }
        )

        return result

    def reset(self) -> None:
        """Reset metric state including extended indicators."""
        super().reset()

        # Extended indicator storage
        self.cv_values: List[float] = []
        self.sobel_mean_values: List[float] = []
        self.sobel_std_values: List[float] = []
        self.laplacian_var_values: List[float] = []
        self.glcm_homo_values: List[float] = []
        self.glcm_corr_values: List[float] = []
        self.glcm_energy_values: List[float] = []
        self.glcm_dissim_values: List[float] = []
        self.skewness_values: List[float] = []
        self.local_var_mean_values: List[float] = []
        self.edge_density_values: List[float] = []
        self.hf_content_values: List[float] = []
        self.grad_var_values: List[float] = []
