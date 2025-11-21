"""Physical Indicators Metric Component for Image Analysis.

This metric computes various physical indicators from images along with
prediction entropy to analyze correlations between image properties and
model uncertainty.
"""

from typing import Dict, List

import numpy as np
import torch
from scipy import stats
from scipy.ndimage import sobel
from skimage.feature import graycomatrix, graycoprops

from cvlabkit.component.base import Metric


class PhysicalIndicators(Metric):
    """Computes physical indicators and prediction entropy for each test sample.

    Physical Indicators:
    - ECR (Energy Compaction Ratio): LF energy / HF energy ratio
    - TV (Total Variation): Sum of image gradients (spikiness measure)
    - Kurtosis: Distribution peakedness (outlier indicator)
    - Shannon Entropy: Spectral disorder/complexity
    - GLCM Contrast: Spectral texture roughness
    - GLCM Entropy: Spectral texture randomness
    - ENL (Equivalent Number of Looks): Image smoothness (mean²/variance)

    Model Uncertainty Measures:
    - Prediction Entropy: Overall output distribution uncertainty H(p)
    - Cross Entropy Loss: Confidence on correct class -log(p_correct)
    """

    def __init__(self, cfg):
        """Initializes the PhysicalIndicators metric.

        Args:
            cfg: Configuration object with optional parameters:
                - lf_ratio (float): Ratio of image size for low-frequency region (default: 0.25)
                - glcm_distances (list): Distances for GLCM calculation (default: [1])
                - glcm_angles (list): Angles for GLCM in radians (default: [0, π/4, π/2, 3π/4])
                - num_bins (int): Number of bins for entropy histogram (default: 256)
                - save_logits (bool): Save raw logits for future analysis (default: False)
                - save_indices (bool): Save sample indices for tracking (default: False)
        """
        super().__init__()
        self.lf_ratio = cfg.get("lf_ratio", 0.25)
        self.glcm_distances = cfg.get("glcm_distances", [1])
        self.glcm_angles = cfg.get(
            "glcm_angles", [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        )
        self.num_bins = cfg.get("num_bins", 256)
        self.save_logits = cfg.get("save_logits", False)
        self.save_indices = cfg.get("save_indices", False)
        self.reset()

    def update(
        self, images: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Updates the metric state with a batch of images and predictions.

        Args:
            images: Input images [B, C, H, W]
            preds: Model predictions (logits) [B, num_classes]
            targets: Ground truth labels [B]
        """
        # Convert to numpy for processing (detach to avoid gradient error)
        images_np = images.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        batch_size = images_np.shape[0]

        for i in range(batch_size):
            # Track sample index (auto-increment counter)
            current_idx = self.sample_counter
            self.sample_counter += 1
            # Get single image [C, H, W] -> convert to grayscale if needed
            img = images_np[i]
            if img.shape[0] == 3:  # RGB -> grayscale
                img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            elif img.shape[0] == 1:  # Already grayscale
                img = img[0]
            else:
                raise ValueError(f"Unexpected number of channels: {img.shape[0]}")

            # Compute physical indicators
            ecr = self._compute_ecr(img)
            tv = self._compute_tv(img)
            kurtosis = self._compute_kurtosis(img)
            entropy = self._compute_shannon_entropy(img)
            glcm_contrast = self._compute_glcm_contrast(img)
            glcm_entropy = self._compute_glcm_entropy(img)
            enl = self._compute_enl(img)

            # Compute prediction entropy and cross entropy loss
            pred_entropy = self._compute_prediction_entropy(preds_np[i])
            ce_loss = self._compute_cross_entropy_loss(preds_np[i], targets_np[i])

            # Store results
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

            # Store prediction correctness
            pred_class = np.argmax(preds_np[i])
            self.correct_list.append(int(pred_class == targets_np[i]))

            # Optionally save logits and indices
            if self.save_logits:
                self.logits_list.append(preds_np[i].tolist())

            if self.save_indices:
                self.indices_list.append(current_idx)

    def _compute_ecr(self, img: np.ndarray) -> float:
        """Computes Energy Compaction Ratio (ECR).

        ECR = LF_energy / HF_energy
        where LF is the center region and HF is the outer region in frequency space.

        For spatial images, we approximate by using the center region vs periphery.
        """
        h, w = img.shape
        center_h = int(h * self.lf_ratio)
        center_w = int(w * self.lf_ratio)

        # Define center region (LF approximation)
        start_h = (h - center_h) // 2
        start_w = (w - center_w) // 2
        end_h = start_h + center_h
        end_w = start_w + center_w

        # Compute energies
        lf_region = img[start_h:end_h, start_w:end_w]
        lf_energy = np.sum(lf_region**2)

        # HF energy: total - LF
        total_energy = np.sum(img**2)
        hf_energy = total_energy - lf_energy

        # Avoid division by zero
        if hf_energy < 1e-10:
            return float("inf")

        return lf_energy / hf_energy

    def _compute_tv(self, img: np.ndarray) -> float:
        """Computes Total Variation (TV).

        TV = sum(sqrt(|∇x|² + |∇y|²))

        Measures image spikiness/roughness.
        """
        # Compute gradients using Sobel operator
        grad_x = sobel(img, axis=1)
        grad_y = sobel(img, axis=0)

        # TV is the sum of gradient magnitudes
        tv = np.sum(np.sqrt(grad_x**2 + grad_y**2))

        return float(tv)

    def _compute_kurtosis(self, img: np.ndarray) -> float:
        """Computes excess kurtosis of pixel intensity distribution.

        Kurtosis measures the "tailedness" of the distribution.
        High kurtosis indicates extreme outliers (e.g., DC peak in k-space).
        """
        flat_img = img.flatten()
        return float(
            stats.kurtosis(flat_img, fisher=True)
        )  # Excess kurtosis (normal=0)

    def _compute_shannon_entropy(self, img: np.ndarray) -> float:
        """Computes Shannon entropy of pixel intensity distribution.

        H = -Σ p(i) * log₂(p(i))

        Measures spectral disorder/complexity.
        """
        # Create histogram
        hist, _ = np.histogram(img.flatten(), bins=self.num_bins, density=True)

        # Normalize to get probabilities
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / np.sum(hist)

        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        return float(entropy)

    def _compute_glcm_contrast(self, img: np.ndarray) -> float:
        """Computes GLCM (Gray-Level Co-occurrence Matrix) Contrast.

        Contrast measures local variations in the gray-level co-occurrence matrix.
        High contrast indicates rough spectral texture.
        """
        # Normalize image to 0-255 range for GLCM
        img_normalized = (
            (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
        ).astype(np.uint8)

        # Compute GLCM
        glcm = graycomatrix(
            img_normalized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True,
        )

        # Compute contrast property
        contrast = graycoprops(glcm, "contrast")

        # Average over all distances and angles
        return float(np.mean(contrast))

    def _compute_glcm_entropy(self, img: np.ndarray) -> float:
        """Computes GLCM (Gray-Level Co-occurrence Matrix) Entropy.

        Entropy measures randomness/disorder in the texture.
        H = -Σ p(i,j) * log₂(p(i,j))

        High entropy indicates complex, random texture patterns.
        """
        # Normalize image to 0-255 range for GLCM
        img_normalized = (
            (img - img.min()) / (img.max() - img.min() + 1e-10) * 255
        ).astype(np.uint8)

        # Compute GLCM
        glcm = graycomatrix(
            img_normalized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True,
        )

        # Compute entropy from GLCM
        # GLCM shape: (levels, levels, distances, angles)
        entropies = []
        for d_idx in range(len(self.glcm_distances)):
            for a_idx in range(len(self.glcm_angles)):
                glcm_slice = glcm[:, :, d_idx, a_idx]
                # Avoid log(0)
                glcm_slice = glcm_slice + 1e-10
                # Compute entropy
                entropy = -np.sum(glcm_slice * np.log2(glcm_slice + 1e-10))
                entropies.append(entropy)

        # Average over all distances and angles
        return float(np.mean(entropies))

    def _compute_enl(self, img: np.ndarray) -> float:
        """Computes ENL (Equivalent Number of Looks).

        ENL = mean² / variance

        ENL measures image smoothness/speckle noise.
        - High ENL: Smooth image (low speckle)
        - Low ENL: Noisy image (high speckle)

        Commonly used in SAR imagery analysis.
        """
        mean = np.mean(img)
        variance = np.var(img)

        # Avoid division by zero
        if variance < 1e-10:
            return float("inf")

        enl = (mean**2) / variance

        return float(enl)

    def _compute_prediction_entropy(self, logits: np.ndarray) -> float:
        """Computes prediction entropy from model logits.

        H = -Σ p(c) * log₂(p(c))

        where p(c) = softmax(logits)

        High entropy indicates model uncertainty.
        """
        # Compute softmax probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return float(entropy)

    def _compute_cross_entropy_loss(self, logits: np.ndarray, target: int) -> float:
        """Computes cross entropy loss for the correct class.

        CE = -log(p_correct)

        where p_correct = softmax(logits)[target]

        High CE indicates low confidence on the correct class (difficult sample).
        Low CE indicates high confidence on the correct class (easy sample).
        """
        # Compute softmax probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        # Get probability of correct class
        p_correct = probs[target]

        # Compute cross entropy loss
        ce_loss = -np.log(p_correct + 1e-10)

        return float(ce_loss)

    def compute(self) -> Dict[str, any]:
        """Computes final statistics and returns all collected data.

        Returns:
            Dictionary containing:
                - Raw values for each indicator
                - Summary statistics
                - Prediction accuracy
        """
        if len(self.ecr_values) == 0:
            return {"num_samples": 0, "warning": "No samples processed"}

        # Convert lists to numpy arrays
        ecr = np.array(self.ecr_values)
        tv = np.array(self.tv_values)
        kurtosis = np.array(self.kurtosis_values)
        entropy = np.array(self.entropy_values)
        glcm_contrast = np.array(self.glcm_contrast_values)
        glcm_entropy = np.array(self.glcm_entropy_values)
        enl = np.array(self.enl_values)
        pred_entropy = np.array(self.pred_entropy_values)
        ce_loss = np.array(self.ce_loss_values)
        correct = np.array(self.correct_list)

        # Compute summary statistics
        result = {
            # Raw data for correlation analysis
            "ecr": ecr.tolist(),
            "tv": tv.tolist(),
            "kurtosis": kurtosis.tolist(),
            "shannon_entropy": entropy.tolist(),
            "glcm_contrast": glcm_contrast.tolist(),
            "glcm_entropy": glcm_entropy.tolist(),
            "enl": enl.tolist(),
            "prediction_entropy": pred_entropy.tolist(),
            "cross_entropy_loss": ce_loss.tolist(),
            "targets": self.targets_list,
            "correct": correct.tolist(),
            # Summary statistics
            "num_samples": len(self.ecr_values),
            "accuracy": float(np.mean(correct)),
            # Physical indicators statistics
            "ecr_mean": float(np.mean(ecr)),
            "ecr_std": float(np.std(ecr)),
            "tv_mean": float(np.mean(tv)),
            "tv_std": float(np.std(tv)),
            "kurtosis_mean": float(np.mean(kurtosis)),
            "kurtosis_std": float(np.std(kurtosis)),
            "entropy_mean": float(np.mean(entropy)),
            "entropy_std": float(np.std(entropy)),
            "glcm_contrast_mean": float(np.mean(glcm_contrast)),
            "glcm_contrast_std": float(np.std(glcm_contrast)),
            "glcm_entropy_mean": float(np.mean(glcm_entropy)),
            "glcm_entropy_std": float(np.std(glcm_entropy)),
            "enl_mean": float(np.mean(enl)),
            "enl_std": float(np.std(enl)),
            # Model uncertainty statistics
            "pred_entropy_mean": float(np.mean(pred_entropy)),
            "pred_entropy_std": float(np.std(pred_entropy)),
            "ce_loss_mean": float(np.mean(ce_loss)),
            "ce_loss_std": float(np.std(ce_loss)),
            # Correlations with Prediction Entropy (Spearman rank correlation)
            "corr_ecr_pred_entropy": float(stats.spearmanr(ecr, pred_entropy)[0]),
            "corr_tv_pred_entropy": float(stats.spearmanr(tv, pred_entropy)[0]),
            "corr_kurtosis_pred_entropy": float(
                stats.spearmanr(kurtosis, pred_entropy)[0]
            ),
            "corr_entropy_pred_entropy": float(
                stats.spearmanr(entropy, pred_entropy)[0]
            ),
            "corr_glcm_contrast_pred_entropy": float(
                stats.spearmanr(glcm_contrast, pred_entropy)[0]
            ),
            "corr_glcm_entropy_pred_entropy": float(
                stats.spearmanr(glcm_entropy, pred_entropy)[0]
            ),
            "corr_enl_pred_entropy": float(stats.spearmanr(enl, pred_entropy)[0]),
            # Correlations with Cross Entropy Loss (Spearman rank correlation)
            "corr_ecr_ce_loss": float(stats.spearmanr(ecr, ce_loss)[0]),
            "corr_tv_ce_loss": float(stats.spearmanr(tv, ce_loss)[0]),
            "corr_kurtosis_ce_loss": float(stats.spearmanr(kurtosis, ce_loss)[0]),
            "corr_entropy_ce_loss": float(stats.spearmanr(entropy, ce_loss)[0]),
            "corr_glcm_contrast_ce_loss": float(
                stats.spearmanr(glcm_contrast, ce_loss)[0]
            ),
            "corr_glcm_entropy_ce_loss": float(
                stats.spearmanr(glcm_entropy, ce_loss)[0]
            ),
            "corr_enl_ce_loss": float(stats.spearmanr(enl, ce_loss)[0]),
            # Correlation between the two uncertainty measures
            "corr_pred_entropy_ce_loss": float(
                stats.spearmanr(pred_entropy, ce_loss)[0]
            ),
        }

        # Add optional data
        if self.save_logits and len(self.logits_list) > 0:
            result["logits"] = self.logits_list

        if self.save_indices and len(self.indices_list) > 0:
            result["indices"] = self.indices_list

        return result

    def reset(self) -> None:
        """Resets the metric state."""
        self.ecr_values: List[float] = []
        self.tv_values: List[float] = []
        self.kurtosis_values: List[float] = []
        self.entropy_values: List[float] = []
        self.glcm_contrast_values: List[float] = []
        self.glcm_entropy_values: List[float] = []
        self.enl_values: List[float] = []
        self.pred_entropy_values: List[float] = []
        self.ce_loss_values: List[float] = []
        self.targets_list: List[int] = []
        self.correct_list: List[int] = []

        # Optional storage
        self.logits_list: List[List[float]] = []  # [N, num_classes]
        self.indices_list: List[int] = []  # [N]
        self.sample_counter: int = 0  # Auto-increment counter
