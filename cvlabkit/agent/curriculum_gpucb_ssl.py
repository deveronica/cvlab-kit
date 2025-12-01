"""Curriculum-Based SSL with GP-UCB Online Hyperparameter Optimization.

This agent extends the 3-Factor Curriculum SSL with Gaussian Process Upper Confidence
Bound (GP-UCB) for online hyperparameter optimization during training.

Optimization targets:
1. curriculum_ratio: Weighted sampling bias (0.2 ~ 1.0)
   - Low ratio → prefer easy samples, High ratio → uniform sampling
2. scale_a: Augmentation sensitivity (5.0 ~ 15.0, default 10.0)
3. tau_base: Pseudo-label threshold (0.85 ~ 0.98)

Reference:
- Srinivas et al., "Gaussian Process Optimization in the Bandit Setting" (ICML 2010)
- Thesis Section 3.5: Hyperparameter Optimization
"""

import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats
from scipy.ndimage import sobel
from scipy.spatial.distance import cdist
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

from cvlabkit.core.agent import Agent


# =============================================================================
# Gaussian Process with UCB Acquisition
# =============================================================================


class GaussianProcessUCB:
    """Gaussian Process with Upper Confidence Bound for online optimization.

    Implements GP-UCB algorithm for continuous hyperparameter optimization
    during training. Uses RBF kernel and maintains a history of observations.

    Attributes:
        param_bounds: Dict of {param_name: (min, max)}
        beta: Exploration-exploitation trade-off parameter
        length_scale: RBF kernel length scale
        noise_var: Observation noise variance
    """

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]],
        beta: float = 2.0,
        length_scale: float = 0.2,
        noise_var: float = 1e-4,
        seed: int = 42,
    ):
        """Initialize GP-UCB optimizer.

        Args:
            param_bounds: Parameter search space {name: (min, max)}
            beta: UCB exploration coefficient (higher = more exploration)
            length_scale: RBF kernel length scale (relative to [0,1] normalized space)
            noise_var: Observation noise variance
            seed: Random seed
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(param_bounds)
        self.beta = beta
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.rng = np.random.RandomState(seed)

        # History storage
        self.X: list[np.ndarray] = []  # Normalized parameters [0, 1]
        self.y: list[float] = []  # Observed rewards (delta accuracy)

        # Precompute bounds for normalization
        self.bounds_min = np.array([b[0] for b in param_bounds.values()])
        self.bounds_max = np.array([b[1] for b in param_bounds.values()])

    def _normalize(self, params: dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] space."""
        x = np.array([params[name] for name in self.param_names])
        return (x - self.bounds_min) / (self.bounds_max - self.bounds_min + 1e-8)

    def _denormalize(self, x: np.ndarray) -> dict[str, float]:
        """Denormalize from [0, 1] to original parameter space."""
        values = x * (self.bounds_max - self.bounds_min) + self.bounds_min
        return {name: float(values[i]) for i, name in enumerate(self.param_names)}

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF (Squared Exponential) kernel matrix.

        K(x, x') = exp(-||x - x'||^2 / (2 * l^2))
        """
        sq_dist = cdist(X1, X2, metric="sqeuclidean")
        return np.exp(-sq_dist / (2 * self.length_scale**2))

    def _gp_posterior(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute GP posterior mean and variance at test points.

        Args:
            X_test: Test points, shape (n_test, n_params)

        Returns:
            mean: Posterior mean, shape (n_test,)
            var: Posterior variance, shape (n_test,)
        """
        if len(self.X) == 0:
            # Prior: zero mean, unit variance
            return np.zeros(len(X_test)), np.ones(len(X_test))

        X_train = np.array(self.X)
        y_train = np.array(self.y)

        # Kernel matrices
        K = self._rbf_kernel(X_train, X_train)  # (n, n)
        K += self.noise_var * np.eye(len(K))  # Add noise
        K_s = self._rbf_kernel(X_train, X_test)  # (n, n_test)
        K_ss = self._rbf_kernel(X_test, X_test)  # (n_test, n_test)

        # Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            v = np.linalg.solve(L, K_s)

            # Posterior mean and variance
            mean = K_s.T @ alpha
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-8)  # Numerical stability
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails
            mean = np.zeros(len(X_test))
            var = np.ones(len(X_test))

        return mean, var

    def _ucb_acquisition(self, x: np.ndarray, t: int) -> float:
        """Compute UCB acquisition value.

        UCB(x) = μ(x) + β_t * σ(x)

        where β_t = sqrt(2 * log(t^(d/2 + 2) * π^2 / 3δ))
        Simplified: β_t = beta * sqrt(log(t + 1))
        """
        mean, var = self._gp_posterior(x.reshape(1, -1))
        std = np.sqrt(var[0])

        # Time-varying beta for theoretical guarantees
        beta_t = self.beta * math.sqrt(math.log(t + 2))

        return mean[0] + beta_t * std

    def suggest(self) -> dict[str, float]:
        """Suggest next hyperparameter configuration using GP-UCB.

        Returns:
            Dict of suggested parameters
        """
        t = len(self.X) + 1

        # Random sampling for first few observations
        if len(self.X) < 3:
            x = self.rng.uniform(0, 1, self.n_params)
            return self._denormalize(x)

        # Optimize UCB via random search (simple but effective)
        n_candidates = 100
        candidates = self.rng.uniform(0, 1, (n_candidates, self.n_params))

        best_ucb = -float("inf")
        best_x = candidates[0]

        for x in candidates:
            ucb = self._ucb_acquisition(x, t)
            if ucb > best_ucb:
                best_ucb = ucb
                best_x = x

        return self._denormalize(best_x)

    def update(self, params: dict[str, float], reward: float) -> None:
        """Update GP with new observation.

        Args:
            params: Parameter configuration that was evaluated
            reward: Observed reward (e.g., validation accuracy improvement)
        """
        x = self._normalize(params)
        self.X.append(x)
        self.y.append(reward)

    def get_best(self) -> tuple[dict[str, float], float]:
        """Get best observed parameters.

        Returns:
            Tuple of (best_params, best_reward)
        """
        if len(self.y) == 0:
            return self._denormalize(np.array([0.5] * self.n_params)), 0.0

        best_idx = np.argmax(self.y)
        return self._denormalize(self.X[best_idx]), self.y[best_idx]

    def get_history(self) -> list[dict]:
        """Get optimization history."""
        history = []
        for i, (x, reward) in enumerate(zip(self.X, self.y)):
            params = self._denormalize(x)
            history.append({"step": i, **params, "reward": reward})
        return history


# =============================================================================
# Dynamic Threshold Scheduler (from curriculum_adaptive_ssl)
# =============================================================================


class DynamicThresholdScheduler:
    """FlexMatch-style class-wise dynamic threshold scheduler."""

    def __init__(
        self,
        num_classes: int,
        base_threshold: float = 0.95,
        flex_coefficient: float = 0.95,
        momentum: float = 0.999,
    ):
        self.num_classes = num_classes
        self.base_threshold = base_threshold
        self.flex_coefficient = flex_coefficient
        self.momentum = momentum

        self.class_learning_status = torch.ones(num_classes)
        self.class_counts = torch.zeros(num_classes)
        self.class_above_threshold = torch.zeros(num_classes)
        self.current_step = 0

    def update(
        self,
        probs: torch.Tensor,
        pseudo_labels: torch.Tensor,
        max_probs: torch.Tensor,
    ) -> None:
        """Update learning status based on current batch predictions."""
        self.current_step += 1

        batch_counts = torch.zeros(self.num_classes, device=probs.device)
        batch_above = torch.zeros(self.num_classes, device=probs.device)

        for c in range(self.num_classes):
            class_mask = pseudo_labels == c
            batch_counts[c] = class_mask.sum().float()
            batch_above[c] = (max_probs[class_mask] >= self.base_threshold).sum().float()

        batch_counts = batch_counts.cpu()
        batch_above = batch_above.cpu()

        batch_status = torch.zeros(self.num_classes)
        for c in range(self.num_classes):
            if batch_counts[c] > 0:
                batch_status[c] = batch_above[c] / batch_counts[c]
            else:
                batch_status[c] = self.class_learning_status[c]

        self.class_learning_status = (
            self.momentum * self.class_learning_status + (1 - self.momentum) * batch_status
        )
        self.class_counts += batch_counts
        self.class_above_threshold += batch_above

    def get_thresholds(self) -> torch.Tensor:
        """Get current per-class thresholds."""
        max_status = self.class_learning_status.max()
        if max_status > 0:
            normalized_status = self.class_learning_status / max_status
        else:
            normalized_status = torch.ones(self.num_classes)

        thresholds = self.base_threshold * (
            self.flex_coefficient * normalized_status + (1 - self.flex_coefficient)
        )
        return torch.clamp(thresholds, 0.5, 0.99)

    def compute_mask(
        self,
        max_probs: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mask for pseudo-label selection."""
        thresholds = self.get_thresholds().to(pseudo_labels.device)
        per_sample_thresholds = thresholds[pseudo_labels]
        mask = (max_probs >= per_sample_thresholds).float()
        return mask


# =============================================================================
# SAR Difficulty Score Calculator
# =============================================================================


class SARDifficultyScore:
    """Training-free difficulty metric for SAR images."""

    WEIGHTS = {
        "fisher": {
            "ecr": 0.059268,
            "tv": 0.212757,
            "kurtosis": 0.111432,
            "shannon_entropy": 0.188688,
            "glcm_contrast": 0.171590,
            "glcm_entropy": 0.184704,
            "enl": 0.071561,
        },
    }

    MEAN = {
        "ecr": 0.697683,
        "tv": 5894.310279,
        "kurtosis": 39.904791,
        "shannon_entropy": 5.916306,
        "glcm_contrast": 548.930053,
        "glcm_entropy": 11.214768,
        "enl": 1.614446,
    }

    STD = {
        "ecr": 0.644195,
        "tv": 3024.751877,
        "kurtosis": 32.557467,
        "shannon_entropy": 0.775157,
        "glcm_contrast": 521.660444,
        "glcm_entropy": 1.406019,
        "enl": 0.598112,
    }

    def __init__(self, cfg: dict):
        strategy = cfg.get("weight_strategy", "fisher")
        self.weights = self.WEIGHTS.get(strategy, self.WEIGHTS["fisher"])
        self.glcm_distances = [1]
        self.glcm_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    def _compute_single(self, image) -> float:
        """Compute difficulty score for a single image."""
        if isinstance(image, Image.Image):
            img = np.array(image.convert("L")).astype(np.float32)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                img = image.mean(dim=0).cpu().numpy().astype(np.float32)
            else:
                img = image.cpu().numpy().astype(np.float32)
        else:
            img = np.array(image).astype(np.float32)

        indicators = {
            "ecr": self._compute_ecr(img),
            "tv": self._compute_tv(img),
            "kurtosis": self._compute_kurtosis(img),
            "shannon_entropy": self._compute_shannon_entropy(img),
            "glcm_contrast": self._compute_glcm_contrast(img),
            "glcm_entropy": self._compute_glcm_entropy(img),
            "enl": self._compute_enl(img),
        }

        difficulty = 0.0
        for name, value in indicators.items():
            if not np.isfinite(value):
                value = self.MEAN[name]
            normalized = (value - self.MEAN[name]) / self.STD[name]
            difficulty += self.weights[name] * normalized

        return float(difficulty)

    def _compute_ecr(self, img: np.ndarray) -> float:
        h, w = img.shape
        center_h, center_w = h // 4, w // 4
        center = img[center_h : 3 * center_h, center_w : 3 * center_w]
        e_center = np.sum(center**2)
        e_total = np.sum(img**2) + 1e-8
        return e_center / e_total

    def _compute_tv(self, img: np.ndarray) -> float:
        dx = np.abs(np.diff(img, axis=1))
        dy = np.abs(np.diff(img, axis=0))
        return np.sum(dx) + np.sum(dy)

    def _compute_kurtosis(self, img: np.ndarray) -> float:
        return float(stats.kurtosis(img.flatten()))

    def _compute_shannon_entropy(self, img: np.ndarray) -> float:
        hist, _ = np.histogram(img.flatten(), bins=256, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))

    def _compute_glcm_contrast(self, img: np.ndarray) -> float:
        img_uint8 = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint8, self.glcm_distances, self.glcm_angles, 256, symmetric=True, normed=True)
        return float(graycoprops(glcm, "contrast").mean())

    def _compute_glcm_entropy(self, img: np.ndarray) -> float:
        img_uint8 = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint8, self.glcm_distances, self.glcm_angles, 256, symmetric=True, normed=True)
        glcm_flat = glcm.flatten()
        glcm_flat = glcm_flat[glcm_flat > 0]
        return -np.sum(glcm_flat * np.log2(glcm_flat + 1e-10))

    def _compute_enl(self, img: np.ndarray) -> float:
        mean_val = np.mean(img)
        var_val = np.var(img) + 1e-8
        return (mean_val**2) / var_val


# =============================================================================
# Difficulty-Weighted Sampler
# =============================================================================


class DifficultyWeightedSampler(torch.utils.data.Sampler):
    """Weighted sampler based on difficulty scores.

    Samples are weighted based on their difficulty and curriculum_ratio:
    - Low curriculum_ratio → prefer easy samples (high weight on low difficulty)
    - High curriculum_ratio → uniform sampling (equal weights)

    Weight formula:
        w_i = exp(-difficulty_i * temperature * (1 - curriculum_ratio))

    When curriculum_ratio=1.0, all weights become exp(0)=1 (uniform).
    When curriculum_ratio=0.0, weights heavily favor easy samples.
    """

    def __init__(
        self,
        indices: list[int],
        difficulty_scores: dict[int, float],
        curriculum_ratio: float = 0.5,
        temperature: float = 5.0,
        num_samples: int | None = None,
    ):
        self.indices = indices
        self.difficulty_scores = difficulty_scores
        self.curriculum_ratio = curriculum_ratio
        self.temperature = temperature
        self.num_samples = num_samples or len(indices)

        # Normalize difficulty scores to [0, 1]
        scores = [difficulty_scores.get(idx, 0.0) for idx in indices]
        min_score, max_score = min(scores), max(scores)
        self.normalized_difficulties = {
            idx: (difficulty_scores.get(idx, 0.0) - min_score) / (max_score - min_score + 1e-8)
            for idx in indices
        }

        self._update_weights()

    def _update_weights(self):
        """Recompute sampling weights based on current curriculum_ratio."""
        weights = []
        for idx in self.indices:
            diff = self.normalized_difficulties[idx]
            # Higher curriculum_ratio → more uniform (less weight difference)
            # Lower curriculum_ratio → prefer easy samples
            w = np.exp(-diff * self.temperature * (1 - self.curriculum_ratio))
            weights.append(w)

        weights = np.array(weights)
        self.weights = weights / weights.sum()

    def set_curriculum_ratio(self, ratio: float):
        """Update curriculum ratio and recompute weights."""
        self.curriculum_ratio = np.clip(ratio, 0.0, 1.0)
        self._update_weights()

    def __iter__(self):
        indices = np.random.choice(
            self.indices,
            size=self.num_samples,
            replace=True,
            p=self.weights,
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


# =============================================================================
# Collate Function
# =============================================================================


def pil_collate(batch):
    """Custom collate function to handle PIL Images."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


# =============================================================================
# Main Agent
# =============================================================================


class CurriculumGPUCBSSL(Agent):
    """Curriculum SSL with GP-UCB Online Hyperparameter Optimization.

    This agent combines:
    1. Physical indicator-based curriculum sampling
    2. Entropy-adaptive augmentation
    3. Dynamic threshold scheduling
    4. GP-UCB online optimization of (curriculum_ratio, scale_a, tau_base)

    The GP-UCB optimizer adjusts hyperparameters every N epochs based on
    validation accuracy improvement, balancing exploration and exploitation.
    """

    def setup(self):
        """Initialize all components."""
        # Device setup
        device_id = self.cfg.get("device", 0)
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.current_epoch = 0
        self.current_step = 0

        # Core components
        self.model = self.create.model().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())

        # Transforms
        self.weak_transform = self.create.transform.weak()
        self.strong_transform = self.create.transform.strong()
        self.val_transform = self.create.transform.val()

        # Loss functions
        self.sup_loss_fn = self.create.loss.supervised()
        self.unsup_loss_fn = self.create.loss.unsupervised()
        self.kl_div_fn = torch.nn.KLDivLoss(reduction="batchmean")

        # Metrics & Logger
        self.metric = self.create.metric.val()
        if self.cfg.get("logger"):
            self.logger = self.create.logger()

        # Dataset preparation
        train_dataset = self.create.dataset.train()
        val_dataset = self.create.dataset.val()
        self.num_classes = self.cfg.get("num_classes", 10)

        # Stratified split
        num_labeled = self.cfg.num_labeled
        targets = np.array(train_dataset.targets)
        labeled_indices, unlabeled_indices = self._stratified_split(targets, num_labeled)

        # Setup curriculum (Factor 1)
        self._setup_curriculum(train_dataset, unlabeled_indices)

        # Setup adaptive augmentation params (Factor 2)
        self.scale_a = self.cfg.get("adaptive_augment", {}).get("scale_a", 10.0)

        # Setup dynamic threshold (Factor 3)
        self._setup_dynamic_threshold()

        # Setup dataloaders
        self._setup_dataloaders(train_dataset, val_dataset, labeled_indices, unlabeled_indices)

        # Setup GP-UCB optimizer
        self._setup_gpucb()

        print(f"\n[CurriculumGPUCBSSL] Setup complete")
        print(f"  Device: {self.device}")
        print(f"  Labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")

    def _stratified_split(self, targets: np.ndarray, num_labeled: int):
        """Stratified split for labeled/unlabeled data."""
        indices_by_class = defaultdict(list)
        for i, target in enumerate(targets):
            indices_by_class[target].append(i)

        num_classes = len(indices_by_class)
        per_class = num_labeled // num_classes

        labeled_indices = []
        unlabeled_indices = []

        for _, indices in indices_by_class.items():
            np.random.shuffle(indices)
            labeled_indices.extend(indices[:per_class])
            unlabeled_indices.extend(indices[per_class:])

        return labeled_indices, unlabeled_indices

    def _setup_curriculum(self, dataset, unlabeled_indices: list[int]):
        """Setup physical indicator-based curriculum sampling."""
        curriculum_cfg = self.cfg.get("curriculum", {})
        weight_strategy = curriculum_cfg.get("weight_strategy", "fisher")

        self.difficulty_metric = SARDifficultyScore({"weight_strategy": weight_strategy})

        # Cache file for difficulty scores
        log_dir = self.cfg.get("log_dir", "./logs")
        dataset_name = self.cfg.dataset.train.split("(")[0]
        cache_file = Path(log_dir) / f"{dataset_name}_difficulty_scores_{weight_strategy}.json"

        if cache_file.exists():
            print(f"  Loading cached difficulty scores from {cache_file}")
            with open(cache_file) as f:
                cached_scores = json.load(f)
            self.unlabeled_difficulty_scores = {int(k): v for k, v in cached_scores.items()}
        else:
            print(f"  Computing difficulty scores ({weight_strategy})...")
            self.unlabeled_difficulty_scores = {}
            for idx in tqdm(unlabeled_indices, desc="  Computing difficulty"):
                img, _ = dataset[idx]
                score = self.difficulty_metric._compute_single(img)
                self.unlabeled_difficulty_scores[idx] = score

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(self.unlabeled_difficulty_scores, f)

        # Sort by difficulty (easy → hard)
        scores_for_unlabeled = {
            idx: self.unlabeled_difficulty_scores.get(idx, 0.0) for idx in unlabeled_indices
        }
        sorted_items = sorted(scores_for_unlabeled.items(), key=lambda x: x[1])
        self.sorted_unlabeled_indices = [item[0] for item in sorted_items]
        self.unlabeled_indices = unlabeled_indices

        # Initial curriculum ratio (will be adjusted by GP-UCB)
        self.curriculum_ratio = curriculum_cfg.get("start_ratio", 0.3)

    def _setup_dynamic_threshold(self):
        """Setup FlexMatch-style dynamic threshold scheduler."""
        threshold_cfg = self.cfg.get("dynamic_threshold", {})

        self.threshold_scheduler = DynamicThresholdScheduler(
            num_classes=self.num_classes,
            base_threshold=self.cfg.get("confidence_threshold", 0.95),
            flex_coefficient=threshold_cfg.get("flex_coefficient", 0.95),
        )

    def _setup_dataloaders(self, train_dataset, val_dataset, labeled_indices, unlabeled_indices):
        """Setup data loaders with difficulty-weighted sampling."""
        self.train_dataset = train_dataset

        labeled_sampler = self.create.sampler.labeled(indices=labeled_indices)

        # Difficulty-weighted sampler for unlabeled data
        labeled_batch = self.cfg.batch_size
        unlabeled_batch = labeled_batch * self.cfg.get("mu", 7)
        steps_per_epoch = self.cfg.get("steps_per_epoch", 256)

        self.unlabeled_sampler = DifficultyWeightedSampler(
            indices=unlabeled_indices,
            difficulty_scores=self.unlabeled_difficulty_scores,
            curriculum_ratio=self.curriculum_ratio,
            temperature=self.cfg.get("curriculum", {}).get("temperature", 5.0),
            num_samples=unlabeled_batch * steps_per_epoch,  # Total samples per epoch
        )

        self.labeled_loader = self.create.dataloader.labeled(
            dataset=train_dataset,
            sampler=labeled_sampler,
            collate_fn=pil_collate,
            batch_size=labeled_batch,
        )
        self.unlabeled_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=self.unlabeled_sampler,
            collate_fn=pil_collate,
            batch_size=unlabeled_batch,
            num_workers=self.cfg.get("num_workers", 0),
        )
        self.val_loader = self.create.dataloader.val(dataset=val_dataset, collate_fn=pil_collate)

    def _setup_gpucb(self):
        """Setup GP-UCB optimizer."""
        gpucb_cfg = self.cfg.get("gpucb", {})

        param_bounds = {
            "curriculum_ratio": tuple(gpucb_cfg.get("curriculum_ratio_bounds", [0.2, 1.0])),
            "scale_a": tuple(gpucb_cfg.get("scale_a_bounds", [5.0, 15.0])),
            "tau_base": tuple(gpucb_cfg.get("tau_base_bounds", [0.85, 0.98])),
        }

        self.gpucb = GaussianProcessUCB(
            param_bounds=param_bounds,
            beta=gpucb_cfg.get("beta", 2.0),
            length_scale=gpucb_cfg.get("length_scale", 0.2),
        )

        self.prev_train_acc = 0.0  # For GP-UCB reward computation

        print(f"\n[GP-UCB] Initialized")
        print(f"  Update: every epoch (training accuracy based)")
        print(f"  Parameter bounds: {param_bounds}")

    def _apply_gpucb_params(self, params: dict[str, float]):
        """Apply GP-UCB suggested parameters."""
        self.curriculum_ratio = params["curriculum_ratio"]
        self.scale_a = params["scale_a"]
        self.threshold_scheduler.base_threshold = params["tau_base"]

        # Update sampler's curriculum ratio for weighted sampling
        if hasattr(self, "unlabeled_sampler"):
            self.unlabeled_sampler.set_curriculum_ratio(self.curriculum_ratio)

    def get_curriculum_indices(self) -> list[int]:
        """Get current curriculum subset based on curriculum_ratio."""
        n_use = int(len(self.sorted_unlabeled_indices) * self.curriculum_ratio)
        n_use = max(1, n_use)  # At least 1 sample
        return self.sorted_unlabeled_indices[:n_use]

    def train_step(self, labeled_batch, unlabeled_batch) -> dict[str, float]:
        """Single training step."""
        self.model.train()

        labeled_images_pil, labels = labeled_batch
        unlabeled_images_pil, _ = unlabeled_batch

        # Apply weak transform in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            labeled_images_list = list(executor.map(self.weak_transform, labeled_images_pil))
            unlabeled_weak_list = list(executor.map(self.weak_transform, unlabeled_images_pil))

        labeled_images = torch.stack(labeled_images_list).to(self.device)
        labels = labels.to(self.device)
        unlabeled_weak = torch.stack(unlabeled_weak_list).to(self.device)

        # === 1. Supervised Loss ===
        sup_preds = self.model(labeled_images)
        loss_sup = self.sup_loss_fn(sup_preds, labels)

        # Training accuracy (for GP-UCB reward)
        with torch.no_grad():
            train_acc = (sup_preds.argmax(dim=1) == labels).float().mean().item()

        # === 2. Generate pseudo-labels ===
        with torch.no_grad():
            teacher_logits = self.model(unlabeled_weak)
            probs = torch.softmax(teacher_logits, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)

            # Entropy-based difficulty scores (Factor 2)
            entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
            num_c = float(self.num_classes)
            scale = float(self.scale_a)
            c_pow_a = num_c**scale
            difficulty_scores = ((c_pow_a * torch.exp(-scale * entropy)) - 1.0) / (c_pow_a - 1.0 + 1e-12)
            difficulty_scores = difficulty_scores.clamp(0.0, 1.0)

        # === 3. Dynamic threshold mask (Factor 3) ===
        self.threshold_scheduler.update(probs, pseudo_labels, max_probs)
        mask = self.threshold_scheduler.compute_mask(max_probs, pseudo_labels)

        # === 4. Apply adaptive strong augmentation ===
        difficulty_list = difficulty_scores.cpu().tolist()

        def apply_strong_with_difficulty(args):
            img, diff = args
            return self.strong_transform(img, difficulty_score=diff)

        with ThreadPoolExecutor(max_workers=8) as executor:
            unlabeled_strong_list = list(
                executor.map(apply_strong_with_difficulty, zip(unlabeled_images_pil, difficulty_list))
            )

        unlabeled_strong = torch.stack(unlabeled_strong_list).to(self.device)

        # === 5. Unsupervised Loss ===
        student_preds = self.model(unlabeled_strong)
        loss_unsup = self.unsup_loss_fn(student_preds, pseudo_labels)
        loss_unsup = (loss_unsup * mask).mean()

        # === 6. Consistency Loss (KL Divergence) ===
        log_student_probs = torch.log_softmax(student_preds, dim=1)
        teacher_probs = torch.softmax(teacher_logits.detach(), dim=1)
        loss_kl = self.kl_div_fn(log_student_probs, teacher_probs)

        # === Total Loss ===
        lambda_u = self.cfg.get("lambda_u", 1.0)
        lambda_kl = self.cfg.get("lambda_kl", 0.5)
        total_loss = loss_sup + lambda_u * loss_unsup + lambda_kl * loss_kl

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.current_step += 1

        return {
            "total_loss": total_loss.item(),
            "sup_loss": loss_sup.item(),
            "unsup_loss": loss_unsup.item(),
            "kl_loss": loss_kl.item(),
            "mask_ratio": mask.mean().item(),
            "avg_difficulty": difficulty_scores.mean().item(),
            "train_acc": train_acc,
        }

    def evaluate(self) -> dict:
        """Evaluate model on validation set."""
        self.model.eval()
        self.metric.reset()
        total_loss = 0.0

        with torch.no_grad():
            for images_pil, labels in self.val_loader:
                images = torch.stack([self.val_transform(img) for img in images_pil]).to(self.device)
                labels = labels.to(self.device)

                preds = self.model(images)
                loss = self.sup_loss_fn(preds, labels)
                total_loss += loss.item()
                self.metric.update(preds=preds, targets=labels)

        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metric.compute()
        metrics["val_loss"] = avg_loss

        return metrics

    def fit(self):
        """Main training loop with GP-UCB online optimization."""
        total_epochs = self.cfg.get("epochs", 100)
        steps_per_epoch = self.cfg.get("steps_per_epoch", 256)

        print("\n" + "=" * 60)
        print("Curriculum SSL with GP-UCB Optimization")
        print("=" * 60)
        print(f"  Total epochs: {total_epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  GP-UCB update: every epoch (training accuracy based)")
        print("=" * 60 + "\n")

        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)

        # Get initial parameters from GP-UCB
        current_params = self.gpucb.suggest()
        self._apply_gpucb_params(current_params)
        print(f"[GP-UCB] Initial params: {current_params}")

        while self.current_epoch < total_epochs:
            epoch_losses = defaultdict(float)

            # Print current curriculum status
            curr_indices = self.get_curriculum_indices()
            print(f"\nCurriculum: {len(curr_indices)}/{len(self.unlabeled_indices)} ({self.curriculum_ratio:.1%})")

            # Training loop
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch [{self.current_epoch + 1}/{total_epochs}]")

            for _ in pbar:
                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(self.labeled_loader)
                    labeled_batch = next(labeled_iter)

                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)

                loss_dict = self.train_step(labeled_batch, unlabeled_batch)

                for key, value in loss_dict.items():
                    epoch_losses[key] += value

                pbar.set_postfix(
                    loss=f"{loss_dict['total_loss']:.4f}",
                    mask=f"{loss_dict['mask_ratio']:.2f}",
                )

            # Compute epoch training accuracy
            epoch_train_acc = epoch_losses["train_acc"] / steps_per_epoch

            # Evaluate (for logging, not for GP-UCB reward)
            metrics = self.evaluate()
            val_acc = metrics.get("val_accuracy", metrics.get("accuracy", 0.0))
            print(f"Epoch {self.current_epoch + 1} | Train Acc: {epoch_train_acc:.4f} | Val: {metrics}")

            # GP-UCB update every epoch (based on training accuracy)
            delta_train_acc = epoch_train_acc - self.prev_train_acc
            self.gpucb.update(current_params, delta_train_acc)

            # Get new parameters for next epoch
            current_params = self.gpucb.suggest()
            self._apply_gpucb_params(current_params)

            print(f"[GP-UCB] Reward (Δtrain_acc): {delta_train_acc:+.4f} → "
                  f"curriculum={current_params['curriculum_ratio']:.3f}, "
                  f"scale_a={current_params['scale_a']:.2f}, "
                  f"tau={current_params['tau_base']:.3f}")

            self.prev_train_acc = epoch_train_acc

            # Log metrics
            if hasattr(self, "logger") and self.logger is not None:
                log_data = {f"train_{k}": v / steps_per_epoch for k, v in epoch_losses.items()}
                log_data.update(metrics)
                log_data["curriculum_ratio"] = self.curriculum_ratio
                log_data["scale_a"] = self.scale_a
                log_data["tau_base"] = self.threshold_scheduler.base_threshold
                self.logger.log_metrics(log_data, step=self.current_epoch + 1)

            self.current_epoch += 1

        # Save GP-UCB history
        log_dir = Path(self.cfg.get("log_dir", "./logs"))
        history_file = log_dir / "gpucb_history.json"
        with open(history_file, "w") as f:
            json.dump(self.gpucb.get_history(), f, indent=2)
        print(f"\n[GP-UCB] History saved to {history_file}")

        best_params, best_reward = self.gpucb.get_best()
        print(f"[GP-UCB] Best params: {best_params}, Best reward: {best_reward:.4f}")
