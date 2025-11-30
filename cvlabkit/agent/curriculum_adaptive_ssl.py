"""Curriculum-Based Difficulty-Aware Augmentation for SAR Semi-Supervised Learning.

3-Factor Framework:
1. Physical Indicator-based Curriculum Sampling
2. Entropy-based Adaptive Augmentation (scaling)
3. Class-wise Dynamic Threshold (FlexMatch-style)

This agent integrates all three factors for optimal SAR SSL performance.

Reference: Curriculum-Based Difficulty-Aware Augmentation for SAR Semi-Supervised Learning
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats
from scipy.ndimage import sobel
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

from cvlabkit.core.agent import Agent

# =============================================================================
# SAR Difficulty Score Calculator
# =============================================================================


class SARDifficultyScore:
    """Training-free difficulty metric for SAR images.

    Computes difficulty score from 7 physical indicators:
    ECR, TV, Kurtosis, Shannon Entropy, GLCM Contrast, GLCM Entropy, ENL
    """

    # Pre-computed weights and normalization params (from MSTAR analysis)
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
        "xgboost": {
            "ecr": 0.113,
            "tv": 0.139,
            "kurtosis": 0.149,
            "shannon_entropy": 0.152,
            "glcm_contrast": 0.137,
            "glcm_entropy": 0.153,
            "enl": 0.156,
        },
        "equal": dict.fromkeys(["ecr", "tv", "kurtosis", "shannon_entropy", "glcm_contrast", "glcm_entropy", "enl"], 1 / 7),
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
        if isinstance(strategy, dict):
            self.weights = strategy
        else:
            self.weights = self.WEIGHTS.get(strategy, self.WEIGHTS["fisher"])

        self.lf_ratio = 0.25
        self.glcm_distances = [1]
        self.glcm_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.num_bins = 256

    def _compute_single(self, image) -> float:
        """Compute difficulty score for a single image."""
        # Convert to grayscale numpy array
        if isinstance(image, Image.Image):
            img = np.array(image.convert("L")).astype(np.float32)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                img = image.mean(dim=0).cpu().numpy().astype(np.float32)
            else:
                img = image.cpu().numpy().astype(np.float32)
        else:
            img = np.array(image).astype(np.float32)

        # Compute indicators
        indicators = {
            "ecr": self._compute_ecr(img),
            "tv": self._compute_tv(img),
            "kurtosis": self._compute_kurtosis(img),
            "shannon_entropy": self._compute_shannon_entropy(img),
            "glcm_contrast": self._compute_glcm_contrast(img),
            "glcm_entropy": self._compute_glcm_entropy(img),
            "enl": self._compute_enl(img),
        }

        # Normalize and weight
        difficulty = 0.0
        for name, value in indicators.items():
            if not np.isfinite(value):
                value = self.MEAN[name]
            normalized = (value - self.MEAN[name]) / self.STD[name]
            difficulty += self.weights[name] * normalized

        return float(difficulty)

    def _compute_ecr(self, img: np.ndarray) -> float:
        h, w = img.shape
        ch, cw = int(h * self.lf_ratio), int(w * self.lf_ratio)
        sh, sw = (h - ch) // 2, (w - cw) // 2
        lf_energy = np.sum(img[sh : sh + ch, sw : sw + cw] ** 2)
        hf_energy = np.sum(img**2) - lf_energy
        return lf_energy / hf_energy if hf_energy > 1e-10 else float("inf")

    def _compute_tv(self, img: np.ndarray) -> float:
        return float(np.sum(np.sqrt(sobel(img, axis=1) ** 2 + sobel(img, axis=0) ** 2)))

    def _compute_kurtosis(self, img: np.ndarray) -> float:
        return float(stats.kurtosis(img.flatten(), fisher=True))

    def _compute_shannon_entropy(self, img: np.ndarray) -> float:
        hist, _ = np.histogram(img.flatten(), bins=self.num_bins, density=True)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        return float(-np.sum(hist * np.log2(hist + 1e-10)))

    def _compute_glcm_contrast(self, img: np.ndarray) -> float:
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
        glcm = graycomatrix(img_norm, self.glcm_distances, self.glcm_angles, 256, symmetric=True, normed=True)
        return float(np.mean(graycoprops(glcm, "contrast")))

    def _compute_glcm_entropy(self, img: np.ndarray) -> float:
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
        glcm = graycomatrix(img_norm, self.glcm_distances, self.glcm_angles, 256, symmetric=True, normed=True)
        entropies = []
        for d in range(len(self.glcm_distances)):
            for a in range(len(self.glcm_angles)):
                g = glcm[:, :, d, a] + 1e-10
                entropies.append(-np.sum(g * np.log2(g + 1e-10)))
        return float(np.mean(entropies))

    def _compute_enl(self, img: np.ndarray) -> float:
        mean, var = np.mean(img), np.var(img)
        return (mean**2) / var if var > 1e-10 else float("inf")


# =============================================================================
# Dynamic Threshold Scheduler (FlexMatch-style)
# =============================================================================


class DynamicThresholdScheduler:
    """FlexMatch-style dynamic threshold scheduler.

    Adjusts pseudo-label confidence threshold per class based on learning status.
    """

    def __init__(
        self,
        num_classes: int,
        base_threshold: float = 0.95,
        flex_coefficient: float = 0.95,
        momentum: float = 0.999,
        warmup_steps: int = 0,
    ):
        self.num_classes = num_classes
        self.base_threshold = base_threshold
        self.flex_coefficient = flex_coefficient
        self.momentum = momentum
        self.warmup_steps = warmup_steps

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
            batch_above[c] = (
                (max_probs[class_mask] >= self.base_threshold).sum().float()
            )

        batch_counts = batch_counts.cpu()
        batch_above = batch_above.cpu()

        batch_status = torch.zeros(self.num_classes)
        for c in range(self.num_classes):
            if batch_counts[c] > 0:
                batch_status[c] = batch_above[c] / batch_counts[c]
            else:
                batch_status[c] = self.class_learning_status[c]

        self.class_learning_status = (
            self.momentum * self.class_learning_status
            + (1 - self.momentum) * batch_status
        )

        self.class_counts += batch_counts
        self.class_above_threshold += batch_above

    def get_thresholds(self) -> torch.Tensor:
        """Get current per-class thresholds."""
        if self.current_step < self.warmup_steps:
            return torch.full((self.num_classes,), self.base_threshold)

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
        """Compute mask for pseudo-label selection using dynamic thresholds."""
        thresholds = self.get_thresholds()
        per_sample_thresholds = thresholds[pseudo_labels]
        mask = (max_probs >= per_sample_thresholds.to(max_probs.device)).float()
        return mask

    def get_stats(self) -> dict[str, float]:
        """Get current scheduler statistics."""
        thresholds = self.get_thresholds()
        return {
            "threshold_mean": thresholds.mean().item(),
            "threshold_min": thresholds.min().item(),
            "threshold_max": thresholds.max().item(),
        }


class CurriculumThresholdScheduler(DynamicThresholdScheduler):
    """Extended threshold scheduler with curriculum-aware adjustment."""

    def __init__(
        self,
        num_classes: int,
        base_threshold: float = 0.95,
        initial_threshold: float = 0.7,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        flex_coefficient: float = 0.95,
        momentum: float = 0.999,
    ):
        super().__init__(
            num_classes=num_classes,
            base_threshold=base_threshold,
            flex_coefficient=flex_coefficient,
            momentum=momentum,
        )
        self.initial_threshold = initial_threshold
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for curriculum scheduling."""
        self.current_epoch = epoch

    def get_curriculum_threshold(self) -> float:
        """Get base threshold adjusted by curriculum schedule."""
        if self.current_epoch >= self.warmup_epochs:
            return self.base_threshold

        progress = self.current_epoch / self.warmup_epochs
        return self.initial_threshold + progress * (
            self.base_threshold - self.initial_threshold
        )

    def get_thresholds(self) -> torch.Tensor:
        """Get per-class thresholds with curriculum adjustment."""
        curriculum_base = self.get_curriculum_threshold()

        max_status = self.class_learning_status.max()
        if max_status > 0:
            normalized_status = self.class_learning_status / max_status
        else:
            normalized_status = torch.ones(self.num_classes)

        thresholds = curriculum_base * (
            self.flex_coefficient * normalized_status + (1 - self.flex_coefficient)
        )

        return torch.clamp(thresholds, 0.5, 0.99)


# =============================================================================
# Agent
# =============================================================================


def pil_collate(batch):
    """Custom collate function to handle PIL Images."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


class CurriculumAdaptiveSSL(Agent):
    """3-Factor Curriculum-based Adaptive Semi-Supervised Learning Agent.

    Integrates:
    1. Physical indicator-based curriculum sampling (easy → hard)
    2. Entropy-adaptive augmentation (uncertain → weak aug)
    3. Class-wise dynamic thresholding (FlexMatch-style)

    Config Requirements:
        model: Model architecture (e.g., resnet18)
        optimizer: Optimizer (e.g., adam)
        dataset.train: Training dataset
        dataset.val: Validation dataset
        transform.weak: Weak augmentation
        transform.val: Validation transform

        # SSL parameters
        num_labeled: Number of labeled samples
        confidence_threshold: Base confidence threshold (default: 0.95)
        lambda_u: Unsupervised loss weight (default: 1.0)
        mu: Unlabeled batch multiplier (default: 7)

        # 3-Factor parameters
        curriculum:
            enabled: true
            weight_strategy: 'fisher' | 'xgboost' | 'equal'
            start_ratio: 0.3
            saturation_epoch: 50
            schedule: 'linear' | 'exponential'

        adaptive_augment:
            enabled: true
            magnitude_min: 2
            magnitude_max: 10
            lambda_aug: 2.0

        dynamic_threshold:
            enabled: true
            initial_threshold: 0.7
            warmup_epochs: 10
            flex_coefficient: 0.95
    """

    # Ablation study mapping: ablation_name -> (curriculum, adaptive, threshold)
    ABLATION_FLAGS = {
        "baseline": (False, False, False),
        "curriculum_only": (True, False, False),
        "adaptive_only": (False, True, False),
        "threshold_only": (False, False, True),
        "curriculum_adaptive": (True, True, False),
        "curriculum_threshold": (True, False, True),
        "adaptive_threshold": (False, True, True),
        "full_proposed": (True, True, True),
    }

    def _resolve_ablation_flags(self) -> tuple[bool, bool, bool]:
        """Resolve factor enable flags from ablation_name or explicit config."""
        ablation_name = self.cfg.get("ablation_name")
        if ablation_name and ablation_name in self.ABLATION_FLAGS:
            return self.ABLATION_FLAGS[ablation_name]

        # Fall back to explicit config
        curriculum = self.cfg.get("curriculum", {}).get("enabled", True)
        adaptive = self.cfg.get("adaptive_augment", {}).get("enabled", True)
        threshold = self.cfg.get("dynamic_threshold", {}).get("enabled", True)
        return (curriculum, adaptive, threshold)

    def setup(self):
        """Initialize all components for 3-Factor SSL."""
        device_id = self.cfg.get("device", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

        self.current_epoch = 0
        self.current_step = 0

        # --- Core Components ---
        self.model = self.create.model().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())

        # Transforms
        self.weak_transform = self.create.transform.weak()
        self.strong_transform = self.create.transform.strong()
        self.val_transform = self.create.transform.val()

        # Loss functions
        self.sup_loss_fn = self.create.loss.supervised()
        self.unsup_loss_fn = self.create.loss.unsupervised()

        # Metrics
        self.metric = self.create.metric.val()

        # Logger
        if self.cfg.get("logger"):
            self.logger = self.create.logger()

        # --- Dataset Preparation ---
        train_dataset = self.create.dataset.train()
        val_dataset = self.create.dataset.val()
        self.num_classes = self.cfg.get("num_classes", 10)

        # Stratified split for labeled/unlabeled
        num_labeled = self.cfg.num_labeled
        targets = np.array(train_dataset.targets)
        labeled_indices, unlabeled_indices = self._prepare_data_split(
            train_dataset, targets, num_labeled
        )

        # --- Resolve Factor Flags (supports ablation_name) ---
        (
            self.curriculum_enabled,
            self.adaptive_aug_enabled,
            self.dynamic_threshold_enabled,
        ) = self._resolve_ablation_flags()

        # --- Factor 1: Curriculum Sampling ---
        if self.curriculum_enabled:
            self._setup_curriculum(train_dataset, unlabeled_indices)

        # --- Factor 2: Adaptive Augmentation ---
        if self.adaptive_aug_enabled:
            self._setup_adaptive_augment()

        # --- Factor 3: Dynamic Threshold ---
        if self.dynamic_threshold_enabled:
            self._setup_dynamic_threshold()

        # --- Data Loaders ---
        self._setup_dataloaders(
            train_dataset, val_dataset, labeled_indices, unlabeled_indices
        )

        # --- Logging and Stats ---
        self.training_stats = defaultdict(list)

    def _prepare_data_split(
        self, dataset, targets: np.ndarray, num_labeled: int
    ) -> tuple[list[int], list[int]]:
        """Prepare stratified labeled/unlabeled split."""
        log_dir = Path(self.cfg.get("log_dir", "."))
        dataset_name = self.cfg.dataset.train.split("(")[0]
        log_dir.mkdir(parents=True, exist_ok=True)
        index_file = log_dir / f"{dataset_name}_labeled_{num_labeled}.json"

        if index_file.exists():
            print(f"Loading indices from {index_file}")
            with index_file.open() as f:
                labeled_indices = json.load(f)
            unlabeled_indices = list(set(range(len(targets))) - set(labeled_indices))
        else:
            labeled_indices, unlabeled_indices = self._stratified_split(
                targets, num_labeled
            )
            with index_file.open("w") as f:
                json.dump(labeled_indices, f)

        print(
            f"Data split: {len(labeled_indices)} labeled, {len(unlabeled_indices)} unlabeled"
        )
        return labeled_indices, unlabeled_indices

    def _stratified_split(
        self, targets: np.ndarray, num_labeled: int
    ) -> tuple[list[int], list[int]]:
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
        print("\n[Factor 1] Setting up Curriculum Sampling...")
        curriculum_cfg = self.cfg.get("curriculum", {})

        # Difficulty score calculator
        weight_strategy = curriculum_cfg.get("weight_strategy", "fisher")
        self.difficulty_metric = SARDifficultyScore(
            {"weight_strategy": weight_strategy}
        )

        # Compute difficulty scores for unlabeled data
        print(f"  Computing difficulty scores ({weight_strategy})...")
        self.unlabeled_difficulty_scores = {}
        for idx in tqdm(unlabeled_indices, desc="  Computing difficulty"):
            img, _ = dataset[idx]
            score = self.difficulty_metric._compute_single(img)
            self.unlabeled_difficulty_scores[idx] = score

        # Sort indices by difficulty (easy → hard)
        sorted_items = sorted(
            self.unlabeled_difficulty_scores.items(), key=lambda x: x[1]
        )
        self.sorted_unlabeled_indices = [item[0] for item in sorted_items]

        # Curriculum parameters
        self.curriculum_start_ratio = curriculum_cfg.get("start_ratio", 0.3)
        self.curriculum_saturation_epoch = curriculum_cfg.get("saturation_epoch", 50)
        self.curriculum_schedule = curriculum_cfg.get("schedule", "linear")

        print(f"  ✓ Difficulty scores computed: {len(unlabeled_indices)} samples")
        print(f"  Start ratio: {self.curriculum_start_ratio:.1%}")
        print(f"  Saturation epoch: {self.curriculum_saturation_epoch}")
        print(f"  Schedule: {self.curriculum_schedule}")

    def _setup_adaptive_augment(self):
        """Setup entropy-adaptive augmentation with exponential scaling.

        Uses the same exponential mapping as AdaptiveAugmentationFixmatch:
        f(H) = (C^a * exp(-a * H) - 1) / (C^a - 1)

        where C = num_classes, a = scale_a, H = entropy
        """
        print("\n[Factor 2] Setting up Adaptive Augmentation...")
        aug_cfg = self.cfg.get("adaptive_augment", {})

        # Store parameters for exponential scaling
        self.scale_a = aug_cfg.get("scale_a", 10.0)

        print(f"  Exponential scaling: a = {self.scale_a}")
        print(f"  Classes: {self.num_classes}")

    def _setup_dynamic_threshold(self):
        """Setup FlexMatch-style dynamic threshold scheduler."""
        print("\n[Factor 3] Setting up Dynamic Threshold...")
        threshold_cfg = self.cfg.get("dynamic_threshold", {})

        self.threshold_scheduler = CurriculumThresholdScheduler(
            num_classes=self.num_classes,
            base_threshold=self.cfg.get("confidence_threshold", 0.95),
            initial_threshold=threshold_cfg.get("initial_threshold", 0.7),
            warmup_epochs=threshold_cfg.get("warmup_epochs", 10),
            total_epochs=self.cfg.get("epochs", 100),
            flex_coefficient=threshold_cfg.get("flex_coefficient", 0.95),
        )

        print(f"  Base threshold: {self.cfg.get('confidence_threshold', 0.95)}")
        print(f"  Initial threshold: {threshold_cfg.get('initial_threshold', 0.7)}")
        print(f"  Flex coefficient: {threshold_cfg.get('flex_coefficient', 0.95)}")

    def _setup_dataloaders(
        self,
        train_dataset,
        val_dataset,
        labeled_indices: list[int],
        unlabeled_indices: list[int],
    ):
        """Setup data loaders."""
        self.train_dataset = train_dataset
        self.unlabeled_indices = unlabeled_indices

        labeled_sampler = self.create.sampler.labeled(indices=labeled_indices)
        unlabeled_sampler = self.create.sampler.unlabeled(indices=unlabeled_indices)

        labeled_batch = self.cfg.batch_size
        unlabeled_batch = labeled_batch * self.cfg.get("mu", 7)

        self.labeled_loader = self.create.dataloader.labeled(
            dataset=train_dataset,
            sampler=labeled_sampler,
            collate_fn=pil_collate,
            batch_size=labeled_batch,
        )
        self.unlabeled_loader = self.create.dataloader.unlabeled(
            dataset=train_dataset,
            sampler=unlabeled_sampler,
            collate_fn=pil_collate,
            batch_size=unlabeled_batch,
        )
        self.val_loader = self.create.dataloader.val(
            dataset=val_dataset, collate_fn=pil_collate
        )

    def get_curriculum_indices(self) -> list[int]:
        """Get current curriculum subset of unlabeled indices."""
        if not self.curriculum_enabled:
            return self.unlabeled_indices

        # Compute current ratio
        if self.curriculum_schedule == "linear":
            progress = min(self.current_epoch / self.curriculum_saturation_epoch, 1.0)
            ratio = self.curriculum_start_ratio + progress * (
                1.0 - self.curriculum_start_ratio
            )
        elif self.curriculum_schedule == "exponential":
            progress = min(self.current_epoch / self.curriculum_saturation_epoch, 1.0)
            ratio = self.curriculum_start_ratio + (1 - np.exp(-3 * progress)) * (
                1.0 - self.curriculum_start_ratio
            ) / (1 - np.exp(-3))
        else:
            ratio = 1.0

        n_use = int(len(self.sorted_unlabeled_indices) * ratio)
        return self.sorted_unlabeled_indices[:n_use]

    def train_step(self, labeled_batch, unlabeled_batch) -> dict[str, float]:
        """Single training step with 3-Factor optimization."""
        self.model.train()

        labeled_images_pil, labels = labeled_batch
        unlabeled_images_pil, _ = unlabeled_batch

        # Apply weak transform to labeled data
        labeled_images = torch.stack(
            [self.weak_transform(img) for img in labeled_images_pil]
        ).to(self.device)
        labels = labels.to(self.device)

        # Apply weak transform to unlabeled data (for pseudo-label generation)
        unlabeled_weak = torch.stack(
            [self.weak_transform(img) for img in unlabeled_images_pil]
        ).to(self.device)

        # === 1. Supervised Loss ===
        sup_preds = self.model(labeled_images)
        loss_sup = self.sup_loss_fn(sup_preds, labels)

        # === 2. Generate pseudo-labels with weak augmentation ===
        with torch.no_grad():
            teacher_logits = self.model(unlabeled_weak)
            probs = torch.softmax(teacher_logits, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)

            # Factor 2: Compute entropy and difficulty scores (exponential scaling)
            if self.adaptive_aug_enabled:
                entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
                # Exponential mapping: f(H) = (C^a * exp(-a*H) - 1) / (C^a - 1)
                num_c = float(self.num_classes)
                scale = float(self.scale_a)
                c_pow_a = num_c**scale
                difficulty_scores = ((c_pow_a * torch.exp(-scale * entropy)) - 1.0) / (
                    c_pow_a - 1.0 + 1e-12
                )
                difficulty_scores = difficulty_scores.clamp(0.0, 1.0)
            else:
                difficulty_scores = None

        # === Factor 3: Dynamic threshold mask ===
        if self.dynamic_threshold_enabled:
            # Update threshold scheduler
            self.threshold_scheduler.update(probs, pseudo_labels, max_probs)
            mask = self.threshold_scheduler.compute_mask(max_probs, pseudo_labels)
        else:
            # Fixed threshold
            threshold = self.cfg.get("confidence_threshold", 0.95)
            mask = max_probs.ge(threshold).float()

        # === Factor 2: Apply entropy-adaptive strong augmentation ===
        unlabeled_strong_list = []

        for i, img in enumerate(unlabeled_images_pil):
            if self.adaptive_aug_enabled and difficulty_scores is not None:
                # Pass difficulty score to strong transform
                aug_img = self.strong_transform(
                    img, difficulty_score=difficulty_scores[i].item()
                )
            else:
                # Fixed strong augmentation
                aug_img = self.strong_transform(img)

            unlabeled_strong_list.append(aug_img)

        unlabeled_strong = torch.stack(unlabeled_strong_list).to(self.device)

        # === 3. Unsupervised Loss ===
        student_preds = self.model(unlabeled_strong)
        loss_unsup = self.unsup_loss_fn(student_preds, pseudo_labels)
        loss_unsup = (loss_unsup * mask).mean()

        # === Total Loss ===
        lambda_u = self.cfg.get("lambda_u", 1.0)
        total_loss = loss_sup + lambda_u * loss_unsup

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.current_step += 1

        # Collect stats
        stats = {
            "total_loss": total_loss.item(),
            "sup_loss": loss_sup.item(),
            "unsup_loss": loss_unsup.item(),
            "mask_ratio": mask.mean().item(),
        }

        if self.adaptive_aug_enabled and difficulty_scores is not None:
            stats["avg_difficulty"] = difficulty_scores.mean().item()

        if self.dynamic_threshold_enabled:
            threshold_stats = self.threshold_scheduler.get_stats()
            stats["threshold_mean"] = threshold_stats["threshold_mean"]

        return stats

    def fit(self):
        """Main training loop with curriculum progression."""
        total_epochs = self.cfg.get("epochs", 100)
        steps_per_epoch = self.cfg.get("steps_per_epoch", 1024)

        print("\n" + "=" * 60)
        print("Starting 3-Factor Curriculum SSL Training")
        print("=" * 60)
        print(f"  Factor 1 (Curriculum): {self.curriculum_enabled}")
        print(f"  Factor 2 (Adaptive Aug): {self.adaptive_aug_enabled}")
        print(f"  Factor 3 (Dynamic τ): {self.dynamic_threshold_enabled}")
        print("=" * 60 + "\n")

        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)

        while self.current_epoch < total_epochs:
            epoch_losses = defaultdict(float)

            # Update curriculum and threshold for new epoch
            if self.curriculum_enabled:
                curr_indices = self.get_curriculum_indices()
                print(
                    f"Curriculum: {len(curr_indices)}/{len(self.unlabeled_indices)} "
                    f"({100 * len(curr_indices) / len(self.unlabeled_indices):.1f}%)"
                )

            if self.dynamic_threshold_enabled:
                self.threshold_scheduler.set_epoch(self.current_epoch)

            # Training loop
            pbar = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch [{self.current_epoch + 1}/{total_epochs}]",
            )

            for _step in pbar:
                # Get batches
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

                # Train step
                loss_dict = self.train_step(labeled_batch, unlabeled_batch)

                for key, value in loss_dict.items():
                    epoch_losses[key] += value

                pbar.set_postfix(
                    loss=f"{loss_dict['total_loss']:.4f}",
                    mask=f"{loss_dict['mask_ratio']:.2f}",
                )

            # Log epoch metrics
            if hasattr(self, "logger") and self.logger is not None:
                log_data = {}
                for key, value in epoch_losses.items():
                    log_data[f"train_{key}"] = value / steps_per_epoch
                self.logger.log_metrics(log_data, step=self.current_epoch + 1)

            # Evaluate
            self.evaluate()
            self.current_epoch += 1

    def evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        self.metric.reset()

        total_loss = 0.0

        with torch.no_grad():
            for images_pil, labels in self.val_loader:
                images = torch.stack(
                    [self.val_transform(img) for img in images_pil]
                ).to(self.device)
                labels = labels.to(self.device)

                preds = self.model(images)
                loss = self.sup_loss_fn(preds, labels)
                total_loss += loss.item()

                self.metric.update(preds=preds, targets=labels)

        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metric.compute()
        metrics["val_loss"] = avg_loss

        print(f"Epoch {self.current_epoch + 1} Validation: {metrics}")

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.log_metrics(metrics, step=self.current_epoch + 1)
