# cvlabkit/agent/generative_augmentation_fixmatch.py
"""FixMatch with difficulty-conditioned generator for adaptive augmentation.

Architecture:
    - Classifier M: ResNet18 for classification
    - Generator G: Conditional U-Net (time + difficulty conditioning)
    - Difficulty: Computed from weak predictions (exponential scaling)

Training Strategy:
    - Two-stage: Generator → Classifier (single graph, retain_graph backward)
    - Batch split: First half (conditional), Second half (unconditional)
    - Direct forward at t=1.0 (no ODE during training)

Losses:
    Generator:
        - L_ratio: Difficulty ratio lower bound (conditional only)
        - L_lpips: LPIPS perceptual upper bound (conditional only)
        - L_identity: Identity preservation + difficulty equality (unconditional only)

    Classifier:
        - L_sup: Supervised CE (labeled data)
        - L_pl: Pseudo-label CE (conditional & high confidence)
        - L_cons: Consistency KL (conditional only, no confidence filter)
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cvlabkit.core.agent import Agent


def pil_collate(batch):
    """Custom collate function to handle PIL Images."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


class GenerativeAugmentationFixmatch(Agent):
    """FixMatch with Rectified Flow generator for difficulty-conditioned augmentation."""

    def setup(self):
        """Initialize components."""
        device_id = self.cfg.get("device", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")
        self.current_epoch = 0

        # --- Models ---
        self.model = self.create.model.classifier().to(self.device)
        self.optimizer = self.create.optimizer.classifier(self.model.parameters())

        # Generator: Rectified Flow U-Net for difficulty-conditioned augmentation
        self.generator = self.create.model.generator().to(self.device)
        self.optimizer_gen = self.create.optimizer.generator(
            self.generator.parameters()
        )

        # --- Transforms ---
        self.weak_transform = self.create.transform.weak()
        self.val_transform = self.create.transform.val()

        # --- Loss Functions ---
        self.loss_supervised = self.create.loss.supervised()
        self.loss_cond_upper = self.create.loss.cond_upper().to(self.device)

        # --- Metrics & Logger ---
        self.metric = self.create.metric.val()
        if self.cfg.get("logger"):
            self.logger = self.create.logger()

        # --- Hyperparameters ---
        self.confidence_threshold = self.cfg.get("confidence_threshold", 0.95)
        self.num_classes = self.cfg.get("num_classes", 10)
        self.scale_a = float(self.cfg.get("scale_a", 10.0))

        # Loss weights
        self.lambda_pl = self.cfg.get("lambda_pl", 1.0)
        self.lambda_cons = self.cfg.get("lambda_cons", 0.5)
        self.lambda_ratio = self.cfg.get("lambda_ratio", 1.0)
        self.lambda_lpips = self.cfg.get("lambda_lpips", 1.0)
        self.lambda_identity = self.cfg.get("lambda_identity", 1.0)

        # Difficulty constraints
        self.ode_steps = self.cfg.get("ode_steps", 16)  # Legacy default

        # ODE Solver
        self.solver = self.create.solver()

        # --- Data Handling ---
        train_dataset = self.create.dataset.train()
        val_dataset = self.create.dataset.val()

        num_labeled = self.cfg.num_labeled
        targets = np.array(train_dataset.targets)

        # Load or create labeled indices
        log_dir = self.cfg.get("log_dir", ".")
        dataset_name = self.cfg.dataset.train.split("(")[0]
        os.makedirs(log_dir, exist_ok=True)
        index_file_path = os.path.join(
            log_dir, f"{dataset_name}_labeled_indices_{num_labeled}.json"
        )

        if os.path.exists(index_file_path):
            with open(index_file_path) as f:
                labeled_indices = json.load(f)
            all_indices = set(range(len(targets)))
            unlabeled_indices = list(all_indices - set(labeled_indices))
            np.random.shuffle(unlabeled_indices)
        else:
            labeled_indices, unlabeled_indices = self._stratified_split(
                targets, num_labeled
            )
            with open(index_file_path, "w") as f:
                json.dump(labeled_indices, f)

        labeled_sampler = self.create.sampler.labeled(indices=labeled_indices)
        unlabeled_sampler = self.create.sampler.unlabeled(indices=unlabeled_indices)

        labeled_batch_size = self.cfg.batch_size
        unlabeled_batch_size = labeled_batch_size * self.cfg.get("mu", 7)

        self.labeled_loader = self.create.dataloader.labeled(
            dataset=train_dataset,
            sampler=labeled_sampler,
            collate_fn=pil_collate,
            batch_size=labeled_batch_size,
        )
        self.unlabeled_loader = self.create.dataloader.unlabeled(
            dataset=train_dataset,
            sampler=unlabeled_sampler,
            collate_fn=pil_collate,
            batch_size=unlabeled_batch_size,
        )
        self.val_loader = self.create.dataloader.val(
            dataset=val_dataset, collate_fn=pil_collate
        )

    def _stratified_split(self, targets: np.ndarray, num_labeled: int):
        """Stratified split of labeled/unlabeled indices."""
        indices_by_class = defaultdict(list)
        for i, target in enumerate(targets):
            indices_by_class[target].append(i)

        num_classes = len(indices_by_class)
        num_labeled_per_class = int(np.floor(num_labeled / num_classes))

        labeled_indices = []
        unlabeled_indices = []

        for _class_idx, indices in indices_by_class.items():
            np.random.shuffle(indices)
            actual_num_labeled = min(num_labeled_per_class, len(indices))
            labeled_indices.extend(indices[:actual_num_labeled])
            unlabeled_indices.extend(indices)

        np.random.shuffle(unlabeled_indices)

        print(
            f"Dataset split: {len(labeled_indices)} labeled, "
            f"{len(unlabeled_indices)} unlabeled."
        )
        return labeled_indices, unlabeled_indices

    def compute_difficulty(self, logits):
        """Compute difficulty score using exponential scaling.

        s = ((C^a * exp(-a*H)) - 1) / (C^a - 1)
        where H is entropy, C is num_classes, a is scale_a
        """
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)

        c_pow_a = float(self.num_classes) ** self.scale_a
        difficulty = ((c_pow_a * torch.exp(-self.scale_a * entropy)) - 1.0) / (
            c_pow_a - 1.0 + 1e-12
        )
        return difficulty.clamp(0.0, 1.0)

    def ode_sample(self, x_weak, difficulty=None):
        """Generate augmented image using ODE solver.

        Identity mapping: x_0 = x_1 = x_weak
        Difficulty conditions deviation from identity.
        """
        # Null conditioning: use zeros for unconditional generation
        if difficulty is None:
            difficulty = torch.zeros(x_weak.shape[0], device=self.device)

        times = torch.linspace(0.0, 1.0, self.ode_steps, device=self.device)

        def ode_fn(t, x_t):
            t_batch = t.expand(x_t.shape[0])
            return self.generator(x_t, t_batch, difficulty)

        # Start from weak (identity mapping)
        x_result = self.solver(ode_fn, x_weak, times)

        return x_result

    def train_step(self, labeled_batch, unlabeled_batch):
        """Performs a single training step with two stages: Generator → Classifier."""
        labeled_images_pil, labels = labeled_batch
        unlabeled_images_pil, _ = unlabeled_batch

        # Apply weak augmentation to labeled and unlabeled data
        labeled_images = torch.stack(
            [self.weak_transform(img) for img in labeled_images_pil]
        ).to(self.device)
        labels = labels.to(self.device)

        weak_aug_images = torch.stack(
            [self.weak_transform(img) for img in unlabeled_images_pil]
        ).to(self.device)

        # ========================================
        # Optimized Forward Passes
        # ========================================
        self.generator.train()
        self.model.train()

        # 1. NO_GRAD: Weak forward → Difficulty → Pseudo-labels (no grad needed)
        with torch.no_grad():
            weak_logits = self.model(weak_aug_images)
            difficulty_scores = self.compute_difficulty(weak_logits)
            teacher_probs = torch.softmax(weak_logits, dim=1)
            max_probs, pseudo_labels = torch.max(teacher_probs, dim=1)
            mask_conf = max_probs.ge(self.confidence_threshold).float()

        # 2. Split batch into conditional (first half) and unconditional (second half)
        batch_size = weak_aug_images.shape[0]
        split_idx = batch_size // 2

        # Create masks for conditional and unconditional samples
        mask_cond = torch.zeros(batch_size, device=self.device)
        mask_cond[:split_idx] = 1.0

        # Conditional samples (with difficulty)
        # Unconditional samples (without difficulty)
        weak_aug_images_cond = weak_aug_images[:split_idx]
        weak_aug_images_uncond = weak_aug_images[split_idx:]
        difficulty_scores_cond = difficulty_scores[:split_idx]
        difficulty_scores_uncond = difficulty_scores[split_idx:]

        # 3. GRAD: Generate strong augmentations (conditional) and identity (unconditional)
        # Training: Direct forward at t=1.0 (no ODE needed for image-to-image RF)
        t_final_cond = torch.ones(split_idx, device=self.device)
        t_final_uncond = torch.ones(batch_size - split_idx, device=self.device)
        null_scores_uncond = torch.zeros(batch_size - split_idx, device=self.device)

        strong_aug_images_cond = self.generator(
            weak_aug_images_cond, t_final_cond, difficulty_scores_cond
        )
        strong_aug_images_cond = strong_aug_images_cond.clamp(0, 1)  # Prevent explosion

        identity_images_uncond = self.generator(
            weak_aug_images_uncond, t_final_uncond, null_scores_uncond
        )
        identity_images_uncond = identity_images_uncond.clamp(0, 1)  # Prevent explosion

        # 4. GRAD: Classifier forwards
        sup_preds = self.model(labeled_images)

        # For Generator training: use grad-enabled generated images
        generated_images_gen = torch.cat(
            [strong_aug_images_cond, identity_images_uncond], dim=0
        )
        generated_logits_gen = self.model(generated_images_gen)

        # For Classifier training: detach to prevent graph issues with retain_graph
        generated_images_cls = generated_images_gen.detach()
        generated_logits_cls = self.model(generated_images_cls)

        # Split logits for Generator training (constraint losses)
        strong_logits_cond_gen = generated_logits_gen[:split_idx]
        identity_logits_uncond_gen = generated_logits_gen[split_idx:]

        # Difficulty from generated images (detach for stability)
        strong_difficulty_cond = self.compute_difficulty(strong_logits_cond_gen)
        identity_difficulty_uncond = self.compute_difficulty(identity_logits_uncond_gen)

        # ========================================
        # Generator Loss Computation
        # ========================================

        # Conditional constraint losses (ratio lower bound & LPIPS upper bound)
        lower_bound = 1.0 / self.num_classes  # 1/C in log space

        # Prevent log explosion by clamping difficulties
        strong_diff_safe = strong_difficulty_cond.clamp(1e-3, 1.0 - 1e-3)
        weak_diff_safe = difficulty_scores_cond.detach().clamp(1e-3, 1.0 - 1e-3)
        log_ratio = torch.log((strong_diff_safe + 1e-7) / (weak_diff_safe + 1e-7))
        log_ratio = log_ratio.clamp(-5, 5)  # Prevent extreme values

        loss_ratio = (lower_bound - log_ratio).pow(2).sum() / batch_size

        lpips_distance = self.loss_cond_upper(
            strong_aug_images_cond, weak_aug_images_cond.detach(), reduction="none"
        )
        lpips_mask = (log_ratio >= lower_bound).float().detach()
        loss_lpips = (lpips_mask * lpips_distance).sum() / batch_size

        # Unconditional constraint losses (identity & difficulty equality)
        # Identity: d=0 should output weak (clear signal at t=1.0)
        loss_identity = (
            F.mse_loss(identity_images_uncond, weak_aug_images_uncond, reduction="none")
            .mean(dim=[1, 2, 3])
            .sum()
            / batch_size
        )
        loss_diff_eq = (
            identity_difficulty_uncond - difficulty_scores_uncond.detach()
        ).pow(2).sum() / batch_size

        # Total generator loss
        loss_gen = (
            self.lambda_ratio * loss_ratio
            + self.lambda_lpips * loss_lpips
            + self.lambda_identity * (loss_identity + loss_diff_eq)
        )

        # ========================================
        # Classifier Loss Computation
        # ========================================

        # Supervised loss
        loss_sup = self.loss_supervised(sup_preds, labels)

        # Pseudo-label loss (conditional & high confidence only)
        # Use detached logits for Classifier training
        loss_pl_per_sample = F.cross_entropy(
            generated_logits_cls, pseudo_labels, reduction="none"
        )
        loss_pl = (loss_pl_per_sample * mask_conf).sum() / batch_size

        # Consistency loss (conditional only, no confidence filtering)
        kl_per_sample = F.kl_div(
            F.log_softmax(generated_logits_cls, dim=1),
            teacher_probs,
            reduction="none",
        ).sum(dim=1)
        loss_cons = (kl_per_sample * mask_cond).sum() / batch_size

        # Total classifier loss
        loss_model = loss_sup + self.lambda_pl * loss_pl + self.lambda_cons * loss_cons

        # ========================================
        # Optimized Backward (retain_graph)
        # ========================================

        # Step 1: Update Generator only
        self.optimizer_gen.zero_grad()
        loss_gen.backward(retain_graph=True)  # Keep graph for Classifier backward

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)

        self.optimizer_gen.step()

        # Step 2: Clear Classifier gradients (accumulated from Generator backward via constraint losses)
        self.model.zero_grad()

        # Step 3: Update Classifier only (optimizer.zero_grad is redundant after model.zero_grad)
        loss_model.backward()  # Reuse computational graph

        # Gradient clipping for Classifier
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            "total_loss": loss_model.item() + loss_gen.item(),
            "model_total": loss_model.item(),
            "sup_loss": loss_sup.item(),
            "pl_loss": loss_pl.item(),
            "cons_loss": loss_cons.item(),
            "gen_total": loss_gen.item(),
            "ratio_loss": loss_ratio.item(),
            "lpips_loss": loss_lpips.item(),
            "identity_loss": loss_identity.item(),
            "diff_eq_loss": loss_diff_eq.item(),
        }

    def fit(self):
        """Main training loop."""
        train_epochs = self.cfg.get("epochs", 256)
        target_epochs = self.current_epoch + train_epochs
        steps_per_epoch = self.cfg.get("steps_per_epoch", 1024)

        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)

        while self.current_epoch < target_epochs:
            epoch_losses = defaultdict(float)

            progress_bar = tqdm(
                range(steps_per_epoch),
                desc=f"Epoch [{self.current_epoch + 1}/{target_epochs}]",
            )
            for _step in progress_bar:
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

                progress_bar.set_postfix(loss=f"{loss_dict['total_loss']:.4f}")

            if hasattr(self, "logger") and self.logger is not None:
                log_data = {}
                for key, value in epoch_losses.items():
                    log_key = "train_loss" if key == "total_loss" else f"train_{key}"
                    log_data[log_key] = value / steps_per_epoch
                self.logger.log_metrics(metrics=log_data, step=self.current_epoch + 1)

            self.evaluate()
            self.current_epoch += 1

    def evaluate(self):
        """Evaluate classifier on validation set."""
        self.model.eval()
        self.metric.reset()

        total_val_loss = 0

        with torch.no_grad():
            for images_pil, labels in self.val_loader:
                images = torch.stack(
                    [self.val_transform(img) for img in images_pil]
                ).to(self.device)
                labels = labels.to(self.device)
                preds = self.model(images)

                val_loss = self.loss_supervised(preds, labels)
                total_val_loss += val_loss.item()

                self.metric.update(preds=preds, targets=labels)

        avg_val_loss = total_val_loss / len(self.val_loader)

        metrics = self.metric.compute()
        metrics["val_loss"] = avg_val_loss

        print(f"Epoch {self.current_epoch + 1} Validation Metrics: {metrics}")

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.log_metrics(metrics=metrics, step=self.current_epoch + 1)
