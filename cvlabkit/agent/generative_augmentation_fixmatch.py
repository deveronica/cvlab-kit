# cvlabkit/agent/generative_augmentation_fixmatch.py
"""FixMatch with Rectified Flow generator for difficulty-conditioned augmentation.

Architecture:
    - Classifier M: ResNet18 for classification
    - Generator: Rectified Flow U-Net with difficulty conditioning
    - Difficulty: Computed from weak predictions (exponential scaling)

Training Strategy:
    - Two-stage: Generator → Classifier
    - Batch split: First half (conditional), Second half (unconditional)
    - Identity mapping: x_0 = x_1 = weak, difficulty conditions deviation

Losses:
    Generator:
        - L_RF: Rectified Flow velocity matching (identity mapping)
        - L_ratio: Difficulty ratio lower bound (conditional only)
        - L_lpips: LPIPS upper bound (conditional only)
        - L_id: Identity preservation (unconditional only)
        - L_diff_eq: Difficulty equality (unconditional only)

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
        self.optimizer_gen = self.create.optimizer.generator(self.generator.parameters())

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
        self.lambda_rf = self.cfg.get("lambda_rf", 1.0)
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

    @torch.no_grad()
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
        labeled_images = torch.stack([self.weak_transform(img) for img in labeled_images_pil]).to(self.device)
        labels = labels.to(self.device)

        weak_aug_images = torch.stack([self.weak_transform(img) for img in unlabeled_images_pil]).to(self.device)

        # ========================================
        # Stage 1: Generator Training
        # ========================================
        self.generator.train()
        self.model.eval()

        # 1. Compute difficulty scores from weak augmentation predictions
        with torch.no_grad():
            weak_logits = self.model(weak_aug_images)
            weak_probs = torch.softmax(weak_logits, dim=1)

            # Exponential scaling difficulty: s = ((C^a * exp(-a*H)) - 1) / (C^a - 1)
            entropy = -torch.sum(weak_probs * torch.log(weak_probs + 1e-7), dim=1)
            C = self.num_classes
            a = self.scale_a
            Ca = (float(C) ** a)
            difficulty_scores = ((Ca * torch.exp(-a * entropy)) - 1.0) / (Ca - 1.0 + 1e-12)
            difficulty_scores = difficulty_scores.clamp(0.0, 1.0)

        # 2. Split batch into conditional (first half) and unconditional (second half)
        batch_size = weak_aug_images.shape[0]
        split_idx = batch_size // 2

        # Create masks for conditional and unconditional samples
        mask_cond = torch.zeros(batch_size, device=self.device)
        mask_cond[:split_idx] = 1.0
        mask_uncond = 1.0 - mask_cond

        # Conditional samples (with difficulty)
        weak_aug_images_cond = weak_aug_images[:split_idx]
        difficulty_scores_cond = difficulty_scores[:split_idx]

        # Unconditional samples (without difficulty)
        weak_aug_images_uncond = weak_aug_images[split_idx:]
        difficulty_scores_uncond = difficulty_scores[split_idx:]

        # 3. Rectified Flow Loss (identity mapping: x_0 = x_1 = weak_aug_images)
        # For conditional samples: use difficulty conditioning
        time_cond = torch.rand(split_idx, device=self.device)
        time_cond_broadcast = time_cond.view(-1, 1, 1, 1)
        interpolated_cond = (1 - time_cond_broadcast) * weak_aug_images_cond + time_cond_broadcast * weak_aug_images_cond
        target_velocity_cond = weak_aug_images_cond - weak_aug_images_cond  # Zero velocity (identity)
        predicted_velocity_cond = self.generator(interpolated_cond, time_cond, difficulty_scores_cond)
        loss_rf_cond = F.mse_loss(predicted_velocity_cond, target_velocity_cond, reduction="none").mean(dim=[1,2,3])

        # For unconditional samples: use null conditioning
        time_uncond = torch.rand(batch_size - split_idx, device=self.device)
        time_uncond_broadcast = time_uncond.view(-1, 1, 1, 1)
        interpolated_uncond = (1 - time_uncond_broadcast) * weak_aug_images_uncond + time_uncond_broadcast * weak_aug_images_uncond
        target_velocity_uncond = weak_aug_images_uncond - weak_aug_images_uncond  # Zero velocity (identity)
        # Null conditioning: pass None for cond
        null_cond = torch.zeros(batch_size - split_idx, device=self.device)
        predicted_velocity_uncond = self.generator(interpolated_uncond, time_uncond, null_cond)
        loss_rf_uncond = F.mse_loss(predicted_velocity_uncond, target_velocity_uncond, reduction="none").mean(dim=[1,2,3])

        loss_rf = (loss_rf_cond.sum() + loss_rf_uncond.sum()) / batch_size

        # 4. Generate strong augmentations (conditional) and identity (unconditional)
        # ode_sample has @torch.no_grad() decorator (like legacy)
        strong_aug_images_cond = self.ode_sample(weak_aug_images_cond, difficulty_scores_cond)
        strong_logits_cond = self.model(strong_aug_images_cond)
        strong_difficulty_cond = self.compute_difficulty(strong_logits_cond)

        identity_images_uncond = self.ode_sample(weak_aug_images_uncond, difficulty=None)
        identity_logits_uncond = self.model(identity_images_uncond)
        identity_difficulty_uncond = self.compute_difficulty(identity_logits_uncond)

        # 5. Conditional constraint losses (ratio lower bound & LPIPS upper bound)
        # Difficulty ratio lower bound: log(s_strong / s_weak) >= 1/C
        # Target: s_strong / s_weak >= e^(1/C)
        lower_bound = 1.0 / self.num_classes  # 1/C in log space
        log_ratio = torch.log((strong_difficulty_cond + 1e-7) / (difficulty_scores_cond.detach() + 1e-7))
        loss_ratio = F.relu(lower_bound - log_ratio).pow(2).sum() / batch_size

        # LPIPS upper bound: Apply only when lower bound is satisfied
        # Use LPIPS distance directly as penalty (no threshold)
        # Detach generated images since they're from no_grad sampling
        lpips_distance = self.loss_cond_upper(strong_aug_images_cond.detach(), weak_aug_images_cond, reduction="none")
        # Mask: 1 if lower bound satisfied, 0 otherwise
        lpips_mask = (log_ratio >= lower_bound).float().detach()
        loss_lpips = (lpips_mask * lpips_distance).sum() / batch_size

        # 6. Unconditional constraint losses (identity & difficulty equality)
        # Identity preservation: ||x_identity - x_weak||^2
        # Detach generated images since they're from no_grad sampling
        loss_identity = F.mse_loss(identity_images_uncond.detach(), weak_aug_images_uncond, reduction="none").mean(dim=[1,2,3]).sum() / batch_size

        # Difficulty equality: (s_identity - s_weak)^2
        loss_diff_eq = (identity_difficulty_uncond - difficulty_scores_uncond.detach()).pow(2).sum() / batch_size

        # 7. Total generator loss
        loss_gen = (
            self.lambda_rf * loss_rf
            + self.lambda_ratio * loss_ratio
            + self.lambda_lpips * loss_lpips
            + self.lambda_identity * (loss_identity + loss_diff_eq)
        )

        self.optimizer_gen.zero_grad()
        loss_gen.backward()
        self.optimizer_gen.step()

        # ========================================
        # Stage 2: Classifier Training
        # ========================================
        self.model.train()
        self.generator.eval()

        # 1. Supervised loss on labeled data
        sup_preds = self.model(labeled_images)
        loss_sup = self.loss_supervised(sup_preds, labels)

        # 2. Generate pseudo-labels from weak augmentation predictions (teacher)
        with torch.no_grad():
            teacher_preds = self.model(weak_aug_images)
            teacher_probs = torch.softmax(teacher_preds, dim=1)
            max_probs, pseudo_labels = torch.max(teacher_probs, dim=1)

            # Confidence mask (only for pseudo-label loss)
            mask_conf = max_probs.ge(self.confidence_threshold).float()

            # Compute difficulty for generation
            teacher_difficulty = self.compute_difficulty(teacher_preds).detach()

        # 3. Generate augmented images using the generator
        with torch.no_grad():
            # Conditional samples: strong augmentation with difficulty
            generated_strong_cond = self.ode_sample(weak_aug_images[:split_idx], teacher_difficulty[:split_idx])

            # Unconditional samples: identity mapping with null conditioning
            generated_identity_uncond = self.ode_sample(weak_aug_images[split_idx:], difficulty=None)

            # Concatenate: [strong_cond; identity_uncond]
            generated_images = torch.cat([generated_strong_cond, generated_identity_uncond], dim=0)

        # 4. Get student predictions on generated images
        student_preds = self.model(generated_images)

        # 5. Pseudo-label loss (conditional & high confidence only)
        mask_pl = mask_cond * mask_conf
        loss_pl_per_sample = F.cross_entropy(student_preds, pseudo_labels, reduction="none")
        loss_pl = (loss_pl_per_sample * mask_pl).sum() / batch_size

        # 6. Consistency loss (conditional only, no confidence filtering)
        kl_per_sample = F.kl_div(
            F.log_softmax(student_preds, dim=1),
            teacher_probs.detach(),
            reduction="none",
        ).sum(dim=1)
        loss_cons = (kl_per_sample * mask_cond).sum() / batch_size

        # 7. Total classifier loss
        loss_model = loss_sup + self.lambda_pl * loss_pl + self.lambda_cons * loss_cons

        self.optimizer.zero_grad()
        loss_model.backward()
        self.optimizer.step()

        return {
            "total_loss": loss_model.item() + loss_gen.item(),
            "model_total": loss_model.item(),
            "sup_loss": loss_sup.item(),
            "pl_loss": loss_pl.item(),
            "cons_loss": loss_cons.item(),
            "gen_total": loss_gen.item(),
            "rf_loss": loss_rf.item(),
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
                desc=f"Epoch [{self.current_epoch+1}/{target_epochs}]",
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

        print(f"Epoch {self.current_epoch+1} Validation Metrics: {metrics}")

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.log_metrics(metrics=metrics, step=self.current_epoch + 1)
