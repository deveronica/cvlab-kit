# cvlabkit/agent/paper_fixmatch_pretrained_flow.py
"""FixMatch with Pretrained Flow Generator for Adaptive Augmentation."""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cvlabkit.core.agent import Agent
from cvlabkit.core.config import Config


def pil_collate(batch):
    """Custom collate function to handle PIL Images."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


class FlowAugmentationFixmatch(Agent):
    """FixMatch with pretrained flow model for adaptive strong augmentation.

    Uses a pretrained flow model to generate difficulty-adaptive strong augmentations.
    High confidence samples get strong augmentations (curriculum learning).
    """

    def setup(self):
        """Creates and initializes all necessary components for the agent."""
        device_id = self.cfg.get("device", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.current_epoch = 0

        # --- Create Components using the Creator ---
        self.model = self.create.model().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())

        # Transforms
        self.weak_transform = self.create.transform.weak()
        self.weak_unnorm_transform = self.create.transform.weak_unnorm()
        self.normalize_transform = self.create.transform.normalize()
        self.val_transform = self.create.transform.val()
        # Note: No strong_transform - we use flow model instead

        # Loss functions
        self.sup_loss_fn = self.create.loss.supervised()
        self.unsup_loss_fn = self.create.loss.unsupervised()
        self.contrastive_loss_fn = self.create.loss.contrastive()

        # Metric
        self.metric = self.create.metric.val()

        # Logger (optional)
        if self.cfg.get("logger"):
            self.logger = self.create.logger()

        # --- Load Pretrained Flow Model ---
        self._setup_augmentation_flow()

        # --- Data Handling with Stratified Splitting ---
        train_dataset = self.create.dataset.train()
        val_dataset = self.create.dataset.val()

        num_labeled = self.cfg.num_labeled
        targets = np.array(train_dataset.targets)

        # Load or create labeled indices for reproducibility
        log_dir = self.cfg.get("log_dir", "./logs")
        dataset_name = self.cfg.dataset.train.split("(")[0]
        os.makedirs(log_dir, exist_ok=True)
        index_file_path = os.path.join(
            log_dir, f"{dataset_name}_labeled_indices_{num_labeled}.json"
        )

        if os.path.exists(index_file_path):
            print(f"Loading labeled indices from {index_file_path}")
            with open(index_file_path) as f:
                labeled_indices = json.load(f)

            all_indices = set(range(len(targets)))
            unlabeled_indices = list(all_indices - set(labeled_indices))
            np.random.shuffle(unlabeled_indices)
            print(
                f"Loaded {len(labeled_indices)} labeled indices and reconstructed {len(unlabeled_indices)} unlabeled indices."
            )
        else:
            print("Generating new labeled/unlabeled split.")
            labeled_indices, unlabeled_indices = self._stratified_split(
                targets, num_labeled
            )

            print(f"Saving {len(labeled_indices)} labeled indices to {index_file_path}")
            with open(index_file_path, "w") as f:
                json.dump(labeled_indices, f)

        labeled_sampler = self.create.sampler.labeled(indices=labeled_indices)
        unlabeled_sampler = self.create.sampler.unlabeled(indices=unlabeled_indices)

        # Dynamically calculate batch sizes
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

    def _setup_augmentation_flow(self):
        """Load pretrained augmentation flow model."""
        flow_checkpoint = self.cfg.get("flow_checkpoint")

        if flow_checkpoint is None:
            raise ValueError("flow_checkpoint must be specified in config")

        checkpoint_path = Path(flow_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Flow checkpoint not found: {checkpoint_path}")

        print(f"Loading augmentation flow from: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create flow model using creator
        generator_cfg = self.cfg.get("generator")
        if generator_cfg is None:
            raise ValueError("generator must be specified in config")

        # Parse generator string
        if isinstance(generator_cfg, str):
            temp_cfg = Config({"model": generator_cfg})
            temp_create = type(self.create)(temp_cfg)
            self.generator = temp_create.model().to(self.device)
        else:
            raise ValueError(f"generator must be a string, got {type(generator_cfg)}")

        # Load weights
        if "model" in checkpoint:
            self.generator.load_state_dict(checkpoint["model"])
            print("Loaded generator weights")
        else:
            self.generator.load_state_dict(checkpoint)
            print("Loaded checkpoint weights (direct)")

        self.generator.eval()

        # Compile generator for faster inference (PyTorch 2.0+)
        # Note: torch.compile is not stable on MPS yet
        if self.device.type != "mps":
            try:
                self.generator = torch.compile(self.generator)
                print("Generator compiled with torch.compile")
            except Exception as e:
                print(f"torch.compile failed: {e}")
        else:
            print("Skipping torch.compile on MPS (not stable yet)")

        # Ensure generator is on correct device after compile
        self.generator = self.generator.to(self.device)

        # Verify device placement
        gen_device = next(self.generator.parameters()).device
        print(f"Generator device: {gen_device}")
        print(f"Target device: {self.device}")

        # Freeze generator parameters
        for param in self.generator.parameters():
            param.requires_grad = False

        # Flow generation settings
        self.flow_steps = self.cfg.get("flow_steps", 10)

        print(f"Augmentation flow loaded (steps={self.flow_steps})")

    def _stratified_split(self, targets: np.ndarray, num_labeled: int):
        """Performs a stratified split of indices based on a fixed number of labeled samples."""
        indices_by_class = defaultdict(list)
        for i, target in enumerate(targets):
            indices_by_class[target].append(i)

        num_classes = len(indices_by_class)
        num_labeled_per_class = int(np.floor(num_labeled / num_classes))

        labeled_indices = []
        unlabeled_indices = []

        for class_idx, indices in indices_by_class.items():
            np.random.shuffle(indices)
            actual_num_labeled = min(num_labeled_per_class, len(indices))

            labeled_indices.extend(indices[:actual_num_labeled])
            unlabeled_indices.extend(indices)

        np.random.shuffle(unlabeled_indices)

        print(
            f"Dataset split: {len(labeled_indices)} labeled ({num_labeled_per_class} per class target), {len(unlabeled_indices)} unlabeled."
        )
        return labeled_indices, unlabeled_indices

    def _generate_adaptive_augmentation(self, images_weak_tensor, confidence_scores):
        """Generate adaptive strong augmentations using flow model.

        Args:
            images_weak_tensor: Weakly augmented images [B, C, H, W] (already tensor)
            confidence_scores: Confidence scores [B] in range [0, 1]
                              High confidence → High score → Need strong aug

        Returns:
            Adaptively augmented images [B, C, H, W]
        """
        with torch.no_grad():
            # Use confidence directly: high confidence → t=1 (strong aug)
            t_target = confidence_scores

            x_t = images_weak_tensor.clone()

            # Solve ODE from t=0 to t=t_target
            for i in range(self.flow_steps):
                # Current timestep for each sample
                t_current = t_target * (i / self.flow_steps)

                # Predict velocity
                v_t = self.generator(x_t, t_current)

                # Euler step
                dt = t_target / self.flow_steps
                x_t = x_t + v_t * dt.view(-1, 1, 1, 1)

            return x_t

    def train_step(self, labeled_batch, unlabeled_batch):
        """Performs a single training step with flow-based adaptive augmentation."""
        self.model.train()

        labeled_images_pil, labels = labeled_batch
        unlabeled_images_pil, _ = unlabeled_batch

        labeled_images = torch.stack(
            [self.weak_transform(img) for img in labeled_images_pil]
        ).to(self.device)
        labels = labels.to(self.device)

        # 1. Supervised loss
        sup_preds = self.model(labeled_images)
        loss_sup = self.sup_loss_fn(sup_preds, labels)

        # 2. Unsupervised loss with flow-based strong augmentation
        with torch.no_grad():
            weak_aug_images = torch.stack(
                [self.weak_transform(img) for img in unlabeled_images_pil]
            ).to(self.device)
            teacher_preds = self.model(weak_aug_images)

            probs = torch.softmax(teacher_preds, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            mask = max_probs.ge(self.cfg.get("confidence_threshold", 0.95)).float()

            # Calculate confidence scores (high confidence → high score → strong aug)
            entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
            C = self.cfg.get("num_classes", 10)
            a = float(self.cfg.get("scale_a", 10.0))
            Ca = float(C) ** a
            confidence_scores = ((Ca * torch.exp(-a * entropy)) - 1.0) / (
                Ca - 1.0 + 1e-12
            )
            confidence_scores = confidence_scores.clamp(0.0, 1.0)

        # Generate flow-based strong augmentation (PIL → tensor [0,1] → flow → normalize)
        unlabeled_unnorm = torch.stack(
            [self.weak_unnorm_transform(img) for img in unlabeled_images_pil]
        ).to(self.device)
        strong_aug_unnorm = self._generate_adaptive_augmentation(
            unlabeled_unnorm, confidence_scores
        )
        strong_aug_images = torch.stack(
            [self.normalize_transform(img) for img in strong_aug_unnorm]
        )

        student_preds = self.model(strong_aug_images)

        # Pseudo-label loss (with mask)
        loss_pl = self.unsup_loss_fn(student_preds, pseudo_labels)
        loss_pl = (loss_pl * mask).mean()

        # Consistency loss
        loss_cons = self.contrastive_loss_fn(student_preds, teacher_preds).mean()

        # Total loss
        total_loss = (
            loss_sup
            + self.cfg.get("lambda_pl", 1.0) * loss_pl
            + self.cfg.get("lambda_cons", 0.5) * loss_cons
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "sup_loss": loss_sup.item(),
            "pl_loss": loss_pl.item(),
            "cons_loss": loss_cons.item(),
        }

    def fit(self):
        """The main training loop, driven by a fixed number of steps per epoch."""
        train_epochs = self.cfg.get("epochs", 1)
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
            for step in progress_bar:
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
        """Evaluates the model on the validation set."""
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

                val_loss = self.sup_loss_fn(preds, labels)
                total_val_loss += val_loss.item()

                self.metric.update(preds=preds, targets=labels)

        avg_val_loss = total_val_loss / len(self.val_loader)

        metrics = self.metric.compute()
        metrics["val_loss"] = avg_val_loss

        print(f"Epoch {self.current_epoch + 1} Validation Metrics: {metrics}")

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.log_metrics(metrics=metrics, step=self.current_epoch + 1)
