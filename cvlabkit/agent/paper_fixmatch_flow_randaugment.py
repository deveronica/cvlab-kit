# cvlabkit/agent/paper_fixmatch_flow_randaugment.py
"""FixMatch with FlowAddedRandAugment for adaptive augmentation."""

import json
import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from cvlabkit.core.agent import Agent


def pil_collate(batch):
    """Custom collate function to handle PIL Images."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


class PaperFixmatchFlowRandaugment(Agent):
    """FixMatch with FlowAddedRandAugment in augmentation pool.

    Uses AdaptiveFlowRandAugment where Flow is one of 15 augmentation operations.
    The magnitude is adaptively selected based on model confidence scores.
    """

    def setup(self):
        """Creates and initializes all necessary components for the agent."""
        device_id = self.cfg.get("device", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")
        self.current_epoch = 0

        # --- Create Components using the Creator ---
        self.model = self.create.model().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())

        self.weak_transform = self.create.transform.weak()
        self.strong_transform = (
            self.create.transform.strong()
        )  # AdaptiveFlowRandAugment
        self.val_transform = self.create.transform.val()

        # Loss functions
        self.sup_loss_fn = self.create.loss.supervised()
        self.unsup_loss_fn = self.create.loss.unsupervised()
        self.contrastive_loss_fn = self.create.loss.contrastive()

        if self.cfg.get("logger"):
            self.logger = self.create.logger()

        # --- Data Handling with Stratified Splitting ---
        train_dataset = self.create.dataset.train()
        val_dataset = self.create.dataset.val()

        num_labeled = self.cfg.num_labeled
        targets = np.array(train_dataset.targets)

        # Load or create labeled indices for reproducibility
        log_dir = self.cfg.get("log_dir", ".")
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
            unlabeled_indices.extend(indices[actual_num_labeled:])

        np.random.shuffle(unlabeled_indices)
        print(
            f"Dataset split: {len(labeled_indices)} labeled ({num_labeled_per_class} per class target), {len(unlabeled_indices)} unlabeled."
        )
        return labeled_indices, unlabeled_indices

    def train_step(self, labeled_batch, unlabeled_batch):
        """Performs a single training step with AdaptiveFlowRandAugment."""
        self.model.train()

        labeled_images_pil, labels = labeled_batch
        unlabeled_images_pil, _ = unlabeled_batch

        # Apply weak transform to labeled data
        labeled_images = torch.stack(
            [self.weak_transform(img) for img in labeled_images_pil]
        ).to(self.device)
        labels = labels.to(self.device)

        # Apply weak transform to unlabeled data
        unlabeled_images_weak = torch.stack(
            [self.weak_transform(img) for img in unlabeled_images_pil]
        ).to(self.device)

        # 1. Supervised loss
        sup_preds = self.model(labeled_images)
        loss_sup = self.sup_loss_fn(sup_preds, labels)

        # 2. Generate pseudo-labels from weak augmentations
        with torch.no_grad():
            teacher_preds = self.model(unlabeled_images_weak)

            probs = torch.softmax(teacher_preds, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            mask = max_probs.ge(self.cfg.get("confidence_threshold", 0.95)).float()

            # Calculate confidence scores for adaptive augmentation
            # High entropy → low confidence → low score → weak augmentation
            # Low entropy → high confidence → high score → strong augmentation
            entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
            C = self.cfg.get("num_classes", 10)
            a = float(self.cfg.get("scale_a", 10.0))
            Ca = float(C) ** a
            confidence_scores = ((Ca * torch.exp(-a * entropy)) - 1.0) / (
                Ca - 1.0 + 1e-12
            )
            confidence_scores = confidence_scores.clamp(0.0, 1.0)

        # Apply AdaptiveFlowRandAugment with confidence-based difficulty
        unlabeled_images_strong = []
        for i, img in enumerate(unlabeled_images_pil):
            aug_img = self.strong_transform(
                img, difficulty_score=confidence_scores[i].item()
            )
            unlabeled_images_strong.append(aug_img)
        unlabeled_images_strong = torch.stack(unlabeled_images_strong).to(self.device)

        student_preds = self.model(unlabeled_images_strong)

        # Pseudo-label loss (with mask)
        loss_pl = self.unsup_loss_fn(student_preds, pseudo_labels)
        loss_pl = (loss_pl * mask).mean()

        # Consistency loss
        loss_cons = self.contrastive_loss_fn(student_preds, teacher_preds).mean()

        # 3. Total Loss
        lambda_pl = self.cfg.get("lambda_pl", 1.0)
        lambda_cons = self.cfg.get("lambda_cons", 4.5)
        total_loss = loss_sup + lambda_pl * loss_pl + lambda_cons * loss_cons

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "sup_loss": loss_sup.item(),
            "pl_loss": loss_pl.item(),
            "cons_loss": loss_cons.item(),
            "mask_ratio": mask.mean().item(),
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
        self.metric = self.create.metric.val()
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
