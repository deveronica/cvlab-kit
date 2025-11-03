"""Agent for FixMatch with pretrained Augmentation Flow model.

This agent integrates a pretrained augmentation flow model to generate
adaptive strong augmentations based on pseudo-label confidence.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

from cvlabkit.core.agent import Agent


class FlowAugmentationFixmatch(Agent):
    """FixMatch with adaptive augmentation from pretrained flow model.

    Uses a pretrained augmentation flow model to generate strong augmentations
    with controllable strength based on pseudo-label confidence (difficulty).
    """

    def setup(self):
        """Set up all components for training."""
        # Random seed
        seed = self.cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Device setup
        device_cfg = self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device_cfg, int):
            self.device = torch.device(f"cuda:{device_cfg}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)

        # Model and optimizer
        self.model = self.create.model().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())

        # Create datasets
        base_transform = self.create.transform.base()
        train_dataset = self.create.dataset.train(transform=base_transform)
        test_dataset = self.create.dataset.test(transform=base_transform)

        # Split labeled/unlabeled
        num_train = len(train_dataset)
        num_labeled = self.cfg.get("num_labeled", 100)

        all_indices = np.arange(num_train)
        np.random.shuffle(all_indices)

        labeled_indices = all_indices[:num_labeled].tolist()
        unlabeled_indices = all_indices.tolist()  # All for unlabeled

        # Create samplers
        labeled_sampler = self.create.sampler.labeled(indices=labeled_indices)
        unlabeled_sampler = self.create.sampler.unlabeled(indices=unlabeled_indices)

        # Create data loaders
        batch_size = self.cfg.get("batch_size", 4)
        mu = self.cfg.get("mu", 2)

        self.train_loader = self.create.dataloader.labeled(
            dataset=train_dataset,
            sampler=labeled_sampler,
            batch_size=batch_size
        )
        self.unlabeled_loader = self.create.dataloader.unlabeled(
            dataset=train_dataset,
            sampler=unlabeled_sampler,
            batch_size=batch_size * mu
        )
        self.val_loader = self.create.dataloader.test(dataset=test_dataset)

        # Loss functions
        self.loss_supervised = self.create.loss.supervised()
        self.loss_unsupervised = self.create.loss.unsupervised()

        # Logger and metrics
        if self.cfg.get("logger"):
            self.logger = self.create.logger()
        else:
            self.logger = None
        self.accuracy_metric = self.create.metric.val()

        # Load pretrained augmentation flow model
        self._setup_augmentation_flow()

        # Training state
        self.best_acc = 0

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

        # Create flow model using creator (use top-level 'generator' key)
        from cvlabkit.core.config import Config
        generator_cfg = self.cfg.get("generator")
        if generator_cfg is None:
            raise ValueError("generator must be specified in config")

        # Parse generator string (e.g., "unet" or "unet(...)")
        if isinstance(generator_cfg, str):
            # Create a minimal config for the generator
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

        # Freeze generator parameters
        for param in self.generator.parameters():
            param.requires_grad = False

        # Flow generation settings
        self.flow_steps = self.cfg.get("flow_steps", 10)

        print(f"Augmentation flow loaded (steps={self.flow_steps})")

    def set_seed(self):
        """Set random seed for reproducibility."""
        seed = self.cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _generate_adaptive_augmentation(self, images_weak, difficulty_scores):
        """Generate adaptive strong augmentations using flow model.

        Args:
            images_weak: Weakly augmented images [B, C, H, W]
            difficulty_scores: Difficulty scores [B] in range [0, 1]
                              0 = easy (high confidence) → weak augmentation
                              1 = hard (low confidence) → strong augmentation

        Returns:
            Adaptively augmented images [B, C, H, W]
        """
        with torch.no_grad():
            x_0 = images_weak
            x_t = x_0.clone()

            # ODE solver with Euler method
            for i in range(self.flow_steps):
                # Current time for each sample based on difficulty
                t_current = difficulty_scores * (i / self.flow_steps)

                # Predict velocity
                v_t = self.generator(x_t, t_current)

                # Update step
                dt = difficulty_scores / self.flow_steps
                x_t = x_t + v_t * dt.view(-1, 1, 1, 1)

            return x_t

    def train_epoch(self):
        """Train for one epoch with epoch-level logging."""
        self.model.train()
        target_epochs = self.cfg.get("epochs", 1)

        # Accumulators for epoch averages
        epoch_loss_sum = 0.0
        epoch_Lx_sum = 0.0
        epoch_Lpl_sum = 0.0
        epoch_Lcons_sum = 0.0
        epoch_mask_sum = 0.0
        epoch_difficulty_sum = 0.0
        epoch_confidence_sum = 0.0
        num_steps = 0

        # zip stops when the shorter iterable is exhausted
        for batch in tqdm(zip(self.train_loader, self.unlabeled_loader),
                          desc=f"Epoch {self.current_epoch + 1}/{target_epochs}"):
            metrics = self.train_step(batch)

            # Accumulate metrics
            epoch_loss_sum += metrics["total_loss"]
            epoch_Lx_sum += metrics["Lx"]
            epoch_Lpl_sum += metrics["L_pl"]
            epoch_Lcons_sum += metrics["L_cons"]
            epoch_mask_sum += metrics["mask_ratio"]
            epoch_difficulty_sum += metrics["avg_difficulty"]
            epoch_confidence_sum += metrics["avg_confidence"]
            num_steps += 1

            self.current_step += 1

        # Log epoch averages
        if num_steps > 0 and self.logger:
            avg_metrics = {
                "epoch_loss": epoch_loss_sum / num_steps,
                "epoch_Lx": epoch_Lx_sum / num_steps,
                "epoch_L_pl": epoch_Lpl_sum / num_steps,
                "epoch_L_cons": epoch_Lcons_sum / num_steps,
                "epoch_mask_ratio": epoch_mask_sum / num_steps,
                "epoch_avg_difficulty": epoch_difficulty_sum / num_steps,
                "epoch_avg_confidence": epoch_confidence_sum / num_steps,
            }
            print(f"Epoch {self.current_epoch + 1} - Avg Loss: {avg_metrics['epoch_loss']:.6f}")
            self.logger.log_metrics(avg_metrics, step=self.current_epoch)

    def train_step(self, batch):
        """Single training step with adaptive augmentation.

        Args:
            batch: Tuple of (labeled_batch, unlabeled_batch)

        Returns:
            dict: Metrics for this step
        """
        (labeled_batch, unlabeled_batch) = batch

        inputs_x, targets_x = labeled_batch
        # Unlabeled batch: just (images, labels) from basic transform
        inputs_u_w, _ = unlabeled_batch

        inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
        inputs_u_w = inputs_u_w.to(self.device)

        # Get pseudo-labels from weak augmentation
        with torch.no_grad():
            logits_u_w = self.model(inputs_u_w)
            pseudo_label = torch.softmax(logits_u_w, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)

            # Calculate difficulty: low confidence → high difficulty → strong aug
            # difficulty = 1 - confidence
            difficulty_scores = 1.0 - max_probs

            # Optional: Clip difficulty to reasonable range
            min_difficulty = self.cfg.get("min_difficulty", 0.0)
            max_difficulty = self.cfg.get("max_difficulty", 1.0)
            difficulty_scores = torch.clamp(difficulty_scores, min_difficulty, max_difficulty)

            # Generate adaptive strong augmentation
            inputs_u_s = self._generate_adaptive_augmentation(inputs_u_w, difficulty_scores)

        # Forward pass
        logits_x = self.model(inputs_x)
        logits_u_w_forward = self.model(inputs_u_w)  # For consistency loss
        logits_u_s = self.model(inputs_u_s)

        # Supervised loss
        Lx = self.loss_supervised(logits_x, targets_x)

        # Pseudo-labeling loss with confidence masking (Eq. 7)
        threshold = self.cfg.get("confidence_threshold", 0.95)
        mask = max_probs.ge(threshold).float()
        Lu_pl = self.loss_unsupervised(logits_u_s, targets_u)
        L_pl = (Lu_pl * mask).mean()

        # Consistency loss: KL divergence between weak and strong predictions (Eq. 8)
        L_cons = F.kl_div(
            F.log_softmax(logits_u_s, dim=1),
            F.softmax(logits_u_w_forward.detach(), dim=1),
            reduction='batchmean'
        )

        # Total loss (Eq. 9)
        lambda_pl = self.cfg.get("lambda_pl", 1.0)
        lambda_cons = self.cfg.get("lambda_cons", 0.5)
        loss = Lx + lambda_pl * L_pl + lambda_cons * L_cons

        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return metrics for epoch averaging
        return {
            "total_loss": loss.item(),
            "Lx": Lx.item(),
            "L_pl": L_pl.item(),
            "L_cons": L_cons.item(),
            "mask_ratio": mask.mean().item(),
            "avg_difficulty": difficulty_scores.mean().item(),
            "avg_confidence": max_probs.mean().item(),
        }

    def validate_step(self, batch):
        """Single validation step."""
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        self.accuracy_metric.update(outputs, targets)

    def evaluate(self):
        """Evaluate on validation set."""
        self.accuracy_metric.reset()
        super().evaluate()
        val_acc = self.accuracy_metric.compute()

        if self.logger:
            self.logger.log_metrics({"val_accuracy": val_acc.item()}, step=self.current_epoch)

        # Track best accuracy
        is_best = val_acc > self.best_acc
        self.best_acc = max(val_acc, self.best_acc)
