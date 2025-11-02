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
        self.set_seed()

        # Device setup
        device_id = self.cfg.get("device", 0)
        if isinstance(device_id, int) and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        elif isinstance(device_id, str):
            self.device = torch.device(device_id)
        else:
            self.device = torch.device("cpu")

        # Classification model
        self.model = self.create.model.classifier().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())

        # Create transform
        base_transform = self.create.transform.base()

        # Create datasets with transform
        train_dataset = self.create.dataset.train(transform=base_transform)
        test_dataset = self.create.dataset.test(transform=base_transform)

        # Get total number of samples
        num_train = len(train_dataset)
        num_labeled = self.cfg.get("num_labeled", 100)

        # Create indices for labeled and unlabeled splits
        import numpy as np
        all_indices = np.arange(num_train)
        np.random.shuffle(all_indices)

        labeled_indices = all_indices[:num_labeled].tolist()
        unlabeled_indices = all_indices.tolist()  # Use all for unlabeled (includes labeled)

        # Create samplers
        labeled_sampler = self.create.sampler.labeled(indices=labeled_indices)
        unlabeled_sampler = self.create.sampler.unlabeled(indices=unlabeled_indices)

        # Create data loaders with datasets and samplers
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

        # Loss and metrics
        self.lx_fn = self.create.loss.cross_entropy()

        # Logger (optional)
        if self.cfg.get("logger"):
            self.logger = self.create.logger()
        else:
            self.logger = None

        self.accuracy_metric = self.create.metric.accuracy()

        # Load pretrained augmentation flow model
        self.setup_augmentation_flow()

        # Training state
        self.best_acc = 0
        self.early_stopping_counter = 0

    def setup_augmentation_flow(self):
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
        self.flow_model = self.create.model.flow().to(self.device)

        # Load weights (prefer EMA if available)
        state_dict = None

        # Check if checkpoint has 'ema_model' key (EMA wrapper with prefixed keys)
        if "ema_model" in checkpoint:
            ema_dict = checkpoint["ema_model"]
            # Extract weights with 'ema_model.' prefix and strip it
            if any(k.startswith("ema_model.") for k in ema_dict.keys()):
                state_dict = {k.replace("ema_model.", ""): v
                             for k, v in ema_dict.items()
                             if k.startswith("ema_model.")}
                print("Loaded EMA model weights (from EMA wrapper)")
            # Fallback to online_model if ema_model weights not found
            elif any(k.startswith("online_model.") for k in ema_dict.keys()):
                state_dict = {k.replace("online_model.", ""): v
                             for k, v in ema_dict.items()
                             if k.startswith("online_model.")}
                print("Loaded online model weights (from EMA wrapper)")
            else:
                state_dict = ema_dict
                print("Loaded EMA model weights (direct)")
        # Check for top-level prefixed keys
        elif any(k.startswith("ema_model.") for k in checkpoint.keys()):
            state_dict = {k.replace("ema_model.", ""): v
                         for k, v in checkpoint.items()
                         if k.startswith("ema_model.")}
            print("Loaded EMA model weights (top-level prefix)")
        elif any(k.startswith("online_model.") for k in checkpoint.keys()):
            state_dict = {k.replace("online_model.", ""): v
                         for k, v in checkpoint.items()
                         if k.startswith("online_model.")}
            print("Loaded online model weights (top-level prefix)")
        # Standard checkpoint formats
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("Loaded model weights")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("Loaded model_state_dict weights")
        else:
            state_dict = checkpoint
            print("Loaded checkpoint weights (direct)")

        self.flow_model.load_state_dict(state_dict)

        self.flow_model.eval()  # Set to eval mode

        # Freeze flow model parameters
        for param in self.flow_model.parameters():
            param.requires_grad = False

        # Flow generation settings
        self.flow_steps = self.cfg.get("flow_steps", 10)
        self.predict_mode = self.cfg.get("flow_predict_mode", "flow")

        print(f"Augmentation flow loaded (steps={self.flow_steps}, mode={self.predict_mode})")

    def set_seed(self):
        """Set random seed for reproducibility."""
        seed = self.cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_adaptive_augmentation(self, images_weak, difficulty_scores):
        """Generate adaptive strong augmentations using flow model.

        Args:
            images_weak: Weakly augmented images [B, C, H, W]
            difficulty_scores: Difficulty scores [B] in range [0, 1]
                              0 = easy (high confidence) � weak augmentation
                              1 = hard (low confidence) � strong augmentation

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

                # Predict flow/velocity
                v_t = self.flow_model(x_t, t_current)

                # Update step
                dt = difficulty_scores / self.flow_steps
                x_t = x_t + v_t * dt.view(-1, 1, 1, 1)

            return x_t

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        target_epochs = self.cfg.get("epochs", 1)

        # zip stops when the shorter iterable is exhausted
        for batch in tqdm(zip(self.train_loader, self.unlabeled_loader),
                          desc=f"Epoch {self.current_epoch + 1}/{target_epochs}"):
            self.train_step(batch)
            self.current_step += 1

    def train_step(self, batch):
        """Single training step with adaptive augmentation.

        Args:
            batch: Tuple of (labeled_batch, unlabeled_batch)
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

            # Calculate difficulty: low confidence � high difficulty � strong aug
            # difficulty = 1 - confidence
            difficulty_scores = 1.0 - max_probs

            # Optional: Clip difficulty to reasonable range
            min_difficulty = self.cfg.get("min_difficulty", 0.0)
            max_difficulty = self.cfg.get("max_difficulty", 1.0)
            difficulty_scores = torch.clamp(difficulty_scores, min_difficulty, max_difficulty)

            # Generate adaptive strong augmentation
            inputs_u_s = self.generate_adaptive_augmentation(inputs_u_w, difficulty_scores)

        # Forward pass
        logits_x = self.model(inputs_x)
        logits_u_s = self.model(inputs_u_s)

        # Supervised loss
        Lx = self.lx_fn(logits_x, targets_x)

        # Unsupervised loss with confidence masking
        threshold = self.cfg.get("threshold", 0.95)
        mask = max_probs.ge(threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

        # Total loss
        lambda_u = self.cfg.get("lambda_u", 1.0)
        loss = Lx + lambda_u * Lu

        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging
        if self.logger:
            self.logger.log_metrics({
                "total_loss": loss.item(),
                "Lx": Lx.item(),
                "Lu": Lu.item(),
                "mask_ratio": mask.mean().item(),
                "avg_difficulty": difficulty_scores.mean().item(),
                "avg_confidence": max_probs.mean().item(),
            }, step=self.current_step)

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

        # Early stopping
        is_best = val_acc > self.best_acc
        self.best_acc = max(val_acc, self.best_acc)

        if is_best:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            patience = self.cfg.get("early_stopping_patience", 100)
            if self.early_stopping_counter > patience:
                print(f"Early stopping at epoch {self.current_epoch} due to no improvement.")
                self.current_epoch = self.cfg.get("epochs")
