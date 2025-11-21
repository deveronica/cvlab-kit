"""Agent for training Rectified Flow with adaptive augmentation strength.

This agent learns the distribution flow from original images (x_0)
to adaptively augmented images (x_1) using Rectified Flow, where the
augmentation strength is controlled by the timestep t. The trained model
can be frozen and used in SSL frameworks for generative augmentation.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from ema_pytorch import EMA
from torchvision.utils import save_image
from tqdm import tqdm

from cvlabkit.core.agent import Agent


def pil_collate(batch):
    """Custom collate function to handle PIL Images."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


class AdaptiveAugmentationFlow(Agent):
    """Agent for learning augmentation flow with adaptive strength.

    Trains a flow model to transform original images to adaptively augmented
    images. The model learns the ODE path x_0 → x_1 where:
    - x_0: Base augmentation (e.g., resize, crop, normalize)
    - x_1: Adaptive augmentation (strength controlled by t)

    The trained model can be frozen and used for SSL generative augmentation.
    """

    def setup(self):
        """Set up all components for training."""
        # Device setup
        device_id = self.cfg.get("device", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

        # Create model using Creator pattern
        self.model = self.create.model().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())

        # Create transforms
        # Base: resize | random_crop | to_tensor | normalize
        # We need to split into: PIL ops | tensor ops
        self.transform_base_pil = (
            self.create.transform.base_pil()
        )  # resize | random_crop
        self.transform_base_tensor = (
            self.create.transform.base_tensor()
        )  # to_tensor | normalize

        # AdaptiveRandAugment: difficulty_score is passed at call time, not constructor
        self.adaptive_augment = self.create.transform.adaptive_augment()

        # Data normalizer (for model input/output)
        if self.cfg.get("transform", {}).get("normalizer"):
            self.normalizer = self.create.transform.normalizer()
            self.unnormalizer = self.create.transform.unnormalizer()
        else:
            self.normalizer = None
            self.unnormalizer = None

        # Noise schedule (optional)
        if self.cfg.get("transform", {}).get("noise_schedule"):
            self.noise_schedule = self.create.transform.noise_schedule()
        else:
            # Default: identity (linear interpolation)
            self.noise_schedule = lambda t: t

        # Create solver for sampling (ODE integration)
        if self.cfg.get("solver"):
            self.solver = self.create.solver()
        else:
            self.solver = None

        # Loss function
        if self.cfg.get("loss"):
            self.loss_fn = self.create.loss()
        else:
            self.loss_fn = None  # Will use MSE by default

        # Create dataset and dataloader
        from torch.utils.data import DataLoader

        dataset = self.create.dataset()
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.get("num_workers", 4),
            collate_fn=pil_collate,  # Handle PIL Images
        )

        # Validation loader (optional)
        if self.cfg.get("val_dataset"):
            val_dataset = self.create.dataset.val()
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.cfg.get("val_batch_size", self.cfg.batch_size),
                shuffle=False,
                drop_last=False,
                collate_fn=pil_collate,
            )
        else:
            self.val_loader = None

        # Training configuration
        self.predict = self.cfg.get("predict", "flow")  # 'flow' or 'x1'
        self.max_grad_norm = self.cfg.get("max_grad_norm", 1.0)

        # Conditioning mode
        self.use_conditioning = self.cfg.get("use_conditioning", False)
        self.condition_type = self.cfg.get(
            "condition_type", "random"
        )  # 'random', 'fixed', 'difficulty'

        # EMA (optional but recommended for stable generation)
        self.use_ema = self.cfg.get("use_ema", True)
        self.ema_model = None

        if self.use_ema:
            ema_kwargs = self.cfg.get("ema_kwargs", {"beta": 0.995, "update_every": 10})
            self.ema_model = EMA(self.model, **ema_kwargs)
            self.ema_model.to(self.device)

        # Checkpoint components
        if self.cfg.get("checkpoint"):
            self.model_checkpoint = self.create.checkpoint.model()
            self.image_checkpoint = self.create.checkpoint.image()

            # Periodic checkpoint for images (optional)
            if self.cfg.get("checkpoint", {}).get("periodic"):
                self.periodic_checkpoint = self.create.checkpoint.periodic()
            else:
                self.periodic_checkpoint = None
        else:
            # Fallback to manual paths (backward compatibility)
            self.model_checkpoint = None
            self.image_checkpoint = None
            self.periodic_checkpoint = None

        # Always set checkpoint_dir and results_dir
        self.checkpoint_dir = Path(
            self.cfg.get("checkpoint_dir", "./checkpoints/augmentation_flow")
        )
        self.results_dir = Path(
            self.cfg.get("results_dir", "./results/augmentation_flow")
        )
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        self.save_checkpoint_every = self.cfg.get("save_checkpoint_every", 1000)
        self.save_results_every = self.cfg.get("save_results_every", 500)
        self.sample_temperature = self.cfg.get("sample_temperature", 1.0)

        # Sample configuration
        num_samples = self.cfg.get("num_samples", 16)
        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows**2) == num_samples, (
            f"num_samples={num_samples} must be perfect square"
        )
        self.num_samples = num_samples

        # Data shape (set on first batch)
        self.data_shape = None

        # Logger (optional)
        if self.cfg.get("logger"):
            self.logger = self.create.logger()
        else:
            self.logger = None

        print("AugmentationFlow Agent initialized")
        print(f"  Device: {self.device}")
        print(f"  Predict mode: {self.predict}")
        print(f"  Conditioning: {self.use_conditioning} ({self.condition_type})")
        print(f"  EMA: {self.use_ema}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")

    def train_step(self, batch):
        """Perform a single training step.

        Args:
            batch: Batch from dataloader (images_pil, labels)
        """
        self.model.train()

        # Unpack batch from pil_collate
        images_pil, _ = batch  # images_pil is list of PIL Images

        # Sample random times for each image to control augmentation strength
        times = torch.rand(len(images_pil), device=self.device)

        # Apply transforms to PIL images
        with torch.no_grad():
            x_0_list = []
            x_1_list = []

            for img, t in zip(images_pil, times):
                # Apply base PIL transforms (resize, crop)
                img_pil = self.transform_base_pil(img)

                # x_0: Base transform only (no adaptive augmentation)
                img_base_tensor = self.transform_base_tensor(img_pil)
                x_0_list.append(img_base_tensor)

                # x_1: Apply adaptive augmentation, then tensor transform
                # Note: AdaptiveRandAugment uses difficulty_score where higher = less augmentation
                # So we pass t as-is: t=0 → weak aug, t=1 → strong aug
                # But difficulty_score logic is: high score → low magnitude
                # Therefore we need to invert: difficulty_score = t means we want magnitude to increase with t
                # Current AdaptiveRandAugment: magnitude = max - (max-min)*score
                # With max=10, min=0: score=0→mag=10, score=1→mag=0
                # We want: t=0→mag=0, t=1→mag=10, so pass (1-t) as difficulty_score
                img_augmented = self.adaptive_augment(
                    img_pil, difficulty_score=1.0 - t.item()
                )
                img_aug_tensor = self.transform_base_tensor(img_augmented)
                x_1_list.append(img_aug_tensor)

            x_0 = torch.stack(x_0_list).to(self.device)
            x_1 = torch.stack(x_1_list).to(self.device)

        # Set data shape on first batch
        if self.data_shape is None:
            self.data_shape = x_0.shape[1:]
            print(f"Data shape: {self.data_shape}")

        # Normalize if normalizer is provided (already done in transform pipeline)
        # with torch.no_grad():
        #     if self.normalizer is not None:
        #         x_0 = self.normalizer(x_0)
        #         x_1 = self.normalizer(x_1)

        # Note: times already sampled above for augmentation strength

        # Apply noise schedule (e.g., cosine schedule)
        scheduled_times = self.noise_schedule(times)
        padded_times = scheduled_times.view(-1, *([1] * (x_0.ndim - 1)))

        # Interpolate: x_0 → x_1 using scheduled times
        # x_t = (1-t) * x_0 + t * x_1
        x_t = x_0.lerp(x_1, padded_times)

        # Generate conditioning input (optional)
        if self.use_conditioning:
            if self.condition_type == "random":
                # Random conditioning value [0, 1]
                condition = torch.rand(len(x_0), device=self.device)
            elif self.condition_type == "fixed":
                # Fixed conditioning value
                condition = torch.ones(len(x_0), device=self.device) * self.cfg.get(
                    "condition_value", 0.5
                )
            elif self.condition_type == "difficulty":
                # Use interpolation time as difficulty (higher t = harder transformation)
                condition = scheduled_times
            else:
                raise ValueError(f"Unknown condition_type: {self.condition_type}")

            # Model prediction with conditioning
            pred = self.model(x_t, times=scheduled_times, condition=condition)
        else:
            # Model prediction without conditioning
            pred = self.model(x_t, times=scheduled_times)

        # Determine target based on prediction objective
        if self.predict == "flow":
            # Predict the flow vector (x_1 - x_0)
            target = x_1 - x_0
        elif self.predict == "x1":
            # Predict the endpoint (strong augmentation)
            target = x_1
        else:
            raise ValueError(f"Unknown predict mode: {self.predict}")

        # Compute loss
        if self.loss_fn is None:
            loss = F.mse_loss(pred, target)
        else:
            loss = self.loss_fn(pred, target)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Update EMA
        if self.use_ema:
            self.ema_model.update()

        # Periodic sampling
        if self.current_step % self.save_results_every == 0:
            self._sample_and_save(f"step_{self.current_step}")

        # Periodic checkpointing
        if self.current_step % self.save_checkpoint_every == 0:
            if self.model_checkpoint is not None:
                self._save_checkpoint_component(
                    f"checkpoint_step_{self.current_step}.pt"
                )
            else:
                self._save_checkpoint(f"checkpoint_step_{self.current_step}.pt")

        return loss.item()

    def train_epoch(self):
        """Override train_epoch to add epoch-level loss logging."""
        if self.train_loader is None:
            raise ValueError("train_loader must be set before training.")

        self.model.train()
        epoch_loss_sum = 0.0
        num_steps = 0

        for batch in tqdm(
            self.train_loader, desc=f"Epoch {self.current_epoch + 1} Training"
        ):
            loss = self.train_step(batch)
            epoch_loss_sum += loss
            num_steps += 1
            self.current_step += 1

        # Log epoch average loss
        if num_steps > 0:
            avg_loss = epoch_loss_sum / num_steps
            print(f"Epoch {self.current_epoch + 1} - Average Loss: {avg_loss:.6f}")
            if self.logger:
                self.logger.log_metrics(
                    {"epoch_loss": avg_loss}, step=self.current_epoch
                )

    def validate_step(self, batch):
        """Perform validation (generate samples from validation data).

        Args:
            batch: Batch of validation data (images_pil, labels)
        """
        # Unpack batch from pil_collate
        images_pil, _ = batch

        # Apply weak transform to get starting point
        with torch.no_grad():
            images = torch.stack([self.transform_weak(img) for img in images_pil]).to(
                self.device
            )

        self.model.eval()
        eval_model = self.ema_model if self.use_ema else self.model

        with torch.no_grad():
            # x_0 already applied (weak transform), just normalize if needed
            if self.normalizer is not None:
                x_0 = self.normalizer(images)
            else:
                x_0 = images

            # Generate strong augmentation as ground truth
            x_1_true = torch.stack(
                [self.transform_strong(img) for img in images_pil]
            ).to(self.device)
            if self.normalizer is not None:
                x_1_true = self.normalizer(x_1_true)

            # Generate using ODE integration
            if self.solver is not None:
                num_steps = self.cfg.get("sample_steps", 50)
                times = torch.linspace(0.0, 1.0, num_steps, device=self.device)

                # Prepare conditioning for validation
                if self.use_conditioning:
                    val_condition = torch.ones(
                        len(images), device=self.device
                    ) * self.cfg.get("val_condition", 1.0)

                    def ode_fn(t, x):
                        t_batch = t.expand(x.shape[0])
                        return eval_model(x, times=t_batch, condition=val_condition)
                else:

                    def ode_fn(t, x):
                        t_batch = t.expand(x.shape[0])
                        return eval_model(x, times=t_batch)

                x_1_generated = self.solver(ode_fn, x_0, times)
            else:
                # Simple single-step generation
                ones = torch.ones(len(images), device=self.device)
                if self.use_conditioning:
                    val_condition = torch.ones(
                        len(images), device=self.device
                    ) * self.cfg.get("val_condition", 1.0)
                    x_1_generated = eval_model(x_0, times=ones, condition=val_condition)
                else:
                    x_1_generated = eval_model(x_0, times=ones)
                if self.predict == "flow":
                    x_1_generated = x_0 + x_1_generated

            # Compute validation loss (not logged - epoch-level only)
            if self.loss_fn is None:
                val_loss = F.mse_loss(x_1_generated, x_1_true)
            else:
                val_loss = self.loss_fn(x_1_generated, x_1_true)

    def evaluate(self):
        """Evaluate and save periodic images at epoch end.

        This overrides the base Agent.evaluate() to add periodic image saving and checkpointing.
        """
        # Run validation if val_loader exists
        if self.val_loader is not None:
            super().evaluate()

        # Periodic image saving using checkpoint component
        if self.periodic_checkpoint is not None:
            # Generate and save comparison images
            self._save_periodic_images()
            # Generate paper figure (Method Comparison)
            self._save_paper_figure()

        # Save checkpoint at end of epoch
        if self.model_checkpoint is not None:
            self._save_checkpoint_component(f"checkpoint_epoch_{self.current_epoch}.pt")
        else:
            self._save_checkpoint(f"checkpoint_epoch_{self.current_epoch}.pt")

    def _save_periodic_images(self):
        """Generate and save images using periodic checkpoint."""
        if self.data_shape is None:
            return

        eval_model = self.ema_model if self.use_ema else self.model
        eval_model.eval()

        with torch.no_grad():
            # Get a batch from train loader
            try:
                batch = next(iter(self.train_loader))
                images_pil, _ = batch
                images_pil = images_pil[: self.num_samples]
            except StopIteration:
                return

            # Generate augmentations from PIL images
            # For visualization, we sample with fixed t values to show progression
            t_samples = torch.linspace(0.0, 1.0, len(images_pil), device=self.device)

            x_0_list = []
            x_1_true_list = []

            for img, t in zip(images_pil, t_samples):
                # Apply base PIL transforms
                img_pil = self.transform_base_pil(img)

                # x_0: Base transform only
                img_base_tensor = self.transform_base_tensor(img_pil)
                x_0_list.append(img_base_tensor)

                # x_1_true: Apply adaptive augmentation with t
                # Pass (1-t) to get correct magnitude: t=0→mag=0, t=1→mag=10
                img_augmented = self.adaptive_augment(
                    img_pil, difficulty_score=1.0 - t.item()
                )
                img_aug_tensor = self.transform_base_tensor(img_augmented)
                x_1_true_list.append(img_aug_tensor)

            images = torch.stack(x_0_list).to(self.device)
            x_0 = images  # Base augmented
            x_1_true = torch.stack(x_1_true_list).to(self.device)

            if self.normalizer is not None:
                x_0_norm = self.normalizer(x_0)
            else:
                x_0_norm = x_0

            # Generate using model
            if self.solver is not None:
                num_steps = self.cfg.get("sample_steps", 50)
                times = torch.linspace(0.0, 1.0, num_steps, device=self.device)

                if self.use_conditioning:
                    sample_condition = torch.ones(
                        len(x_0_norm), device=self.device
                    ) * self.cfg.get("sample_condition", 1.0)

                    def ode_fn(t, x):
                        t_batch = t.expand(x.shape[0])
                        return eval_model(x, times=t_batch, condition=sample_condition)
                else:

                    def ode_fn(t, x):
                        t_batch = t.expand(x.shape[0])
                        return eval_model(x, times=t_batch)

                x_1_generated = self.solver(ode_fn, x_0_norm, times)
            else:
                ones = torch.ones(len(x_0_norm), device=self.device)
                if self.use_conditioning:
                    sample_condition = torch.ones(
                        len(x_0_norm), device=self.device
                    ) * self.cfg.get("sample_condition", 1.0)
                    x_1_generated = eval_model(
                        x_0_norm, times=ones, condition=sample_condition
                    )
                else:
                    x_1_generated = eval_model(x_0_norm, times=ones)
                if self.predict == "flow":
                    x_1_generated = x_0_norm + x_1_generated

            # Unnormalize
            if self.unnormalizer is not None:
                x_1_generated = self.unnormalizer(x_1_generated)

            # Save using periodic checkpoint
            comparison = torch.stack([images, x_0, x_1_generated, x_1_true])

            # Use periodic checkpoint to save
            def save_fn(path):
                from einops import rearrange
                from torchvision.utils import save_image

                # Rearrange for visualization
                comp = rearrange(comparison, "views b c h w -> (b views) c h w")
                save_image(comp, path, nrow=4, normalize=False)

            self.periodic_checkpoint.on_epoch(
                current_epoch=self.current_epoch,
                save_fn=save_fn,
                prefix="epoch_comparison",
                ext="png",
            )

    def _sample_and_save(self, fname):
        """Sample transformations from weak to strong and save visualizations.

        Args:
            fname: Filename identifier for saving
        """
        if self.data_shape is None:
            print("Warning: data_shape not set, skipping sampling")
            return

        eval_model = self.ema_model if self.use_ema else self.model
        eval_model.eval()

        with torch.no_grad():
            # Get a batch of samples from train loader
            try:
                batch = next(iter(self.train_loader))
                images_pil, _ = batch
                images_pil = images_pil[: self.num_samples]
            except StopIteration:
                print("Warning: cannot sample from empty train_loader")
                return

            # Generate augmentations from PIL images
            # For visualization, we sample with fixed t values to show progression
            t_samples = torch.linspace(0.0, 1.0, len(images_pil), device=self.device)

            x_0_list = []
            x_1_true_list = []

            for img, t in zip(images_pil, t_samples):
                # Apply base PIL transforms
                img_pil = self.transform_base_pil(img)

                # x_0: Base transform only
                img_base_tensor = self.transform_base_tensor(img_pil)
                x_0_list.append(img_base_tensor)

                # x_1_true: Apply adaptive augmentation with t
                # Pass (1-t) to get correct magnitude: t=0→mag=0, t=1→mag=10
                img_augmented = self.adaptive_augment(
                    img_pil, difficulty_score=1.0 - t.item()
                )
                img_aug_tensor = self.transform_base_tensor(img_augmented)
                x_1_true_list.append(img_aug_tensor)

            images = torch.stack(x_0_list).to(self.device)
            x_0 = images  # Base augmented
            x_1_true = torch.stack(x_1_true_list).to(self.device)

            if self.normalizer is not None:
                x_0_norm = self.normalizer(x_0)
            else:
                x_0_norm = x_0

            # Generate using ODE solver if available
            if self.solver is not None:
                num_steps = self.cfg.get("sample_steps", 50)
                times = torch.linspace(0.0, 1.0, num_steps, device=self.device)

                # Prepare conditioning for sampling
                if self.use_conditioning:
                    sample_condition = torch.ones(
                        len(x_0_norm), device=self.device
                    ) * self.cfg.get("sample_condition", 1.0)

                    def ode_fn(t, x):
                        t_batch = t.expand(x.shape[0])
                        return eval_model(x, times=t_batch, condition=sample_condition)
                else:

                    def ode_fn(t, x):
                        t_batch = t.expand(x.shape[0])
                        return eval_model(x, times=t_batch)

                x_1_generated = self.solver(ode_fn, x_0_norm, times)
            else:
                # Simple single-step generation
                ones = torch.ones(len(x_0_norm), device=self.device)
                if self.use_conditioning:
                    sample_condition = torch.ones(
                        len(x_0_norm), device=self.device
                    ) * self.cfg.get("sample_condition", 1.0)
                    x_1_generated = eval_model(
                        x_0_norm, times=ones, condition=sample_condition
                    )
                else:
                    x_1_generated = eval_model(x_0_norm, times=ones)
                if self.predict == "flow":
                    x_1_generated = x_0_norm + x_1_generated

            # Unnormalize
            if self.unnormalizer is not None:
                x_1_generated = self.unnormalizer(x_1_generated)

            # Create comparison grid: [original | weak | generated | true_strong]
            comparison = torch.stack(
                [images, x_0, x_1_generated, x_1_true], dim=1
            )  # [B, 4, C, H, W]

            # Rearrange into grid
            comparison = rearrange(
                comparison,
                "(row col) views c h w -> c (row h) (views col w)",
                row=self.num_sample_rows,
            )
            comparison = comparison.clamp(0.0, 1.0)

            # Save visualization
            save_path = self.results_dir / f"{fname}.png"
            save_image(comparison, save_path)
            print(f"Saved visualization to {save_path}")

    def _save_paper_figure(self):
        """Generate Figure 2: Method Comparison (Flow vs RandAugment)."""
        if self.data_shape is None:
            return

        eval_model = self.ema_model if self.use_ema else self.model
        eval_model.eval()

        with torch.no_grad():
            # Get samples
            try:
                batch = next(iter(self.train_loader))
                images_pil, _ = batch
                # Take more samples to have variety
                images_pil = images_pil[:16]
            except StopIteration:
                return

            # Simulate classifier predictions (confidence scores)
            # In real case, you'd use actual predictions from a classifier
            # For visualization, we create high/low confidence scenarios
            num_samples = len(images_pil)

            # Generate with different difficulties
            original_list = []
            randaugment_list = []
            flow_weak_list = []
            flow_strong_list = []

            for img_pil in images_pil:
                # Original (base transform only)
                img_pil_transformed = self.transform_base_pil(img_pil)
                img_tensor = self.transform_base_tensor(img_pil_transformed)
                original_list.append(img_tensor)

                # RandAugment (fixed strong augmentation)
                img_randaug = self.adaptive_augment(
                    img_pil_transformed, difficulty_score=1.0
                )
                img_randaug_tensor = self.transform_base_tensor(img_randaug)
                randaugment_list.append(img_randaug_tensor)

            # Stack originals
            originals = torch.stack(original_list).to(self.device)
            randaugs = torch.stack(randaugment_list).to(self.device)

            # Normalize for model input
            if self.normalizer is not None:
                originals_norm = self.normalizer(originals)
            else:
                originals_norm = originals

            # Generate flow-based augmentations
            # High confidence scenario (weak augmentation, t=0.2)
            t_weak = torch.ones(num_samples, device=self.device) * 0.2
            flow_weak_norm = eval_model(originals_norm, times=t_weak)
            if self.predict == "flow":
                flow_weak_norm = originals_norm + flow_weak_norm
            if self.unnormalizer is not None:
                flow_weak = self.unnormalizer(flow_weak_norm)
            else:
                flow_weak = flow_weak_norm

            # Low confidence scenario (strong augmentation, t=0.9)
            t_strong = torch.ones(num_samples, device=self.device) * 0.9
            flow_strong_norm = eval_model(originals_norm, times=t_strong)
            if self.predict == "flow":
                flow_strong_norm = originals_norm + flow_strong_norm
            if self.unnormalizer is not None:
                flow_strong = self.unnormalizer(flow_strong_norm)
            else:
                flow_strong = flow_strong_norm

            # Create comparison grid
            # Top half: High confidence (should use weak aug)
            # Bottom half: Low confidence (should use strong aug)
            k = min(4, num_samples // 2)

            high_conf_grid = torch.stack(
                [
                    originals[:k],
                    randaugs[:k],  # Always strong (wasteful!)
                    flow_weak[:k],  # Adaptive weak
                ],
                dim=1,
            )  # [k, 3, C, H, W]

            low_conf_grid = torch.stack(
                [
                    originals[k : k * 2],
                    randaugs[k : k * 2],  # Always strong (correct but not adaptive)
                    flow_strong[k : k * 2],  # Adaptive strong
                ],
                dim=1,
            )  # [k, 3, C, H, W]

            # Combine and rearrange
            comparison = torch.cat(
                [
                    rearrange(high_conf_grid, "b views c h w -> (b views) c h w"),
                    rearrange(low_conf_grid, "b views c h w -> (b views) c h w"),
                ],
                dim=0,
            )

            comparison = comparison.clamp(0.0, 1.0)

            # Save
            save_path = (
                self.results_dir / f"paper_fig2_epoch{self.current_epoch:03d}.png"
            )
            from torchvision.utils import make_grid

            grid = make_grid(
                comparison, nrow=3, normalize=False, padding=2, pad_value=1.0
            )
            save_image(grid, save_path)
            print(f"Saved paper figure to {save_path}")

    def _save_checkpoint(self, filename):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        save_path = self.checkpoint_dir / filename

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "data_shape": self.data_shape,
            "config": {
                "predict": self.predict,
                "use_ema": self.use_ema,
            },
        }

        if self.ema_model is not None:
            checkpoint["ema_model"] = self.ema_model.state_dict()

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

        # Also save as "latest"
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

    def _load_checkpoint(self, path):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "current_epoch" in checkpoint:
            self.current_epoch = checkpoint["current_epoch"]
        if "current_step" in checkpoint:
            self.current_step = checkpoint["current_step"]
        if "data_shape" in checkpoint:
            self.data_shape = checkpoint["data_shape"]

        if self.ema_model is not None and "ema_model" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_model"])

        print(f"Checkpoint loaded: {path}")

    def _save_model_for_ssl(self, filename="augmentation_flow_model.pt"):
        """Save model in a format suitable for SSL usage (frozen model).

        This saves only the model weights (preferably EMA) without optimizer state,
        making it easy to load as a frozen generator in SSL frameworks.

        Args:
            filename: Output filename
        """
        save_path = self.checkpoint_dir / filename

        # Use EMA model if available (better for generation)
        model_to_save = self.ema_model if self.use_ema else self.model

        ssl_package = {
            "model_state_dict": model_to_save.state_dict()
            if self.use_ema
            else self.model.state_dict(),
            "data_shape": self.data_shape,
            "config": {
                "predict": self.predict,
                "model_type": self.cfg.get("model"),
            },
            "metadata": {
                "trained_epochs": self.current_epoch,
                "trained_steps": self.current_step,
                "agent": "AugmentationFlow",
            },
        }

        torch.save(ssl_package, save_path)
        print(f"SSL-ready model saved: {save_path}")
        print(f"  Load in SSL agent with: torch.load('{save_path}')")
        print("  Then: model.load_state_dict(checkpoint['model_state_dict'])")
        print("  And freeze: model.eval() + model.requires_grad_(False)")

        return save_path

    def _save_checkpoint_component(self, filename):
        """Save checkpoint using checkpoint component.

        Args:
            filename: Checkpoint filename
        """
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "data_shape": self.data_shape,
            "config": {
                "predict": self.predict,
                "use_ema": self.use_ema,
                "use_conditioning": self.use_conditioning,
            },
        }

        if self.ema_model is not None:
            state["ema_model"] = self.ema_model.state_dict()

        self.model_checkpoint.save(state, filename)

    def _save_ssl_checkpoint_component(self, filename="augmentation_flow_model.pt"):
        """Save SSL-ready model using checkpoint component.

        Args:
            filename: Output filename
        """
        model_to_save = self.ema_model if self.use_ema else self.model

        ssl_state = {
            "model_state_dict": model_to_save.state_dict()
            if self.use_ema
            else self.model.state_dict(),
            "data_shape": self.data_shape,
            "config": {
                "predict": self.predict,
                "model_type": self.cfg.get("model"),
                "use_conditioning": self.use_conditioning,
            },
            "metadata": {
                "trained_epochs": self.current_epoch,
                "trained_steps": self.current_step,
                "agent": "AugmentationFlow",
            },
        }

        return self.model_checkpoint.save_for_inference(
            model_state=ssl_state["model_state_dict"],
            metadata=ssl_state,
            filename=filename,
        )
