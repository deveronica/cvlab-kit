"""Agent for training Rectified Flow models."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from ema_pytorch import EMA
from torchvision.utils import save_image

from cvlabkit.core.agent import Agent


class RectifiedFlow(Agent):
    """Agent for training Rectified Flow generative models.

    This agent handles training of flow-based generative models using
    ODE integration for sampling.
    """

    def setup(self):
        """Set up all components for training."""
        # Device setup
        device_id = self.cfg.get("device", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

        # Create components using Creator pattern
        self.model = self.create.model().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())

        # Create transforms
        self.normalize = self.create.transform.data_normalizer()
        self.unnormalize = self.create.transform.unnormalize()
        self.noise_schedule = self.create.transform.noise_schedule()

        # Create solver for sampling
        self.solver = self.create.solver()

        # Loss function (if specified, otherwise use MSE)
        if self.cfg.get("loss"):
            self.loss_fn = self.create.loss()
        else:
            self.loss_fn = None

        # Create dataset and dataloader
        dataset = self.create.dataset()
        self.train_loader = self.create.dataloader(
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Training configuration
        self.predict = self.cfg.get("predict", "flow")  # 'flow' or 'noise'
        self.max_grad_norm = self.cfg.get("max_grad_norm", 0.5)

        # EMA (optional)
        self.use_ema = self.cfg.get("use_ema", False)
        self.ema_model = None

        if self.use_ema:
            ema_kwargs = self.cfg.get("ema_kwargs", {})
            self.ema_model = EMA(self.model, **ema_kwargs)
            self.ema_model.to(self.device)

        # Results and checkpoints
        self.results_folder = Path(self.cfg.get("results_folder", "./results"))
        self.checkpoints_folder = Path(
            self.cfg.get("checkpoints_folder", "./checkpoints")
        )

        self.results_folder.mkdir(exist_ok=True, parents=True)
        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)

        self.save_results_every = self.cfg.get("save_results_every", 100)
        self.checkpoint_every = self.cfg.get("checkpoint_every", 1000)
        self.sample_temperature = self.cfg.get("sample_temperature", 1.0)

        num_samples = self.cfg.get("num_samples", 16)
        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows**2) == num_samples, f"{num_samples} must be square"
        self.num_samples = num_samples

        # Data shape (set on first batch)
        self.data_shape = None

        # Logger
        if self.cfg.get("logger"):
            self.logger = self.create.logger()
        else:
            self.logger = None

    def train_step(self, batch):
        """Perform a single training step.

        Args:
            batch: Batch of data from dataloader
        """
        self.model.train()

        data = batch.to(self.device)

        # Set data shape on first batch
        if self.data_shape is None:
            self.data_shape = data.shape[1:]

        # Normalize data
        data = self.normalize(data)

        # Sample noise and times
        noise = torch.randn_like(data)
        times = torch.rand(len(data), device=self.device)

        # Apply noise schedule
        scheduled_times = self.noise_schedule(times)
        padded_times = scheduled_times.view(-1, *([1] * (data.ndim - 1)))

        # Interpolate: noise → data using times
        noised = noise.lerp(data, padded_times)

        # Model prediction
        pred = self.model(noised, times=scheduled_times)

        # Determine target based on prediction objective
        if self.predict == "flow":
            target = data - noise
        elif self.predict == "noise":
            target = noise
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Update EMA
        if self.use_ema:
            self.ema_model.update()

        # Log
        if self.logger:
            self.logger.log({"loss": loss.item()}, step=self.current_step)

        # Periodic sampling
        if self.current_step % self.save_results_every == 0:
            self.sample_and_save(self.current_step)

        # Periodic checkpointing
        if self.current_step % self.checkpoint_every == 0:
            self.save(
                str(self.checkpoints_folder / f"checkpoint.{self.current_step}.pt")
            )

    def validate_step(self, batch):
        """Perform validation (sampling).

        Args:
            batch: Batch of data (unused, just for signature compatibility)
        """
        self.sample_and_save(f"val_epoch_{self.current_epoch}")

    def sample_and_save(self, fname):
        """Sample from model and save images.

        Args:
            fname: Filename or identifier for saving
        """
        if self.data_shape is None:
            print("Warning: data_shape not set, skipping sampling")
            return

        eval_model = self.ema_model if self.use_ema else self.model
        eval_model.eval()

        with torch.no_grad():
            # Initial noise
            noise = torch.randn(
                (self.num_samples, *self.data_shape), device=self.device
            )

            # Time points for ODE
            num_steps = self.cfg.get("sample_steps", 50)
            times = torch.linspace(0.0, 1.0, num_steps, device=self.device)

            # Define ODE function (captures model)
            def ode_fn(t, x):
                # Model expects times as batch
                t_batch = t.expand(x.shape[0])
                return eval_model(x, times=t_batch)

            # Solve ODE using solver component
            sampled = self.solver(ode_fn, noise, times)

            # Unnormalize
            sampled = self.unnormalize(sampled)

        # Arrange in grid
        sampled = rearrange(
            sampled, "(row col) c h w -> c (row h) (col w)", row=self.num_sample_rows
        )
        sampled.clamp_(0.0, 1.0)

        save_path = self.results_folder / f"results.{fname}.png"
        save_image(sampled, save_path)

        if self.logger:
            self.logger.log_image("samples", sampled, step=self.current_step)

    def save(self, path):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        save_package = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "data_shape": self.data_shape,
        }

        if self.ema_model is not None:
            save_package["ema_model"] = self.ema_model.state_dict()

        torch.save(save_package, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path):
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        load_package = torch.load(path, map_location=self.device)

        self.model.load_state_dict(load_package["model"])
        self.optimizer.load_state_dict(load_package["optimizer"])

        if "current_epoch" in load_package:
            self.current_epoch = load_package["current_epoch"]
        if "current_step" in load_package:
            self.current_step = load_package["current_step"]
        if "data_shape" in load_package:
            self.data_shape = load_package["data_shape"]

        if self.ema_model is not None and "ema_model" in load_package:
            self.ema_model.load_state_dict(load_package["ema_model"])

        print(f"Loaded checkpoint from {path}")
