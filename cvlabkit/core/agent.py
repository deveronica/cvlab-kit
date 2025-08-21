"""Defines the abstract base class for all experiment agents.

This module provides the `Agent` class, which serves as the skeleton for
all training, evaluation, and experiment-running logic. Users must subclass
`Agent` and implement its abstract methods to define the specific behavior
of their experiment.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from tqdm import tqdm
import torch

from cvlabkit.core.config import Config


class Agent(ABC):
    """Abstract base class for experiment agents.

    This class provides the main structure for running an experiment. Users must
    subclass it and implement the abstract methods to define the core logic
    for training and validation.

    The agent orchestrates the entire experiment lifecycle, from setting up
    components (models, data, etc.) to running the training loop and evaluating
    the results.

    Attributes:
        cfg (Config): The configuration object for the experiment.
        create (ComponentCreator): The factory object for creating components.
        current_epoch (int): The current epoch number (0-indexed).
        current_step (int): The total number of training steps taken.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
    """

    def __init__(self, cfg: Config, component_creator: 'ComponentCreator'):
        """Initializes the Agent with configuration and component creator.
        Args:
            cfg (Config): Configuration object containing parameters.
            component_creator (ComponentCreator): Creator instance for components.
        """
        self.cfg = cfg
        self.create = component_creator
        self.current_epoch: int = 0
        self.current_step: int = 0

        self.train_loader = None
        self.val_loader = None

        self.setup()

    def setup(self) -> None:
        """Initializes and sets up all components required for the agent.

        This method is called by the agent's `__init__` and should be used to
        create and configure the model, data loaders, optimizer, loss functions,
        and any other components needed for the experiment, using `self.create`.
        """
        pass

    @abstractmethod
    def train_step(self, batch: Any) -> None:
        """Perform a single training step.
        Args:
            batch (Any): A batch of data from the training dataloader.
        """
        pass

    def validate_step(self, batch: Any) -> None:
        """Perform a single validation step.
        Args:
            batch (Any): A batch of data from the validation dataloader.
        """
        pass

    def save(self, path: str) -> None:
        """Save the model and training state to the specified path.
        Args:
            path (str): Path to save the model and state.
        """
        raise NotImplementedError(
            "The save method must be implemented by the subclass."
        )

    def load(self, path: str) -> None:
        """Load the model and training state from the specified path.
        Args:
            path (str): Path to load the model and state from.
        """
        raise NotImplementedError(
            "The load method must be implemented by the subclass."
        )

    def fit(self) -> None:
        """Fitting the model from the current state for cfg.epochs additional epochs.

        If 'checkpoint_path' is specified in the configuration, the checkpoint is loaded.
        If 'checkpoint_dir' and 'checkpoint_interval' are specified, the agent state is saved.
        """

        # Load a checkpoint if a specific path is provided in the config.
        if hasattr(self.cfg, "checkpoint_path") and self.cfg.checkpoint_path:
            self.load(self.cfg.checkpoint_path)

        if not hasattr(self.cfg, "epochs"):
            raise ValueError("cfg.epochs must be defined for fit().")

        train_epochs = self.cfg.get("epochs", 1)
        target_epochs = self.current_epoch + train_epochs

        while self.current_epoch < target_epochs:
            print(f"Starting epoch {self.current_epoch + 1}/{target_epochs}...")
            self.train_epoch()
            self.evaluate()
            self.current_epoch += 1

            # Check if checkpointing is enabled and if it's time to save.
            should_save = (
                hasattr(self.cfg, "checkpoint_dir") and
                hasattr(self.cfg, "checkpoint_interval") and
                self.cfg.checkpoint_interval > 0 and
                self.current_epoch % self.cfg.checkpoint_interval == 0
            )
            if should_save:
                import os
                save_path = os.path.join(
                    self.cfg.checkpoint_dir,
                    f"checkpoint_{self.current_epoch}.pt"
                )
                if not os.path.exists(self.cfg.checkpoint_dir):
                    os.makedirs(self.cfg.checkpoint_dir)

                print(f"Saving checkpoint to {save_path}...")
                # Assuming self.save() handles saving the agent and state
                self.save(save_path)
                print(f"Checkpoint saved to {save_path}")

    def train_epoch(self) -> None:
        """Agent train for one epoch."""
        if self.train_loader is None:
            raise ValueError("train_loader must be set before training.")

        self.model.train()
        for batch in tqdm(self.train_loader, 
                          desc=f"Epoch {self.current_epoch + 1} Training"):
            self.train_step(batch)
            self.current_step += 1

    def evaluate(self) -> None:
        """Evaluate the model on the validation set.
        
        Raises:
            ValueError: If val_loader is not defined.
        """
        if self.val_loader is None:
            raise ValueError("val_loader must be set before evaluation.")

        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                self.validate_step(batch)
