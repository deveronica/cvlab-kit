"""Provides a standard agent for basic image classification tasks."""

import torch
from tqdm import tqdm

from cvlabkit.core.agent import Agent
from cvlabkit.core.config import Config
from cvlabkit.core.creator import ComponentCreator


class Classification(Agent):
    """A general-purpose agent for standard image classification.

    This agent implements a typical training and validation workflow. It handles
    the creation of components like the model, optimizer, data loaders, and loss
    function based on the provided configuration. It also includes logic for
    checkpointing and evaluation.
    """

    def setup(self):
        """Creates and sets up all components required for classification.

        This method uses the component creator to instantiate the model, optimizer,
        loss function, data loaders, and metrics. It also handles the creation
        of transforms and associates them with the correct datasets.
        """
        print("Initializing components...")
        self.device = (
            f"cuda:{self.cfg.get('device', 0)}" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.create.model().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())
        self.loss_fn = self.create.loss()

        transform = self.create.transform() if "transform" in self.cfg else None

        # Create datasets using the named configurations from the config file.
        train_dataset = self.create.dataset.train(transform=transform)
        val_dataset = self.create.dataset.val(transform=transform)

        # Dataloaders receive the dataset object as a positional argument.
        self.train_loader = self.create.dataloader.train(train_dataset)
        self.val_loader = self.create.dataloader.val(val_dataset)
        self.metric = self.create.metric()
        print(f"Components initialized successfully. Using device: {self.device}")

    def train_step(self, batch):
        """Performs a single training step on a batch of data.

        The process is as follows:
        1.  Move data and labels to the configured device.
        2.  Perform a forward pass through the model.
        3.  Calculate the loss.
        4.  Perform a backward pass and update the optimizer.

        Args:
            batch: A tuple containing the input data and corresponding labels.

        Returns:
            A dictionary containing the loss for this training step.
        """
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, batch):
        """Performs a single validation step on a batch of data.

        The process is as follows:
        1.  Move data and labels to the configured device.
        2.  Perform a forward pass through the model.
        3.  Calculate the loss.
        4.  Update the metric with the predictions and labels.

        Args:
            batch: A tuple containing the input data and corresponding labels.

        Returns:
            A dictionary containing the loss for this validation step.
        """
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)

        self.metric.update(preds=outputs, targets=labels)

        return {"loss": loss.item()}

    def evaluate(self):
        """Evaluates the model on the validation set and prints the metrics."""
        self.metric.reset()
        super().evaluate()
        metrics = self.metric.compute()
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(
            f"Epoch {self.current_epoch}/{self.cfg.epochs} - "
            f"Validation Metrics: {metric_str}"
        )

    def save(self, path: str):
        """Saves the model, optimizer, and current epoch to a checkpoint file."""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Loads the model, optimizer, and epoch from a checkpoint file."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {path}")
