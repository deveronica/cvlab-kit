import torch
from tqdm import tqdm

from cvlabkit.core.agent import Agent
from cvlabkit.core.config import Config
from cvlabkit.core.creator import ComponentCreator


class Classification(Agent):
    """An agent for standard image classification tasks."""

    def setup(self):
        """Creates and initializes all necessary components for the agent."""
        print("Initializing components...")
        self.device = (
            f"cuda:{self.cfg.get('device', 0)}" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.create.model().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())
        self.loss_fn = self.create.loss()

        transform = self.create.transform() if "transform" in self.cfg else None

        # Create datasets using the named configurations from example.yaml
        train_dataset = self.create.dataset.train(transform=transform)
        val_dataset = self.create.dataset.val(transform=transform)

        # Dataloaders receive the dataset object as a positional argument.
        self.train_loader = self.create.dataloader.train(train_dataset)
        self.val_loader = self.create.dataloader.val(val_dataset)
        self.metric = self.create.metric()
        print(f"Components initialized successfully. Using device: {self.device}")

    def train_step(self, batch):
        """Performs a single training step."""
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, batch):
        """Performs a single validation step."""
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)

        self.metric.update(preds=outputs, targets=labels)

        return {"loss": loss.item()}

    def evaluate(self):
        """Evaluates the model on the validation set."""
        self.metric.reset()
        super().evaluate()
        metrics = self.metric.compute()
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(
            f"Epoch {self.current_epoch}/{self.cfg.epochs} - "
            f"Validation Metrics: {metric_str}"
        )

    def save(self, path: str):
        """Saves the model and optimizer state."""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Loads the model and optimizer state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {path}")