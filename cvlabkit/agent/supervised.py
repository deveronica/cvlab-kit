# cvlabkit/agent/supervised.py

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from cvlabkit.core.agent import Agent


class Supervised(Agent):
    """
    A baseline agent that performs standard supervised learning on a labeled dataset.
    
    This agent serves as a baseline to compare against semi-supervised methods.
    It uses a subset of the training data designated as 'labeled' by the config.
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
        self.train_transform = self.create.transform.train()
        self.val_transform = self.create.transform.val()
        self.loss_fn = self.create.loss()
        self.metric = self.create.metric()
        
        if self.cfg.get("logger"):
            self.logger = self.create.logger()

        # --- Data Handling with Stratified Splitting ---
        # Pass the transform objects to the dataset constructor.
        train_dataset = self.create.dataset.train(transform=self.train_transform)
        val_dataset = self.create.dataset.val(transform=self.val_transform)

        num_labeled = self.cfg.get("num_labeled", len(train_dataset))
        targets = np.array(train_dataset.targets)
        
        # We only need the labeled indices for supervised training.
        labeled_indices, _ = self._stratified_split(targets, num_labeled)
        
        labeled_sampler = self.create.sampler(indices=labeled_indices)

        self.train_loader = self.create.dataloader.train(
            dataset=train_dataset,
            sampler=labeled_sampler
        )
        self.val_loader = self.create.dataloader.val(
            dataset=val_dataset
        )

    def _stratified_split(self, targets: np.ndarray, num_labeled: int):
        """Performs a stratified split of indices to get a representative labeled set."""
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

        print(f"Dataset split: Using {len(labeled_indices)} labeled samples for training.")
        return labeled_indices, unlabeled_indices

    def train_step(self, batch):
        """Performs a single training step."""
        self.model.train()
        images, labels = batch
        
        # The dataloader should handle device placement if possible,
        # but we ensure it here for robustness.
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        preds = self.model(images)
        loss = self.loss_fn(preds, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}

    def fit(self):
        """The main training loop."""
        train_epochs = self.cfg.get("epochs", 1)
        target_epochs = self.current_epoch + train_epochs

        while self.current_epoch < target_epochs:
            epoch_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch [{self.current_epoch+1}/{target_epochs}]")
            for batch in progress_bar:
                loss_dict = self.train_step(batch)
                epoch_loss += loss_dict['loss']
                progress_bar.set_postfix(loss=f"{loss_dict['loss']:.4f}")

            if hasattr(self, 'logger') and self.logger is not None:
                avg_train_loss = epoch_loss / len(self.train_loader)
                self.logger.log_metrics(metrics={'train_loss': avg_train_loss}, step=self.current_epoch + 1)

            self.evaluate()
            self.current_epoch += 1

    def evaluate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        self.metric.reset()
        
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                preds = self.model(images)
                
                val_loss = self.loss_fn(preds, labels)
                total_val_loss += val_loss.item()
                
                self.metric.update(preds=preds, targets=labels)
        
        avg_val_loss = total_val_loss / len(self.val_loader)
        
        metrics = self.metric.compute()
        metrics['val_loss'] = avg_val_loss
        
        print(f"Epoch {self.current_epoch+1} Validation Metrics: {metrics}")

        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.log_metrics(metrics=metrics, step=self.current_epoch + 1)
