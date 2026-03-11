import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cvlabkit.core.agent import Agent


class FixmatchTrainer(Agent):
    def setup(self):
        self.set_seed()

        self.device = self.cfg.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.create.model().to(self.device)
        self.optimizer = self.create.optimizer(self.model.parameters())

        self.train_loader = self.create.dataloader.labeled()
        self.unlabeled_loader = self.create.dataloader.unlabeled()
        self.val_loader = self.create.dataloader.test()

        self.lx_fn = self.create.loss.cross_entropy()
        self.logger = self.create.logger()
        self.accuracy_metric = self.create.metric.accuracy()

        self.best_acc = 0
        self.early_stopping_counter = 0

    def set_seed(self):
        seed = self.cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.cfg.get("n_gpu", 0) > 0 and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_epoch(self):
        self.model.train()
        target_epochs = self.cfg.get("epochs", 1)
        # zip stops when the shorter iterable is exhausted
        for batch in tqdm(
            zip(self.train_loader, self.unlabeled_loader),
            desc=f"Epoch {self.current_epoch + 1}/{target_epochs}",
        ):
            self.train_step(batch)
            self.current_step += 1

    def train_step(self, batch):
        (labeled_batch, unlabeled_batch) = batch

        inputs_x, targets_x = labeled_batch
        # The unlabeled batch format depends on the transform used.
        # Assuming (weak_aug, strong_aug1, strong_aug2) for AugMix or (weak, strong) for FixMatch
        inputs_u_w, inputs_u_s1, *rest = unlabeled_batch[0]
        inputs_u_s2 = rest[0] if rest else inputs_u_s1  # Handle both cases

        inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
        inputs_u_w = inputs_u_w.to(self.device)
        inputs_u_s1 = inputs_u_s1.to(self.device)
        inputs_u_s2 = inputs_u_s2.to(self.device)

        logits_x = self.model(inputs_x)
        with torch.no_grad():
            logits_u_w = self.model(inputs_u_w)
        logits_u_s1 = self.model(inputs_u_s1)
        logits_u_s2 = self.model(inputs_u_s2)

        Lx = self.lx_fn(logits_x, targets_x)

        pseudo_label = torch.softmax(logits_u_w, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.cfg.get("threshold", 0.95)).float()

        Lu = (F.cross_entropy(logits_u_s1, targets_u, reduction="none") * mask).mean()

        Ljsd = 0
        if self.cfg.get("use_jsd", False):
            p_u_w = torch.softmax(logits_u_w, dim=1)
            p_u_s1 = torch.softmax(logits_u_s1, dim=1)
            p_u_s2 = torch.softmax(logits_u_s2, dim=1)
            p_mix = torch.clamp((p_u_w + p_u_s1 + p_u_s2) / 3.0, 1e-7, 1).log()
            Ljsd = (
                self.cfg.get("lambda_jsd", 1.0)
                * (
                    F.kl_div(p_mix, p_u_w, reduction="batchmean")
                    + F.kl_div(p_mix, p_u_s1, reduction="batchmean")
                    + F.kl_div(p_mix, p_u_s2, reduction="batchmean")
                )
                / 3.0
            )

        lambda_u = self.cfg.get("lambda_u", 1.0)
        loss = Lx + lambda_u * Lu + Ljsd

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.log_metrics(
            {
                "total_loss": loss.item(),
                "Lx": Lx.item(),
                "Lu": Lu.item(),
                "Ljsd": Ljsd.item() if isinstance(Ljsd, torch.Tensor) else Ljsd,
                "mask_ratio": mask.mean().item(),
            },
            step=self.current_step,
        )

    def validate_step(self, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        self.accuracy_metric.update(outputs, targets)

    def evaluate(self):
        self.accuracy_metric.reset()
        super().evaluate()
        val_acc = self.accuracy_metric.compute()
        self.logger.log_metrics(
            {"val_accuracy": val_acc.item()}, step=self.current_epoch
        )

        is_best = val_acc > self.best_acc
        self.best_acc = max(val_acc, self.best_acc)

        if is_best:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            patience = self.cfg.get("early_stopping_patience", 100)
            if self.early_stopping_counter > patience:
                print(
                    f"Early stopping at epoch {self.current_epoch} due to no improvement."
                )
                self.current_epoch = self.cfg.get("epochs")
