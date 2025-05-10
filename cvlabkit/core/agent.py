from __future__ import annotations

from cvlabkit.core.config import Config


class Agent:
    """
    Base class for experiment agents.
    Subclasses must implement train_step, train_epoch, and evaluate.
    Optional: override dry_run() or fit() for custom behavior.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def train_step(self, batch):
        raise NotImplementedError("train_step(batch) must be implemented by subclass.")

    def train_epoch(self, epoch: int):
        raise NotImplementedError("train_epoch(epoch) must be implemented by subclass.")

    def evaluate(self, loader):
        raise NotImplementedError("evaluate(loader) must be implemented by subclass.")

    def dry_run(self) -> None:
        if not hasattr(self, "loader"):
            raise RuntimeError("dry_run() requires self.loader to be defined in __init__.")
        batch = next(iter(self.loader))
        self.train_step(batch)

    def fit(self) -> None:
        if not hasattr(self.cfg, "epochs"):
            raise ValueError("Config must define 'epochs' for fit().")
        for epoch in range(self.cfg.epochs):
            self.train_epoch(epoch)
            print(f"[Epoch {epoch}] Eval: {self.evaluate(self.loader)}")
