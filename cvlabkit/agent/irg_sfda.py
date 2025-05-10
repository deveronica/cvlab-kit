import torch

from cvlabkit.core.agent import Agent
from cvlabkit.core.creator import Creator


class IrgSfdaAgent(Agent):
    """Instance-Relation Graph Source-Free DA Agent"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.create = Creator(cfg)

        self.detector = self.create.model.detector()
        self.irg_head = self.create.model.irg(backbone=self.detector.backbone)

        self.loss_fn = self.create.loss()
        self.optimizer = self.create.optimizer(
            params=list(self.detector.parameters()) + list(self.irg_head.parameters())
        )
        self.scheduler = self.create.scheduler(self.optimizer)
        self.metric = self.create.metric()
        self.loader = self.create.dataloader.target()

    # ── mandatory API (diagram) ────────────
    def train_step(self, batch):
        images, targets = batch
        preds = self.detector(images, targets)
        irg_feats = self.irg_head(preds)
        loss = self.loss_fn(preds, irg_feats, targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return loss.detach()

    def train_epoch(self, epoch):
        for batch in self.loader:
            self.train_step(batch)
        self.scheduler.step()

    def evaluate(self, loader):
        self.metric.update(self.detector, loader)
        return self.metric.compute()

    # ── public entrypoints ─────────────────
    def dry_run(self):
        self.train_step(next(iter(self.loader)))

    def fit(self):
        for epoch in range(self.cfg.epochs):
            self.train_epoch(epoch)
            print(self.evaluate(self.loader))

if __name__ == "__main__":
    import inspect
    from cvlabkit.agent.base import Agent as BaseAgent

    print("Found subclasses:")
    for name, cls in inspect.getmembers(__import__(__name__), inspect.isclass):
        if issubclass(cls, BaseAgent) and cls is not BaseAgent:
            print("-", name)