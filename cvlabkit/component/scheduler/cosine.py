# cvlabkit/component/scheduler/cosine.py

import torch

from cvlabkit.component.base import Scheduler


class CosineScheduler(Scheduler):
    def __init__(self, cfg, optimizer):
        base_opt = getattr(optimizer, "opt", optimizer)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            base_opt, T_max=int(cfg.epochs) if hasattr(cfg, "epochs") else 10
        )

    def step(self):
        self.scheduler.step()
