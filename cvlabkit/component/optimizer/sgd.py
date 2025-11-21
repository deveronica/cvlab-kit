import torch

from cvlabkit.component.base import Optimizer


class SgdOptimizer(Optimizer):
    def __init__(self, cfg, params):
        lr = float(cfg.lr) if hasattr(cfg, "lr") else 0.01
        momentum = float(cfg.momentum) if hasattr(cfg, "momentum") else 0.9
        weight_decay = float(cfg.weight_decay) if hasattr(cfg, "weight_decay") else 0.0

        self.opt = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()
