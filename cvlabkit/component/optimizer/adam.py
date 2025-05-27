import torch
from cvlabkit.component.base import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, cfg, params):
        lr = float(cfg.lr) if hasattr(cfg, "lr") else 0.01
        betas = float(cfg.betas) if hasattr(cfg, "betas") else (0.9, 0.999)
        eps = float(cfg.eps) if hasattr(cfg, "eps") else 1e-8
        weight_decay = float(cfg.weight_decay) if hasattr(cfg, "weight_decay") else 0.0

        self.opt = torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()
