import torch

from cvlabkit.component.base import Optimizer


class Adam(Optimizer):
    def __init__(self, cfg, params):
        lr = cfg.get("lr", 0.001)
        betas = cfg.get("betas", (0.9, 0.999))
        eps = cfg.get("eps", 1e-8)
        weight_decay = cfg.get("weight_decay", 0)

        self.opt = torch.optim.Adam(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()
