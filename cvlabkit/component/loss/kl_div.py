import torch
import torch.nn as nn
import torch.nn.functional as F

from cvlabkit.component.base import Loss
from cvlabkit.core.config import Config


class KlDiv(Loss):
    """Computes the Kullback-Leibler divergence loss.

    This component wraps `torch.nn.KLDivLoss`. It expects raw logits as input
    and applies the log-softmax function to the predictions before computing
    the loss.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.reduction = cfg.get("reduction", "batchmean")  # 'batchmean'|'mean'|'sum'|'none'
        self.dim = int(cfg.get("dim", 1))
        self.eps = float(cfg.get("eps", 1e-12))

    def _to_probs(self, t: torch.Tensor) -> torch.Tensor:
        # clamp & renormalize for numerical safety
        t = t.clamp_min(self.eps)
        t = t / t.sum(dim=self.dim, keepdim=True).clamp_min(self.eps)
        return t

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # q: preds (logits)
        logq = F.log_softmax(preds, dim=self.dim)
        # p: targets (logits -> probs, stop-grad)
        p = F.softmax(targets.detach(), dim=self.dim)
        p = self._to_probs(p).to(dtype=logq.dtype, device=logq.device)

        # KL(p || q) = -H(p) + H(p, q)
        loss = F.kl_div(logq, p, reduction=self.reduction)     # KL(p || q)
        return torch.nan_to_num(loss)
