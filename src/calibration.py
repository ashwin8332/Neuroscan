"""Post-hoc probability calibration (temperature scaling, Guo et al. 2017).

A single scalar T divides the logits; it is fitted on the validation set by
minimising the negative log-likelihood. It never changes which class wins, it
only makes the displayed confidence match the observed accuracy - essential
when a UI shows "97% confidence" to non-experts.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MIN_TEMPERATURE = 0.5
MAX_TEMPERATURE = 10.0


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    logits = logits.detach().double().cpu()
    labels = labels.detach().long().cpu()
    if logits.numel() == 0:
        return 1.0
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.exp().item())
    if not torch.isfinite(torch.tensor(temperature)):
        return 1.0
    # A perfectly separated validation set pushes T -> 0; clamp to a sane range.
    return float(min(MAX_TEMPERATURE, max(MIN_TEMPERATURE, temperature)))


def nll(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> float:
    return float(F.cross_entropy(logits.double() / temperature, labels.long()).item())
