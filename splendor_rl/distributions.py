from __future__ import annotations

import torch


class MaskedCategorical:
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.bool()
        if logits.shape != mask.shape:
            raise ValueError("logits and mask shapes differ")
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("masked categorical has an empty legal-action set")
        self.mask = mask
        self.masked_logits = logits.float().masked_fill(~mask, -1e9)
        self._distribution = torch.distributions.Categorical(logits=self.masked_logits)

    @property
    def probs(self) -> torch.Tensor:
        return self._distribution.probs.masked_fill(~self.mask, 0.0)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()

    def mode(self) -> torch.Tensor:
        return self.masked_logits.argmax(dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        p = self.probs
        return -(
            torch.where(p > 0, p * torch.log(p.clamp_min(1e-30)), torch.zeros_like(p))
        ).sum(-1)
