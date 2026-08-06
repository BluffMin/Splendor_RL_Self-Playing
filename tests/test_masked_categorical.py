import pytest
import torch

from splendor_rl.distributions import MaskedCategorical


def test_masked_distribution_is_finite_and_legal():
    logits = torch.randn(2, 4, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.bool)
    dist = MaskedCategorical(logits, mask)
    assert torch.all(dist.probs[~mask] == 0) and dist.entropy()[1] == 0
    samples = torch.stack([dist.sample() for _ in range(100)])
    assert torch.all((samples[:, 0] == 0) | (samples[:, 0] == 2)) and torch.all(
        samples[:, 1] == 3
    )
    dist.entropy().sum().backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError):
        MaskedCategorical(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
