import pytest
import torch

from splendor_rl.schedules import (
    apply_learning_rate,
    entropy_coefficient,
    linear_learning_rate,
)


def test_linear_lr_and_entropy_are_transition_based():
    assert linear_learning_rate(3e-4, 0, 0, 100) == 3e-4
    assert linear_learning_rate(3e-4, 0, 50, 100) == 1.5e-4
    assert linear_learning_rate(3e-4, 1e-5, 200, 100) == pytest.approx(1e-5)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1)
    apply_learning_rate(optimizer, 1.5e-4)
    assert optimizer.param_groups[0]["lr"] == 1.5e-4
    assert entropy_coefficient(0.01, 0.001, 0.8, 40, 100) == 0.0055
