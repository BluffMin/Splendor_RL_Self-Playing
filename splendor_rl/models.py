from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(
    input_size: int, output_size: int, hidden: list[int], output_gain: float
) -> nn.Sequential:
    layers = []
    previous = input_size
    for size in hidden:
        linear = nn.Linear(previous, size)
        nn.init.orthogonal_(linear.weight, math.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.extend([linear, nn.LayerNorm(size), nn.SiLU()])
        previous = size
    output = nn.Linear(previous, output_size)
    nn.init.orthogonal_(output.weight, output_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class SharedActor(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.network = _mlp(
            observation_size, action_size, hidden_sizes or [512] * 3, 0.01
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class PrivilegedCritic(nn.Module):
    def __init__(self, state_size: int, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        self.network = _mlp(state_size, 1, hidden_sizes or [512] * 3, 1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)
