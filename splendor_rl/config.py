from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PPOConfig:
    seed: int = 42
    num_players: int = 4
    num_envs: int = 128
    collector_backend: str = "single_process"
    payment_mode: str = "canonical"
    hidden_sizes: list[int] = field(default_factory=lambda: [512, 512, 512])
    transitions_per_update: int = 65536
    update_epochs: int = 4
    minibatch_size: int = 8192
    learning_rate: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    value_coef: float = 0.5
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    entropy_anneal_fraction: float = 0.8
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    normalize_advantages: bool = True
    total_transitions: int = 10_000_000
    checkpoint_interval: int = 500_000
    evaluation_interval: int = 250_000
    max_turns: int | None = 300
    device: str = "cpu"

    @classmethod
    def load(cls, path: str | Path) -> PPOConfig:
        # JSON is valid YAML; configs deliberately use this dependency-free subset.
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if value is not None:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
