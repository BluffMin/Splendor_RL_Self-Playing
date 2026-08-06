from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .player_count import validate_num_players


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
    linear_lr_decay: bool = True
    min_learning_rate: float = 0.0
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    value_coef: float = 0.5
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    entropy_anneal_fraction: float = 0.8
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    target_kl_mode: str = "mean_epoch"
    normalize_advantages: bool = True
    total_transitions: int = 10_000_000
    checkpoint_interval: int = 500_000
    evaluation_interval: int = 250_000
    evaluate_initial_policy: bool = True
    evaluation_games_per_matchup: int = 100
    evaluation_deterministic: bool = True
    evaluation_seed_base: int = 100_000
    continue_on_evaluation_error: bool = False
    max_turns: int | None = 300
    device: str = "cpu"
    current_entropy_coef: float | None = None

    @classmethod
    def load(cls, path: str | Path) -> PPOConfig:
        # JSON is valid YAML; configs deliberately use this dependency-free subset.
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**data)
        config.validate()
        return config

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if value is not None:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        validate_num_players(self.num_players)
        for name in ("checkpoint_interval", "evaluation_interval"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        if self.evaluation_games_per_matchup <= 0:
            raise ValueError("evaluation_games_per_matchup must be positive")
        if not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate must be between zero and learning_rate")
        if self.entropy_coef_start < 0 or self.entropy_coef_end < 0:
            raise ValueError("entropy coefficients must be non-negative")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be positive or None")
        if self.target_kl_mode not in {"mean_epoch", "max_minibatch"}:
            raise ValueError("invalid target_kl_mode")
        if (
            self.evaluation_interval
            and self.evaluation_interval > self.total_transitions
        ):
            warnings.warn("evaluation_interval exceeds total_transitions", stacklevel=2)
