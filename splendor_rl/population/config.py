from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROLES = (
    "main",
    "main_exploiter_0",
    "main_exploiter_1",
    "league_exploiter_0",
    "league_exploiter_1",
)


@dataclass
class PopulationConfig:
    seed: int = 42
    training_mode: str = "population_league_2p"
    num_players: int = 2
    num_envs: int = 8
    payment_mode: str = "canonical"
    hidden_sizes: list[int] = field(default_factory=lambda: [512, 512, 512])
    transitions_per_update: int = 16_384
    update_epochs: int = 4
    minibatch_size: int = 2048
    learning_rate: float = 3e-4
    min_learning_rate: float = 0.0
    linear_lr_decay: bool = True
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
    current_entropy_coef: float | None = None
    max_turns: int = 300
    device: str = "cpu"
    total_population_transitions: int = 50_000_000
    scheduler: dict[str, float] = field(
        default_factory=lambda: {
            "main": 0.6,
            "main_exploiter_0": 0.1,
            "main_exploiter_1": 0.1,
            "league_exploiter_0": 0.1,
            "league_exploiter_1": 0.1,
        }
    )
    main_training: dict[str, float] = field(
        default_factory=lambda: {
            "current_selfplay": 0.2,
            "champion": 0.15,
            "historical_pfsp": 0.2,
            "meta_strategy": 0.2,
            "main_exploiter": 0.15,
            "league_exploiter": 0.1,
        }
    )
    main_exploiter: dict[str, float] = field(
        default_factory=lambda: {
            "current_champion": 0.7,
            "current_main": 0.2,
            "hall_of_fame": 0.1,
        }
    )
    league_exploiter: dict[str, float] = field(
        default_factory=lambda: {
            "meta_strategy": 0.7,
            "historical_pfsp": 0.2,
            "current_champion": 0.1,
        }
    )
    meta_update_interval: int = 1_000_000
    meta_max_policies: int = 24
    meta_matchup_games_per_pair: int = 100
    meta_solver_iterations: int = 20_000
    cycle_score_threshold: float = 0.55
    exploiter_evaluation_interval: int = 500_000
    exploiter_pair_count: int = 300
    exploiter_bootstrap_samples: int = 10_000
    exploiter_min_target_score: float = 0.55
    exploiter_min_ci_lower: float = 0.50
    exploiter_max_transitions_without_success: int = 2_000_000
    main_promotion_interval: int = 1_000_000
    main_promotion_pair_count: int = 500
    main_promotion_bootstrap_samples: int = 10_000
    main_promotion_min_score: float = 0.55
    main_promotion_min_ci_lower: float = 0.50
    main_promotion_min_meta_delta: float = 0.0
    main_promotion_max_exploiter_regression: float = 0.03
    promotion_anchor_games: int = 200
    recent_snapshot_interval: int = 250_000
    checkpoint_interval: int = 250_000
    pfsp_prior_alpha: float = 1.0
    pfsp_prior_beta: float = 1.0
    pfsp_alpha: float = 1.0
    pfsp_epsilon: float = 0.05

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        meta = raw.pop("meta", {})
        solver = meta.pop("solver", {})
        exploiter = raw.pop("exploiter", {})
        promotion = raw.pop("main_promotion", {})
        aliases = {
            **raw,
            **{f"meta_{k}": v for k, v in meta.items()},
            "meta_solver_iterations": solver.get("iterations", 20_000),
            **{f"exploiter_{k}": v for k, v in exploiter.items()},
            **{f"main_promotion_{k}": v for k, v in promotion.items()},
        }
        value = cls(**aliases)
        value.validate()
        return value

    def validate(self):
        if self.training_mode != "population_league_2p" or self.num_players != 2:
            raise ValueError("population league supports exactly two players")
        if (
            set(self.scheduler) - set(ROLES)
            or abs(sum(self.scheduler.values()) - 1) > 1e-9
        ):
            raise ValueError("scheduler roles must be known and sum to one")
        for mixture in (self.main_training, self.main_exploiter, self.league_exploiter):
            if (
                any(value < 0 for value in mixture.values())
                or abs(sum(mixture.values()) - 1) > 1e-9
            ):
                raise ValueError("opponent mixture must be non-negative and sum to one")
        if self.total_population_transitions <= 0 or self.transitions_per_update <= 0:
            raise ValueError("transition counts must be positive")

    @property
    def total_transitions(self):
        return self.total_population_transitions

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)
