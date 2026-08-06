from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from splendor_rl.config import PPOConfig


@dataclass
class LeagueConfig(PPOConfig):
    training_mode: str = "league_2p"
    current_selfplay_fraction: float = 0.20
    champion_fraction: float = 0.40
    historical_pfsp_fraction: float = 0.40
    recent_snapshot_interval: int = 100_000
    max_recent_snapshots: int = 8
    pfsp_recent_window_games: int = 400
    pfsp_prior_alpha: float = 1.0
    pfsp_prior_beta: float = 1.0
    pfsp_alpha: float = 1.0
    pfsp_epsilon: float = 0.05
    promotion_interval: int = 250_000
    promotion_pair_count: int = 500
    promotion_bootstrap_samples: int = 10_000
    promotion_confidence: float = 0.95
    promotion_min_score: float = 0.55
    promotion_min_lower_bound: float = 0.50
    promotion_anchor_games_per_opponent: int = 200
    promotion_historical_anchor_count: int = 3
    max_anchor_aggregate_regression: float = 0.03
    max_single_anchor_regression: float = 0.07
    matchup_matrix_interval: int = 500_000
    matchup_matrix_games_per_pair: int = 100
    matchup_matrix_max_policies: int = 12
    tutor_trace_enabled: bool = False

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("league config must contain a YAML mapping")
        value = cls(**data)
        value.validate()
        return value

    def validate(self):
        super().validate()
        if self.training_mode != "league_2p":
            raise ValueError("training_mode must be league_2p")
        if self.num_players != 2:
            raise ValueError("league_2p currently supports exactly two players.")
        fractions = (
            self.current_selfplay_fraction,
            self.champion_fraction,
            self.historical_pfsp_fraction,
        )
        if any(value < 0 for value in fractions) or abs(sum(fractions) - 1) > 1e-9:
            raise ValueError(
                "league episode fractions must be non-negative and sum to one"
            )
        positive = (
            "recent_snapshot_interval",
            "max_recent_snapshots",
            "pfsp_recent_window_games",
            "promotion_interval",
            "promotion_pair_count",
            "promotion_bootstrap_samples",
            "promotion_anchor_games_per_opponent",
            "matchup_matrix_interval",
            "matchup_matrix_games_per_pair",
            "matchup_matrix_max_policies",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.promotion_confidence < 1:
            raise ValueError("promotion_confidence must be between zero and one")
