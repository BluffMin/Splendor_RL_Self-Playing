from __future__ import annotations

import pytest

from splendor_env.agents import GreedyAgent
from splendor_env.core import SplendorGame


@pytest.mark.slow
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_one_hundred_seeded_rule_validation_games(num_players: int) -> None:
    for seed in range(100):
        game = SplendorGame(num_players, seed=10_000 + seed)
        agent = GreedyAgent()
        while not game.done:
            game.step(agent.act(game))
            game.validate_invariants()
            assert game.decision_id < 5_000
        assert game.end_reason == "official_game_end"
        assert game.turns_completed % num_players == 0
