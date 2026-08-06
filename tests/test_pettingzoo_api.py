from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from pettingzoo.test import api_test

from splendor_env.actions import N_ACTIONS
from splendor_env.core import OBSERVATION_SIZE
from splendor_env.pettingzoo_env import raw_env


@pytest.mark.parametrize("players", [2, 3, 4])
def test_masked_rollout_and_spaces(players: int) -> None:
    environment = raw_env(num_players=players, max_turns=100)
    environment.reset(seed=players)
    rng = np.random.default_rng(players)
    assert environment.action_space("player_0").n == N_ACTIONS == 373
    assert environment.state().shape == (OBSERVATION_SIZE,) == (475,)
    for agent in environment.agent_iter(max_iter=5000):
        obs, _, terminated, truncated, info = environment.last()
        assert obs["observation"].shape == (475,)
        assert obs["action_mask"].shape == (373,)
        assert {
            "phase",
            "decision_id",
            "turn_id",
            "round_id",
            "acting_player",
            "turn_completed",
            "automatic_resolution",
        } <= info.keys()
        action = (
            None
            if terminated or truncated
            else int(rng.choice(np.flatnonzero(obs["action_mask"])))
        )
        environment.step(action)
    assert not environment.agents


def test_api() -> None:
    api_test(
        raw_env(num_players=2, max_turns=80), num_cycles=1000, verbose_progress=False
    )


def test_max_turns_is_truncation_not_termination() -> None:
    environment = raw_env(num_players=2, max_turns=1)
    environment.reset(seed=0)
    obs, *_ = environment.last()
    environment.step(int(np.flatnonzero(obs["action_mask"])[0]))
    assert all(environment.truncations.values())
    assert not any(environment.terminations.values())
    assert environment.game.end_reason == "max_turns_truncation"
