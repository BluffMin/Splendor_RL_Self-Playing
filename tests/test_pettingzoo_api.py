from __future__ import annotations

import numpy as np
import pytest

pettingzoo = pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from pettingzoo.test import api_test

from splendor_env.actions import N_ACTIONS, action_id
from splendor_env.core import OBSERVATION_SIZE
from splendor_env.pettingzoo_env import raw_env


def test_api() -> None:
    environment = raw_env(num_players=2, max_turns=80, allow_deadlock_pass=True)
    api_test(environment, num_cycles=1000, verbose_progress=False)


def test_masked_rollout() -> None:
    environment = raw_env(num_players=2, max_turns=100, allow_deadlock_pass=True)
    environment.reset(seed=0)
    rng = np.random.default_rng(0)
    for agent in environment.agent_iter(max_iter=3000):
        obs, _, terminated, truncated, _ = environment.last()
        if terminated or truncated:
            action = None
        else:
            legal = np.flatnonzero(obs["action_mask"])
            assert legal.size > 0
            action = int(rng.choice(legal))
        environment.step(action)
    assert not environment.agents


def test_spaces_match_v020_shapes() -> None:
    environment = raw_env(num_players=3)
    environment.reset(seed=1)
    assert OBSERVATION_SIZE == 454
    assert environment.action_space("player_0").n == N_ACTIONS == 324
    for agent in environment.possible_agents:
        assert environment.observe(agent)["observation"].shape == (454,)
    assert environment.state().shape == (454,)


def test_render_omniscient_option() -> None:
    hidden = raw_env(num_players=2, render_mode="ansi")
    hidden.reset(seed=2)
    hidden.step(action_id("reserve_deck", 1))
    assert "[Tier 2 hidden card]" in hidden.game.render(perspective=1)

    debug = raw_env(
        num_players=2,
        render_mode="ansi",
        render_omniscient=True,
    )
    debug.reset(seed=2)
    debug.step(action_id("reserve_deck", 1))
    rendered = debug.render()
    assert rendered is not None
    assert "[deck-private]" in rendered
    assert "[Tier 2 hidden card]" not in rendered
