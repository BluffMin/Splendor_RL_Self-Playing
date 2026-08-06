from __future__ import annotations

import numpy as np
import pytest

pettingzoo = pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from pettingzoo.test import api_test

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
