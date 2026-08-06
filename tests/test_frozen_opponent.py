import numpy as np
from league_helpers import models

from splendor_env.wrappers import SelfPlayWrapper
from splendor_rl.league.pool import FrozenOpponent
from splendor_rl.league.types import OpponentMetadata


def test_frozen_opponent_is_actor_only_eval_and_legal():
    actor, _ = models()
    metadata = OpponentMetadata("x", "champion", 0, 0, 1, 475, 373, 2, "x.pt", "hash")
    opponent = FrozenOpponent(actor, metadata, "cpu")
    env = SelfPlayWrapper(2, seed=3)
    mask = env.action_mask()
    first = opponent.act(env.actor_observation(0), mask, deterministic=True)
    second = opponent.act(env.actor_observation(0), mask, deterministic=True)
    assert first == second and mask[first]
    assert not opponent.actor.training
    assert all(not parameter.requires_grad for parameter in opponent.actor.parameters())
    assert not hasattr(opponent, "critic")
    assert np.array_equal(
        env.actor_observation(0), env.game.observation(0, omniscient=False)
    )
