import numpy as np
import torch

from splendor_rl.config import PPOConfig
from splendor_rl.models import PrivilegedCritic, SharedActor
from splendor_rl.ppo import ppo_update
from splendor_rl.rollout import PlayerTransition


def test_synthetic_update_changes_parameters():
    actor = SharedActor(4, 3, [8])
    critic = PrivilegedCritic(5, [8])
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-3)
    config = PPOConfig(hidden_sizes=[8], update_epochs=1, minibatch_size=4)
    transitions = []
    for i in range(8):
        obs = np.random.rand(4).astype("float32")
        state = np.random.rand(5).astype("float32")
        mask = np.ones(3, bool)
        with torch.no_grad():
            logits = actor(torch.tensor(obs).unsqueeze(0))
            action = i % 3
            log = torch.log_softmax(logits, -1)[0, action].item()
            value = critic(torch.tensor(state).unsqueeze(0)).item()
        transitions.append(
            PlayerTransition(
                obs,
                state,
                mask,
                action,
                log,
                value,
                0,
                0.9,
                False,
                False,
                0,
                0,
                i,
                i,
                0,
                "normal",
            )
        )
    before = next(actor.parameters()).detach().clone()
    stats = ppo_update(
        actor,
        critic,
        optimizer,
        transitions,
        np.ones(8, dtype="float32"),
        np.ones(8, dtype="float32"),
        config,
    )
    numeric = [v for v in stats.values() if isinstance(v, (int, float))]
    assert not torch.equal(before, next(actor.parameters())) and all(
        np.isfinite(v) for v in numeric
    )
