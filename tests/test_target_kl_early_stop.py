import numpy as np
import torch

from splendor_rl.config import PPOConfig
from splendor_rl.models import PrivilegedCritic, SharedActor
from splendor_rl.ppo import ppo_update
from splendor_rl.rollout import PlayerTransition


def test_epoch_kl_stats_are_reported_and_finite():
    actor = SharedActor(2, 2, [4])
    critic = PrivilegedCritic(2, [4])
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-3)
    items = []
    for i in range(4):
        obs = np.ones(2, dtype="float32") * i
        mask = np.ones(2, bool)
        with torch.no_grad():
            logits = actor(torch.tensor(obs).unsqueeze(0))
            action = i % 2
            log = torch.log_softmax(logits, -1)[0, action].item()
            value = critic(torch.tensor(obs).unsqueeze(0)).item()
        items.append(
            PlayerTransition(
                obs,
                obs,
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
    cfg = PPOConfig(
        hidden_sizes=[4],
        update_epochs=2,
        minibatch_size=2,
        target_kl=1e-12,
        target_kl_mode="mean_epoch",
    )
    stats = ppo_update(
        actor,
        critic,
        opt,
        items,
        np.ones(4, dtype="float32"),
        np.ones(4, dtype="float32"),
        cfg,
    )
    assert (
        1 <= stats["ppo_completed_epochs"] <= 2
        and np.isfinite(stats["approx_kl_mean"])
        and np.isfinite(stats["approx_kl_max"])
    )
