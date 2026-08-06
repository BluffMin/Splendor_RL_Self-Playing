from __future__ import annotations

import numpy as np
import torch

from .distributions import MaskedCategorical


def ppo_update(actor, critic, optimizer, transitions, advantages, returns, config):
    device = next(actor.parameters()).device
    n = len(transitions)
    obs = torch.as_tensor(np.stack([t.actor_obs for t in transitions]), device=device)
    states = torch.as_tensor(
        np.stack([t.critic_state for t in transitions]), device=device
    )
    masks = torch.as_tensor(
        np.stack([t.action_mask for t in transitions]), device=device
    )
    actions = torch.as_tensor([t.action for t in transitions], device=device)
    old_log = torch.as_tensor([t.log_prob for t in transitions], device=device)
    old_values = torch.as_tensor([t.value for t in transitions], device=device)
    adv = torch.as_tensor(advantages, device=device)
    targets = torch.as_tensor(returns, device=device)
    if config.normalize_advantages:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    stats = []
    indices = np.arange(n)
    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, config.minibatch_size):
            ix = torch.as_tensor(
                indices[start : start + config.minibatch_size], device=device
            )
            dist = MaskedCategorical(actor(obs[ix]), masks[ix])
            new_log = dist.log_prob(actions[ix])
            values = critic(states[ix])
            ratio = (new_log - old_log[ix]).exp()
            policy_loss = -torch.minimum(
                ratio * adv[ix],
                ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef) * adv[ix],
            ).mean()
            if config.clip_value_loss:
                clipped = old_values[ix] + (values - old_values[ix]).clamp(
                    -config.clip_coef, config.clip_coef
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        (values - targets[ix]).square(),
                        (clipped - targets[ix]).square(),
                    ).mean()
                )
            else:
                value_loss = 0.5 * (values - targets[ix]).square().mean()
            entropy = dist.entropy().mean()
            entropy_coef = getattr(
                config, "current_entropy_coef", config.entropy_coef_start
            )
            loss = policy_loss + config.value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                config.max_grad_norm,
            )
            optimizer.step()
            kl = ((ratio - 1) - (new_log - old_log[ix])).mean()
            clip = ((ratio - 1).abs() > config.clip_coef).float().mean()
            stats.append(
                [
                    policy_loss.item(),
                    value_loss.item(),
                    entropy.item(),
                    kl.item(),
                    clip.item(),
                    float(grad),
                ]
            )
        if stats[-1][3] > config.target_kl:
            break
    mean = np.asarray(stats).mean(0)
    variance = np.var(returns)
    explained = (
        1 - np.var(returns - np.asarray([t.value for t in transitions])) / variance
        if variance > 1e-8
        else 0
    )
    return dict(
        zip(
            (
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "gradient_norm",
            ),
            mean,
            strict=True,
        )
    ) | {"explained_variance": float(explained)}
