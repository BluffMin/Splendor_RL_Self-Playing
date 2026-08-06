from __future__ import annotations

import numpy as np


def variable_discount_gae(rewards, values, next_values, discounts, gae_lambda: float):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    discounts = np.asarray(discounts, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    following = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + discounts[index] * next_values[index] - values[index]
        following = delta + discounts[index] * gae_lambda * following
        advantages[index] = following
    return advantages, advantages + values


def normalize_advantages(values, epsilon: float = 1e-8):
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + epsilon)
