from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(
    path, actor, critic, optimizer, config, global_transition_count, update_index, sizes
):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "0.4.0",
        "engine_version": "0.3.2",
        "global_transition_count": global_transition_count,
        "update_index": update_index,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "observation_sizes": sizes,
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        },
    }
    torch.save(payload, target)
    return target


def load_checkpoint(
    path,
    actor,
    critic=None,
    optimizer=None,
    expected_sizes=None,
    map_location="cpu",
    restore_rng=True,
):
    data = torch.load(path, map_location=map_location, weights_only=False)
    if data.get("schema_version") != "0.4.0":
        raise ValueError("unsupported checkpoint schema")
    if expected_sizes and data["observation_sizes"] != expected_sizes:
        raise ValueError("checkpoint observation/action sizes differ")
    actor.load_state_dict(data["actor_state_dict"])
    if critic is not None:
        critic.load_state_dict(data["critic_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer_state_dict"])
    if restore_rng:
        random.setstate(data["rng_states"]["python"])
        np.random.set_state(data["rng_states"]["numpy"])
        torch.set_rng_state(data["rng_states"]["torch_cpu"])
        if torch.cuda.is_available() and data["rng_states"]["torch_cuda"]:
            torch.cuda.set_rng_state_all(data["rng_states"]["torch_cuda"])
    return data
