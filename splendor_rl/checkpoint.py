from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
import torch


def _cpu_byte_rng_state(state):
    """Normalize RNG tensors after a checkpoint-wide CUDA map_location."""
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state)
    return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def save_checkpoint(
    path,
    actor,
    critic,
    optimizer,
    config,
    global_transition_count,
    update_index,
    sizes,
    *,
    best_metrics=None,
):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    current_lr = float(optimizer.param_groups[0]["lr"])
    payload = {
        "schema_version": "0.5.1",
        "engine_version": "0.3.2",
        "rl_version": "0.5.1",
        "num_players": config.num_players,
        "max_players_in_observation": 4,
        "training_mode": getattr(config, "training_mode", "shared_current"),
        "payment_mode": config.payment_mode,
        "global_transition_count": global_transition_count,
        "update_index": update_index,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "schedule_state": {
            "initial_learning_rate": config.learning_rate,
            "current_learning_rate": current_lr,
            "min_learning_rate": config.min_learning_rate,
            "linear_lr_decay": config.linear_lr_decay,
            "current_entropy_coef": config.current_entropy_coef
            if config.current_entropy_coef is not None
            else config.entropy_coef_start,
        },
        "best_metrics": best_metrics or {},
        "observation_sizes": sizes,
        "critic_information_scope": {
            "private_reserved_cards": True,
            "full_deck_order": False,
        },
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
    version = data.get("schema_version")
    if version not in {"0.4.0", "0.4.1", "0.4.2", "0.4.3", "0.5.0", "0.5.1"}:
        raise ValueError("unsupported checkpoint schema")
    config_players = data.get("config", {}).get("num_players")
    top_players = data.get("num_players", config_players)
    if top_players is None:
        raise ValueError("checkpoint does not contain num_players")
    if config_players is not None and top_players != config_players:
        raise ValueError("checkpoint num_players metadata is inconsistent")
    data["num_players"] = top_players
    if expected_sizes and data["observation_sizes"] != expected_sizes:
        raise ValueError("checkpoint observation/action sizes differ")
    if version == "0.4.0":
        warnings.warn(
            "Loaded a v0.4.0 checkpoint. Schedule and best-checkpoint metadata were initialized with v0.4.1 defaults.",
            stacklevel=2,
        )
        cfg = data.get("config", {})
        optimizer_lr = (
            data.get("optimizer_state_dict", {})
            .get("param_groups", [{}])[0]
            .get("lr", cfg.get("learning_rate", 3e-4))
        )
        data["schedule_state"] = {
            "initial_learning_rate": cfg.get("learning_rate", 3e-4),
            "current_learning_rate": optimizer_lr,
            "min_learning_rate": 0.0,
            "linear_lr_decay": cfg.get("linear_lr_decay", True),
            "current_entropy_coef": cfg.get(
                "current_entropy_coef", cfg.get("entropy_coef_start", 0.01)
            ),
        }
        data["best_metrics"] = {}
        data["critic_information_scope"] = {
            "private_reserved_cards": True,
            "full_deck_order": False,
        }
    actor.load_state_dict(data["actor_state_dict"])
    if critic is not None:
        critic.load_state_dict(data["critic_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer_state_dict"])
    if restore_rng:
        random.setstate(data["rng_states"]["python"])
        np.random.set_state(data["rng_states"]["numpy"])
        # map_location=device also maps this CPU RNG tensor to CUDA. PyTorch's
        # CPU generator strictly requires a CPU ByteTensor.
        torch.set_rng_state(_cpu_byte_rng_state(data["rng_states"]["torch_cpu"]))
        if torch.cuda.is_available() and data["rng_states"]["torch_cuda"]:
            torch.cuda.set_rng_state_all(
                [
                    _cpu_byte_rng_state(state)
                    for state in data["rng_states"]["torch_cuda"]
                ]
            )
    return data
