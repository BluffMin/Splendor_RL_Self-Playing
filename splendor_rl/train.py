from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from splendor_env.wrappers import SelfPlayWrapper

from .checkpoint import load_checkpoint, save_checkpoint
from .config import PPOConfig
from .metrics import JsonlMetrics
from .models import PrivilegedCritic, SharedActor
from .ppo import ppo_update
from .rollout import RolloutCollector


def train(config: PPOConfig, run_dir, resume=None):
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config_resolved.yaml").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(config.device)
    sizes = {
        "actor": SelfPlayWrapper.actor_observation_size,
        "critic": SelfPlayWrapper.critic_state_size,
        "action": SelfPlayWrapper.action_size,
    }
    actor = SharedActor(sizes["actor"], sizes["action"], config.hidden_sizes).to(device)
    critic = PrivilegedCritic(sizes["critic"], config.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.learning_rate,
        eps=1e-5,
    )
    transitions = updates = 0
    if resume:
        data = load_checkpoint(
            resume, actor, critic, optimizer, sizes, map_location=device
        )
        transitions = data["global_transition_count"]
        updates = data["update_index"]
    collector = RolloutCollector(
        actor,
        critic,
        num_envs=config.num_envs,
        num_players=config.num_players,
        seed=config.seed,
        gamma=config.gamma,
        payment_mode=config.payment_mode,
        device=device,
        max_turns=config.max_turns,
    )
    metrics = JsonlMetrics(run / "metrics" / "training.jsonl")
    while transitions < config.total_transitions:
        target = min(
            config.transitions_per_update, config.total_transitions - transitions
        )
        batch, adv, ret, roll = collector.collect(target, config.gae_lambda)
        progress = min(
            transitions
            / max(1, config.total_transitions * config.entropy_anneal_fraction),
            1,
        )
        config.current_entropy_coef = config.entropy_coef_start + progress * (
            config.entropy_coef_end - config.entropy_coef_start
        )
        start = time.perf_counter()
        update = ppo_update(actor, critic, optimizer, batch, adv, ret, config)
        update_seconds = time.perf_counter() - start
        transitions += len(batch)
        updates += 1
        recent = collector.episodes[-100:]
        row = {
            "global_transitions": transitions,
            "update": updates,
            **roll,
            **update,
            "update_seconds": update_seconds,
            "rollout_to_update_ratio": roll["rollout_seconds"]
            / max(update_seconds, 1e-9),
            "entropy_coef": config.current_entropy_coef,
            "episodes": len(collector.episodes),
            "truncation_rate": float(np.mean([e["truncated"] for e in recent]))
            if recent
            else 0,
            "mean_episode_turns": float(np.mean([e["turns"] for e in recent]))
            if recent
            else 0,
            "mean_episode_decisions": float(np.mean([e["decisions"] for e in recent]))
            if recent
            else 0,
            "average_final_score": float(
                np.mean([score for e in recent for score in e["scores"]])
            )
            if recent
            else 0,
        }
        metrics.write(row)
        print(json.dumps(row))
        save_checkpoint(
            run / "checkpoints" / "latest.pt",
            actor,
            critic,
            optimizer,
            config,
            transitions,
            updates,
            sizes,
        )
    return actor, critic, collector
