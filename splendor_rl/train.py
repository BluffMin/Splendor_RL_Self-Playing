from __future__ import annotations

import json
import os
import random
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from splendor_env.wrappers import SelfPlayWrapper

from .checkpoint import load_checkpoint, save_checkpoint
from .config import PPOConfig
from .evaluation import evaluate_ladder
from .metrics import JsonlMetrics
from .models import PrivilegedCritic, SharedActor
from .orchestration import load_best_state, update_best_checkpoints
from .ppo import ppo_update
from .progress import ProgressConfig, make_training_progress
from .rollout import RolloutCollector
from .schedules import (
    apply_learning_rate,
    entropy_coefficient,
    linear_learning_rate,
    next_interval_threshold,
)

SCHEDULE_FIELDS = (
    "learning_rate",
    "min_learning_rate",
    "linear_lr_decay",
    "total_transitions",
    "entropy_coef_start",
    "entropy_coef_end",
    "entropy_anneal_fraction",
)


def train(
    config: PPOConfig,
    run_dir,
    resume=None,
    *,
    allow_schedule_override=False,
    progress_config: ProgressConfig | None = None,
    stop_at_transitions: int | None = None,
):
    config.validate()
    progress_config = progress_config or ProgressConfig()
    started = datetime.now(timezone.utc).isoformat()
    run = Path(run_dir)
    checkpoints = run / "checkpoints"
    evaluations = run / "evaluations"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (run / "config_resolved.yaml").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
        [*actor.parameters(), *critic.parameters()], lr=config.learning_rate, eps=1e-5
    )
    transitions = updates = 0
    checkpoint_best = {}
    if resume:
        data = load_checkpoint(
            resume, actor, critic, optimizer, sizes, map_location=device
        )
        if data["num_players"] != config.num_players:
            raise ValueError(
                f"checkpoint num_players={data['num_players']} does not match config num_players={config.num_players}"
            )
        transitions = data["global_transition_count"]
        updates = data["update_index"]
        checkpoint_best = data.get("best_metrics", {})
        differences = [
            name
            for name in SCHEDULE_FIELDS
            if name in data.get("config", {})
            and data["config"][name] != getattr(config, name)
        ]
        if differences and not allow_schedule_override:
            raise ValueError(
                f"resume schedule config mismatch: {', '.join(differences)}"
            )
    if stop_at_transitions is not None:
        if stop_at_transitions > config.total_transitions:
            raise ValueError("stop_at_transitions must not exceed total_transitions")
        if stop_at_transitions <= transitions:
            raise ValueError(
                f"Checkpoint already contains {transitions:,} transitions, which is not below stop target {stop_at_transitions:,}."
            )
    run_target = stop_at_transitions or config.total_transitions
    (run / "invocation_metadata.json").write_text(
        json.dumps(
            {
                "progress_mode": progress_config.mode.value,
                "progress_refresh_seconds": progress_config.refresh_seconds,
                "stop_at_transitions": stop_at_transitions,
                "schedule_total_transitions": config.total_transitions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    progress = make_training_progress(
        run_target, transitions, config.total_transitions, progress_config
    )
    best = load_best_state(checkpoints, checkpoint_best)
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
    config.current_entropy_coef = entropy_coefficient(
        config.entropy_coef_start,
        config.entropy_coef_end,
        config.entropy_anneal_fraction,
        transitions,
        config.total_transitions,
    )
    current_lr = linear_learning_rate(
        config.learning_rate,
        config.min_learning_rate,
        transitions,
        config.total_transitions,
        config.linear_lr_decay,
    )
    apply_learning_rate(optimizer, current_lr)
    initial_path = checkpoints / "step_000000000.pt"
    if transitions == 0 and not initial_path.exists():
        save_checkpoint(
            initial_path,
            actor,
            critic,
            optimizer,
            config,
            0,
            0,
            sizes,
            best_metrics=best,
        )
        shutil.copy2(initial_path, checkpoints / "latest.pt")

    def save_at(count):
        path = checkpoints / f"step_{count:09d}.pt"
        progress.status(f"Saving {path.name}")
        save_checkpoint(
            path,
            actor,
            critic,
            optimizer,
            config,
            transitions,
            updates,
            sizes,
            best_metrics=best,
        )
        shutil.copy2(path, checkpoints / "latest.pt")
        progress.write(f"Checkpoint saved: {path}")
        return path

    def evaluate_at(count, checkpoint_path):
        target = evaluations / f"step_{count:09d}"
        progress.status(f"Evaluating step {count:,}")
        try:
            progress.pause()
            try:
                summary = evaluate_ladder(
                    actor,
                    output_dir=target,
                    games_per_matchup=config.evaluation_games_per_matchup,
                    evaluation_seed_base=config.evaluation_seed_base,
                    device=device,
                    num_players=config.num_players,
                    deterministic=config.evaluation_deterministic,
                    checkpoint_path=str(checkpoint_path),
                    transition_count=count,
                    progress_config=progress_config,
                )
            finally:
                progress.resume()
            updates_best = update_best_checkpoints(
                checkpoint_path, summary, checkpoints, best
            )
            for name in updates_best:
                metric = summary["aggregate"].get("average_rank")
                progress.write(f"New {name}: {metric:.3g} at step {count:,}")
            return target, updates_best
        except Exception as exc:
            target.mkdir(parents=True, exist_ok=True)
            (target / "evaluation_failed.json").write_text(
                json.dumps(
                    {"error": str(exc), "traceback": traceback.format_exc()}, indent=2
                ),
                encoding="utf-8",
            )
            if not config.continue_on_evaluation_error:
                raise
            return target, []

    if (
        transitions == 0
        and config.evaluate_initial_policy
        and not (evaluations / "step_000000000" / "summary.json").exists()
    ):
        evaluate_at(0, initial_path)
    next_checkpoint = next_interval_threshold(transitions, config.checkpoint_interval)
    next_evaluation = next_interval_threshold(transitions, config.evaluation_interval)
    try:
        while transitions < run_target:
            target = min(
                config.transitions_per_update, config.total_transitions - transitions
            )
            progress.status(f"Collecting rollout (update {updates + 1})")
            batch, adv, ret, roll = collector.collect(target, config.gae_lambda)
            current_lr = linear_learning_rate(
                config.learning_rate,
                config.min_learning_rate,
                transitions,
                config.total_transitions,
                config.linear_lr_decay,
            )
            apply_learning_rate(optimizer, current_lr)
            config.current_entropy_coef = entropy_coefficient(
                config.entropy_coef_start,
                config.entropy_coef_end,
                config.entropy_anneal_fraction,
                transitions,
                config.total_transitions,
            )
            progress.status(f"PPO update {updates + 1}")
            start = time.perf_counter()
            update = ppo_update(actor, critic, optimizer, batch, adv, ret, config)
            update_seconds = time.perf_counter() - start
            collected = len(batch)
            transitions += collected
            updates += 1
            numbered_path = None
            if next_checkpoint is not None and transitions >= next_checkpoint:
                numbered_path = save_at(transitions)
                while next_checkpoint is not None and transitions >= next_checkpoint:
                    next_checkpoint += config.checkpoint_interval
            latest = checkpoints / "latest.pt"
            progress.status(f"Saving {latest.name}")
            save_checkpoint(
                latest,
                actor,
                critic,
                optimizer,
                config,
                transitions,
                updates,
                sizes,
                best_metrics=best,
            )
            evaluation_path = None
            best_updates = []
            should_stop = transitions >= run_target
            evaluation_due = (
                next_evaluation is not None and transitions >= next_evaluation
            )
            if evaluation_due or (stop_at_transitions is not None and should_stop):
                evaluation_checkpoint = numbered_path or save_at(transitions)
                evaluation_path, best_updates = evaluate_at(
                    transitions, evaluation_checkpoint
                )
                while next_evaluation is not None and transitions >= next_evaluation:
                    next_evaluation += config.evaluation_interval
                # Persist best-metric metadata in latest after evaluation.
                save_checkpoint(
                    latest,
                    actor,
                    critic,
                    optimizer,
                    config,
                    transitions,
                    updates,
                    sizes,
                    best_metrics=best,
                )
            recent = collector.episodes[-100:]
            row = {
                "global_transition_count": transitions,
                "update_index": updates,
                **roll,
                **update,
                "update_seconds": update_seconds,
                "rollout_to_update_ratio": roll["rollout_seconds"]
                / max(update_seconds, 1e-9),
                "learning_rate": current_lr,
                "entropy_coef": config.current_entropy_coef,
                "checkpoint_saved": numbered_path is not None,
                "checkpoint_path": str(numbered_path) if numbered_path else None,
                "evaluation_triggered": evaluation_path is not None,
                "evaluation_path": str(evaluation_path) if evaluation_path else None,
                "best_checkpoint_updates": best_updates,
                "episodes": len(collector.episodes),
                "truncation_rate": float(np.mean([e["truncated"] for e in recent]))
                if recent
                else 0,
            }
            metrics.write(row)
            progress.update_training(
                collected,
                transitions=transitions,
                update_index=updates,
                episodes=len(collector.episodes),
                metrics={
                    "tr/s": f"{roll['decisions_per_second']:.0f}",
                    "lr": f"{current_lr:.2e}",
                    "ent": f"{config.current_entropy_coef:.2e}",
                    "kl": f"{update['approx_kl_mean']:.2e}",
                    "illegal": collector.illegal_actions,
                    "inv": collector.invariant_violations,
                },
            )
            progress.write(json.dumps(row))
    except BaseException:
        progress.close()
        raise
    final_lr = linear_learning_rate(
        config.learning_rate,
        config.min_learning_rate,
        transitions,
        config.total_transitions,
        config.linear_lr_decay,
    )
    apply_learning_rate(optimizer, final_lr)
    config.current_entropy_coef = entropy_coefficient(
        config.entropy_coef_start,
        config.entropy_coef_end,
        config.entropy_anneal_fraction,
        transitions,
        config.total_transitions,
    )
    progress.status("Finalizing")
    save_at(transitions)
    summary = {
        "rl_version": "0.5.0",
        "engine_version": "0.3.2",
        "num_players": config.num_players,
        "training_mode": "shared_current",
        "payment_mode": config.payment_mode,
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "resolved_config": config.to_dict(),
        "total_transitions": transitions,
        "updates": updates,
        "initial_learning_rate": config.learning_rate,
        "final_learning_rate": final_lr,
        "initial_entropy_coefficient": config.entropy_coef_start,
        "final_entropy_coefficient": config.current_entropy_coef,
        "checkpoints": [p.name for p in sorted(checkpoints.glob("step_*.pt"))],
        "evaluation_steps": [p.name for p in sorted(evaluations.glob("step_*"))],
        "best_checkpoints": best,
        "truncations": sum(e["truncated"] for e in collector.episodes),
        "illegal_actions": collector.illegal_actions,
        "invariant_violations": collector.invariant_violations,
        "critic_information_scope": {
            "private_reserved_cards": True,
            "full_deck_order": False,
        },
    }
    (run / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run / "training_report.md").write_text(
        "# Training report\n\n"
        + "\n".join(
            f"- {k}: {v}" for k, v in summary.items() if k != "best_checkpoints"
        ),
        encoding="utf-8",
    )
    progress.close()
    return actor, critic, collector
