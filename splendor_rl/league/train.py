from __future__ import annotations

import copy
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from splendor_env.wrappers import SelfPlayWrapper
from splendor_rl.checkpoint import load_checkpoint, save_checkpoint
from splendor_rl.metrics import JsonlMetrics
from splendor_rl.models import PrivilegedCritic, SharedActor
from splendor_rl.ppo import ppo_update
from splendor_rl.progress import (
    ProgressConfig,
    make_evaluation_progress,
    make_training_progress,
)
from splendor_rl.schedules import (
    apply_learning_rate,
    entropy_coefficient,
    linear_learning_rate,
    next_interval_threshold,
)

from .config import LeagueConfig
from .matrix import build_matchup_matrix
from .pool import FrozenOpponent, OpponentPool, actor_sha256
from .promotion import (
    actor_vs_bot_score,
    anchor_group_regression_decision,
    bootstrap_confidence_interval,
    paired_actor_evaluation,
    promotion_decision,
    regression_decision,
)
from .records import MatchRecords
from .rollout import LeagueRolloutCollector
from .state import atomic_json_write, load_league_state
from .types import OpponentMetadata


def _frozen_actor(actor, opponent_id, transition, device):
    metadata = OpponentMetadata(
        opponent_id,
        "candidate",
        transition,
        None,
        0,
        SelfPlayWrapper.actor_observation_size,
        SelfPlayWrapper.action_size,
        2,
        "",
        actor_sha256(actor),
    )
    return FrozenOpponent(copy.deepcopy(actor), metadata, device)


def _save_state(
    path,
    *,
    transitions,
    updates,
    champion_version,
    champion_id,
    pool,
    collector,
    thresholds,
):
    atomic_json_write(
        path,
        {
            "schema_version": "0.5.1",
            "num_players": 2,
            "candidate": {
                "latest_checkpoint": "candidate/checkpoints/latest.pt",
                "global_transition_count": transitions,
                "update_index": updates,
            },
            "champion": {
                "version": champion_version,
                "opponent_id": champion_id,
                "actor_path": pool.metadata[champion_id].file_name,
                "promoted_at_transition": pool.metadata[champion_id].created_transition,
            },
            "pool": {
                "hall_of_fame_ids": pool.hall_of_fame_ids,
                "recent_ids": pool.recent_ids,
            },
            "thresholds": thresholds,
            "rng_state": collector.rng_state(),
            "episode_indices": collector.episode_indices,
            "candidate_seat_counts": dict(collector.candidate_seat_counts),
            "promotion_attempts": thresholds.get("promotion_attempts", 0),
            "promotion_successes": thresholds.get("promotion_successes", 0),
            "match_records_path": "matchup_records.json",
        },
    )


def _update_current_champion(run, pool, opponent_id):
    metadata = pool.metadata[opponent_id]
    champion_dir = Path(run) / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pool.root / metadata.file_name, champion_dir / "current_actor.pt")
    atomic_json_write(champion_dir / "current_metadata.json", metadata.to_dict())


def _adopt_checkpoint_architecture(config, checkpoint_path, expected_sizes):
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sizes = data.get("observation_sizes")
    if sizes != expected_sizes:
        raise ValueError(
            f"initial checkpoint observation/action sizes {sizes} do not match {expected_sizes}"
        )
    if data.get("num_players", data.get("config", {}).get("num_players")) != 2:
        raise ValueError(
            "league_2p initial checkpoint must contain a two-player policy"
        )
    hidden_sizes = data.get("config", {}).get("hidden_sizes")
    if not hidden_sizes:
        raise ValueError("checkpoint does not contain actor hidden_sizes metadata")
    config.hidden_sizes = list(hidden_sizes)
    return data


def train_league(
    config: LeagueConfig,
    run_dir,
    *,
    initial_checkpoint=None,
    bootstrap_manifest=None,
    resume=None,
    stop_at_transitions=None,
    progress_config=None,
):
    config.validate()
    progress_config = progress_config or ProgressConfig()
    run = Path(run_dir)
    candidate_checkpoints = run / "candidate" / "checkpoints"
    pool_root = run / "opponent_pool"
    promotion_root = run / "promotion_evaluations"
    matrix_root = run / "league_evaluations"
    candidate_checkpoints.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    manifest = None
    bootstrap_checkpoint = None
    if bootstrap_manifest:
        manifest = json.loads(Path(bootstrap_manifest).read_text(encoding="utf-8"))
        if manifest.get("num_players") != 2 or not manifest.get("policies"):
            raise ValueError("bootstrap manifest must contain two-player policies")
        selected_id = manifest.get("selected_champion_id")
        selected = next(
            (
                item
                for item in manifest["policies"]
                if item["candidate_id"] == selected_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("selected bootstrap Champion is missing from manifest")
        bootstrap_checkpoint = selected["checkpoint_path"]
    architecture_data = None
    if resume or initial_checkpoint or bootstrap_checkpoint:
        architecture_data = _adopt_checkpoint_architecture(
            config, resume or initial_checkpoint or bootstrap_checkpoint, sizes
        )
    actor = SharedActor(sizes["actor"], sizes["action"], config.hidden_sizes).to(device)
    critic = PrivilegedCritic(sizes["critic"], config.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(
        [*actor.parameters(), *critic.parameters()], lr=config.learning_rate, eps=1e-5
    )
    transitions = updates = 0
    state_path = run / "league_state.json"
    state = load_league_state(state_path)
    if resume:
        resolved = Path(resume).resolve()
        expected_root = candidate_checkpoints.resolve()
        if expected_root not in resolved.parents:
            raise ValueError(
                "resume checkpoint does not belong to this league run directory"
            )
        data = load_checkpoint(
            resume, actor, critic, optimizer, sizes, map_location=device
        )
        if data.get("training_mode") != "league_2p":
            raise ValueError(
                "resume checkpoint is not a league_2p candidate checkpoint"
            )
        transitions, updates = data["global_transition_count"], data["update_index"]
        if (
            state is None
            or state["candidate"]["global_transition_count"] != transitions
        ):
            raise ValueError(
                "candidate checkpoint and league_state.json are inconsistent"
            )
    elif initial_checkpoint or bootstrap_checkpoint:
        actor.load_state_dict(architecture_data["actor_state_dict"])
        if "critic_state_dict" in architecture_data:
            critic.load_state_dict(architecture_data["critic_state_dict"])
    if stop_at_transitions is not None:
        if stop_at_transitions > config.total_transitions:
            raise ValueError("stop_at_transitions must not exceed total_transitions")
        if stop_at_transitions <= transitions:
            raise ValueError("checkpoint transition count is not below stop target")
    target = stop_at_transitions or config.total_transitions
    (run / "config_resolved.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    pool = OpponentPool(pool_root, config.hidden_sizes, device)
    if not pool.metadata:
        pool.add_snapshot(
            actor,
            opponent_id="champion_0000",
            source_type="champion",
            created_transition=0,
            champion_version=0,
            training_seed=config.seed,
            actor_obs_size=sizes["actor"],
            action_size=sizes["action"],
            bootstrap_champion=True,
            source_checkpoint=str(bootstrap_checkpoint or initial_checkpoint or ""),
        )
        if manifest:
            selected_hash = actor_sha256(actor)
            for item in manifest["policies"]:
                if item["candidate_id"] == manifest["selected_champion_id"]:
                    if item["actor_sha256"] != selected_hash:
                        raise ValueError("bootstrap Champion actor hash mismatch")
                    continue
                data = torch.load(
                    item["checkpoint_path"], map_location=device, weights_only=False
                )
                if data.get("observation_sizes") != sizes:
                    raise ValueError("bootstrap historical observation schema mismatch")
                if (
                    data.get("num_players", data.get("config", {}).get("num_players"))
                    != 2
                ):
                    raise ValueError("bootstrap historical policy must be two-player")
                if list(data["config"]["hidden_sizes"]) != list(config.hidden_sizes):
                    raise ValueError("bootstrap policies must share one architecture")
                historical_actor = SharedActor(
                    sizes["actor"], sizes["action"], config.hidden_sizes
                ).to(device)
                historical_actor.load_state_dict(data["actor_state_dict"])
                if actor_sha256(historical_actor) != item["actor_sha256"]:
                    raise ValueError("bootstrap historical actor hash mismatch")
                pool.add_snapshot(
                    historical_actor,
                    opponent_id=item.get(
                        "opponent_id", f"bootstrap_{item['candidate_id']}"
                    ),
                    source_type="bootstrap_historical",
                    created_transition=item["transition_count"],
                    champion_version=None,
                    training_seed=config.seed,
                    actor_obs_size=sizes["actor"],
                    action_size=sizes["action"],
                    source_checkpoint=item["checkpoint_path"],
                )
    champion_version = state["champion"]["version"] if state else 0
    champion_id = state["champion"]["opponent_id"] if state else "champion_0000"
    _update_current_champion(run, pool, champion_id)
    records = MatchRecords(
        run / "matchup_records.json", config.pfsp_recent_window_games
    )
    collector = LeagueRolloutCollector(
        actor, critic, pool=pool, records=records, config=config, device=device
    )
    collector.champion_id = champion_id
    if state:
        collector.restore_rng_state(state["rng_state"])
        collector.candidate_seat_counts.update(
            {int(k): v for k, v in state.get("candidate_seat_counts", {}).items()}
        )
        collector.episode_indices = [
            int(value) + 1
            for value in state.get("episode_indices", [0] * config.num_envs)
        ]
        collector.assignments.clear()
        collector.envs = [
            SelfPlayWrapper(
                2,
                seed=config.seed + env_id * 100_003 + collector.episode_indices[env_id],
                payment_mode=config.payment_mode,
                max_turns=config.max_turns,
            )
            for env_id in range(config.num_envs)
        ]
        collector.env_assignments = [
            collector._assign(env_id) for env_id in range(config.num_envs)
        ]
    next_snapshot = (
        state["thresholds"]["next_snapshot"]
        if state
        else next_interval_threshold(transitions, config.recent_snapshot_interval)
    )
    next_promotion = (
        state["thresholds"]["next_promotion"]
        if state
        else next_interval_threshold(transitions, config.promotion_interval)
    )
    next_matrix = (
        state["thresholds"]["next_matrix"]
        if state
        else next_interval_threshold(transitions, config.matchup_matrix_interval)
    )
    attempts = state.get("promotion_attempts", 0) if state else 0
    successes = state.get("promotion_successes", 0) if state else 0
    progress = make_training_progress(
        target, transitions, config.total_transitions, progress_config
    )
    metrics = JsonlMetrics(run / "metrics" / "league_training.jsonl")

    def save_candidate(count, numbered=True):
        path = candidate_checkpoints / (
            f"step_{count:09d}.pt" if numbered else "latest.pt"
        )
        save_checkpoint(
            path, actor, critic, optimizer, config, transitions, updates, sizes
        )
        if numbered:
            shutil.copy2(path, candidate_checkpoints / "latest.pt")
        return path

    def promotion_attempt(count):
        nonlocal champion_version, champion_id, attempts, successes
        attempts += 1
        candidate = _frozen_actor(actor, f"candidate_step_{count:09d}", count, device)
        champion = pool.load(champion_id)
        output = promotion_root / f"step_{count:09d}"
        output.mkdir(parents=True, exist_ok=True)
        evaluation_progress = make_evaluation_progress(
            config.promotion_pair_count * 2, count, progress_config
        )
        try:
            paired = paired_actor_evaluation(
                candidate,
                champion,
                pair_count=config.promotion_pair_count,
                seed_base=config.seed + 500_000 + count,
                progress=evaluation_progress,
                replay_dir=output / "replays",
                replay_metadata={
                    "candidate_checkpoint": f"candidate/checkpoints/step_{count:09d}.pt",
                    "candidate_transition": count,
                    "opponent_id": champion_id,
                    "opponent_source_type": "champion",
                    "opponent_created_transition": pool.metadata[
                        champion_id
                    ].created_transition,
                    "champion_version": champion_version,
                    "acting_policy_id": f"candidate_step_{count:09d}",
                    "opponent_policy_id": champion_id,
                },
            )
        finally:
            evaluation_progress.close()
        ci = bootstrap_confidence_interval(
            paired.pop("pair_scores"),
            samples=config.promotion_bootstrap_samples,
            confidence=config.promotion_confidence,
            seed=config.seed + 600_000 + count,
        )
        head_passed, reasons = promotion_decision(
            ci["mean_score"],
            ci["lower_confidence_bound"],
            min_score=config.promotion_min_score,
            min_lower_bound=config.promotion_min_lower_bound,
            identical_hash=actor_sha256(actor) == pool.metadata[champion_id].sha256,
        )
        anchors = {}
        candidate_anchor_scores, champion_anchor_scores = {}, {}
        ordered_bots = (*config.hard_anchors, *config.saturated_anchors)
        for index, bot in enumerate(ordered_bots):
            seed = config.seed + 700_000 + count + index * 10_000
            candidate_result = actor_vs_bot_score(
                candidate,
                bot,
                games=config.promotion_anchor_games_per_opponent,
                seed_base=seed,
            )
            champion_result = actor_vs_bot_score(
                champion,
                bot,
                games=config.promotion_anchor_games_per_opponent,
                seed_base=seed,
            )
            anchors[bot] = {"candidate": candidate_result, "champion": champion_result}
            candidate_anchor_scores[bot], champion_anchor_scores[bot] = (
                candidate_result["score"],
                champion_result["score"],
            )
        anchor_gate = anchor_group_regression_decision(
            candidate_anchor_scores,
            champion_anchor_scores,
            hard_anchors=config.hard_anchors,
            saturated_anchors=config.saturated_anchors,
            max_hard_aggregate=config.promotion_max_hard_anchor_aggregate_regression,
            max_single_hard=config.promotion_max_single_hard_anchor_regression,
            max_saturated=config.promotion_max_saturated_anchor_regression,
        )
        historical = [item for item in pool.historical_ids(champion_id)][
            : config.promotion_historical_anchor_count
        ]
        historical_candidate_scores, historical_champion_scores = {}, {}
        for index, opponent_id in enumerate(historical):
            opponent = pool.load(opponent_id)
            seed = config.seed + 800_000 + count + index * 10_000
            c = paired_actor_evaluation(
                candidate,
                opponent,
                pair_count=max(1, config.promotion_anchor_games_per_opponent // 2),
                seed_base=seed,
            )
            h = paired_actor_evaluation(
                champion,
                opponent,
                pair_count=max(1, config.promotion_anchor_games_per_opponent // 2),
                seed_base=seed,
            )
            historical_candidate_scores[opponent_id] = float(np.mean(c["pair_scores"]))
            historical_champion_scores[opponent_id] = float(np.mean(h["pair_scores"]))
        regression = regression_decision(
            historical_candidate_scores,
            historical_champion_scores,
            max_aggregate=config.max_anchor_aggregate_regression,
            max_single=config.max_single_anchor_regression,
        )
        if not regression["passed"]:
            reasons.append("regression_gate_failed")
        if not anchor_gate["passed"]:
            reasons.extend(anchor_gate["reasons"])
        promoted = head_passed and regression["passed"] and anchor_gate["passed"]
        old_champion = champion_id
        if promoted:
            champion_version += 1
            successes += 1
            champion_id = f"champion_{champion_version:04d}"
            pool.add_snapshot(
                actor,
                opponent_id=champion_id,
                source_type="champion",
                created_transition=count,
                champion_version=champion_version,
                training_seed=config.seed,
                actor_obs_size=sizes["actor"],
                action_size=sizes["action"],
            )
            collector.champion_id = champion_id
            _update_current_champion(run, pool, champion_id)
        result = {
            "schema_version": "0.5.1",
            "candidate_transition": count,
            "champion_version": champion_version,
            "evaluated_champion_id": old_champion,
            "head_to_head": {
                **paired,
                "mean_score": ci["mean_score"],
                "ci_lower": ci["lower_confidence_bound"],
                "ci_upper": ci["upper_confidence_bound"],
            },
            "anchors": anchors,
            "anchor_groups": {
                "hard": list(config.hard_anchors),
                "saturated": list(config.saturated_anchors),
            },
            "hard_anchor_gate": anchor_gate,
            "regression_gate": regression,
            "promoted": promoted,
            "reasons": reasons,
        }
        atomic_json_write(output / "promotion_result.json", result)
        return result

    started = datetime.now(timezone.utc).isoformat()
    try:
        while transitions < target:
            count = min(
                config.transitions_per_update, config.total_transitions - transitions
            )
            collector.current_transition = transitions
            progress.status(f"League collecting (champion v{champion_version})")
            batch, advantages, returns, rollout = collector.collect(
                count, config.gae_lambda
            )
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
            progress.status(f"League PPO update {updates + 1}")
            before = time.perf_counter()
            update = ppo_update(
                actor, critic, optimizer, batch, advantages, returns, config
            )
            update_seconds = time.perf_counter() - before
            transitions += len(batch)
            updates += 1
            save_candidate(transitions, numbered=False)
            snapshot_created = None
            if next_snapshot is not None and transitions >= next_snapshot:
                snapshot_created = f"recent_step_{transitions:09d}"
                pool.add_snapshot(
                    actor,
                    opponent_id=snapshot_created,
                    source_type="recent",
                    created_transition=transitions,
                    champion_version=None,
                    training_seed=config.seed,
                    actor_obs_size=sizes["actor"],
                    action_size=sizes["action"],
                )
                pool.trim_recent(config.max_recent_snapshots)
                while transitions >= next_snapshot:
                    next_snapshot += config.recent_snapshot_interval
            promotion = None
            if next_promotion is not None and transitions >= next_promotion:
                save_candidate(transitions)
                promotion = promotion_attempt(transitions)
                while transitions >= next_promotion:
                    next_promotion += config.promotion_interval
            matrix = None
            if next_matrix is not None and transitions >= next_matrix:
                policies = {
                    f"candidate_{transitions}": _frozen_actor(
                        actor, "candidate", transitions, device
                    ),
                    champion_id: pool.load(champion_id),
                }
                for opponent_id in reversed(pool.hall_of_fame_ids + pool.recent_ids):
                    if opponent_id not in policies:
                        policies[opponent_id] = pool.load(opponent_id)
                    if len(policies) >= config.matchup_matrix_max_policies:
                        break
                matrix = build_matchup_matrix(
                    policies,
                    games_per_pair=config.matchup_matrix_games_per_pair,
                    seed_base=config.seed + 900_000 + transitions,
                    output_dir=matrix_root / f"step_{transitions:09d}",
                )
                while transitions >= next_matrix:
                    next_matrix += config.matchup_matrix_interval
            records.save()
            thresholds = {
                "next_snapshot": next_snapshot,
                "next_promotion": next_promotion,
                "next_matrix": next_matrix,
                "promotion_attempts": attempts,
                "promotion_successes": successes,
            }
            _save_state(
                state_path,
                transitions=transitions,
                updates=updates,
                champion_version=champion_version,
                champion_id=champion_id,
                pool=pool,
                collector=collector,
                thresholds=thresholds,
            )
            row = {
                "global_transition_count": transitions,
                "update_index": updates,
                **rollout,
                **update,
                "update_seconds": update_seconds,
                "learning_rate": current_lr,
                "entropy_coef": config.current_entropy_coef,
                "champion_version": champion_version,
                "champion_id": champion_id,
                "pool_size": len(pool.metadata),
                "hall_of_fame_size": len(pool.hall_of_fame_ids),
                "recent_snapshot_count": len(pool.recent_ids),
                "promotion_attempts": attempts,
                "promotion_successes": successes,
                "recent_snapshot_created": snapshot_created,
                "promotion_result": promotion,
                "matchup_matrix_created": matrix is not None,
            }
            metrics.write(row)
            progress.update_training(
                len(batch),
                transitions=transitions,
                update_index=updates,
                episodes=len(collector.episodes),
                metrics={
                    "champ": champion_version,
                    "pool": len(pool.metadata),
                    "opp": collector.assignments[-1].opponent_id or "current",
                    "mode": collector.assignments[-1].mode,
                    "cscore": f"{records.score(champion_id):.2f}",
                    "promos": successes,
                    "lr": f"{current_lr:.2e}",
                    "kl": f"{update['approx_kl_mean']:.2e}",
                    "illegal": collector.illegal_actions,
                    "inv": collector.invariant_violations,
                },
            )
    finally:
        progress.close()
    save_candidate(transitions)
    rounds = [episode["rounds"] for episode in collector.episodes]
    candidate_games = [
        episode
        for episode in collector.episodes
        if episode.get("candidate_score") is not None
    ]
    win_rounds = [
        episode["rounds"]
        for episode in candidate_games
        if episode["candidate_score"] == 1
    ]
    loss_rounds = [
        episode["rounds"]
        for episode in candidate_games
        if episode["candidate_score"] == 0
    ]
    tie_rounds = [
        episode["rounds"]
        for episode in candidate_games
        if episode["candidate_score"] == 0.5
    ]
    summary = {
        "rl_version": "0.5.1",
        "training_mode": "league_2p",
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "total_transitions": transitions,
        "updates": updates,
        "champion_version": champion_version,
        "champion_id": champion_id,
        "promotion_attempts": attempts,
        "promotion_successes": successes,
        "pool_size": len(pool.metadata),
        "hall_of_fame_size": len(pool.hall_of_fame_ids),
        "recent_snapshot_count": len(pool.recent_ids),
        "average_final_round": float(np.mean(rounds)) if rounds else None,
        "average_rounds_on_candidate_wins": float(np.mean(win_rounds))
        if win_rounds
        else None,
        "average_rounds_on_candidate_losses": float(np.mean(loss_rounds))
        if loss_rounds
        else None,
        "average_rounds_on_ties": float(np.mean(tie_rounds)) if tie_rounds else None,
        "illegal_actions": collector.illegal_actions,
        "invariant_violations": collector.invariant_violations,
        "ppo_finite": True,
        "game_length_is_reward": False,
    }
    atomic_json_write(run / "training_summary.json", summary)
    (run / "training_report.md").write_text(
        "# PPO-based league self-play report\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary.items()),
        encoding="utf-8",
    )
    return actor, critic, collector
