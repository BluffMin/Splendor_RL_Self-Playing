from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from splendor_rl.league.pool import OpponentPool, actor_sha256
from splendor_rl.league.promotion import (
    actor_vs_bot_score,
    anchor_group_regression_decision,
    bootstrap_confidence_interval,
    paired_actor_evaluation,
    promotion_decision,
    regression_decision,
)
from splendor_rl.league.records import MatchRecords
from splendor_rl.metrics import JsonlMetrics
from splendor_rl.ppo import ppo_update
from splendor_rl.profiling import WallProfiler
from splendor_rl.progress import ProgressConfig, make_training_progress
from splendor_rl.schedules import (
    apply_learning_rate,
    entropy_coefficient,
    linear_learning_rate,
)

from .bootstrap import bootstrap_population
from .config import ROLES, PopulationConfig
from .learner import make_learner, reset_learner
from .meta import (
    antisymmetrize_score_matrix,
    detect_cycles,
    empirical_exploitability_proxy,
    farthest_point_selection,
    solve_symmetric_meta_strategy,
)
from .multiprocess_collector import MultiprocessBatchedPopulationCollector
from .rollout import PopulationRolloutCollector, weighted_selector
from .scheduler import DeficitScheduler


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_learner(path, learner, config, bootstrap):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "0.6.1",
            "rl_version": "0.6.1",
            "training_mode": "population_league_2p",
            "role": learner.role,
            "generation": learner.generation,
            "population_transition": bootstrap["population_transition"],
            "learner_transitions": learner.transitions,
            "update_index": learner.updates,
            "source_rl_version": bootstrap["source_rl_version"],
            "source_transition": bootstrap["source_transition"],
            "actor_state_dict": learner.actor.state_dict(),
            "critic_state_dict": learner.critic.state_dict(),
            "optimizer_state_dict": learner.optimizer.state_dict(),
            "config": config.to_dict(),
            "observation_sizes": bootstrap["sizes"],
            "learner_summary": learner.state_summary(),
        },
        path,
    )


def _load_learner(path, config, sizes, device):
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("schema_version") not in {"0.6.0", "0.6.1"} or data.get("observation_sizes") != sizes:
        raise ValueError(f"incompatible population learner checkpoint: {path}")
    learner = make_learner(
        data["role"],
        data["actor_state_dict"],
        data["critic_state_dict"],
        config,
        sizes,
        device,
    )
    learner.optimizer.load_state_dict(data["optimizer_state_dict"])
    for key, value in data["learner_summary"].items():
        if hasattr(learner, key):
            setattr(learner, key, value)
    learner.transitions = int(data["learner_summary"]["learner_transitions"])
    return learner


def _replace_snapshot(pool, actor, opponent_id, source_type, transition, config, sizes):
    if opponent_id in pool.metadata:
        pool.remove(opponent_id)
    return pool.add_snapshot(
        actor,
        opponent_id=opponent_id,
        source_type=source_type,
        created_transition=transition,
        champion_version=None,
        training_seed=config.seed,
        actor_obs_size=sizes["actor"],
        action_size=sizes["action"],
    )


def _add_snapshot_idempotent(
    pool, actor, opponent_id, source_type, transition, config, sizes
):
    """Finish a snapshot write safely after a crash before state commit.

    The pool index can be one orchestration boundary ahead of population_state.json.
    A deterministic resume therefore sees the same ID and actor hash again.
    """
    if opponent_id in pool.metadata:
        expected = actor_sha256(actor)
        existing = pool.metadata[opponent_id]
        if existing.sha256 != expected or existing.created_transition != transition:
            raise ValueError(
                f"snapshot ID collision with different actor/state: {opponent_id}"
            )
        pool.load(opponent_id)  # Also validates the persisted file and checksum.
        return existing
    return pool.add_snapshot(
        actor,
        opponent_id=opponent_id,
        source_type=source_type,
        created_transition=transition,
        champion_version=None,
        training_seed=config.seed,
        actor_obs_size=sizes["actor"],
        action_size=sizes["action"],
    )


def _reconcile_pool_to_committed_state(pool, learners, transition, config, sizes):
    """Roll back pool writes that happened after the last atomic state commit."""
    removed = []
    for opponent_id, metadata in list(pool.metadata.items()):
        if metadata.created_transition <= transition:
            continue
        population_owned = opponent_id.startswith(("recent_main_", "active_")) or (
            "_exploiter_" in opponent_id
        )
        if population_owned:
            path = pool.root / metadata.file_name
            if path.exists():
                path.unlink()
            pool.metadata.pop(opponent_id)
            pool.loaded.pop(opponent_id, None)
            removed.append(opponent_id)
    if removed:
        pool._save_index()
    _replace_snapshot(
        pool,
        learners["main"].actor,
        "current_main_snapshot",
        "recent_main",
        transition,
        config,
        sizes,
    )
    for role in ROLES[1:]:
        _replace_snapshot(
            pool,
            learners[role].actor,
            f"active_{role}",
            "active_exploiter",
            transition,
            config,
            sizes,
        )
    return removed


def _scores_for(ids, records):
    scores = np.asarray([records.score(item) for item in ids], dtype=float)
    weights = 0.05 + 4 * scores * (1 - scores)
    return weights / weights.sum() if len(weights) else weights


def _selector(role, config, pool, records, champion_id, meta):
    historical = [
        item
        for item in pool.historical_ids(champion_id)
        if not item.startswith("active_")
    ]
    main_exploiters = [
        item
        for item in pool.metadata
        if item.startswith(("active_main_exploiter", "main_exploiter_"))
    ]
    league_exploiters = [
        item
        for item in pool.metadata
        if item.startswith(("active_league_exploiter", "league_exploiter_"))
    ]
    current_main = (
        ["current_main_snapshot"]
        if "current_main_snapshot" in pool.metadata
        else [champion_id]
    )
    if role == "main":
        mix = config.main_training
        categories = [
            ("current_selfplay", current_main, mix["current_selfplay"], None),
            ("champion", [champion_id], mix["champion"], None),
            (
                "historical_pfsp",
                historical,
                mix["historical_pfsp"],
                _scores_for(historical, records),
            ),
            (
                "meta_strategy",
                meta["policy_ids"],
                mix["meta_strategy"],
                meta["probabilities"],
            ),
            ("main_exploiter", main_exploiters, mix["main_exploiter"], None),
            ("league_exploiter", league_exploiters, mix["league_exploiter"], None),
        ]
    elif role.startswith("main_exploiter"):
        mix = config.main_exploiter
        hof = [item for item in pool.hall_of_fame_ids if item != champion_id]
        categories = [
            ("current_champion", [champion_id], mix["current_champion"], None),
            ("current_main", current_main, mix["current_main"], None),
            ("hall_of_fame", hof, mix["hall_of_fame"], None),
        ]
    else:
        mix = config.league_exploiter
        categories = [
            (
                "meta_strategy",
                meta["policy_ids"],
                mix["meta_strategy"],
                meta["probabilities"],
            ),
            (
                "historical_pfsp",
                historical,
                mix["historical_pfsp"],
                _scores_for(historical, records),
            ),
            ("current_champion", [champion_id], mix["current_champion"], None),
        ]
    return weighted_selector(categories, fallback_id=champion_id)


def _population_ids(pool, champion_id, maximum):
    required = [
        champion_id,
        *pool.hall_of_fame_ids,
        *[key for key in pool.metadata if key.startswith("active_")],
    ]
    ids = list(dict.fromkeys([*required, *pool.recent_ids]))
    if len(ids) <= maximum:
        return ids
    # Unknown fingerprints are evaluated below; transition diversity is the deterministic initial proxy.
    transitions = np.asarray(
        [
            [
                pool.metadata[a].created_transition
                / max(1, pool.metadata[b].created_transition or 1)
                for b in ids
            ]
            for a in ids
        ]
    )
    return farthest_point_selection(
        ids,
        transitions,
        required,
        maximum,
        {key: pool.metadata[key].sha256 for key in ids},
    )


def _meta_update(run, pool, champion_id, config, transition, device):
    ids = _population_ids(pool, champion_id, config.meta_max_policies)
    actors = {key: pool.load(key) for key in ids}
    hashes = {key: pool.metadata[key].sha256 for key in ids}
    matrix = np.full((len(ids), len(ids)), 0.5)
    games = np.zeros_like(matrix, dtype=int)
    errors = np.zeros_like(matrix)
    wins = np.zeros_like(matrix, dtype=int)
    ties = np.zeros_like(matrix, dtype=int)
    losses = np.zeros_like(matrix, dtype=int)
    score_sums = np.zeros_like(matrix)
    last_updated = np.zeros_like(matrix, dtype=int)
    cache = _load_meta_pair_cache(run, pool, replay_transition=transition)
    reused_pairs = evaluated_pairs = 0
    for row in range(len(ids)):
        for col in range(row + 1, len(ids)):
            cache_key = tuple(sorted((hashes[ids[row]], hashes[ids[col]])))
            cached = cache.get(cache_key)
            refresh_due = cached is not None and (
                transition - cached["last_updated_transition"]
                >= config.meta_refresh_interval
            )
            target_games = max(
                config.meta_min_games_per_pair,
                config.meta_matchup_games_per_pair,
            )
            if cached is not None and cached["games"] >= target_games and not refresh_due:
                score = cached["score"]
                cached_wins, cached_losses = cached["wins"], cached["losses"]
                if hashes[ids[row]] != cache_key[0]:
                    score = 1.0 - score
                    cached_wins, cached_losses = cached_losses, cached_wins
                matrix[row, col], matrix[col, row] = score, 1.0 - score
                games[row, col] = games[col, row] = cached["games"]
                errors[row, col] = errors[col, row] = cached["standard_error"]
                wins[row, col], losses[row, col] = cached_wins, cached_losses
                wins[col, row], losses[col, row] = cached_losses, cached_wins
                ties[row, col] = ties[col, row] = cached["ties"]
                score_sums[row, col] = score * cached["games"]
                score_sums[col, row] = (1.0 - score) * cached["games"]
                last_updated[row, col] = last_updated[col, row] = cached[
                    "last_updated_transition"
                ]
                reused_pairs += 1
                continue
            existing_games = cached["games"] if cached else 0
            requested_games = (
                config.meta_refresh_games_per_pair
                if refresh_due and existing_games >= target_games
                else target_games - existing_games
            )
            evaluation_pairs = max(1, (requested_games + 1) // 2)
            result = paired_actor_evaluation(
                actors[ids[row]],
                actors[ids[col]],
                pair_count=evaluation_pairs,
                seed_base=config.seed
                + 30_000_000
                + transition
                + row * 10_000
                + col * 100,
            )
            values = np.asarray(result["pair_scores"])
            new_games = int(result["games"])
            row_new_score = float(values.mean())
            if cached:
                cached_score = cached["score"]
                if hashes[ids[row]] != cache_key[0]:
                    cached_score = 1.0 - cached_score
                score = (
                    cached_score * existing_games + row_new_score * new_games
                ) / (existing_games + new_games)
            else:
                score = row_new_score
            matrix[row, col] = score
            matrix[col, row] = 1 - score
            total_games = existing_games + new_games
            games[row, col] = games[col, row] = total_games
            errors[row, col] = errors[col, row] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.5
            )
            raw_scores = result["raw_game_scores"]
            new_wins = sum(value == 1.0 for value in raw_scores)
            new_ties = sum(value == 0.5 for value in raw_scores)
            new_losses = sum(value == 0.0 for value in raw_scores)
            old_wins = old_ties = old_losses = 0
            if cached:
                old_wins, old_losses = cached["wins"], cached["losses"]
                old_ties = cached["ties"]
                if hashes[ids[row]] != cache_key[0]:
                    old_wins, old_losses = old_losses, old_wins
            wins[row, col], ties[row, col], losses[row, col] = (
                old_wins + new_wins,
                old_ties + new_ties,
                old_losses + new_losses,
            )
            wins[col, row], ties[col, row], losses[col, row] = (
                losses[row, col], ties[row, col], wins[row, col]
            )
            score_sums[row, col] = score * total_games
            score_sums[col, row] = (1.0 - score) * total_games
            last_updated[row, col] = last_updated[col, row] = transition
            evaluated_pairs += 1
    solver_matrix = antisymmetrize_score_matrix(matrix)
    solved = solve_symmetric_meta_strategy(
        solver_matrix,
        iterations=config.meta_solver_iterations,
        seed=config.seed + transition,
    )
    cycles = detect_cycles(matrix, ids, config.cycle_score_threshold)
    output = run / "meta" / f"step_{transition:09d}"
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("raw_matrix.csv", matrix),
        ("solver_matrix.csv", solver_matrix),
    ):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["policy", *ids])
            for key, row in zip(ids, values, strict=True):
                writer.writerow([key, *row])
    payload = {
        "schema_version": "0.6.1",
        "population_transition": transition,
        "policy_ids": ids,
        "policy_hashes": hashes,
        "raw_score_matrix": matrix.tolist(),
        "solver_payoff_matrix": solver_matrix.tolist(),
        "games": games.tolist(),
        "standard_errors": errors.tolist(),
        "wins": wins.tolist(),
        "ties": ties.tolist(),
        "losses": losses.tolist(),
        "score_sums": score_sums.tolist(),
        "last_updated_transitions": last_updated.tolist(),
        "reused_pairs": reused_pairs,
        "evaluated_pairs": evaluated_pairs,
        **solved.to_dict(),
        "non_transitive_cycles": cycles,
    }
    _atomic_json(output / "meta_strategy.json", payload)
    (output / "report.md").write_text(
        "# PSRO-lite Meta Strategy\n\n"
        + "\n".join(
            f"- {key}: {prob:.6f}"
            for key, prob in zip(ids, solved.probabilities, strict=True)
        )
        + f"\n\nEstimated value: {solved.estimated_value}\nConvergence gap: {solved.convergence_gap}\nCycles: {cycles}\n",
        encoding="utf-8",
    )
    return {
        "policy_ids": ids,
        "probabilities": solved.probabilities.tolist(),
        "estimated_value": solved.estimated_value,
        "convergence_gap": solved.convergence_gap,
        "step": transition,
        "cycles": cycles,
    }


def _load_meta_pair_cache(run, pool, replay_transition=None):
    cache = {}
    for path in sorted((run / "meta").glob("step_*/meta_strategy.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            ids = payload["policy_ids"]
            hashes = payload.get("policy_hashes", {})
            same_replayed_boundary = (
                replay_transition is not None
                and int(payload.get("population_transition", -1))
                == replay_transition
            )
            for row in range(len(ids)):
                for col in range(row + 1, len(ids)):
                    left, right = ids[row], ids[col]
                    left_hash = hashes.get(left)
                    right_hash = hashes.get(right)
                    # Immutable snapshot IDs are stable. Active IDs are safe only
                    # when deterministically replaying the exact committed boundary.
                    if (
                        not left_hash
                        and (same_replayed_boundary or not left.startswith("active_"))
                        and left in pool.metadata
                    ):
                        left_hash = pool.metadata[left].sha256
                    if (
                        not right_hash
                        and (same_replayed_boundary or not right.startswith("active_"))
                        and right in pool.metadata
                    ):
                        right_hash = pool.metadata[right].sha256
                    if not left_hash or not right_hash:
                        continue
                    key = tuple(sorted((left_hash, right_hash)))
                    score = float(payload["raw_score_matrix"][row][col])
                    cached_wins = (
                        int(payload["wins"][row][col]) if "wins" in payload else 0
                    )
                    cached_losses = (
                        int(payload["losses"][row][col])
                        if "losses" in payload
                        else 0
                    )
                    if left_hash != key[0]:
                        score = 1.0 - score
                        cached_wins, cached_losses = cached_losses, cached_wins
                    cache[key] = {
                        "score": score,
                        "games": int(payload["games"][row][col]),
                        "standard_error": float(payload["standard_errors"][row][col]),
                        "wins": cached_wins,
                        "ties": int(payload.get("ties", [[0]])[row][col])
                        if "ties" in payload
                        else 0,
                        "losses": cached_losses,
                        "last_updated_transition": int(
                            payload.get("last_updated_transitions", [])[row][col]
                        )
                        if payload.get("last_updated_transitions")
                        else int(payload.get("population_transition", 0)),
                    }
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return cache


def _mixture_score(actor, pool, meta, pair_count, seed):
    rng = np.random.default_rng(seed)
    probabilities = np.asarray(meta["probabilities"], dtype=float)
    probabilities /= probabilities.sum()
    scores = []
    sampled = rng.choice(
        len(meta["policy_ids"]), size=max(1, pair_count), p=probabilities
    )
    for index, opponent_index in enumerate(sampled):
        opponent_id = meta["policy_ids"][int(opponent_index)]
        result = paired_actor_evaluation(
            actor,
            pool.load(opponent_id),
            pair_count=1,
            seed_base=seed + index,
        )
        scores.append(float(result["pair_scores"][0]))
    return float(np.mean(scores))


def _archive_or_respawn(
    run,
    learner,
    pool,
    champion_id,
    main,
    meta,
    config,
    sizes,
    population_transition,
    device,
):
    frozen = _as_frozen(learner, device)
    if learner.role.startswith("main_exploiter"):
        result = paired_actor_evaluation(
            frozen,
            pool.load(champion_id),
            pair_count=config.exploiter_pair_count,
            seed_base=config.seed
            + 40_000_000
            + population_transition
            + learner.generation * 100,
        )
        scores = result["pair_scores"]
    else:
        mixture_seed = config.seed + 41_000_000 + population_transition
        rng = np.random.default_rng(mixture_seed + learner.generation)
        probabilities = np.asarray(meta["probabilities"], dtype=float)
        probabilities /= probabilities.sum()
        sampled = rng.choice(
            len(meta["policy_ids"]),
            size=config.exploiter_pair_count,
            p=probabilities,
        )
        scores = []
        for index, opponent_index in enumerate(sampled):
            evaluated = paired_actor_evaluation(
                frozen,
                pool.load(meta["policy_ids"][int(opponent_index)]),
                pair_count=1,
                seed_base=mixture_seed + index,
            )
            scores.append(float(evaluated["pair_scores"][0]))
    ci = bootstrap_confidence_interval(
        scores,
        samples=config.exploiter_bootstrap_samples,
        confidence=0.95,
        seed=config.seed + 42_000_000 + population_transition,
    )
    accepted = (
        ci["mean_score"] >= config.exploiter_min_target_score
        and ci["lower_confidence_bound"] > config.exploiter_min_ci_lower
    )
    stale = (
        learner.transitions_since_success
        >= config.exploiter_max_transitions_without_success
    )
    event = {
        "role": learner.role,
        "generation": learner.generation,
        "population_transition": population_transition,
        "target": champion_id
        if learner.role.startswith("main_exploiter")
        else "meta_strategy",
        **ci,
        "archived": accepted,
        "stale_respawn": stale and not accepted,
    }
    if accepted:
        archive_id = f"{learner.role}_gen{learner.generation:04d}"
        pool.add_snapshot(
            learner.actor,
            opponent_id=archive_id,
            source_type="archived_exploiter",
            created_transition=population_transition,
            champion_version=None,
            training_seed=config.seed,
            actor_obs_size=sizes["actor"],
            action_size=sizes["action"],
        )
        archive = (
            run
            / "population/exploiters"
            / ("main" if learner.role.startswith("main_exploiter") else "league")
        )
        archive.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "0.6.1",
                "opponent_id": archive_id,
                "role": learner.role,
                "generation": learner.generation,
                "actor_state_dict": learner.actor.state_dict(),
                "actor_sha256": actor_sha256(learner.actor),
            },
            archive / f"{archive_id}.pt",
        )
        learner.archive_successes += 1
    if accepted or stale:
        if stale and not accepted:
            learner.stale_respawns += 1
        reset_learner(
            learner,
            pool.load(champion_id).actor.state_dict(),
            main.critic.state_dict(),
            config,
        )
        _replace_snapshot(
            pool,
            learner.actor,
            f"active_{learner.role}",
            "active_exploiter",
            population_transition,
            config,
            sizes,
        )
    return event


def _as_frozen(learner, device):
    from splendor_rl.league.promotion import frozen_copy

    return frozen_copy(learner.actor, learner.role, device)


def _promotion(
    run,
    main,
    pool,
    champion_id,
    champion_version,
    meta,
    config,
    population_transition,
    sizes,
    device,
):
    candidate = _as_frozen(main, device)
    champion = pool.load(champion_id)
    paired = paired_actor_evaluation(
        candidate,
        champion,
        pair_count=config.main_promotion_pair_count,
        seed_base=config.seed + 50_000_000 + population_transition,
    )
    ci = bootstrap_confidence_interval(
        paired["pair_scores"],
        samples=config.main_promotion_bootstrap_samples,
        confidence=0.95,
        seed=config.seed + 51_000_000 + population_transition,
    )
    head, reasons = promotion_decision(
        ci["mean_score"],
        ci["lower_confidence_bound"],
        min_score=config.main_promotion_min_score,
        min_lower_bound=config.main_promotion_min_ci_lower,
        identical_hash=actor_sha256(main.actor) == pool.metadata[champion_id].sha256,
    )
    candidate_scores = {}
    champion_scores = {}
    anchors = {}
    for index, bot in enumerate(("greedy", "noble", "blocking", "random", "shortest")):
        seed = config.seed + 52_000_000 + population_transition + index * 10000
        c = actor_vs_bot_score(
            candidate, bot, games=config.promotion_anchor_games, seed_base=seed
        )
        h = actor_vs_bot_score(
            champion, bot, games=config.promotion_anchor_games, seed_base=seed
        )
        anchors[bot] = {"candidate": c, "champion": h}
        candidate_scores[bot] = c["score"]
        champion_scores[bot] = h["score"]
    anchor = anchor_group_regression_decision(candidate_scores, champion_scores)
    historical_candidate = {}
    historical_champion = {}
    for index, opponent_id in enumerate(
        [key for key in pool.hall_of_fame_ids if key != champion_id][-3:]
    ):
        opponent = pool.load(opponent_id)
        seed = config.seed + 52_500_000 + population_transition + index * 1000
        historical_candidate[opponent_id] = paired_actor_evaluation(
            candidate,
            opponent,
            pair_count=max(1, config.main_promotion_pair_count // 4),
            seed_base=seed,
        )["paired_score"]
        historical_champion[opponent_id] = paired_actor_evaluation(
            champion,
            opponent,
            pair_count=max(1, config.main_promotion_pair_count // 4),
            seed_base=seed,
        )["paired_score"]
    historical_gate = regression_decision(historical_candidate, historical_champion)
    candidate_meta = _mixture_score(
        candidate,
        pool,
        meta,
        config.main_promotion_pair_count,
        config.seed + 53_000_000 + population_transition,
    )
    champion_meta = _mixture_score(
        champion,
        pool,
        meta,
        config.main_promotion_pair_count,
        config.seed + 53_000_000 + population_transition,
    )
    meta_delta = candidate_meta - champion_meta
    meta_pass = meta_delta >= config.main_promotion_min_meta_delta
    exploiters = [key for key in pool.metadata if "exploiter" in key]
    regressions = []
    for index, key in enumerate(exploiters):
        seed = config.seed + 55_000_000 + population_transition + index * 1000
        c = _mixture_score(
            candidate, pool, {"policy_ids": [key], "probabilities": [1]}, 1, seed
        )
        h = _mixture_score(
            champion, pool, {"policy_ids": [key], "probabilities": [1]}, 1, seed
        )
        regressions.append(h - c)
    exploiter_reg = max(regressions, default=0.0)
    exploiter_pass = exploiter_reg <= config.main_promotion_max_exploiter_regression
    if not anchor["passed"]:
        reasons.extend(anchor["reasons"])
    if not historical_gate["passed"]:
        reasons.append("hall_of_fame_regression")
    if not meta_pass:
        reasons.append("meta_strategy_regression")
    if not exploiter_pass:
        reasons.append("exploiter_robustness_regression")
    promoted = (
        head
        and anchor["passed"]
        and historical_gate["passed"]
        and meta_pass
        and exploiter_pass
    )
    new_id = champion_id
    if promoted:
        champion_version += 1
        new_id = f"champion_{champion_version:04d}"
        pool.add_snapshot(
            main.actor,
            opponent_id=new_id,
            source_type="champion",
            created_transition=population_transition,
            champion_version=champion_version,
            training_seed=config.seed,
            actor_obs_size=sizes["actor"],
            action_size=sizes["action"],
        )
    result = {
        "schema_version": "0.6.1",
        "population_transition": population_transition,
        "evaluated_champion_id": champion_id,
        "head_to_head": {**paired, **ci},
        "anchors": anchors,
        "hard_anchor_gate": anchor,
        "hall_of_fame_gate": historical_gate,
        "meta_strategy_gate": {
            "candidate_meta_score": candidate_meta,
            "champion_meta_score": champion_meta,
            "meta_delta": meta_delta,
            "passed": meta_pass,
        },
        "exploiter_robustness_gate": {
            "max_regression": exploiter_reg,
            "passed": exploiter_pass,
        },
        "promoted": promoted,
        "new_champion_id": new_id,
        "reasons": reasons,
    }
    output = run / "promotions" / f"step_{population_transition:09d}"
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "promotion_result.json", result)
    return new_id, champion_version, result


def _state(
    run,
    transition,
    learners,
    scheduler,
    champion_id,
    champion_version,
    pool,
    meta,
    thresholds,
    bootstrap,
):
    value = {
        "schema_version": "0.6.1",
        "population_transition": transition,
        "source": {
            "source_rl_version": bootstrap["source_rl_version"],
            "source_transition": bootstrap["source_transition"],
            "source_actor_sha256": bootstrap["source_actor_sha256"],
            "population_seed": bootstrap.get("population_seed", 42),
        },
        "main": {
            "candidate_checkpoint": "learners/main/latest.pt",
            "champion_id": champion_id,
            "champion_version": champion_version,
            "learner_transitions": learners["main"].transitions,
        },
        "learners": {
            role: learner.state_summary() | {"checkpoint": f"learners/{role}/latest.pt"}
            for role, learner in learners.items()
        },
        "population": {
            "hall_of_fame_size": len(pool.hall_of_fame_ids),
            "recent_size": len(pool.recent_ids),
            "archived_main_exploiters": len(
                [k for k in pool.metadata if k.startswith("main_exploiter")]
            ),
            "archived_league_exploiters": len(
                [k for k in pool.metadata if k.startswith("league_exploiter")]
            ),
        },
        "meta_strategy": meta,
        "scheduler": scheduler.state_dict(),
        "rng_state": {
            "scheme": "counter_derived_v1",
            "base_seed": bootstrap.get("population_seed", 42),
            "scheduler": "deterministic_deficit",
            "role_update_counters": {
                role: learner.updates for role, learner in learners.items()
            },
            "role_generations": {
                role: learner.generation for role, learner in learners.items()
            },
        },
        "thresholds": thresholds,
    }
    _atomic_json(run / "population_state.json", value)
    return value


def _next_boundary(current, interval):
    return ((int(current) // int(interval)) + 1) * int(interval)


def _migrate_thresholds(thresholds, learners, transition, config):
    """Move orchestration-only cadence forward without replaying missed gates."""
    result = dict(thresholds)
    result["next_meta"] = _next_boundary(transition, config.meta_update_interval)
    result["next_promotion"] = _next_boundary(
        transition, config.main_promotion_interval
    )
    result["next_recent"] = max(
        int(result.get("next_recent", 0)),
        _next_boundary(transition, config.recent_snapshot_interval),
    )
    result["next_full_checkpoint"] = _next_boundary(
        transition, config.checkpoint_full_interval
    )
    result["next_lightweight_state"] = _next_boundary(
        transition, config.checkpoint_lightweight_interval
    )
    interval = (
        config.exploiter_evaluation_interval_learner_transitions
        or config.exploiter_evaluation_interval
    )
    existing = result.get("next_exploiter_eval_by_role", {})
    result["next_exploiter_eval_by_role"] = {
        role: max(
            int(existing.get(role, 0)),
            _next_boundary(learners[role].transitions, interval),
        )
        for role in ROLES[1:]
    }
    result.pop("next_exploiter_eval", None)
    return result


def _write_migration_manifest(run, source_state, transition, backend):
    source_state = Path(source_state).resolve()
    digest = hashlib.sha256(source_state.read_bytes()).hexdigest()
    backup = Path(run) / "pre_v061_backup"
    backup.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_version": "0.6.0",
        "target_version": "0.6.1",
        "population_transition": transition,
        "source_checkpoint": str(source_state),
        "source_checkpoint_sha256": digest,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "collector_backend": backend,
    }
    _atomic_json(backup / "migration_manifest.json", manifest)
    return manifest


def train_population(
    config: PopulationConfig,
    run_dir,
    *,
    bootstrap_run_dir=None,
    resume=None,
    stop_at_population_transitions=None,
    device=None,
    progress_config=None,
    resume_dry_run=False,
    profile_output=None,
):
    config.validate()
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    source = None
    if resume:
        state = json.loads(Path(resume).read_text(encoding="utf-8"))
        source_schema = state.get("schema_version")
        if source_schema not in {"0.6.0", "0.6.1"}:
            raise ValueError("unsupported population state")
        # Pool and architecture are reconstructed from the persisted run.
        first = torch.load(
            run / state["learners"]["main"]["checkpoint"],
            map_location=device,
            weights_only=False,
        )
        config.hidden_sizes = list(first["config"]["hidden_sizes"])
        sizes = first["observation_sizes"]
        pool = OpponentPool(
            run / "population/pool",
            config.hidden_sizes,
            device,
            max_cached_actors=config.frozen_actor_cache_size,
        )
        learners = {
            role: _load_learner(run / item["checkpoint"], config, sizes, device)
            for role, item in state["learners"].items()
        }
        transition = state["population_transition"]
        scheduler = DeficitScheduler.from_state_dict(state["scheduler"])
        champion_id = state["main"]["champion_id"]
        champion_version = state["main"]["champion_version"]
        meta = state["meta_strategy"]
        # Do not thrash the LRU while resuming a legacy 24-policy meta mixture;
        # the next fast meta update naturally contracts it to max_policies=16.
        pool.max_cached_actors = max(
            config.frozen_actor_cache_size, len(meta.get("policy_ids", [])) + 5
        )
        thresholds = state["thresholds"]
        source = {
            **state["source"],
            "sizes": sizes,
            "population_transition": transition,
        }
        thresholds = _migrate_thresholds(thresholds, learners, transition, config)
        if resume_dry_run:
            print("Loaded v0.6.0 state successfully." if source_schema == "0.6.0" else "Loaded v0.6.1 state successfully.")
            print(f"Population transition: {transition:,}")
            print(f"Current champion: {champion_id}")
            print("Main optimizer: restored")
            print("Exploiters: restored")
            print("Meta strategy: restored")
            print(f"Next promotion threshold: {thresholds['next_promotion']:,}")
            print(f"Next meta threshold: {thresholds['next_meta']:,}")
            print("Migration target schema: v0.6.1")
            print("Population horizon: 50,000,000")
            print("SAFE TO RESUME\nNo training executed.")
            return {"state": state, "thresholds": thresholds, "safe_to_resume": True}
        if source_schema == "0.6.0":
            _write_migration_manifest(run, resume, transition, config.collector_backend)
        recovered_pool_entries = _reconcile_pool_to_committed_state(
            pool, learners, transition, config, sizes
        )
        if recovered_pool_entries:
            _atomic_json(
                run / "recovery_report.json",
                {
                    "schema_version": "0.6.1",
                    "committed_population_transition": transition,
                    "removed_uncommitted_pool_entries": recovered_pool_entries,
                },
            )
    else:
        if not bootstrap_run_dir:
            raise ValueError("--bootstrap-run-dir is required for a new population run")
        source = bootstrap_population(
            bootstrap_run_dir, run / "population/pool", config, device
        )
        sizes = source["sizes"]
        pool = source["pool"]
        learners = {
            role: make_learner(
                role,
                source["actor_state_dict"],
                source["critic_state_dict"],
                config,
                sizes,
                device,
            )
            for role in ROLES
        }
        transition = 0
        scheduler = DeficitScheduler(config.scheduler)
        champion_id = source["champion_id"]
        champion_version = source["champion_version"]
        meta = {
            "policy_ids": [champion_id],
            "probabilities": [1.0],
            "estimated_value": 0.0,
            "convergence_gap": 0.0,
            "step": 0,
            "cycles": [],
        }
        thresholds = {
            "next_meta": config.meta_update_interval,
            "next_exploiter_eval": config.exploiter_evaluation_interval,
            "next_promotion": config.main_promotion_interval,
            "next_recent": config.recent_snapshot_interval,
            "promotion_attempts": 0,
            "promotion_successes": 0,
            "next_full_checkpoint": config.checkpoint_full_interval,
            "next_lightweight_state": config.checkpoint_lightweight_interval,
            "next_exploiter_eval_by_role": {
                role: config.exploiter_evaluation_interval_learner_transitions
                or config.exploiter_evaluation_interval
                for role in ROLES[1:]
            },
        }
        source["population_transition"] = 0
        source["population_seed"] = config.seed
        _atomic_json(
            run / "bootstrap_report.json",
            {
                key: value
                for key, value in source.items()
                if key not in {"actor_state_dict", "critic_state_dict", "pool"}
            }
            | {
                "schema_version": "0.6.1",
                "optimizer_reset": True,
                "learning_rate_schedule_reset": True,
                "entropy_schedule_reset": True,
                "population_transition": 0,
            },
        )
        _replace_snapshot(
            pool,
            learners["main"].actor,
            "current_main_snapshot",
            "recent_main",
            0,
            config,
            sizes,
        )
        for role in ROLES[1:]:
            _replace_snapshot(
                pool,
                learners[role].actor,
                f"active_{role}",
                "active_exploiter",
                0,
                config,
                sizes,
            )
    target = stop_at_population_transitions or config.total_population_transitions
    if not transition < target <= config.total_population_transitions:
        raise ValueError("invalid population stop target")
    records = MatchRecords(run / "metrics/matchup_records.json", 400)
    metrics = JsonlMetrics(run / "metrics/population_training.jsonl")
    started = datetime.now(timezone.utc).isoformat()
    promotion_attempts = thresholds.get("promotion_attempts", 0)
    promotion_successes = thresholds.get("promotion_successes", 0)
    archive_events = []
    illegal_actions = invariant_violations = 0
    round_metrics = []
    profiler = WallProfiler(config.profiling_enabled)
    progress = make_training_progress(
        target,
        transition,
        config.total_population_transitions,
        progress_config or ProgressConfig(),
    )
    while transition < target:
        role = scheduler.next()
        learner = learners[role]
        update_seed = (
            config.seed
            + ROLES.index(role) * 1_000_003
            + learner.updates * 10_007
            + learner.generation * 100_000_007
        )
        torch.manual_seed(update_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(update_seed)
        count = min(config.transitions_per_update, target - transition)
        role_config = copy.copy(config)
        role_config.seed = update_seed
        selector = _selector(role, config, pool, records, champion_id, meta)
        collector_class = (
            MultiprocessBatchedPopulationCollector
            if config.collector_backend == "multiprocess_batched"
            else PopulationRolloutCollector
        )
        collector_kwargs = {
            "role": role,
            "pool": pool,
            "records": records,
            "config": role_config,
            "selector": selector,
            "update_index": learner.updates,
            "device": device,
        }
        if collector_class is MultiprocessBatchedPopulationCollector:
            collector_kwargs["profiler"] = profiler
        collector = collector_class(learner.actor, learner.critic, **collector_kwargs)
        try:
            batch, advantages, returns, rollout = collector.collect(
                count, config.gae_lambda
            )
        finally:
            close = getattr(collector, "close", None)
            if close:
                close()
        illegal_actions += collector.illegal_actions
        invariant_violations += collector.invariant_violations
        round_metrics.extend(collector.episodes)
        budget = config.total_population_transitions * config.scheduler[role]
        schedule_position = (
            learner.transitions if role == "main" else learner.generation_transitions
        )
        lr = linear_learning_rate(
            config.learning_rate,
            config.min_learning_rate,
            schedule_position,
            budget,
            config.linear_lr_decay,
        )
        apply_learning_rate(learner.optimizer, lr)
        config.current_entropy_coef = entropy_coefficient(
            config.entropy_coef_start,
            config.entropy_coef_end,
            config.entropy_anneal_fraction,
            schedule_position,
            budget,
        )
        with profiler.measure("PPO_forward"):
            update = ppo_update(
                learner.actor,
                learner.critic,
                learner.optimizer,
                batch,
                advantages,
                returns,
                config,
            )
        learner.transitions += len(batch)
        learner.updates += 1
        learner.generation_transitions += len(batch)
        learner.generation_updates += 1
        learner.games += len(collector.episodes)
        learner.transitions_since_success += len(batch)
        transition += len(batch)
        if role == "main":
            _replace_snapshot(
                pool,
                learner.actor,
                "current_main_snapshot",
                "recent_main",
                transition,
                config,
                sizes,
            )
        else:
            _replace_snapshot(
                pool,
                learner.actor,
                f"active_{role}",
                "active_exploiter",
                transition,
                config,
                sizes,
            )
        if transition >= thresholds["next_recent"]:
            _add_snapshot_idempotent(
                pool,
                learners["main"].actor,
                f"recent_main_{transition:09d}",
                "recent",
                transition,
                config,
                sizes,
            )
            while transition >= thresholds["next_recent"]:
                thresholds["next_recent"] += config.recent_snapshot_interval
        if transition >= thresholds["next_meta"]:
            with profiler.measure("meta_matchup_evaluation"):
                meta = _meta_update(run, pool, champion_id, config, transition, device)
            while transition >= thresholds["next_meta"]:
                thresholds["next_meta"] += config.meta_update_interval
        for exploiter_role in ROLES[1:]:
            local_threshold = thresholds["next_exploiter_eval_by_role"][exploiter_role]
            if learners[exploiter_role].transitions >= local_threshold:
                with profiler.measure("exploiter_evaluation"):
                    event = _archive_or_respawn(
                        run,
                        learners[exploiter_role],
                        pool,
                        champion_id,
                        learners["main"],
                        meta,
                        config,
                        sizes,
                        transition,
                        device,
                    )
                archive_events.append(
                    event
                )
                interval = (
                    config.exploiter_evaluation_interval_learner_transitions
                    or config.exploiter_evaluation_interval
                )
                while learners[exploiter_role].transitions >= local_threshold:
                    local_threshold += interval
                thresholds["next_exploiter_eval_by_role"][exploiter_role] = local_threshold
        promotion = None
        if transition >= thresholds["next_promotion"]:
            promotion_attempts += 1
            with profiler.measure("promotion_evaluation"):
                champion_id, champion_version, promotion = _promotion(
                    run,
                    learners["main"],
                    pool,
                    champion_id,
                    champion_version,
                    meta,
                    config,
                    transition,
                    sizes,
                    device,
                )
            promotion_successes += int(promotion["promoted"])
            thresholds["promotion_attempts"] = promotion_attempts
            thresholds["promotion_successes"] = promotion_successes
            while transition >= thresholds["next_promotion"]:
                thresholds["next_promotion"] += config.main_promotion_interval
        source["population_transition"] = transition
        checkpoint_due = (
            transition >= thresholds["next_full_checkpoint"] or transition >= target
        )
        if checkpoint_due:
            with profiler.measure("checkpoint_save"):
                records.save()
                for item in learners.values():
                    _save_learner(
                        run / f"learners/{item.role}/latest.pt", item, config, source
                    )
                state = _state(
                    run, transition, learners, scheduler, champion_id,
                    champion_version, pool, meta, thresholds, source,
                )
            while transition >= thresholds["next_full_checkpoint"]:
                thresholds["next_full_checkpoint"] += config.checkpoint_full_interval
        if transition >= thresholds["next_lightweight_state"]:
            _atomic_json(
                run / "lightweight_state.json",
                {
                    "schema_version": "0.6.1",
                    "diagnostic_only": True,
                    "authoritative_resume": "population_state.json",
                    "population_transition": transition,
                    "learner_transitions": {
                        key: value.transitions for key, value in learners.items()
                    },
                },
            )
            while transition >= thresholds["next_lightweight_state"]:
                thresholds["next_lightweight_state"] += config.checkpoint_lightweight_interval
        metrics.write(
            {
                "population_transition": transition,
                "selected_learner_role": role,
                "role_population_transitions": learner.transitions,
                "role_updates": learner.updates,
                "role_games": learner.games,
                "role_learning_rate": lr,
                "role_entropy": update["entropy"],
                "role_kl": update["approx_kl_mean"],
                "learner_allocations": {
                    key: value.transitions for key, value in learners.items()
                },
                **rollout,
                **update,
                "champion_id": champion_id,
                "champion_version": champion_version,
                "meta_population_size": len(meta["policy_ids"]),
                "promotion_result": promotion,
                "checkpoint_saved": checkpoint_due,
            }
        )
        if profile_output and config.profiling_enabled:
            profiler.write(profile_output)
        progress.update_training(
            len(batch),
            transitions=transition,
            update_index=scheduler.steps,
            episodes=sum(item.games for item in learners.values()),
            metrics={
                "learner": role,
                "gen": learner.generation,
                "champ": champion_version,
                "hof": len(pool.hall_of_fame_ids),
                "meta": len(meta["policy_ids"]),
                "main": learners["main"].transitions,
                "me0": learners["main_exploiter_0"].transitions,
                "me1": learners["main_exploiter_1"].transitions,
                "le0": learners["league_exploiter_0"].transitions,
                "le1": learners["league_exploiter_1"].transitions,
            },
        )
    progress.close()
    win_rounds = [
        item["rounds"] for item in round_metrics if item.get("candidate_score") == 1
    ]
    loss_rounds = [
        item["rounds"] for item in round_metrics if item.get("candidate_score") == 0
    ]
    tie_rounds = [
        item["rounds"] for item in round_metrics if item.get("candidate_score") == 0.5
    ]
    mode_counts = {
        name: sum(
            item.get("population_role") == "main" and item.get("league_mode") == name
            for item in round_metrics
        )
        for name in config.main_training
    }
    mode_total = max(1, sum(mode_counts.values()))
    exploiter_scores = []
    final_champion = pool.load(champion_id)
    for index, key in enumerate([key for key in pool.metadata if "exploiter" in key]):
        result = paired_actor_evaluation(
            pool.load(key),
            final_champion,
            pair_count=1,
            seed_base=config.seed + 70_000_000 + transition + index * 1000,
        )
        exploiter_scores.append(float(result["paired_score"]))
    summary = {
        "schema_version": "0.6.1",
        "rl_version": "0.6.1",
        "training_mode": "population_league_2p",
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "source_rl_version": source["source_rl_version"],
        "source_transition": source["source_transition"],
        "population_transition": transition,
        "champion_id": champion_id,
        "champion_version": champion_version,
        "learner_allocations": {
            key: value.transitions for key, value in learners.items()
        },
        "learner_updates": {key: value.updates for key, value in learners.items()},
        "promotion_attempts": promotion_attempts,
        "promotion_successes": promotion_successes,
        "archive_events": archive_events,
        "archive_successes_total": sum(
            item.archive_successes for key, item in learners.items() if key != "main"
        ),
        "respawns_total": sum(
            item.respawns for key, item in learners.items() if key != "main"
        ),
        "stale_respawns_total": sum(
            item.stale_respawns for key, item in learners.items() if key != "main"
        ),
        "meta_strategy": meta,
        "best_exploiter_score_against_champion": max(exploiter_scores, default=0.5),
        "empirical_exploitability_proxy": empirical_exploitability_proxy(
            exploiter_scores
        ),
        "exploitability_note": "Empirical best-response proxy over discovered exploiters; not exact Nash exploitability.",
        "average_final_round": float(
            np.mean([item["rounds"] for item in round_metrics])
        )
        if round_metrics
        else None,
        "average_rounds_on_wins": float(np.mean(win_rounds)) if win_rounds else None,
        "average_rounds_on_losses": float(np.mean(loss_rounds))
        if loss_rounds
        else None,
        "average_rounds_on_ties": float(np.mean(tie_rounds)) if tie_rounds else None,
        "average_player_turns": float(
            np.mean([item["turns"] for item in round_metrics])
        )
        if round_metrics
        else None,
        "average_decisions": float(
            np.mean([item["decisions"] for item in round_metrics])
        )
        if round_metrics
        else None,
        "main_opponent_fractions_actual": {
            key: value / mode_total for key, value in mode_counts.items()
        },
        "illegal_actions": illegal_actions,
        "invariant_violations": invariant_violations,
        "ppo_finite": True,
        "game_length_is_reward": False,
    }
    _atomic_json(run / "training_summary.json", summary)
    (run / "training_report.md").write_text(
        "# Population league report\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary.items()),
        encoding="utf-8",
    )
    return state
