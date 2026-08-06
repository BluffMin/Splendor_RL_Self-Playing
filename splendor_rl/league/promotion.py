from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from splendor_env.core import NoLegalActionError, SplendorGame
from splendor_env.recording import EpisodeRecorder
from splendor_env.wrappers import CanonicalPaymentWrapper
from splendor_rl.evaluation import make_bot

from .pool import FrozenOpponent
from .types import OpponentMetadata


def bootstrap_confidence_interval(values, *, samples, confidence, seed):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("bootstrap requires at least one pair score")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(
        axis=1
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_score": float(values.mean()),
        "lower_confidence_bound": float(np.quantile(means, tail)),
        "upper_confidence_bound": float(np.quantile(means, 1.0 - tail)),
    }


def promotion_decision(
    mean_score,
    lower_bound,
    *,
    min_score=0.55,
    min_lower_bound=0.50,
    identical_hash=False,
):
    reasons = []
    if identical_hash:
        reasons.append("candidate_and_champion_actor_hash_identical")
    if mean_score < min_score:
        reasons.append("head_to_head_mean_below_minimum")
    if lower_bound <= min_lower_bound:
        reasons.append("head_to_head_lower_bound_not_above_0.50")
    return not reasons, reasons


def regression_decision(
    candidate_scores,
    champion_scores,
    *,
    max_aggregate=0.03,
    max_single=0.07,
):
    common = sorted(set(candidate_scores) & set(champion_scores))
    regressions = {key: champion_scores[key] - candidate_scores[key] for key in common}
    aggregate = float(np.mean(list(regressions.values()))) if regressions else 0.0
    maximum = max(regressions.values(), default=0.0)
    passed = aggregate <= max_aggregate and maximum <= max_single
    return {
        "passed": passed,
        "aggregate_regression": aggregate,
        "max_single_regression": maximum,
        "anchor_regressions": regressions,
    }


def frozen_copy(actor, opponent_id, device="cpu"):
    cloned = copy.deepcopy(actor)
    metadata = OpponentMetadata(
        opponent_id,
        "candidate",
        0,
        None,
        0,
        next(iter(actor.parameters())).shape[-1],
        373,
        2,
        "",
        "",
    )
    return FrozenOpponent(cloned, metadata, device)


def _game_score(game, candidate_seat):
    ranking = next(
        group for group in game.final_ranking() if candidate_seat in group["players"]
    )
    if ranking["rank"] != 1:
        return 0.0
    return 1.0 if len(ranking["players"]) == 1 else 0.5


def play_actor_game(
    left,
    right,
    *,
    seed,
    candidate_seat=0,
    max_turns=300,
    replay_path=None,
    replay_metadata=None,
):
    game = SplendorGame(2, seed=seed)
    wrapper = CanonicalPaymentWrapper(game)
    recorder = EpisodeRecorder(replay_path) if replay_path else None
    policy_trace = []
    if recorder:
        recorder.attach(game)
    actors = [left, right]
    while not game.done:
        if game.turns_completed >= max_turns:
            game.truncate("league_evaluation_max_turns")
            break
        player = game.current_player
        try:
            observation = game.observation(player, omniscient=False)
            mask = game.legal_action_mask().astype(bool)
            action = actors[player].act(
                observation,
                mask,
                deterministic=True,
            )
            if recorder:
                probabilities = actors[player].action_probabilities(observation, mask)
                policy_trace.append(
                    {
                        "decision_id": game.decision_id,
                        "acting_policy_id": actors[player].metadata.opponent_id,
                        "selected_action": action,
                        "legal_action_probabilities": {
                            str(index): float(probabilities[index])
                            for index in np.flatnonzero(mask)
                        },
                    }
                )
            wrapper.policy_step(action)
        except NoLegalActionError:
            game.truncate("league_evaluation_no_legal_action")
            break
    score = _game_score(game, candidate_seat) if game.terminated else 0.5
    if recorder:
        document = recorder.finalize()
        if game.truncated and document["events"]:
            replay_hash = document["events"][-1]["post_state_hash"]
            document["final_state_hash"] = replay_hash
            document["result"]["final_state_hash"] = replay_hash
            document["final_summary"]["state_hash"] = replay_hash
        document["game_metadata"].update(replay_metadata or {})
        document["game_metadata"].update(
            {
                "rl_version": "0.5.0",
                "training_mode": "league_2p",
                "actor_public_observation_version": "0.3.2",
                "action_mask_version": "0.3.2",
                "final_result": score,
                "final_rounds": game.round_id,
                "policy_trace": policy_trace,
            }
        )
        Path(replay_path).write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return score, game


def paired_actor_evaluation(
    candidate,
    opponent,
    *,
    pair_count,
    seed_base,
    progress=None,
    replay_dir=None,
    replay_metadata=None,
    matchup_label="candidate_vs_champion",
):
    pair_scores = []
    p0_scores, p1_scores, rounds = [], [], []
    for index in range(pair_count):
        seed = seed_base + index
        replay_a = (
            Path(replay_dir) / "pair_0000_candidate_p0.json"
            if replay_dir and index == 0
            else None
        )
        replay_b = (
            Path(replay_dir) / "pair_0000_candidate_p1.json"
            if replay_dir and index == 0
            else None
        )
        if replay_dir:
            Path(replay_dir).mkdir(parents=True, exist_ok=True)
        p0, game_a = play_actor_game(
            candidate,
            opponent,
            seed=seed,
            candidate_seat=0,
            replay_path=replay_a,
            replay_metadata={**(replay_metadata or {}), "candidate_seat": 0},
        )
        opponent_score, game_b = play_actor_game(
            opponent,
            candidate,
            seed=seed,
            candidate_seat=0,
            replay_path=replay_b,
            replay_metadata={**(replay_metadata or {}), "candidate_seat": 1},
        )
        p1 = 1.0 - opponent_score
        p0_scores.append(p0)
        p1_scores.append(p1)
        pair_scores.append((p0 + p1) / 2.0)
        rounds.extend([game_a.round_id, game_b.round_id])
        if progress:
            progress.update(2, matchup=matchup_label, seat="paired")
    return {
        "pairs": pair_count,
        "games": pair_count * 2,
        "pair_scores": pair_scores,
        "candidate_as_p0_score": float(np.mean(p0_scores)),
        "candidate_as_p1_score": float(np.mean(p1_scores)),
        "seat_gap": float(abs(np.mean(p0_scores) - np.mean(p1_scores))),
        "average_final_round": float(np.mean(rounds)),
    }


def actor_vs_bot_score(actor, bot_name, *, games, seed_base, progress=None, label=None):
    scores, ranks, final_scores, rounds, win_rounds = [], [], [], [], []
    for index in range(games):
        seat = index % 2
        seed = seed_base + index // 2
        game = SplendorGame(2, seed=seed)
        wrapper = CanonicalPaymentWrapper(game)
        bot = make_bot(bot_name, seed * 17 + (1 - seat))
        while not game.done:
            if game.turns_completed >= 300:
                game.truncate("league_anchor_max_turns")
                break
            player = game.current_player
            try:
                action = (
                    actor.act(
                        game.observation(player, omniscient=False),
                        game.legal_action_mask().astype(bool),
                        deterministic=True,
                    )
                    if player == seat
                    else bot.act(game)
                )
                wrapper.policy_step(action)
            except NoLegalActionError:
                game.truncate("league_anchor_no_legal_action")
                break
        score = _game_score(game, seat) if game.terminated else 0.5
        scores.append(score)
        ranking = next(
            group for group in game.final_ranking() if seat in group["players"]
        )
        ranks.append(ranking["rank"])
        final_scores.append(game.players[seat].score)
        rounds.append(game.round_id)
        if score == 1:
            win_rounds.append(game.round_id)
        if progress:
            progress.update(1, matchup=label or bot_name, seat=f"P{seat}")
    return {
        "score": float(np.mean(scores)),
        "fractional_first_place_rate": float(np.mean(scores)),
        "average_rank": float(np.mean(ranks)),
        "average_score": float(np.mean(final_scores)),
        "average_final_round": float(np.mean(rounds)),
        "average_rounds_on_wins": float(np.mean(win_rounds)) if win_rounds else None,
    }
