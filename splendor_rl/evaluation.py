from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from splendor_env.actions import ACTIONS
from splendor_env.agents import GreedyAgent, RandomLegalAgent, ShortestAgent
from splendor_env.core import NoLegalActionError, Phase, SplendorGame
from splendor_env.recording import EpisodeRecorder
from splendor_env.replay import replay_text
from splendor_env.visualization.html_export import export_replay
from splendor_env.wrappers import CanonicalPaymentWrapper

from .distributions import MaskedCategorical


class NobleAgent(GreedyAgent):
    def act(self, game):
        return (
            min(game.legal_actions())
            if game.phase == Phase.NOBLE
            else super().act(game)
        )


class BlockingAgent(GreedyAgent):
    def act(self, game):
        legal = game.legal_actions()
        reserves = [a for a in legal if ACTIONS[a].kind == "reserve_visible"]
        return max(reserves) if reserves else super().act(game)


def make_bot(name, seed):
    return {
        "random": lambda: RandomLegalAgent(seed),
        "greedy": GreedyAgent,
        "shortest": ShortestAgent,
        "noble": NobleAgent,
        "blocking": BlockingAgent,
    }[name]()


def actor_action(actor, game, player, device="cpu", deterministic=True):
    obs = torch.as_tensor(
        game.observation(player, omniscient=False), device=device
    ).unsqueeze(0)
    mask = torch.as_tensor(
        game.legal_action_mask().astype(bool), device=device
    ).unsqueeze(0)
    with torch.no_grad():
        dist = MaskedCategorical(actor(obs), mask)
        action = dist.mode() if deterministic else dist.sample()
    return int(action.item())


def _stats(rows):
    result = {
        "games": len(rows),
        "average_rank": float(np.mean([r["rank"] for r in rows])),
        "fractional_first_place_rate": float(
            np.mean([r["fractional_win"] for r in rows])
        ),
        "average_score": float(np.mean([r["score"] for r in rows])),
        "tie_rate": float(np.mean([r["winner_count"] > 1 for r in rows])),
        "average_turns": float(np.mean([r["turns"] for r in rows])),
    }
    for seat in range(4):
        values = [r for r in rows if r["policy_seat"] == seat]
        result[f"seat_{seat}_average_rank"] = (
            float(np.mean([r["rank"] for r in values])) if values else None
        )
    return result


def evaluate_ladder(
    actor,
    output_dir,
    games_per_matchup=8,
    seed=100000,
    device="cpu",
    save_replays=True,
    *,
    deterministic=True,
    checkpoint_path="",
    transition_count=0,
):
    was_training = actor.training
    actor.eval()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    matchups = {
        "policy_vs_random": ["random"] * 3,
        "policy_vs_greedy": ["greedy"] * 3,
        "policy_vs_shortest": ["shortest"] * 3,
        "policy_vs_noble": ["noble"] * 3,
        "policy_vs_blocking": ["blocking"] * 3,
        "mixed_ladder": ["greedy", "shortest", "blocking"],
    }
    actual_games = max(1, games_per_matchup)
    try:
        with torch.no_grad():
            for matchup_index, (matchup, opponents) in enumerate(matchups.items()):
                target = output / matchup
                target.mkdir(exist_ok=True)
                for index in range(actual_games):
                    seat = index % 4
                    game = SplendorGame(4, seed=seed + matchup_index * 100_000 + index)
                    wrapper = CanonicalPaymentWrapper(game)
                    recorder = (
                        EpisodeRecorder(target / "game_0000.json")
                        if save_replays and index == 0
                        else None
                    )
                    if recorder:
                        recorder.attach(game)
                    bot_names = iter(opponents)
                    bots = [
                        None
                        if p == seat
                        else make_bot(
                            next(bot_names),
                            seed + matchup_index * 100_000 + index * 17 + p,
                        )
                        for p in range(4)
                    ]
                    while not game.done:
                        if game.turns_completed >= 300:
                            game.truncate("evaluation_max_turns")
                            break
                        try:
                            action = (
                                actor_action(actor, game, seat, device, deterministic)
                                if game.current_player == seat
                                else bots[game.current_player].act(game)
                            )
                            wrapper.policy_step(action)
                        except NoLegalActionError:
                            game.truncate("evaluation_no_legal_action")
                            break
                    ranking = next(
                        g for g in game.final_ranking() if seat in g["players"]
                    )
                    fractional = (
                        (1 / len(ranking["players"])) if ranking["rank"] == 1 else 0
                    )
                    rows.append(
                        {
                            "matchup": matchup,
                            "game": index,
                            "policy_seat": seat,
                            "rank": ranking["rank"],
                            "score": game.players[seat].score,
                            "fractional_win": fractional,
                            "winner_count": len(game.winner_ids()),
                            "turns": game.turns_completed,
                            "seed": seed + matchup_index * 100_000 + index,
                        }
                    )
                    if recorder:
                        doc = recorder.finalize()
                        doc["game_metadata"].update(
                            {
                                "engine_version": "0.3.2",
                                "rl_version": "0.4.1",
                                "payment_mode": "canonical",
                                "checkpoint": Path(checkpoint_path).name,
                            }
                        )
                        (target / "game_0000.json").write_text(
                            json.dumps(doc, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        export_replay(
                            target / "game_0000.json", target / "game_0000_viewer.html"
                        )
                        (target / "game_0000_turns.txt").write_text(
                            replay_text(doc, turn_only=True), encoding="utf-8"
                        )
                        (target / "game_0000_final_summary.txt").write_text(
                            game.render_final_summary(), encoding="utf-8"
                        )
    finally:
        actor.train(was_training)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matchup_stats = {
        name: _stats([r for r in rows if r["matchup"] == name]) for name in matchups
    }
    aggregate = _stats(rows)
    summary = {
        "rl_version": "0.4.1",
        "engine_version": "0.3.2",
        "checkpoint_path": str(checkpoint_path),
        "transition_count": transition_count,
        "evaluation_seed_base": seed,
        "games_per_matchup_requested": games_per_matchup,
        "games_per_matchup_actual": actual_games,
        "deterministic": deterministic,
        "matchups": matchup_stats,
        "aggregate": aggregate,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(
        "# Fixed-bot evaluation\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in aggregate.items()),
        encoding="utf-8",
    )
    return summary
