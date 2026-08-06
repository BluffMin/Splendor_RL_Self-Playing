from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from splendor_env.actions import ACTIONS
from splendor_env.agents import GreedyAgent, RandomLegalAgent, ShortestAgent
from splendor_env.core import Phase, SplendorGame
from splendor_env.recording import EpisodeRecorder
from splendor_env.replay import replay_text
from splendor_env.visualization.html_export import export_replay
from splendor_env.wrappers import CanonicalPaymentWrapper

from .distributions import MaskedCategorical


class NobleAgent(GreedyAgent):
    def act(self, game):
        if game.phase == Phase.NOBLE:
            return min(game.legal_actions())
        return super().act(game)


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


def evaluate_ladder(
    actor, output_dir, games_per_matchup=8, seed=100000, device="cpu", save_replays=True
):
    actor.eval()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for matchup in ("random", "greedy", "shortest", "noble", "blocking"):
        target = output / f"policy_vs_{matchup}"
        target.mkdir(exist_ok=True)
        for index in range(games_per_matchup):
            seat = index % 4
            game = SplendorGame(
                4,
                seed=seed
                + index
                + 1000
                * ["random", "greedy", "shortest", "noble", "blocking"].index(matchup),
            )
            wrapper = CanonicalPaymentWrapper(game)
            recorder = (
                EpisodeRecorder(target / "game_0000.json")
                if save_replays and index == 0
                else None
            )
            if recorder:
                recorder.attach(game)
            bots = [make_bot(matchup, seed + index * 17 + p) for p in range(4)]
            while not game.done:
                action = (
                    actor_action(actor, game, seat, device)
                    if game.current_player == seat
                    else bots[game.current_player].act(game)
                )
                wrapper.policy_step(action)
            ranking = next(g for g in game.final_ranking() if seat in g["players"])
            fractional = (1 / len(ranking["players"])) if ranking["rank"] == 1 else 0
            rows.append(
                {
                    "matchup": matchup,
                    "game": index,
                    "policy_seat": seat,
                    "rank": ranking["rank"],
                    "score": game.players[seat].score,
                    "fractional_win": fractional,
                    "turns": game.turns_completed,
                }
            )
            if recorder:
                doc = recorder.finalize()
                doc["game_metadata"].update(
                    {
                        "engine_version": "0.3.2",
                        "rl_version": "0.4.0",
                        "payment_mode": "canonical",
                    }
                )
                (target / "game_0000.json").write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
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
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "games": len(rows),
        "average_rank": float(np.mean([r["rank"] for r in rows])),
        "first_place_fractional_rate": float(
            np.mean([r["fractional_win"] for r in rows])
        ),
        "average_score": float(np.mean([r["score"] for r in rows])),
        "by_seat": {
            str(s): {
                "average_rank": float(
                    np.mean([r["rank"] for r in rows if r["policy_seat"] == s])
                ),
                "average_score": float(
                    np.mean([r["score"] for r in rows if r["policy_seat"] == s])
                ),
            }
            for s in range(4)
        },
    }
    (output / "report.md").write_text(
        "# Fixed-bot evaluation\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in summary.items()),
        encoding="utf-8",
    )
    return summary
