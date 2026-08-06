from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splendor_env.core import SplendorGame
from splendor_env.wrappers import CanonicalPaymentWrapper
from splendor_rl.checkpoint import load_checkpoint
from splendor_rl.evaluation import actor_action, make_bot
from splendor_rl.models import SharedActor
from splendor_rl.player_count import validate_num_players


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--opponents", nargs="*", default=["greedy", "shortest", "blocking"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--perspective", default="omniscient")
    p.add_argument("--manual", action="store_true")
    p.add_argument("--step-mode", choices=("decision", "turn"), default="turn")
    p.add_argument("--show-policy", action="store_true")
    p.add_argument("--top-k-actions", type=int, default=5)
    a = p.parse_args()
    validate_num_players(a.players)
    if len(a.opponents) != a.players - 1:
        p.error(
            f"--opponents requires exactly {a.players - 1} names for {a.players} players"
        )
    data = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_players = data.get(
        "num_players", data.get("config", {}).get("num_players")
    )
    if checkpoint_players != a.players:
        p.error(
            f"checkpoint num_players={checkpoint_players} does not match --players {a.players}"
        )
    sizes = data["observation_sizes"]
    actor = SharedActor(sizes["actor"], sizes["action"], data["config"]["hidden_sizes"])
    load_checkpoint(a.checkpoint, actor, expected_sizes=sizes, restore_rng=False)
    actor.eval()
    game = SplendorGame(a.players, seed=a.seed)
    wrapper = CanonicalPaymentWrapper(game)
    bots = [None] + [
        make_bot(a.opponents[i - 1], a.seed + i) for i in range(1, a.players)
    ]
    while not game.done:
        before = game.turns_completed
        action = (
            actor_action(actor, game, 0)
            if game.current_player == 0
            else bots[game.current_player].act(game)
        )
        wrapper.policy_step(action)
        if a.step_mode == "decision" or game.turns_completed > before:
            print(game.render(0, omniscient=a.perspective == "omniscient"))
        if a.manual:
            input("Enter for next...")
    print(game.render_final_summary())


if __name__ == "__main__":
    main()
