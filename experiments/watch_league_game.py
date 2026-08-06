from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splendor_env.core import SplendorGame
from splendor_env.wrappers import CanonicalPaymentWrapper
from splendor_rl.league.pool import OpponentPool
from splendor_rl.league.state import load_league_state
from splendor_rl.league.train import _frozen_actor
from splendor_rl.models import SharedActor


def main(argv=None):
    parser = argparse.ArgumentParser(description="Watch a Candidate league game")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--left", default="candidate")
    parser.add_argument("--right", default="champion")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--candidate-seat", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--perspective", choices=("omniscient", "candidate"), default="omniscient"
    )
    parser.add_argument("--step-mode", choices=("decision", "turn"), default="turn")
    parser.add_argument("--show-policy", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    state = load_league_state(run / "league_state.json")
    checkpoint = torch.load(
        run / state["candidate"]["latest_checkpoint"],
        map_location=args.device,
        weights_only=False,
    )
    sizes = checkpoint["observation_sizes"]
    hidden = checkpoint["config"]["hidden_sizes"]
    actor = SharedActor(sizes["actor"], sizes["action"], hidden).to(args.device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    candidate = _frozen_actor(
        actor, "candidate", state["candidate"]["global_transition_count"], args.device
    )
    pool = OpponentPool(run / "opponent_pool", hidden, args.device)
    right_id = (
        state["champion"]["opponent_id"] if args.right == "champion" else args.right
    )
    opponent = pool.load(right_id)
    policies = (
        [candidate, opponent] if args.candidate_seat == 0 else [opponent, candidate]
    )
    game = SplendorGame(2, seed=args.seed)
    wrapper = CanonicalPaymentWrapper(game)
    last_turn = -1
    while not game.done:
        player = game.current_player
        mask = game.legal_action_mask().astype(bool)
        action = policies[player].act(
            game.observation(player, omniscient=False), mask, deterministic=True
        )
        if args.show_policy:
            print(
                f"P{player} policy={policies[player].metadata.opponent_id} selected_action={action} (policy probability, not win rate)"
            )
        wrapper.policy_step(action)
        if args.step_mode == "decision" or game.turns_completed != last_turn:
            print(
                game.render(
                    args.candidate_seat, omniscient=args.perspective == "omniscient"
                )
            )
            last_turn = game.turns_completed
    print(game.render_final_summary())


if __name__ == "__main__":
    main()
