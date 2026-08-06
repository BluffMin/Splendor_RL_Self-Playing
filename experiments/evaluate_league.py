from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splendor_rl.league.matrix import build_matchup_matrix
from splendor_rl.league.pool import OpponentPool
from splendor_rl.league.promotion import actor_vs_bot_score
from splendor_rl.league.state import load_league_state
from splendor_rl.league.train import _frozen_actor
from splendor_rl.models import SharedActor
from splendor_rl.progress import ProgressConfig, ProgressMode, make_evaluation_progress


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a two-player PPO league")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--games-per-matchup", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--progress", choices=("auto", "always", "never"), default="auto"
    )
    parser.add_argument("--progress-refresh-seconds", type=float, default=1.0)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--champion", action="store_true")
    parser.add_argument("--hall-of-fame", action="store_true")
    parser.add_argument("--matchup-matrix", action="store_true")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    state = load_league_state(run / "league_state.json")
    if state is None:
        raise ValueError("league_state.json is missing")
    checkpoint = torch.load(
        run / state["candidate"]["latest_checkpoint"],
        map_location=args.device,
        weights_only=False,
    )
    sizes = checkpoint["observation_sizes"]
    hidden = checkpoint["config"]["hidden_sizes"]
    actor = SharedActor(sizes["actor"], sizes["action"], hidden).to(args.device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    pool = OpponentPool(run / "opponent_pool", hidden, args.device)
    policies = {
        "candidate": _frozen_actor(
            actor,
            "candidate",
            state["candidate"]["global_transition_count"],
            args.device,
        )
    }
    if args.champion or not any(
        (args.candidate, args.champion, args.hall_of_fame, args.matchup_matrix)
    ):
        policies[state["champion"]["opponent_id"]] = pool.load(
            state["champion"]["opponent_id"]
        )
    if args.hall_of_fame:
        policies.update(
            {
                opponent_id: pool.load(opponent_id)
                for opponent_id in pool.hall_of_fame_ids
            }
        )
    progress_config = ProgressConfig(
        ProgressMode(args.progress), args.progress_refresh_seconds
    )
    evaluation_progress = make_evaluation_progress(
        len(policies) * 5 * args.games_per_matchup,
        state["candidate"]["global_transition_count"],
        progress_config,
    )
    results = {
        policy_id: {
            bot: actor_vs_bot_score(
                policy,
                bot,
                games=args.games_per_matchup,
                seed_base=1_000_000 + index * 100_000,
                progress=evaluation_progress,
                label=f"{policy_id}_vs_{bot}",
            )
            for index, bot in enumerate(
                ("random", "greedy", "shortest", "noble", "blocking")
            )
        }
        for policy_id, policy in policies.items()
    }
    evaluation_progress.close()
    output = run / "final_league_evaluation"
    output.mkdir(exist_ok=True)
    if args.matchup_matrix or len(policies) > 1:
        pair_total = len(policies) * (len(policies) - 1) // 2
        matrix_progress = make_evaluation_progress(
            pair_total * 2 * max(1, (args.games_per_matchup + 1) // 2),
            state["candidate"]["global_transition_count"],
            progress_config,
        )
        results["matchup_matrix"] = build_matchup_matrix(
            policies,
            games_per_pair=args.games_per_matchup,
            seed_base=2_000_000,
            output_dir=output,
            progress=matrix_progress,
        )
        matrix_progress.close()
    (output / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
