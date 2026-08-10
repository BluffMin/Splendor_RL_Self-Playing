from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splendor_rl.league.pool import OpponentPool
from splendor_rl.league.promotion import play_actor_game


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="population_game.json")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    state = json.loads((run / "population_state.json").read_text(encoding="utf-8"))
    checkpoint = __import__("torch").load(
        run / state["learners"]["main"]["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    pool = OpponentPool(run / "population/pool", checkpoint["config"]["hidden_sizes"])
    left = args.left or state["main"]["champion_id"]
    right = args.right or state["meta_strategy"]["policy_ids"][0]
    score, game = play_actor_game(
        pool.load(left),
        pool.load(right),
        seed=args.seed,
        replay_path=run / args.output,
        replay_metadata={
            "population_role": "evaluation",
            "left_policy_id": left,
            "right_policy_id": right,
        },
    )
    print(
        json.dumps(
            {
                "left": left,
                "right": right,
                "left_score": score,
                "final_round": game.round_id,
            }
        )
    )


if __name__ == "__main__":
    main()
