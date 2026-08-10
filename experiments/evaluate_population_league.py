from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splendor_rl.league.pool import OpponentPool
from splendor_rl.league.promotion import actor_vs_bot_score


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    state = json.loads((run / "population_state.json").read_text(encoding="utf-8"))
    checkpoint = __import__("torch").load(
        run / state["learners"]["main"]["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    pool = OpponentPool(
        run / "population/pool", checkpoint["config"]["hidden_sizes"], args.device
    )
    champion = pool.load(state["main"]["champion_id"])
    results = {
        bot: actor_vs_bot_score(
            champion, bot, games=args.games, seed_base=60_000_000 + i * 100_000
        )
        for i, bot in enumerate(("random", "shortest", "greedy", "noble", "blocking"))
    }
    output = run / "evaluations/final_fixed_bots.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
