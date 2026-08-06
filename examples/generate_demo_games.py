from __future__ import annotations

import argparse
from pathlib import Path

from splendor_env.agents import GreedyAgent, RandomLegalAgent
from splendor_env.core import SplendorGame
from splendor_env.recording import EpisodeRecorder, append_games_summary_csv
from splendor_env.replay import replay_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output-dir", default="runs/demo_games")
    parser.add_argument(
        "--record-level", choices=("summary", "actions", "full"), default="full"
    )
    parser.add_argument("--agents", nargs="*", choices=("greedy", "random"))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_csv = output / "games_summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    names = args.agents or ["greedy", "greedy"] + ["random"] * (args.players - 2)
    if len(names) != args.players:
        parser.error("--agents must provide exactly --players entries")

    for game_index in range(args.games):
        seed = args.seed + game_index
        game = SplendorGame(args.players, seed=seed)
        agents = [
            GreedyAgent()
            if name == "greedy"
            else RandomLegalAgent(seed + i * 1000, avoid_deadlock=True)
            for i, name in enumerate(names)
        ]
        json_path = output / f"game_{game_index:04d}.json"
        recorder = EpisodeRecorder(json_path, args.record_level, omniscient=True)
        recorder.attach(game)
        while not game.done:
            if game.decision_id >= 20_000:
                raise RuntimeError(
                    "demo exceeded 20,000 decisions without official end"
                )
            game.step(agents[game.current_player].act(game))
        document = recorder.finalize()
        (output / f"game_{game_index:04d}_final_summary.txt").write_text(
            game.render_final_summary(), encoding="utf-8"
        )
        append_games_summary_csv(summary_csv, f"game_{game_index:04d}", seed, game)
        if game_index == 0:
            (output / "game_0000_replay.txt").write_text(
                replay_text(document, omniscient=True, turn_only=True), encoding="utf-8"
            )
        print(
            f"game_{game_index:04d}: turns={game.turns_completed} "
            f"decisions={game.decision_id} winners={game.winner_ids()} "
            f"scores={[p.score for p in game.players]}"
        )


if __name__ == "__main__":
    main()
