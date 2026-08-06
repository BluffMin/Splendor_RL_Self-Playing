from __future__ import annotations

import argparse
from pathlib import Path

from splendor_env.agents import GreedyAgent, RandomLegalAgent, ShortestAgent
from splendor_env.core import SplendorGame
from splendor_env.recording import EpisodeRecorder
from splendor_env.visualization import export_game_view
from splendor_env.visualization.html_export import export_replay


def make_agent(name: str, seed: int):
    if name == "greedy":
        return GreedyAgent()
    if name == "shortest":
        return ShortestAgent()
    return RandomLegalAgent(seed, avoid_deadlock=True)


def export_one(players: int, names: list[str], seed: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    game = SplendorGame(players, seed=seed)
    agents = [make_agent(name, seed + index * 1000) for index, name in enumerate(names)]
    recorder = EpisodeRecorder(output / "game.json", "full", omniscient=True)
    recorder.attach(game)
    while not game.done:
        game.step(agents[game.current_player].act(game))
    recorder.finalize()
    export_replay(output / "game.json", output / "game_viewer.html")
    export_game_view(game, output / "final_board.html", perspective="omniscient")
    (output / "final_summary.txt").write_text(
        game.render_final_summary(), encoding="utf-8"
    )
    print(f"{players}p: {output / 'game_viewer.html'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--players", type=int, nargs="+", required=True, choices=(2, 3, 4)
    )
    parser.add_argument("--agents", nargs="*", choices=("greedy", "shortest", "random"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for players in args.players:
        names = args.agents or (["greedy", "shortest"] + ["random"] * (players - 2))
        if len(names) != players:
            parser.error("--agents must match each requested player count")
        target = (
            Path(args.output_dir)
            if len(args.players) == 1
            else Path(args.output_dir) / f"{players}p"
        )
        export_one(players, names, args.seed, target)


if __name__ == "__main__":
    main()
