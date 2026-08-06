from __future__ import annotations

from pathlib import Path

from splendor_env.agents import GreedyAgent
from splendor_env.core import SplendorGame
from splendor_env.visualization import export_game_view


def main() -> None:
    game = SplendorGame(2, seed=7)
    agent = GreedyAgent()
    output = Path("runs/live/current_board.html")
    for _ in range(10):
        export_game_view(game, output, perspective="omniscient")
        if game.done:
            break
        game.step(agent.act(game))
    print(f"latest board: {output.resolve()}")


if __name__ == "__main__":
    main()
