from __future__ import annotations

from splendor_env import SplendorGame, describe_action
from splendor_env.agents import GreedyAgent


def main() -> None:
    game = SplendorGame(num_players=2, seed=123)
    agent = GreedyAgent()

    while not game.done:
        action = agent.act(game)
        result = game.step(action)
        if result.turn_ended:
            print(
                f"turn={game.turns_completed:03d} actor=P{result.actor} "
                f"action={describe_action(action)} score_delta={result.score_delta}"
            )

    print(game.render())


if __name__ == "__main__":
    main()
