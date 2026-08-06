from __future__ import annotations

from splendor_env.actions import action_id
from splendor_env.core import SplendorGame


def main() -> None:
    game = SplendorGame(num_players=2, seed=7)

    game.step(action_id("reserve_visible", 0))
    game.step(action_id("reserve_deck", 1))

    print("=== P0 perspective ===")
    print(game.render(perspective=0))

    print("\n=== P1 perspective ===")
    print(game.render(perspective=1))

    print("\n=== Omniscient debug perspective ===")
    print(game.render(perspective=0, omniscient=True))


if __name__ == "__main__":
    main()
