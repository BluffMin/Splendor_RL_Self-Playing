from __future__ import annotations

from splendor_env.core import SplendorGame
from splendor_env.visualization.view_model import board_view_from_game


def test_no_empty_player_panels_for_all_counts() -> None:
    for count in (2, 3, 4):
        view = board_view_from_game(SplendorGame(count, seed=count))
        assert len(view.players) == count
        assert {player.player_id for player in view.players} == set(range(count))
