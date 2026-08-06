from __future__ import annotations

import pytest

from splendor_env.core import SplendorGame
from splendor_env.visualization.view_model import (
    board_view_from_game,
    board_view_from_snapshot,
)


@pytest.mark.parametrize("players", [2, 3, 4])
def test_player_and_market_layout(players: int) -> None:
    game = SplendorGame(players, seed=players)
    view = board_view_from_game(game)
    assert len(view.players) == players
    assert [p.player_id for p in view.players] == list(range(players))
    assert sum(p.is_current for p in view.players) == 1
    assert len(view.market_cards) == 12
    assert len(view.nobles) == players + 1


def test_empty_market_and_snapshot_conversion() -> None:
    game = SplendorGame(2, seed=3)
    game.decks[0].clear()
    game.visible[0][0] = None
    view = board_view_from_game(game)
    assert view.market_cards[0].card_id is None
    snapshot_view = board_view_from_snapshot(game.to_state_dict())
    assert snapshot_view.market_cards[0].card_id is None
    assert snapshot_view.num_players == 2


def test_table_and_egocentric_order_keep_seat_labels() -> None:
    game = SplendorGame(3, seed=4)
    game.current_player = 2
    table = board_view_from_game(game, perspective=1, layout="table")
    ego = board_view_from_game(game, perspective=1, layout="egocentric")
    assert [p.player_id for p in table.players] == [0, 1, 2]
    assert [p.player_id for p in ego.players] == [1, 2, 0]
    assert [p.display_name for p in ego.players] == ["Player 1", "Player 2", "Player 0"]


def test_terminal_and_joint_winners() -> None:
    game = SplendorGame(2, seed=5)
    game.players[0].score = game.players[1].score = 15
    game.truncate("visual-test")
    view = board_view_from_game(game, perspective="omniscient")
    assert view.is_terminal
    assert view.winner_ids == (0, 1)
