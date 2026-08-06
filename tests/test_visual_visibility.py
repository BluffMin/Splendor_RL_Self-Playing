from __future__ import annotations

from splendor_env.actions import action_id
from splendor_env.core import SplendorGame
from splendor_env.visualization.view_model import board_view_from_game


def test_private_reservation_visibility() -> None:
    game = SplendorGame(2, seed=6)
    game.step(action_id("reserve_deck", 1))
    card_id = game.players[0].reserved[0].card.card_id
    own = board_view_from_game(game, perspective=0).players[0].reserved_cards[0]
    opponent = board_view_from_game(game, perspective=1).players[0].reserved_cards[0]
    omniscient = (
        board_view_from_game(game, perspective="omniscient")
        .players[0]
        .reserved_cards[0]
    )
    assert own.card_id == omniscient.card_id == card_id
    assert opponent.tier == 2 and opponent.reservation_origin == "deck"
    assert (
        opponent.card_id
        is opponent.cost
        is opponent.points
        is opponent.bonus_color
        is None
    )


def test_visible_reservation_is_public() -> None:
    game = SplendorGame(2, seed=7)
    game.step(action_id("reserve_visible", 0))
    card = board_view_from_game(game, perspective=1).players[0].reserved_cards[0]
    assert card.visible and card.card_id and card.cost is not None
    assert card.reservation_origin == "visible"
