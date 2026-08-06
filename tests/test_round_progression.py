from splendor_env.core import SplendorGame


def test_round_is_seat_cycle():
    game = SplendorGame(4, seed=2)
    while game.turns_completed < 5:
        game.step(game.legal_actions()[0])
    assert game.round_id == 1
