from splendor_env.core import SplendorGame


def test_initial_and_completed_turn_semantics():
    game = SplendorGame(4, seed=1)
    events = []
    game.add_event_listener(events.append)
    assert (
        game.decision_id,
        game.player_turn_id,
        game.round_id,
        game.current_player,
    ) == (0, 0, 0, 0)
    for actor in range(4):
        while game.current_player == actor:
            game.step(game.legal_actions()[0])
    assert [
        (e["player_turn_id"], e["round_id"], e["acting_player"])
        for e in events
        if e["turn_completed"]
    ] == [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3)]
    assert game.player_turn_id == 4 and game.round_id == 1
