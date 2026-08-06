from splendor_rl.rollout import PlayerTransition


def test_transition_carries_actor_and_turn_identity():
    t = PlayerTransition(
        None, None, None, 1, 0, 0, 0, 1, False, 2, 3, 4, 5, 1, "discard"
    )
    assert (t.env_id, t.player_id, t.player_turn_id, t.phase) == (2, 3, 5, "discard")
