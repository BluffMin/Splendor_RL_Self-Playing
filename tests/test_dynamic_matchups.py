import pytest

from splendor_rl.player_count import build_fixed_bot_matchups, validate_num_players


def test_matchups_follow_player_count():
    for players in (2, 3, 4):
        matchups = build_fixed_bot_matchups(players)
        assert all(len(v) == players - 1 for v in matchups.values())
    assert "mixed_ladder" not in build_fixed_bot_matchups(2)
    assert (
        "mixed_ladder" in build_fixed_bot_matchups(3)
        and len(build_fixed_bot_matchups(4)) == 6
    )
    with pytest.raises(ValueError):
        validate_num_players(1)
