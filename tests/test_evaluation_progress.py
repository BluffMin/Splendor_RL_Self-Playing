from splendor_rl.player_count import build_fixed_bot_matchups


def test_evaluation_totals_are_player_count_aware():
    assert len(build_fixed_bot_matchups(2)) * 100 == 500
    assert len(build_fixed_bot_matchups(4)) * 100 == 600
