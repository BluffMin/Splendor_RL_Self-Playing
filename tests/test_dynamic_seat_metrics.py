from splendor_rl.evaluation import _stats


def test_two_player_stats_have_no_padded_seats():
    rows = [
        {
            "rank": 1,
            "fractional_win": 1,
            "score": 15,
            "winner_count": 1,
            "turns": 30,
            "policy_seat": 0,
        },
        {
            "rank": 2,
            "fractional_win": 0,
            "score": 10,
            "winner_count": 1,
            "turns": 30,
            "policy_seat": 1,
        },
    ]
    stats = _stats(rows, 2)
    assert set(stats["seat_metrics"]) == {"0", "1"}
    assert not any(
        key.startswith(("seat_2", "seat_3")) for key in stats
    )
