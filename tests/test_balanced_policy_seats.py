from collections import Counter

from splendor_rl.player_count import balanced_policy_seats


def test_balanced_seats_for_two_to_four_players():
    assert Counter(balanced_policy_seats(2, 100)) == {0: 50, 1: 50}
    assert Counter(balanced_policy_seats(2, 101)) == {0: 51, 1: 50}
    for players in (3, 4):
        counts = Counter(balanced_policy_seats(players, 101)).values()
        assert max(counts) - min(counts) <= 1
