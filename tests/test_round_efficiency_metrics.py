from splendor_rl.league.promotion import actor_vs_bot_score


def test_round_metrics_are_distinct_from_turns(monkeypatch):
    class Actor:
        def act(self, observation, mask, **kwargs):
            return int(mask.nonzero()[0][0])

    result = actor_vs_bot_score(Actor(), "random", games=2, seed_base=91)
    assert result["average_final_round"] > 0
    assert result["average_player_turns"] >= result["average_final_round"]
    assert result["average_decisions"] >= result["average_player_turns"]
