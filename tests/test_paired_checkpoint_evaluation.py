from splendor_rl.league.promotion import paired_actor_evaluation


def test_paired_result_exposes_seat_and_round_metrics(monkeypatch):
    class Game:
        round_id = 10
        turns_completed = 19
        decision_id = 21

    values = iter([(1.0, Game()), (0.0, Game())])
    monkeypatch.setattr(
        "splendor_rl.league.promotion.play_actor_game",
        lambda *args, **kwargs: next(values),
    )
    result = paired_actor_evaluation(object(), object(), pair_count=1, seed_base=7)
    assert result["pair_scores"] == [1.0]
    assert result["raw_candidate_p0_score"] == result["raw_candidate_p1_score"] == 1.0
    assert result["average_final_round"] == 10
